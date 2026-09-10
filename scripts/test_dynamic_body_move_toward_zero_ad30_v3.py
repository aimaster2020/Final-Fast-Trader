from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median

EPS = 1e-12
DEFAULT_LOOKBACK = 50
DEFAULT_MIN_HISTORY = 15
DEFAULT_CAPITAL = 1000.0
DEFAULT_COMMISSION = 0.0013


@dataclass
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float


def load(path: Path) -> list[Candle]:
    out: list[Candle] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                out.append(Candle(str(r.get("timestamp", "")), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def body(c: Candle) -> float:
    return c.close - c.open


def ad(c: Candle) -> int:
    b = body(c)
    hc = c.high - c.close
    ho = c.high - c.open
    lc = c.low - c.close
    return int(hc > b) + int(ho < hc) + int(lc > b)


def move_to_zero_ratio(cur: float, nxt: float) -> float | None:
    if abs(cur) <= EPS:
        return None
    return math.copysign(1.0, cur) * (cur - nxt) / abs(cur)


def predict_ratio(rows: list[Candle], i: int, lookback: int, min_history: int) -> tuple[float | None, int]:
    score = ad(rows[i])
    values: list[float] = []
    start = max(1, i - lookback)
    for j in range(start, i):
        if ad(rows[j]) != score:
            continue
        x = move_to_zero_ratio(body(rows[j]), body(rows[j + 1]))
        if x is not None and math.isfinite(x):
            values.append(x)
    if len(values) < min_history:
        return None, len(values)
    return median(values), len(values)


def gross_return(side: str, entry: float, exit_price: float) -> float:
    if side == "LONG":
        return exit_price / entry - 1.0
    return 1.0 - exit_price / entry


def net_return(side: str, entry: float, exit_price: float, commission: float) -> float:
    # Exact fee-on-notional model: pay commission on entry notional and exit proceeds.
    # For 100% capital, buy/sell factors are represented explicitly rather than by
    # subtracting a fixed return from the trade.
    if side == "LONG":
        qty = 1.0 / (entry * (1.0 + commission))
        cash_after_exit = qty * exit_price * (1.0 - commission)
        return cash_after_exit - 1.0
    # Short: approximate symmetric fee model on entry and exit notionals.
    # The strategy is a research backtest; short PnL is represented as gross return
    # with the same two-side fee drag applied multiplicatively.
    gross = gross_return(side, entry, exit_price)
    return (1.0 + gross) * (1.0 - commission) * (1.0 - commission) - 1.0


def evaluate(
    rows: list[Candle],
    symbol: str,
    ad_filter: int | None,
    min_expected_move: float,
    lookback: int,
    min_history: int,
    capital0: float,
    commission: float,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    capital = capital0
    trades: list[dict[str, object]] = []
    candidates = signals = eligible = history_skips = predicted_direction = 0
    gross_sum = net_sum = 0.0
    gross_wins = net_wins = 0
    gross_profit = gross_loss = net_profit = net_loss = 0.0
    peak = capital0
    max_dd = 0.0
    predicted_edges: list[float] = []
    actual_edges: list[float] = []

    for i in range(1, len(rows) - 1):
        score = ad(rows[i])
        if score not in (0, 3) or (ad_filter is not None and score != ad_filter):
            continue
        candidates += 1

        pred_ratio, hist_n = predict_ratio(rows, i, lookback, min_history)
        if pred_ratio is None:
            history_skips += 1
            continue
        signals += 1

        cur_body = body(rows[i])
        pred_next_body = cur_body - math.copysign(abs(cur_body) * pred_ratio, cur_body)
        if abs(pred_next_body) <= EPS:
            continue

        side = "LONG" if pred_next_body > 0 else "SHORT"
        expected_move = abs(pred_next_body) / rows[i].close
        predicted_edges.append(expected_move)
        actual_next_body = body(rows[i + 1])
        actual_edges.append(math.copysign(1.0, actual_next_body) * actual_next_body / rows[i].close if abs(actual_next_body) > EPS else 0.0)
        if (pred_next_body > 0) == (actual_next_body > 0):
            predicted_direction += 1

        if expected_move < min_expected_move:
            continue
        eligible += 1

        entry = rows[i].close
        exit_price = rows[i + 1].close
        if entry <= EPS:
            continue

        g = gross_return(side, entry, exit_price)
        n = net_return(side, entry, exit_price, commission)
        before = capital
        after = before * (1.0 + n)
        capital = after

        gross_sum += g
        net_sum += n
        gross_wins += int(g > 0)
        net_wins += int(n > 0)
        if g > 0:
            gross_profit += g
        elif g < 0:
            gross_loss += g
        if n > 0:
            net_profit += n
        elif n < 0:
            net_loss += n
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak if peak > EPS else 0.0)

        trades.append({
            "signal_index": i,
            "entry_timestamp": rows[i].timestamp,
            "exit_timestamp": rows[i + 1].timestamp,
            "side": side,
            "ad": score,
            "current_body": cur_body,
            "predicted_move_ratio": pred_ratio,
            "predicted_next_body": pred_next_body,
            "expected_move_pct": expected_move * 100.0,
            "entry_price": entry,
            "exit_price": exit_price,
            "actual_next_body": actual_next_body,
            "actual_price_move_pct": ((exit_price / entry) - 1.0) * (1.0 if side == "LONG" else -1.0) * 100.0,
            "gross_return": g,
            "net_return": n,
            "capital_before": before,
            "capital_after": after,
        })

    count = len(trades)
    return {
        "candidates": candidates,
        "signals": signals,
        "eligible": eligible,
        "trades": count,
        "gross_wr": gross_wins / count * 100.0 if count else 0.0,
        "net_wr": net_wins / count * 100.0 if count else 0.0,
        "gross_avg": gross_sum / count * 100.0 if count else 0.0,
        "net_avg": net_sum / count * 100.0 if count else 0.0,
        "return": capital / capital0 - 1.0,
        "final": capital,
        "gross_pf": gross_profit / abs(gross_loss) if gross_loss < -EPS else math.inf,
        "net_pf": net_profit / abs(net_loss) if net_loss < -EPS else math.inf,
        "dd": max_dd,
        "pred_dir": predicted_direction / signals * 100.0 if signals else 0.0,
        "pred_edge_med": median(predicted_edges) * 100.0 if predicted_edges else 0.0,
        "actual_signed_edge_med": median(actual_edges) * 100.0 if actual_edges else 0.0,
        "history_skips": history_skips,
    }, trades


def write_log(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "signal_index", "entry_timestamp", "exit_timestamp", "side", "ad",
        "current_body", "predicted_move_ratio", "predicted_next_body", "expected_move_pct",
        "entry_price", "exit_price", "actual_next_body", "actual_price_move_pct",
        "gross_return", "net_return", "capital_before", "capital_after",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def pf(x: float) -> str:
    return "INF" if math.isinf(x) else f"{x:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Move-to-zero strategy with tradable expected-price-move thresholds")
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    ap.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    ap.add_argument("--thresholds", default="0.0005,0.0010,0.0015,0.0020,0.0025,0.0030,0.0040,0.0050,0.0075,0.0100")
    ap.add_argument("--log-dir", default="reports/move_to_zero_trades_ad30_v3")
    args = ap.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    root = Path(args.input_dir)
    log_root = Path(args.log_dir)

    print(f"MODEL=causal rolling median move-to-zero | LB={args.lookback} MINH={args.min_history} CAPITAL={args.capital:.2f}")
    print(f"FEE=0 and {args.commission*100:.3f}%/side | fee break-even ~= {2*args.commission*100:.3f}% round-trip")
    print("ENTRY=current close EXIT=next close | AD3=LONG AD0=SHORT | 100% capital | one-bar hold")
    print("NEW FILTER=abs(predicted_next_body)/current_close; threshold is expected tradable price move")
    print()

    for symbol in symbols:
        rows = load(root / f"{symbol}_1h.csv")
        print(f"=== {symbol} rows={len(rows)} ===")
        for score_filter, label in ((3, "AD3"), (0, "AD0"), (None, "ALL")):
            print(f"[{label}]")
            for threshold in thresholds:
                r0, log0 = evaluate(rows, symbol, score_filter, threshold, args.lookback, args.min_history, args.capital, 0.0)
                rf, logf = evaluate(rows, symbol, score_filter, threshold, args.lookback, args.min_history, args.capital, args.commission)
                print(
                    f"T={threshold*100:.2f}% C={int(r0['candidates'])} SIG={int(r0['signals'])} ELIG={int(r0['eligible'])} "
                    f"| F0 TR={int(r0['trades'])} WR={r0['gross_wr']:.1f}% RET={r0['return']*100:.2f}% FIN={r0['final']:.2f} PF={pf(float(r0['gross_pf']))} DD={r0['dd']*100:.2f}% "
                    f"| Ffee TR={int(rf['trades'])} WR={rf['net_wr']:.1f}% RET={rf['return']*100:.2f}% FIN={rf['final']:.2f} PF={pf(float(rf['net_pf']))} DD={rf['dd']*100:.2f}% "
                    f"| PDIR={r0['pred_dir']:.1f}% PMED={r0['pred_edge_med']:.3f}% AMED={r0['actual_signed_edge_med']:.3f}%"
                )
                write_log(log_root / f"{symbol}_{label}_T{threshold:.4f}_F0.csv", log0)
                write_log(log_root / f"{symbol}_{label}_T{threshold:.4f}_F0_13.csv", logf)
            print()


if __name__ == "__main__":
    main()
