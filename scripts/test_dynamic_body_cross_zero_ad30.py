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
DEFAULT_THRESHOLDS = "0.00,0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50"


@dataclass
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float


@dataclass
class Trade:
    index: int
    entry_timestamp: str
    exit_timestamp: str
    side: str
    ad: int
    body: float
    pred_ratio: float
    pred_next_body: float
    pred_next_body_pct: float
    actual_next_body: float
    entry: float
    exit: float
    gross_return: float
    fee_return: float
    net_return: float
    capital_before: float
    capital_after: float


def load(path: Path) -> list[Candle]:
    rows: list[Candle] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append(Candle(str(r.get("timestamp", "")), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


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


def rolling_prediction(rows: list[Candle], i: int, lookback: int, min_history: int) -> tuple[float | None, int, float | None]:
    score = ad(rows[i])
    vals: list[float] = []
    start = max(1, i - lookback)
    for j in range(start, i):
        if ad(rows[j]) != score:
            continue
        r = move_to_zero_ratio(body(rows[j]), body(rows[j + 1]))
        if r is not None and math.isfinite(r):
            vals.append(r)
    if len(vals) < min_history:
        return None, len(vals), None
    pred_ratio = median(vals)
    b = body(rows[i])
    pred_next = b - math.copysign(abs(b) * pred_ratio, b)
    return pred_ratio, len(vals), pred_next


def trade_return(side: str, entry: float, exit_price: float, fee: float) -> tuple[float, float]:
    if side == "LONG":
        gross = exit_price / entry - 1.0
    else:
        gross = 1.0 - exit_price / entry
    return gross, gross - 2.0 * fee


def run(rows: list[Candle], threshold: float, fee: float, lookback: int, minh: int, capital0: float, ad_filter: int | None, side_filter: str | None):
    capital = capital0
    trades: list[Trade] = []
    candidates = signals = eligible = hist_skips = 0
    correct_direction = 0
    pred_pct: list[float] = []
    actual_pct: list[float] = []

    for i in range(1, len(rows) - 1):
        score = ad(rows[i])
        if score not in (0, 3):
            continue
        if ad_filter is not None and score != ad_filter:
            continue
        candidates += 1

        pred_ratio, hist_n, pred_next = rolling_prediction(rows, i, lookback, minh)
        if pred_ratio is None or pred_next is None:
            hist_skips += 1
            continue
        signals += 1

        entry = rows[i].close
        if entry <= EPS:
            continue
        pred_pct_value = abs(pred_next) / entry
        actual_body = body(rows[i + 1])
        actual_ratio = move_to_zero_ratio(body(rows[i]), actual_body)
        if actual_ratio is not None:
            pred_pct.append(pred_pct_value)
            actual_pct.append(abs(actual_body) / entry)

        if pred_next > 0:
            side = "LONG"
        elif pred_next < 0:
            side = "SHORT"
        else:
            continue
        if side_filter is not None and side != side_filter:
            continue
        if pred_pct_value < threshold:
            continue
        eligible += 1

        exit_price = rows[i + 1].close
        gross, net = trade_return(side, entry, exit_price, fee)
        before = capital
        after = before * (1.0 + net)
        correct_direction += int((side == "LONG") == (exit_price > entry))
        trades.append(Trade(i, rows[i].timestamp, rows[i + 1].timestamp, side, score, body(rows[i]), pred_ratio, pred_next, pred_pct_value, actual_body, entry, exit_price, gross, 2.0 * fee, net, before, after))
        capital = after

    wins = sum(1 for t in trades if t.net_return > 0)
    gross_profit = sum(t.gross_return for t in trades if t.gross_return > 0)
    gross_loss = sum(t.gross_return for t in trades if t.gross_return < 0)
    net_profit = sum(t.net_return for t in trades if t.net_return > 0)
    net_loss = sum(t.net_return for t in trades if t.net_return < 0)
    peak = capital0
    dd = 0.0
    path = capital0
    for t in trades:
        path = t.capital_after
        peak = max(peak, path)
        dd = max(dd, (peak - path) / peak if peak > EPS else 0.0)

    return {
        "candidates": candidates,
        "signals": signals,
        "eligible": eligible,
        "trades": len(trades),
        "wins": wins,
        "wr": wins / len(trades) * 100.0 if trades else 0.0,
        "direction_acc": correct_direction / len(trades) * 100.0 if trades else 0.0,
        "return": capital / capital0 - 1.0,
        "final": capital,
        "pf": net_profit / abs(net_loss) if net_loss < -EPS else math.inf,
        "dd": dd,
        "hist_skips": hist_skips,
        "pred_pct_med": median(pred_pct) if pred_pct else 0.0,
        "actual_body_pct_med": median(actual_pct) if actual_pct else 0.0,
        "trades_data": trades,
    }


def write_csv(path: Path, trades: list[Trade]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "entry_timestamp", "exit_timestamp", "side", "ad", "body", "pred_ratio", "pred_next_body", "pred_next_body_pct", "actual_next_body", "entry", "exit", "gross_return", "fee_return", "net_return", "capital_before", "capital_after"])
        for t in trades:
            w.writerow([t.index, t.entry_timestamp, t.exit_timestamp, t.side, t.ad, f"{t.body:.12g}", f"{t.pred_ratio:.12g}", f"{t.pred_next_body:.12g}", f"{t.pred_next_body_pct:.12g}", f"{t.actual_next_body:.12g}", f"{t.entry:.12g}", f"{t.exit:.12g}", f"{t.gross_return:.12g}", f"{t.fee_return:.12g}", f"{t.net_return:.12g}", f"{t.capital_before:.12g}", f"{t.capital_after:.12g}"])


def pf(x: float) -> str:
    return "INF" if math.isinf(x) else f"{x:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-zero body prediction strategy test")
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    ap.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--log-dir", default="reports/cross_zero_trades_ad30")
    args = ap.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    root = Path(args.input_dir)
    log_root = Path(args.log_dir)

    print(f"MODEL=causal rolling median move-to-zero -> predicted next body sign | LB={args.lookback} MINH={args.min_history} CAPITAL={args.capital:.2f}")
    print(f"FEE=0 and {args.commission*100:.3f}%/side | entry=current close exit=next close | AD3/AD0 only")
    print("LONG if predicted_next_body>0; SHORT if predicted_next_body<0; threshold=abs(predicted_next_body)/entry")

    for symbol in symbols:
        rows = load(root / f"{symbol}_1h.csv")
        print(f"\n=== {symbol} rows={len(rows)} ===")
        for side_filter, side_label in [(None, "ALL"), ("LONG", "LONG"), ("SHORT", "SHORT")]:
            print(f"[{side_label}]")
            for t in thresholds:
                r0 = run(rows, t, 0.0, args.lookback, args.min_history, args.capital, None, side_filter)
                rf = run(rows, t, args.commission, args.lookback, args.min_history, args.capital, None, side_filter)
                print(f"T={t*100:.2f}% C={r0['candidates']} SIG={r0['signals']} ELIG={r0['eligible']} | F0 TR={r0['trades']} WR={r0['wr']:.1f}% DIR={r0['direction_acc']:.1f}% R={r0['return']*100:.2f}% F={r0['final']:.2f} PF={pf(float(r0['pf']))} DD={r0['dd']*100:.2f}% | Ffee TR={rf['trades']} WR={rf['wr']:.1f}% DIR={rf['direction_acc']:.1f}% R={rf['return']*100:.2f}% F={rf['final']:.2f} PF={pf(float(rf['pf']))} DD={rf['dd']*100:.2f}% | PNB={r0['pred_pct_med']*100:.3f}% ANB={r0['actual_body_pct_med']*100:.3f}% SKIP={r0['hist_skips']}")
                write_csv(log_root / f"{symbol}_{side_label}_T{t:.4f}_F0.csv", r0['trades_data'])
                write_csv(log_root / f"{symbol}_{side_label}_T{t:.4f}_F0_13.csv", rf['trades_data'])


if __name__ == "__main__":
    main()
