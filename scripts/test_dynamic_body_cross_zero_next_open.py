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
DEFAULT_THRESHOLDS = "0.00,0.0001,0.0002,0.0003,0.0004,0.0005,0.00075,0.0010,0.0015,0.0020,0.0030"


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
    current_body: float
    pred_ratio: float
    pred_next_body: float
    pred_body_pct: float
    actual_next_body: float
    actual_body_pct: float
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
                rows.append(
                    Candle(
                        str(r.get("timestamp", "")),
                        float(r["open"]),
                        float(r["high"]),
                        float(r["low"]),
                        float(r["close"]),
                    )
                )
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


def rolling_prediction(rows: list[Candle], i: int, lookback: int, minh: int) -> tuple[float | None, int, float | None]:
    score = ad(rows[i])
    vals: list[float] = []
    start = max(1, i - lookback)
    for j in range(start, i):
        if ad(rows[j]) != score:
            continue
        r = move_to_zero_ratio(body(rows[j]), body(rows[j + 1]))
        if r is not None and math.isfinite(r):
            vals.append(r)
    if len(vals) < minh:
        return None, len(vals), None
    pred_ratio = median(vals)
    current = body(rows[i])
    pred_next = current - math.copysign(abs(current) * pred_ratio, current)
    return pred_ratio, len(vals), pred_next


def trade_return(side: str, entry: float, exit_price: float, fee: float) -> tuple[float, float]:
    if side == "LONG":
        gross = exit_price / entry - 1.0
    else:
        gross = 1.0 - exit_price / entry
    return gross, gross - 2.0 * fee


def run(
    rows: list[Candle],
    threshold: float,
    fee: float,
    lookback: int,
    minh: int,
    capital0: float,
    side_filter: str | None = None,
    cross_only: bool = False,
) -> dict:
    capital = capital0
    trades: list[Trade] = []
    candidates = 0
    signals = 0
    eligible = 0
    history_skips = 0
    direction_correct = 0
    cross_correct = 0
    pred_pct_values: list[float] = []
    actual_body_pct_values: list[float] = []

    for i in range(1, len(rows) - 1):
        score = ad(rows[i])
        if score not in (0, 3):
            continue
        candidates += 1

        pred_ratio, hist_n, pred_next = rolling_prediction(rows, i, lookback, minh)
        if pred_ratio is None or pred_next is None:
            history_skips += 1
            continue
        signals += 1

        current = body(rows[i])
        nxt = body(rows[i + 1])
        predicted_sign = 1 if pred_next > EPS else -1 if pred_next < -EPS else 0
        actual_sign = 1 if nxt > EPS else -1 if nxt < -EPS else 0
        if predicted_sign and actual_sign:
            direction_correct += int(predicted_sign == actual_sign)

        crosses = predicted_sign != 0 and current * pred_next < -EPS
        if crosses:
            cross_correct += int(predicted_sign == actual_sign)

        entry = rows[i + 1].open
        exit_price = rows[i + 1].close
        if entry <= EPS:
            continue

        pred_pct = abs(pred_next) / entry
        actual_pct = abs(nxt) / entry
        pred_pct_values.append(pred_pct)
        actual_body_pct_values.append(actual_pct)

        if cross_only and not crosses:
            continue
        if side_filter is not None:
            side = "LONG" if predicted_sign > 0 else "SHORT" if predicted_sign < 0 else "FLAT"
            if side != side_filter:
                continue
        if predicted_sign == 0 or pred_pct < threshold:
            continue

        side = "LONG" if predicted_sign > 0 else "SHORT"
        eligible += 1
        gross, net = trade_return(side, entry, exit_price, fee)
        before = capital
        after = before * (1.0 + net)
        trades.append(
            Trade(
                i,
                rows[i + 1].timestamp,
                rows[i + 1].timestamp,
                side,
                score,
                current,
                pred_ratio,
                pred_next,
                pred_pct,
                nxt,
                actual_pct,
                entry,
                exit_price,
                gross,
                2.0 * fee,
                net,
                before,
                after,
            )
        )
        capital = after

    wins = sum(1 for t in trades if t.net_return > 0)
    profit = sum(t.net_return for t in trades if t.net_return > 0)
    loss = sum(t.net_return for t in trades if t.net_return < 0)
    peak = capital0
    dd = 0.0
    for t in trades:
        peak = max(peak, t.capital_after)
        dd = max(dd, (peak - t.capital_after) / peak if peak > EPS else 0.0)

    return {
        "candidates": candidates,
        "signals": signals,
        "eligible": eligible,
        "trades": len(trades),
        "wins": wins,
        "wr": wins / len(trades) * 100.0 if trades else 0.0,
        "dir": direction_correct / signals * 100.0 if signals else 0.0,
        "cross_dir": cross_correct / max(1, sum(1 for _ in [])) * 100.0,
        "return": capital / capital0 - 1.0,
        "final": capital,
        "pf": profit / abs(loss) if loss < -EPS else math.inf,
        "dd": dd,
        "hist_skips": history_skips,
        "pred_pct_med": median(pred_pct_values) if pred_pct_values else 0.0,
        "actual_body_pct_med": median(actual_body_pct_values) if actual_body_pct_values else 0.0,
        "trades_data": trades,
    }


def write_csv(path: Path, trades: list[Trade]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "index", "entry_timestamp", "exit_timestamp", "side", "ad",
            "current_body", "pred_ratio", "pred_next_body", "pred_body_pct",
            "actual_next_body", "actual_body_pct", "entry", "exit",
            "gross_return", "fee_return", "net_return", "capital_before", "capital_after"
        ])
        for t in trades:
            w.writerow([
                t.index, t.entry_timestamp, t.exit_timestamp, t.side, t.ad,
                f"{t.current_body:.12g}", f"{t.pred_ratio:.12g}",
                f"{t.pred_next_body:.12g}", f"{t.pred_body_pct:.12g}",
                f"{t.actual_next_body:.12g}", f"{t.actual_body_pct:.12g}",
                f"{t.entry:.12g}", f"{t.exit:.12g}",
                f"{t.gross_return:.12g}", f"{t.fee_return:.12g}",
                f"{t.net_return:.12g}", f"{t.capital_before:.12g}",
                f"{t.capital_after:.12g}",
            ])


def fmt_pf(x: float) -> str:
    return "INF" if math.isinf(x) else f"{x:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-zero body strategy: next open to next close")
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    ap.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--mode", choices=["all", "cross_only", "same_only"], default="all")
    ap.add_argument("--log-dir", default="reports/cross_zero_next_open_trades")
    args = ap.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    root = Path(args.input_dir)
    log_root = Path(args.log_dir)

    print(f"MODEL=causal rolling median move-to-zero -> predicted next body | LB={args.lookback} MINH={args.min_history} CAPITAL={args.capital:.2f}")
    print(f"FEE=0 and {args.commission*100:.3f}%/side | ENTRY=next open EXIT=next close | MODE={args.mode}")
    print("AD=3/0 only | LONG if predicted body>0 | SHORT if predicted body<0")
    print("threshold=abs(predicted_next_body)/next_open")

    for symbol in symbols:
        rows = load(root / f"{symbol}_1h.csv")
        print(f"\n=== {symbol} rows={len(rows)} ===")
        for side_filter, label in [(None, "ALL"), ("LONG", "LONG"), ("SHORT", "SHORT")]:
            print(f"[{label}]")
            for threshold in thresholds:
                r0 = run(rows, threshold, 0.0, args.lookback, args.min_history, args.capital, side_filter, args.mode == "cross_only")
                rf = run(rows, threshold, args.commission, args.lookback, args.min_history, args.capital, side_filter, args.mode == "cross_only")
                print(
                    f"T={threshold*100:.3f}% C={r0['candidates']} SIG={r0['signals']} ELIG={r0['eligible']} | "
                    f"F0 TR={r0['trades']} WR={r0['wr']:.1f}% DIR={r0['dir']:.1f}% R={r0['return']*100:.2f}% F={r0['final']:.2f} PF={fmt_pf(float(r0['pf']))} DD={r0['dd']*100:.2f}% | "
                    f"Ffee TR={rf['trades']} WR={rf['wr']:.1f}% DIR={rf['dir']:.1f}% R={rf['return']*100:.2f}% F={rf['final']:.2f} PF={fmt_pf(float(rf['pf']))} DD={rf['dd']*100:.2f}% | "
                    f"PNB={r0['pred_pct_med']*100:.3f}% ANB={r0['actual_body_pct_med']*100:.3f}% SKIP={r0['hist_skips']}"
                )
                write_csv(log_root / f"{symbol}_{label}_T{threshold:.5f}_F0.csv", r0["trades_data"])
                write_csv(log_root / f"{symbol}_{label}_T{threshold:.5f}_F0_13.csv", rf["trades_data"])


if __name__ == "__main__":
    main()
