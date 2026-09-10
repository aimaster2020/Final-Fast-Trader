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
DEFAULT_THRESHOLDS = "0.0000,0.0001,0.0002,0.0003,0.0004,0.0005,0.00075,0.0010,0.0015,0.0020,0.0030"


@dataclass
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float


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
    cur = body(rows[i])
    pred_next = cur - math.copysign(abs(cur) * pred_ratio, cur)
    return pred_ratio, len(vals), pred_next


def trade_return(side: str, entry: float, exit_price: float, fee: float) -> tuple[float, float]:
    gross = exit_price / entry - 1.0 if side == "LONG" else 1.0 - exit_price / entry
    return gross, gross - 2.0 * fee


def run(rows: list[Candle], threshold: float, fee: float, lookback: int, minh: int, capital0: float, mode: str):
    capital = capital0
    trades = []
    candidates = signals = hist_skips = 0
    direction_correct = 0
    pred_pct_values: list[float] = []
    actual_pct_values: list[float] = []

    for i in range(1, len(rows) - 1):
        score = ad(rows[i])
        if score not in (0, 3):
            continue
        candidates += 1
        pred_ratio, hist_n, pred_next = rolling_prediction(rows, i, lookback, minh)
        if pred_ratio is None or pred_next is None:
            hist_skips += 1
            continue
        signals += 1
        cur = body(rows[i])
        actual = body(rows[i + 1])
        entry = rows[i].close
        if entry <= EPS:
            continue
        pred_sign = 1 if pred_next > 0 else -1 if pred_next < 0 else 0
        if pred_sign == 0:
            continue

        pred_pct = abs(pred_next) / entry
        actual_pct = abs(actual) / entry
        pred_pct_values.append(pred_pct)
        actual_pct_values.append(actual_pct)
        direction_correct += int((pred_sign > 0) == (actual > 0))

        current_sign = 1 if cur > 0 else -1 if cur < 0 else 0
        crossed = current_sign != pred_sign
        if mode == "cross_only" and not crossed:
            continue
        if mode == "same_only" and crossed:
            continue
        if pred_pct < threshold:
            continue

        side = "LONG" if pred_sign > 0 else "SHORT"
        exit_price = rows[i + 1].close
        gross, net = trade_return(side, entry, exit_price, fee)
        before = capital
        after = before * (1.0 + net)
        trades.append({
            "index": i, "entry_timestamp": rows[i].timestamp, "exit_timestamp": rows[i + 1].timestamp,
            "side": side, "ad": score, "current_body": cur, "pred_ratio": pred_ratio,
            "pred_next_body": pred_next, "pred_next_body_pct": pred_pct, "actual_next_body": actual,
            "actual_next_body_pct": actual_pct, "entry": entry, "exit": exit_price,
            "gross_return": gross, "fee_return": 2.0 * fee, "net_return": net,
            "capital_before": before, "capital_after": after,
        })
        capital = after

    wins = sum(1 for t in trades if t["net_return"] > 0)
    net_profit = sum(t["net_return"] for t in trades if t["net_return"] > 0)
    net_loss = sum(t["net_return"] for t in trades if t["net_return"] < 0)
    peak = capital0
    dd = 0.0
    for t in trades:
        peak = max(peak, t["capital_after"])
        dd = max(dd, (peak - t["capital_after"]) / peak if peak > EPS else 0.0)

    return {
        "candidates": candidates, "signals": signals, "eligible": len(trades), "trades": len(trades),
        "wins": wins, "wr": wins / len(trades) * 100.0 if trades else 0.0,
        "dir": direction_correct / signals * 100.0 if signals else 0.0,
        "return": capital / capital0 - 1.0, "final": capital,
        "pf": net_profit / abs(net_loss) if net_loss < -EPS else math.inf,
        "dd": dd, "hist_skips": hist_skips,
        "pred_pct_med": median(pred_pct_values) if pred_pct_values else 0.0,
        "actual_body_pct_med": median(actual_pct_values) if actual_pct_values else 0.0,
        "trades_data": trades,
    }


def write_csv(path: Path, trades: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["index", "entry_timestamp", "exit_timestamp", "side", "ad", "current_body", "pred_ratio", "pred_next_body", "pred_next_body_pct", "actual_next_body", "actual_next_body_pct", "entry", "exit", "gross_return", "fee_return", "net_return", "capital_before", "capital_after"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for t in trades:
            w.writerow([t[k] for k in fields])


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-zero body strategy v2")
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    ap.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--mode", choices=["all", "cross_only", "same_only"], default="all")
    ap.add_argument("--log-dir", default="reports/cross_zero_trades_ad30_v2")
    args = ap.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    root = Path(args.input_dir)
    log_root = Path(args.log_dir)

    print(f"MODEL=causal rolling median move-to-zero -> predicted next body sign | LB={args.lookback} MINH={args.min_history} CAPITAL={args.capital:.2f}")
    print(f"FEE=0 and {args.commission*100:.3f}%/side | ENTRY=current close EXIT=next close | MODE={args.mode}")
    print("AD=3/0 only | LONG if predicted_next_body>0 | SHORT if <0")
    print("threshold=abs(predicted_next_body)/entry; fractions of price")

    for symbol in symbols:
        rows = load(root / f"{symbol}_1h.csv")
        print(f"\n=== {symbol} rows={len(rows)} ===")
        for t in thresholds:
            r0 = run(rows, t, 0.0, args.lookback, args.min_history, args.capital, args.mode)
            rf = run(rows, t, args.commission, args.lookback, args.min_history, args.capital, args.mode)
            print(
                f"T={t*100:.3f}% SIG={r0['signals']} ELIG={r0['eligible']} | "
                f"F0 TR={r0['trades']} WR={r0['wr']:.1f}% DIR={r0['dir']:.1f}% R={r0['return']*100:.2f}% F={r0['final']:.2f} PF={fmt_pf(float(r0['pf']))} DD={r0['dd']*100:.2f}% | "
                f"Ffee TR={rf['trades']} WR={rf['wr']:.1f}% DIR={rf['dir']:.1f}% R={rf['return']*100:.2f}% F={rf['final']:.2f} PF={fmt_pf(float(rf['pf']))} DD={rf['dd']*100:.2f}% | "
                f"PNB={r0['pred_pct_med']*100:.3f}% ANB={r0['actual_body_pct_med']*100:.3f}% SKIP={r0['hist_skips']}"
            )
            write_csv(log_root / f"{symbol}_{args.mode}_T{t:.5f}_F0.csv", r0["trades_data"])
            write_csv(log_root / f"{symbol}_{args.mode}_T{t:.5f}_F0_13.csv", rf["trades_data"])


def fmt_pf(x: float) -> str:
    return "INF" if math.isinf(x) else f"{x:.2f}"


if __name__ == "__main__":
    main()
