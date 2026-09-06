from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.july_r_composite_walkforward import discover_symbols, find_month_file, load_binance, resample

# Pure time-based baseline: no R rules, no indicators, no signal model.
# For every horizon, enter at the first candle of each period and exit after
# the requested elapsed horizon. The monthly test is one trade from the first
# available candle of July to the last available candle of July.
HORIZONS = [
    ("1m", 1),
    ("5m", 5),
    ("15m", 15),
    ("30m", 30),
    ("1h", 60),
    ("2h", 120),
    ("4h", 240),
    ("1d", 1440),
    ("1w", 10080),
    ("2w", 20160),
]


def build_frames(raw, minutes: int):
    return resample(raw, minutes)


def run_horizon(series: dict[str, list], horizon_name: str, horizon_minutes: int, fee_side_pct: float) -> dict:
    initial = 1000.0
    per_symbol = initial / len(series)
    total = 0.0
    trades = wins = 0
    fees_total = 0.0
    max_dd = 0.0

    for symbol, raw in series.items():
        # Use 1-minute closes for exact elapsed-time exits. Entry is the first
        # available 1m close; exit is the first 1m close at/after the horizon.
        # This keeps the experiment independent of strategy signals and candle
        # construction. For 1w/2w, each symbol has one trade starting at the
        # first available July minute if the horizon fits; the explicit 1m
        # monthly case is handled separately below.
        capital = per_symbol
        peak = capital
        c1 = build_frames(raw, 1)
        if not c1:
            total += capital
            continue

        # For sub-month horizons, repeatedly enter immediately after each exit
        # so the whole month is tested as a pure fixed-holding-period strategy.
        # This answers: "what if we always entered, held X, then exited?"
        i = 0
        while i < len(c1):
            exit_ts = c1[i].timestamp + horizon_minutes * 60
            j = i + 1
            while j < len(c1) and c1[j].timestamp < exit_ts:
                j += 1
            if j >= len(c1):
                break
            entry = c1[i].close
            exit_price = c1[j].close
            if entry <= 0:
                i = j
                continue

            # Long-only buy-and-hold for the fixed interval.
            move = exit_price / entry - 1.0
            gross = capital * move
            fee = capital * 2.0 * fee_side_pct / 100.0
            capital = max(0.0, capital + gross - fee)
            fees_total += fee
            trades += 1
            wins += int(gross > 0)
            peak = max(peak, capital)
            max_dd = max(max_dd, 100.0 * (peak - capital) / peak if peak else 0.0)
            i = j

        total += capital

    return {
        "horizon": horizon_name,
        "final": total,
        "ret": 100.0 * (total / initial - 1.0),
        "trades": trades,
        "win": 100.0 * wins / trades if trades else 0.0,
        "dd": max_dd,
        "fees": fees_total,
    }


def run_month_buy_hold(series: dict[str, list], fee_side_pct: float) -> dict:
    initial = 1000.0
    per_symbol = initial / len(series)
    total = 0.0
    trades = wins = 0
    fees_total = 0.0
    max_dd = 0.0

    for symbol, raw in series.items():
        c1 = build_frames(raw, 1)
        capital = per_symbol
        if len(c1) < 2:
            total += capital
            continue
        entry = c1[0].close
        exit_price = c1[-1].close
        if entry <= 0:
            total += capital
            continue
        move = exit_price / entry - 1.0
        gross = capital * move
        fee = capital * 2.0 * fee_side_pct / 100.0
        capital = max(0.0, capital + gross - fee)
        fees_total += fee
        trades += 1
        wins += int(gross > 0)
        max_dd = 0.0
        total += capital

    return {
        "horizon": "1mo",
        "final": total,
        "ret": 100.0 * (total / initial - 1.0),
        "trades": trades,
        "win": 100.0 * wins / trades if trades else 0.0,
        "dd": max_dd,
        "fees": fees_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--fee-side-pct", type=float, default=0.0)
    args = ap.parse_args()

    data_root = Path(args.input_dir)
    symbols = sorted(discover_symbols(data_root, args.test_month)) if args.symbols.upper() == "ALL" else [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    series = {}
    for symbol in symbols:
        p = find_month_file(data_root, symbol, args.test_month)
        if p:
            cs = load_binance(p)
            if len(cs) >= 2:
                series[symbol] = cs
    if not series:
        raise RuntimeError("No test data found")

    print(f"JULY_TIME_HORIZON_BASELINE test={args.test_month} assets={len(series)} fee={args.fee_side_pct:.3f}%/side")
    print("NO STRATEGY: entry/exit are determined only by elapsed holding time")
    print("Capital: $1000 total, equal split across symbols, long-only, 1x")
    print("HORIZON FINAL RET T WIN DD FEES")
    print("------- ----- ------ - ---- ---- -----")

    results = []
    for name, minutes in HORIZONS:
        r = run_horizon(series, name, minutes, args.fee_side_pct)
        results.append(r)
        print(f"{r['horizon']:>7} {r['final']:>7.2f} {r['ret']:>+6.2f}% {r['trades']:>4} {r['win']:>5.2f}% {r['dd']:>5.2f}% {r['fees']:>6.2f}")

    r = run_month_buy_hold(series, args.fee_side_pct)
    results.append(r)
    print(f"{r['horizon']:>7} {r['final']:>7.2f} {r['ret']:>+6.2f}% {r['trades']:>4} {r['win']:>5.2f}% {r['dd']:>5.2f}% {r['fees']:>6.2f}")

    out = ROOT / "reports" / "july_time_horizon_baseline.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["horizon", "final", "ret", "trades", "win", "dd", "fees"])
        w.writeheader()
        w.writerows(results)
    print(f"SAVED {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
