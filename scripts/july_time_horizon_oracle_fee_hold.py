from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.july_r_composite_walkforward import discover_symbols, find_month_file, load_binance

HORIZONS = [
    ("1m", 1), ("5m", 5), ("15m", 15), ("30m", 30),
    ("1h", 60), ("2h", 120), ("4h", 240), ("1d", 1440),
    ("1w", 10080), ("2w", 20160),
]
DEFAULT_FEE_SIDE_PCT = 0.13


def load_series(data_root: Path, test_month: str, symbols_arg: str) -> dict[str, list]:
    symbols = (sorted(discover_symbols(data_root, test_month))
               if symbols_arg.upper() == "ALL"
               else [x.strip().upper() for x in symbols_arg.split(",") if x.strip()])
    series = {}
    for symbol in symbols:
        p = find_month_file(data_root, symbol, test_month)
        if p:
            cs = load_binance(p)
            if len(cs) >= 2:
                series[symbol] = cs
    if not series:
        raise RuntimeError("No test data found")
    return series


def charge_fee(capital: float, fee_side_pct: float) -> tuple[float, float]:
    f = capital * fee_side_pct / 100.0
    return capital - f, f


def run_horizon(series: dict[str, list], name: str, minutes: int, fee_side_pct: float) -> dict:
    initial = 1000.0
    per_symbol = initial / len(series)
    total = 0.0
    intervals = directional = 0
    move_abs_sum = 0.0
    longs = shorts = flats = 0
    reversals = entries = fee_events = 0
    total_fees = 0.0

    for raw in series.values():
        capital = per_symbol
        i = 0
        position = 0

        while i < len(raw):
            entry = raw[i].close
            target_ts = raw[i].timestamp + minutes * 60
            j = i + 1
            while j < len(raw) and raw[j].timestamp < target_ts:
                j += 1
            if j >= len(raw) or entry <= 0:
                break

            exit_price = raw[j].close
            move = exit_price / entry - 1.0
            intervals += 1
            move_abs_sum += abs(move)

            if move > 0:
                direction = 1
                longs += 1
                directional += 1
            elif move < 0:
                direction = -1
                shorts += 1
                directional += 1
            else:
                direction = 0
                flats += 1

            # Oracle sees the direction before this interval starts.
            # Same direction => keep the position, zero commission.
            # Flip => close old + open new, two commission sides at entry.
            if direction == 0:
                if position != 0:
                    capital, f = charge_fee(capital, fee_side_pct)
                    total_fees += f
                    fee_events += 1
                    position = 0
            elif position == 0:
                capital, f = charge_fee(capital, fee_side_pct)
                total_fees += f
                fee_events += 1
                entries += 1
                position = direction
            elif direction != position:
                capital, f = charge_fee(capital, fee_side_pct)
                total_fees += f
                fee_events += 1
                capital, f = charge_fee(capital, fee_side_pct)
                total_fees += f
                fee_events += 1
                reversals += 1
                entries += 1
                position = direction

            # Apply the current interval under the oracle-selected direction.
            if position == 1:
                capital *= exit_price / entry
            elif position == -1:
                capital *= entry / exit_price

            i = j

        # Close any remaining position at the end: one commission side.
        if position != 0:
            capital, f = charge_fee(capital, fee_side_pct)
            total_fees += f
            fee_events += 1

        total += capital

    return {
        "horizon": name,
        "final": total,
        "net_ret_pct": 100.0 * (total / initial - 1.0),
        "intervals": intervals,
        "directional_pct": 100.0 * directional / intervals if intervals else 0.0,
        "avg_abs_move_pct": 100.0 * move_abs_sum / intervals if intervals else 0.0,
        "longs": longs,
        "shorts": shorts,
        "flats": flats,
        "reversals": reversals,
        "entries": entries,
        "fee_events": fee_events,
        "fees": total_fees,
    }


def run_month(series: dict[str, list], fee_side_pct: float) -> dict:
    initial = 1000.0
    per_symbol = initial / len(series)
    total = 0.0
    intervals = directional = 0
    move_abs_sum = 0.0
    longs = shorts = flats = 0
    entries = fee_events = 0
    total_fees = 0.0

    for raw in series.values():
        entry = raw[0].close
        exit_price = raw[-1].close
        capital = per_symbol
        if entry <= 0:
            total += capital
            continue

        move = exit_price / entry - 1.0
        intervals += 1
        move_abs_sum += abs(move)
        if move > 0:
            direction = 1; longs += 1; directional += 1
        elif move < 0:
            direction = -1; shorts += 1; directional += 1
        else:
            direction = 0; flats += 1

        if direction != 0:
            capital, f = charge_fee(capital, fee_side_pct)
            total_fees += f; fee_events += 1; entries += 1
            capital *= exit_price / entry if direction == 1 else entry / exit_price
            capital, f = charge_fee(capital, fee_side_pct)
            total_fees += f; fee_events += 1
        total += capital

    return {
        "horizon": "1mo",
        "final": total,
        "net_ret_pct": 100.0 * (total / initial - 1.0),
        "intervals": intervals,
        "directional_pct": 100.0 * directional / intervals if intervals else 0.0,
        "avg_abs_move_pct": 100.0 * move_abs_sum / intervals if intervals else 0.0,
        "longs": longs, "shorts": shorts, "flats": flats,
        "reversals": 0, "entries": entries,
        "fee_events": fee_events, "fees": total_fees,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--fee-side-pct", type=float, default=DEFAULT_FEE_SIDE_PCT)
    args = ap.parse_args()

    series = load_series(Path(args.input_dir), args.test_month, args.symbols)
    print(f"JULY_TIME_HORIZON_ORACLE_FEE_HOLD test={args.test_month} assets={len(series)}")
    print("LOOK-AHEAD 100%-ACCURACY UPPER BOUND")
    print(f"fee={args.fee_side_pct:.3f}%/side | SAME DIRECTION=HOLD | FLIP=CLOSE+REVERSE | 1x | $1000")
    print("HORIZON FINAL NET_RET INTERVALS DIR% AVG_MOVE REVERSALS ENTRIES FEES")
    print("------- ----- ------- --------- ----- --------- --------- ------- -----")

    results = []
    for name, minutes in HORIZONS:
        r = run_horizon(series, name, minutes, args.fee_side_pct)
        results.append(r)
        print(f"{r['horizon']:>7} {r['final']:>9.2f} {r['net_ret_pct']:>+8.2f}% "
              f"{r['intervals']:>9} {r['directional_pct']:>5.1f}% {r['avg_abs_move_pct']:>8.4f}% "
              f"{r['reversals']:>9} {r['entries']:>7} {r['fees']:>8.2f}")

    r = run_month(series, args.fee_side_pct)
    results.append(r)
    print(f"{r['horizon']:>7} {r['final']:>9.2f} {r['net_ret_pct']:>+8.2f}% "
          f"{r['intervals']:>9} {r['directional_pct']:>5.1f}% {r['avg_abs_move_pct']:>8.4f}% "
          f"{r['reversals']:>9} {r['entries']:>7} {r['fees']:>8.2f}")

    out = ROOT / "reports" / "july_time_horizon_oracle_fee_hold.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"SAVED {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
