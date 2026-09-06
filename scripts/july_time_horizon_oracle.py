from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.july_r_composite_walkforward import discover_symbols, find_month_file, load_binance

# Oracle / upper-bound experiment.
# Direction is chosen AFTER seeing the future endpoint. This is intentionally
# look-ahead and measures the theoretical profit available in each horizon.
#
# Commission model:
# - 0.13% per side by default.
# - If the next oracle direction is the SAME as the current position,
#   the position stays open: no exit/re-entry and no commission.
# - If direction FLIPS, close the old position and open the new one:
#   2 sides of commission are charged at that boundary.
# - One final close is charged at the end of the test.
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

DEFAULT_FEE_SIDE_PCT = 0.13


def load_series(data_root: Path, test_month: str, symbols_arg: str) -> dict[str, list]:
    symbols = (
        sorted(discover_symbols(data_root, test_month))
        if symbols_arg.upper() == "ALL"
        else [x.strip().upper() for x in symbols_arg.split(",") if x.strip()]
    )
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


def apply_fee(capital: float, fee_side_pct: float) -> float:
    return capital * (1.0 - fee_side_pct / 100.0)


def run_horizon(
    series: dict[str, list],
    name: str,
    minutes: int,
    fee_side_pct: float,
) -> dict:
    initial = 1000.0
    per_symbol = initial / len(series)
    total = 0.0
    interval_count = 0
    winning_intervals = 0
    move_abs_sum = 0.0
    longs = shorts = flats = 0
    reversals = 0
    entries = 0
    fee_events = 0
    total_fees = 0.0

    for raw in series.values():
        capital = per_symbol
        i = 0
        position = 0
        last_price = None

        while i < len(raw):
            entry_price = raw[i].close
            target_ts = raw[i].timestamp + minutes * 60
            j = i + 1
            while j < len(raw) and raw[j].timestamp < target_ts:
                j += 1
            if j >= len(raw) or entry_price <= 0:
                break

            exit_price = raw[j].close
            raw_move = exit_price / entry_price - 1.0
            if raw_move > 0:
                direction = 1
                longs += 1
            elif raw_move < 0:
                direction = -1
                shorts += 1
            else:
                direction = 0
                flats += 1

            interval_count += 1
            winning_intervals += int(direction != 0)
            move_abs_sum += abs(raw_move)

            # Open the first non-flat oracle position.
            if position == 0 and direction != 0:
                fee = capital * fee_side_pct / 100.0
                capital -= fee
                total_fees += fee
                fee_events += 1
                entries += 1
                position = direction
                last_price = entry_price

            # A flat oracle interval closes the position only if there is no
            # directional edge for the next interval. We do not pay a fee for
            # simply waiting flat; if the next interval has a direction,
            # the new entry fee is charged there.
            elif position != 0 and direction == 0:
                # Realize the move through this interval, then close.
                if last_price is not None:
                    if position == 1:
                        capital *= exit_price / entry_price
                    else:
                        capital *= entry_price / exit_price
                fee = capital * fee_side_pct / 100.0
                capital -= fee
                total_fees += fee
                fee_events += 1
                position = 0
                last_price = None

            elif position != 0 and direction != 0 and direction != position:
                # The old position is held through this interval. Then the
                # oracle flips, so close + reverse: two commission sides.
                if position == 1:
                    capital *= exit_price / entry_price
                else:
                    capital *= entry_price / exit_price

                close_fee = capital * fee_side_pct / 100.0
                capital -= close_fee
                total_fees += close_fee
                fee_events += 1

                open_fee = capital * fee_side_pct / 100.0
                capital -= open_fee
                total_fees += open_fee
                fee_events += 1
                reversals += 1
                entries += 1
                position = direction
                last_price = exit_price

            # Same direction: keep the position open, no commission.
            elif position != 0 and direction == position:
                if position == 1:
                    capital *= exit_price / entry_price
                else:
                    capital *= entry_price / exit_price

            i = j

        # Final close: charge one side if a position is still open.
        if position != 0:
            fee = capital * fee_side_pct / 100.0
            capital -= fee
            total_fees += fee
            fee_events += 1

        total += capital

    return {
        "horizon": name,
        "final": total,
        "ret": 100.0 * (total / initial - 1.0),
        "intervals": interval_count,
        "all_directional_pct": 100.0 * winning_intervals / interval_count if interval_count else 0.0,
        "avg_abs_move_pct": 100.0 * move_abs_sum / interval_count if interval_count else 0.0,
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
    intervals = wins = 0
    move_abs_sum = 0.0
    longs = shorts = flats = 0
    entries = fee_events = 0
    total_fees = 0.0

    for raw in series.values():
        capital = per_symbol
        entry = raw[0].close
        exit_price = raw[-1].close
        if entry <= 0:
            total += capital
            continue

        raw_move = exit_price / entry - 1.0
        intervals += 1
        move_abs_sum += abs(raw_move)
        if raw_move > 0:
            longs += 1
            direction = 1
        elif raw_move < 0:
            shorts += 1
            direction = -1
        else:
            flats += 1
            direction = 0

        if direction != 0:
            # First entry commission.
            fee = capital * fee_side_pct / 100.0
            capital -= fee
            total_fees += fee
            fee_events += 1
            entries += 1

            # Perfect hindsight holds to month end, then closes.
            capital *= 1.0 + abs(raw_move)

            fee = capital * fee_side_pct / 100.0
            capital -= fee
            total_fees += fee
            fee_events += 1
        total += capital

    return {
        "horizon": "1mo",
        "final": total,
        "ret": 100.0 * (total / initial - 1.0),
        "intervals": intervals,
        "all_directional_pct": 100.0 * wins / intervals if intervals else 0.0,
        "avg_abs_move_pct": 100.0 * move_abs_sum / intervals if intervals else 0.0,
        "longs": longs,
        "shorts": shorts,
        "flats": flats,
        "reversals": 0,
        "entries": entries,
        "fee_events": fee_events,
        "fees": total_fees,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--fee-side-pct", type=float, default=DEFAULT_FEE_SIDE_PCT)
    args = ap.parse_args()

    series = load_series(Path(args.input_dir), args.test_month, args.symbols)
    print(f"JULY_TIME_HORIZON_ORACLE test={args.test_month} assets={len(series)}")
    print("LOOK-AHEAD UPPER BOUND: direction is chosen from the known future")
    print(
        f"SAME DIRECTION = HOLD; FLIP = CLOSE+REVERSE; "
        f"commission={args.fee_side_pct:.3f}%/side; 1x; $1000"
    )
    print("HORIZON FINAL NET_RET INTERVALS DIRECTIONAL AVG_MOVE REVERSALS ENTRIES FEES")
    print("------- ----- ------- --------- ---------- --------- --------- ------- -----")

    results = []
    for name, minutes in HORIZONS:
        r = run_horizon(series, name, minutes, args.fee_side_pct)
        results.append(r)
        print(
            f"{r['horizon']:>7} {r['final']:>9.2f} {r['ret']:>+8.2f}% "
            f"{r['intervals']:>9} {r['all_directional_pct']:>9.2f}% "
            f"{r['avg_abs_move_pct']:>8.4f}% {r['reversals']:>9} "
            f"{r['entries']:>7} {r['fees']:>8.2f}"
        )

    r = run_month(series, args.fee_side_pct)
    results.append(r)
    print(
        f"{r['horizon']:>7} {r['final']:>9.2f} {r['ret']:>+8.2f}% "
        f"{r['intervals']:>9} {r['all_directional_pct']:>9.2f}% "
        f"{r['avg_abs_move_pct']:>8.4f}% {r['reversals']:>9} "
        f"{r['entries']:>7} {r['fees']:>8.2f}"
    )

    out = ROOT / "reports" / "july_time_horizon_oracle.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"SAVED {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
