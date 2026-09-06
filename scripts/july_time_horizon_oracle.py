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
# For every fixed holding interval, direction is chosen AFTER seeing the
# future: LONG when exit > entry, SHORT when exit < entry. Therefore every
# non-flat interval is a winning trade. This is intentionally look-ahead and
# measures the hidden profit available in the time horizon itself.
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


def run_horizon(series: dict[str, list], name: str, minutes: int) -> dict:
    initial = 1000.0
    per_symbol = initial / len(series)
    total = 0.0
    trades = wins = 0
    move_sum = 0.0
    move_abs_sum = 0.0
    longs = shorts = flats = 0
    max_move = 0.0

    for raw in series.values():
        capital = per_symbol
        i = 0
        while i < len(raw):
            entry = raw[i].close
            target_ts = raw[i].timestamp + minutes * 60
            j = i + 1
            while j < len(raw) and raw[j].timestamp < target_ts:
                j += 1
            if j >= len(raw) or entry <= 0:
                break

            exit_price = raw[j].close
            raw_move = exit_price / entry - 1.0
            if raw_move > 0:
                direction = 1
                realized = raw_move
                longs += 1
            elif raw_move < 0:
                direction = -1
                realized = -raw_move
                shorts += 1
            else:
                direction = 0
                realized = 0.0
                flats += 1

            capital *= 1.0 + realized
            trades += 1
            wins += int(direction != 0)
            move_sum += realized
            move_abs_sum += abs(raw_move)
            max_move = max(max_move, realized)
            i = j

        total += capital

    return {
        "horizon": name,
        "final": total,
        "ret": 100.0 * (total / initial - 1.0),
        "trades": trades,
        "all_win_pct": 100.0 * wins / trades if trades else 0.0,
        "avg_abs_move_pct": 100.0 * move_abs_sum / trades if trades else 0.0,
        "sum_abs_move_pct": 100.0 * move_abs_sum,
        "longs": longs,
        "shorts": shorts,
        "flats": flats,
        "max_move_pct": 100.0 * max_move,
    }


def run_month(series: dict[str, list]) -> dict:
    initial = 1000.0
    per_symbol = initial / len(series)
    total = 0.0
    trades = wins = 0
    move_abs_sum = 0.0
    longs = shorts = flats = 0

    for raw in series.values():
        capital = per_symbol
        entry = raw[0].close
        exit_price = raw[-1].close
        if entry <= 0:
            total += capital
            continue
        raw_move = exit_price / entry - 1.0
        realized = abs(raw_move)
        if raw_move > 0:
            longs += 1
        elif raw_move < 0:
            shorts += 1
        else:
            flats += 1
        capital *= 1.0 + realized
        trades += 1
        wins += int(raw_move != 0)
        move_abs_sum += realized
        total += capital

    return {
        "horizon": "1mo",
        "final": total,
        "ret": 100.0 * (total / initial - 1.0),
        "trades": trades,
        "all_win_pct": 100.0 * wins / trades if trades else 0.0,
        "avg_abs_move_pct": 100.0 * move_abs_sum / trades if trades else 0.0,
        "sum_abs_move_pct": 100.0 * move_abs_sum,
        "longs": longs,
        "shorts": shorts,
        "flats": flats,
        "max_move_pct": 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    args = ap.parse_args()

    series = load_series(Path(args.input_dir), args.test_month, args.symbols)
    print(f"JULY_TIME_HORIZON_ORACLE test={args.test_month} assets={len(series)}")
    print("LOOK-AHEAD UPPER BOUND: direction is chosen from the known future")
    print("LONG if exit>entry, SHORT if exit<entry; commission=0; 1x; $1000")
    print("HORIZON FINAL HIDDEN_RET TRADES ALL_WIN AVG_MOVE LONG SHORT FLAT")
    print("------- ----- ----------- ------ ------- -------- ---- ----- ----")

    results = []
    for name, minutes in HORIZONS:
        r = run_horizon(series, name, minutes)
        results.append(r)
        print(
            f"{r['horizon']:>7} {r['final']:>7.2f} {r['ret']:>+10.2f}% "
            f"{r['trades']:>6} {r['all_win_pct']:>7.2f}% {r['avg_abs_move_pct']:>8.4f}% "
            f"{r['longs']:>4} {r['shorts']:>5} {r['flats']:>4}"
        )

    r = run_month(series)
    results.append(r)
    print(
        f"{r['horizon']:>7} {r['final']:>7.2f} {r['ret']:>+10.2f}% "
        f"{r['trades']:>6} {r['all_win_pct']:>7.2f}% {r['avg_abs_move_pct']:>8.4f}% "
        f"{r['longs']:>4} {r['shorts']:>5} {r['flats']:>4}"
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
