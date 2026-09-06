from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.july_time_horizon_oracle_fee_hold import load_series

HORIZONS = [("30m", 30), ("1h", 60), ("2h", 120), ("4h", 240)]


def direction_runs(raw: list, minutes: int) -> Counter[int]:
    dirs: list[int] = []
    i = 0
    while i < len(raw):
        entry = raw[i].close
        target_ts = raw[i].timestamp + minutes * 60
        j = i + 1
        while j < len(raw) and raw[j].timestamp < target_ts:
            j += 1
        if j >= len(raw) or entry <= 0:
            break
        move = raw[j].close / entry - 1.0
        # Flat intervals are ignored; the run counter is over directional frames.
        if move > 0:
            dirs.append(1)
        elif move < 0:
            dirs.append(-1)
        i = j

    runs: Counter[int] = Counter()
    if not dirs:
        return runs

    current = dirs[0]
    length = 1
    for d in dirs[1:]:
        if d == current:
            length += 1
        else:
            runs[length] += 1
            current = d
            length = 1
    runs[length] += 1
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    args = ap.parse_args()

    series = load_series(Path(args.input_dir), args.test_month, args.symbols)
    print(f"JULY_TIME_HORIZON_DIRECTION_RUNS test={args.test_month} assets={len(series)}")
    print("ORACLE DIRECTION | SAME DIRECTION=ONE CONTINUOUS TRADE | FLATS IGNORED")
    print("HORIZON RUNS 1F 2F 3F 4F 5F 6F 7F 8F 9F 10F 11F+ AVG_F MAX_F")

    for name, minutes in HORIZONS:
        total = Counter()
        for raw in series.values():
            total.update(direction_runs(raw, minutes))

        n_runs = sum(total.values())
        weighted = sum(k * v for k, v in total.items())
        avg = weighted / n_runs if n_runs else 0.0
        max_run = max(total) if total else 0
        vals = [total.get(k, 0) for k in range(1, 11)]
        tail = sum(v for k, v in total.items() if k >= 11)
        print(
            f"{name:>7} {n_runs:>5} "
            + " ".join(f"{v:>4}" for v in vals)
            + f" {tail:>5} {avg:>5.2f} {max_run:>5}"
        )

        # Percent distribution for the same horizon, useful for quick interpretation.
        if n_runs:
            parts = [f"{k}F={100*total.get(k,0)/n_runs:.1f}%" for k in range(1, 6)]
            tail_pct = 100 * tail / n_runs
            print("        " + " ".join(parts) + f" 11F+={tail_pct:.1f}%")

    out = ROOT / "reports" / "july_time_horizon_direction_runs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("horizon,run_length,count\n")
        for name, minutes in HORIZONS:
            total = Counter()
            for raw in series.values():
                total.update(direction_runs(raw, minutes))
            for k in sorted(total):
                f.write(f"{name},{k},{total[k]}\n")
    print(f"SAVED {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
