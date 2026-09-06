from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.july_time_horizon_oracle_fee_hold import load_series

HORIZONS = [("30m", 30), ("1h", 60), ("2h", 120), ("4h", 240)]
DEFAULT_FEE_SIDE_PCT = 0.13


@dataclass
class Run:
    horizon: str
    symbol: str
    length: int
    direction: int
    entry_price: float
    exit_price: float

    @property
    def gross_return(self) -> float:
        if self.direction == 1:
            return self.exit_price / self.entry_price - 1.0
        return self.entry_price / self.exit_price - 1.0


def build_runs(raw: list, minutes: int, horizon: str, symbol: str) -> list[Run]:
    frames: list[tuple[int, int, float, float]] = []
    i = 0
    while i < len(raw):
        entry = raw[i].close
        target_ts = raw[i].timestamp + minutes * 60
        j = i + 1
        while j < len(raw) and raw[j].timestamp < target_ts:
            j += 1
        if j >= len(raw) or entry <= 0 or raw[j].close <= 0:
            break
        move = raw[j].close / entry - 1.0
        if move > 0:
            frames.append((i, j, 1, entry))
        elif move < 0:
            frames.append((i, j, -1, entry))
        i = j

    runs: list[Run] = []
    if not frames:
        return runs

    start_i, prev_j, current_dir, start_price = frames[0]
    length = 1
    exit_price = raw[prev_j].close

    for i, j, direction, entry_price in frames[1:]:
        if direction == current_dir:
            length += 1
            prev_j = j
            exit_price = raw[j].close
        else:
            runs.append(Run(horizon, symbol, length, current_dir, start_price, exit_price))
            start_i = i
            prev_j = j
            current_dir = direction
            start_price = entry_price
            exit_price = raw[j].close
            length = 1

    runs.append(Run(horizon, symbol, length, current_dir, start_price, exit_price))
    return runs


def net_return(gross_return: float, fee_side_pct: float) -> float:
    fee = fee_side_pct / 100.0
    # Standalone continuous trade: one entry fee + one exit fee.
    return (1.0 - fee) * (1.0 + gross_return) * (1.0 - fee) - 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--fee-side-pct", type=float, default=DEFAULT_FEE_SIDE_PCT)
    args = ap.parse_args()

    series = load_series(Path(args.input_dir), args.test_month, args.symbols)
    print(f"JULY_TIME_HORIZON_RUN_PROFIT test={args.test_month} assets={len(series)}")
    print("ORACLE DIRECTION | ONE CONTINUOUS TRADE PER SAME-DIRECTION RUN")
    print(f"fee={args.fee_side_pct:.3f}%/side | 2 fee sides/run | FLATS IGNORED")
    print("HORIZON RUN_LEN TIME COUNT WIN% AVG_GROSS AVG_NET MEDIAN_NET")
    print("------- ------- ---- ----- ----- --------- -------- -----------")

    rows: list[dict] = []
    for horizon, minutes in HORIZONS:
        all_runs: list[Run] = []
        for symbol, raw in series.items():
            all_runs.extend(build_runs(raw, minutes, horizon, symbol))

        grouped: dict[int, list[Run]] = defaultdict(list)
        for run in all_runs:
            grouped[run.length].append(run)

        for length in sorted(grouped):
            bucket = grouped[length]
            gross = [r.gross_return for r in bucket]
            net = [net_return(x, args.fee_side_pct) for x in gross]
            net_sorted = sorted(net)
            n = len(net_sorted)
            median = net_sorted[n // 2] if n % 2 else (net_sorted[n // 2 - 1] + net_sorted[n // 2]) / 2.0
            wins = sum(x > 0 for x in net)
            avg_gross = sum(gross) / n
            avg_net = sum(net) / n
            time_minutes = length * minutes
            time_label = f"{time_minutes / 60:.1f}h" if time_minutes >= 60 else f"{time_minutes}m"

            print(
                f"{horizon:>7} {length:>7} {time_label:>5} {n:>5} "
                f"{100*wins/n:>5.1f}% {100*avg_gross:>+8.4f}% "
                f"{100*avg_net:>+8.4f}% {100*median:>+10.4f}%"
            )

            rows.append({
                "horizon": horizon,
                "run_length": length,
                "holding_minutes": time_minutes,
                "count": n,
                "win_pct_net": 100 * wins / n,
                "avg_gross_pct": 100 * avg_gross,
                "avg_net_pct": 100 * avg_net,
                "median_net_pct": 100 * median,
                "min_net_pct": 100 * min(net),
                "max_net_pct": 100 * max(net),
            })

    out = ROOT / "reports" / "july_time_horizon_run_profit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"SAVED {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
