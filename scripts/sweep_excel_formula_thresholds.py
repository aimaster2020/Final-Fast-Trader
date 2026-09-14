from __future__ import annotations

import argparse
import csv
from pathlib import Path

from test_excel_formula_walkforward import evaluate, load_rows


def value_range(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if end < start:
        raise ValueError("end must be >= start")

    values: list[float] = []
    x = start
    epsilon = step * 1e-9
    while x <= end + epsilon:
        values.append(round(x, 10))
        x += step
    return values


def write_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "G",
        "H",
        "commission_round_trip_pct",
        "long_signals",
        "short_signals",
        "total_signals",
        "long_avg_move_pct",
        "short_avg_move_pct",
        "long_avg_net_pct",
        "short_avg_net_pct",
        "combined_avg_net_pct_per_trade",
        "long_total_net_pct_sum",
        "short_total_net_pct_sum",
        "combined_total_net_pct_sum",
        "net_positive_trades",
        "net_negative_trades",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep G and H thresholds for the Excel-formula strategy")
    p.add_argument("--input", required=True)
    p.add_argument("--g-start", type=float, default=0.0)
    p.add_argument("--g-end", type=float, default=500.0)
    p.add_argument("--g-step", type=float, default=100.0)
    p.add_argument("--h-start", type=float, default=0.0)
    p.add_argument("--h-end", type=float, default=500.0)
    p.add_argument("--h-step", type=float, default=100.0)
    p.add_argument("--commission", type=float, default=0.0026,
                   help="Round-trip commission as decimal; default 0.0026 = 0.26%%")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if args.commission < 0:
        raise ValueError("--commission must be >= 0")
    if args.top <= 0:
        raise ValueError("--top must be > 0")

    rows = load_rows(Path(args.input))
    g_values = value_range(args.g_start, args.g_end, args.g_step)
    h_values = value_range(args.h_start, args.h_end, args.h_step)

    results: list[dict] = []
    for g in g_values:
        for h in h_values:
            _, stats = evaluate(rows, g, h, args.commission)
            results.append({
                "G": g,
                "H": h,
                "commission_round_trip_pct": args.commission * 100.0,
                **{k: stats[k] for k in (
                    "long_signals",
                    "short_signals",
                    "total_signals",
                    "long_avg_move_pct",
                    "short_avg_move_pct",
                    "long_avg_net_pct",
                    "short_avg_net_pct",
                    "combined_avg_net_pct_per_trade",
                    "long_total_net_pct_sum",
                    "short_total_net_pct_sum",
                    "combined_total_net_pct_sum",
                    "net_positive_trades",
                    "net_negative_trades",
                )},
            })

    results.sort(key=lambda r: (
        r["combined_avg_net_pct_per_trade"],
        r["combined_total_net_pct_sum"],
        r["total_signals"],
    ), reverse=True)

    for rank, row in enumerate(results, 1):
        row["rank"] = rank

    write_results(Path(args.output), results)

    print("=" * 88)
    print("G/H THRESHOLD SWEEP")
    print("=" * 88)
    print(f"G={g_values[0]:g}..{g_values[-1]:g} step={args.g_step:g}")
    print(f"H={h_values[0]:g}..{h_values[-1]:g} step={args.h_step:g}")
    print(f"COMMISSION_ROUND_TRIP={args.commission * 100:.4f}%")
    print(f"combinations={len(results)}")
    print(f"output={args.output}")
    print("rank  G      H      signals  avg_net/trade%  total_net_sum%  long  short")

    for row in results[:args.top]:
        print(
            f"{row['rank']:>4}  "
            f"{row['G']:>5.0f}  "
            f"{row['H']:>5.0f}  "
            f"{row['total_signals']:>7}  "
            f"{row['combined_avg_net_pct_per_trade']:>15.6f}  "
            f"{row['combined_total_net_pct_sum']:>15.3f}  "
            f"{row['long_signals']:>5}  "
            f"{row['short_signals']:>5}"
        )


if __name__ == "__main__":
    main()
