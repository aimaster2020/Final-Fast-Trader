from __future__ import annotations

import argparse
import csv
from pathlib import Path


def num(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Sort the complete R confirmation CSV without dropping any rows")
    ap.add_argument("--input", default="reports/july_r_confirmation_all_triggers_wf_v6.csv")
    ap.add_argument("--output", default="reports/july_r_confirmation_all_triggers_wf_v6_sorted.csv")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    with src.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Preserve every row. Ranking is purely presentation order.
    rows.sort(
        key=lambda r: (
            num(r, "test_accuracy_pct"),
            num(r, "test_signals"),
            num(r, "lift_pp"),
            num(r, "horizon_baseline_accuracy_pct"),
        ),
        reverse=True,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"SORTED {dst} rows={len(rows)}")
    print("ORDER: test_accuracy_pct DESC, test_signals DESC, lift_pp DESC, horizon_baseline_accuracy_pct DESC")
    print("NO ROWS REMOVED")

    print("\nRANK ACC SIGNALS LIFT TF TRIGGER SET")
    for rank, r in enumerate(rows, 1):
        print(
            f"{rank:4d} {num(r,'test_accuracy_pct'):6.2f}% "
            f"{int(num(r,'test_signals')):7d} "
            f"{num(r,'lift_pp'):+7.2f} "
            f"{int(num(r,'timeframe_min')):3d}m "
            f"{r.get('trigger_rule','')}:{r.get('trigger_variant','')} "
            f"{r.get('set','')}"
        )


if __name__ == "__main__":
    main()
