#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test BP (predicted direction) against the next candle close direction. BP=0 is excluded from the denominator."
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    try:
        from backtest_exact_excel_formulas import load_rows, calculate
    except ImportError as exc:
        raise SystemExit(
            "Could not import backtest_exact_excel_formulas.py. Run this script from the repository scripts directory or ensure scripts is on PYTHONPATH."
        ) from exc

    source = Path(a.input)
    target = Path(a.output)
    rows = load_rows(source)
    if len(rows) < 22:
        raise SystemExit("Need at least 22 usable OHLC rows.")

    result = calculate(rows)

    records: list[dict[str, object]] = []
    for i in range(20, len(result) - 1):
        bp = int(result[i]["BP_PredictedDirection"])
        if bp == 0:
            continue

        current_close = float(rows[i]["Close"])
        next_close = float(rows[i + 1]["Close"])
        actual = 1 if next_close > current_close else -1 if next_close < current_close else 0
        correct = int(bp == actual)

        records.append(
            {
                "Timestamp": rows[i]["Timestamp"],
                "Current_Close": current_close,
                "Next_Close": next_close,
                "BP_PredictedDirection": bp,
                "NextCandle_ActualDirection": actual,
                "Correct": correct,
            }
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Timestamp",
        "Current_Close",
        "Next_Close",
        "BP_PredictedDirection",
        "NextCandle_ActualDirection",
        "Correct",
    ]
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    total = len(records)
    correct = sum(int(r["Correct"]) for r in records)
    longs = [r for r in records if int(r["BP_PredictedDirection"]) == 1]
    shorts = [r for r in records if int(r["BP_PredictedDirection"]) == -1]
    long_correct = sum(int(r["Correct"]) for r in longs)
    short_correct = sum(int(r["Correct"]) for r in shorts)

    all_rows_with_bp = [
        r for r in result[20:-1] if int(r["BP_PredictedDirection"]) != 0
    ]
    zero_bp = sum(
        1 for r in result[20:-1] if int(r["BP_PredictedDirection"]) == 0
    )

    print(f"input={source}")
    print(f"output={target}")
    print("TEST=BP_CURRENT_CANDLE_PREDICTS_NEXT_CANDLE")
    print("BP=1 -> next close must be above current close")
    print("BP=-1 -> next close must be below current close")
    print("BP=0 -> excluded from statistics")
    print(f"rows={len(rows)}")
    print(f"eligible_rows={len(result[20:-1])}")
    print(f"bp_nonzero={len(all_rows_with_bp)}")
    print(f"bp_zero_excluded={zero_bp}")
    print()
    print("RESULT")
    print(
        f"all={correct}/{total} accuracy={correct / total * 100 if total else 0.0:.4f}%"
    )
    print(
        f"long={long_correct}/{len(longs)} accuracy={long_correct / len(longs) * 100 if longs else 0.0:.4f}%"
    )
    print(
        f"short={short_correct}/{len(shorts)} accuracy={short_correct / len(shorts) * 100 if shorts else 0.0:.4f}%"
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
