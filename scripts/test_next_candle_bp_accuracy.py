#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test Excel BP prediction against the next candle close direction. BP=0 is excluded."
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def exact_initial_trend(rows, i: int) -> int:
    # Excel AW formula uses a progressively available prior window:
    # row 2: AVERAGE(B1:E2) -> header text is ignored, so current row only.
    # row 3: AVERAGE(B2:E2) -> previous 1 row.
    # row 4: AVERAGE(B2:E3) -> previous 2 rows.
    # row 5: AVERAGE(B2:E4) -> previous 3 rows.
    # row 6+: rolling previous 4 rows.
    if i == 0:
        values = rows[0:1]
    else:
        start = max(0, i - 4)
        values = rows[start:i]
    total = 0.0
    count = 0
    for r in values:
        for key in ("Open", "High", "Low", "Close"):
            total += float(r[key])
            count += 1
    avg = total / count if count else 0.0
    return 1 if float(rows[i]["Close"]) > avg else 0


def patch_bp_for_excel_initial_rows(rows, result) -> None:
    # The base calculator already reproduces all candle-pattern columns.
    # Patch only the first four Trend-dependent score rows, where Excel uses
    # shorter historical windows instead of the normal 4-candle window.
    limit = min(4, len(result))
    bullish_keys = list("KLMNOPQRSTUVWXY")
    bearish_keys = list("AFGHIJKLMNOPQRS")

    for i in range(limit):
        trend = exact_initial_trend(rows, i)
        bullish = trend + sum(int(result[i][k]) for k in bullish_keys)
        bearish = -(trend + sum(int(result[i][k]) for k in bearish_keys))
        if bullish + bearish > 0:
            final_score = bullish
        elif bullish + bearish < 0:
            final_score = bearish
        else:
            final_score = 0
        result[i]["Z_BullScore"] = bullish
        result[i]["AU_BearScore"] = bearish
        result[i]["AV_FinalScore"] = final_score
        result[i]["BP_PredictedDirection"] = 1 if final_score > 0 else -1 if final_score < 0 else 0


def main() -> None:
    a = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from backtest_exact_excel_formulas import load_rows, calculate
    except ImportError as exc:
        raise SystemExit("Could not import backtest_exact_excel_formulas.py") from exc

    source = Path(a.input)
    target = Path(a.output)
    rows = load_rows(source)
    if len(rows) < 2:
        raise SystemExit("Need at least 2 usable OHLC rows.")

    result = calculate(rows)
    patch_bp_for_excel_initial_rows(rows, result)

    # Every non-zero BP is eligible. BP=0 is removed from the denominator.
    # BP at candle i predicts candle i+1 direction:
    # next Close > current Close => +1, next Close < current Close => -1.
    records: list[dict[str, object]] = []
    for i in range(0, len(result) - 1):
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
    zero_bp = sum(1 for r in result[:-1] if int(r["BP_PredictedDirection"]) == 0)

    print(f"input={source}")
    print(f"output={target}")
    print("TEST=BP_CURRENT_CANDLE_PREDICTS_NEXT_CANDLE")
    print("BP=1 -> next close must be above current close")
    print("BP=-1 -> next close must be below current close")
    print("BP=0 -> excluded from statistics")
    print("WINDOWS=exact Excel row-relative windows")
    print(f"rows={len(rows)}")
    print(f"bp_nonzero={total}")
    print(f"bp_zero_excluded={zero_bp}")
    print()
    print("RESULT")
    print(f"all={correct}/{total} accuracy={correct / total * 100 if total else 0.0:.4f}%")
    print(f"long={long_correct}/{len(longs)} accuracy={long_correct / len(longs) * 100 if longs else 0.0:.4f}%")
    print(f"short={short_correct}/{len(shorts)} accuracy={short_correct / len(shorts) * 100 if shorts else 0.0:.4f}%")


if __name__ == "__main__":
    main()
