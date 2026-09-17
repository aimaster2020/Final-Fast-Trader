#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test individual Excel BP components against the next candle direction."
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    target = Path(args.output)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from backtest_exact_excel_formulas import load_rows, calculate
    except ImportError as exc:
        raise SystemExit("Could not import backtest_exact_excel_formulas.py") from exc

    rows = load_rows(source)
    if len(rows) < 2:
        raise SystemExit("Need at least 2 usable OHLC rows.")

    result = calculate(rows)

    # These are the actual components used to build Z/AU/AV/BP.
    components = {
        "Trend": ("trend", None),
        "K_Hammer": ("bull", "K"),
        "L_InvertedHammer": ("bull", "L"),
        "M_DragonflyDoji": ("bull", "M"),
        "N_BullishMarubozu": ("bull", "N"),
        "O_BullishEngulfing": ("bull", "O"),
        "P_BullishHarami": ("bull", "P"),
        "Q_PiercingLine": ("bull", "Q"),
        "R_TweezerBottom": ("bull", "R"),
        "S_MorningStar": ("bull", "S"),
        "T_ThreeWhiteSoldiers": ("bull", "T"),
        "U_BullishOutsideBar": ("bull", "U"),
        "V_BullishPinBar": ("bull", "V"),
        "W_BullishBreakout": ("bull", "W"),
        "X_BullishFalseBreakout": ("bull", "X"),
        "Y_BullishRetest": ("bull", "Y"),
        "AF_HangingManLike": ("bear", "AF"),
        "AG_ShootingStar": ("bear", "AG"),
        "AH_GravestoneDoji": ("bear", "AH"),
        "AI_BearishMarubozu": ("bear", "AI"),
        "AJ_BearishEngulfing": ("bear", "AJ"),
        "AK_BearishHarami": ("bear", "AK"),
        "AL_DarkCloud": ("bear", "AL"),
        "AM_TweezerTop": ("bear", "AM"),
        "AN_EveningStar": ("bear", "AN"),
        "AO_ThreeBlackCrows": ("bear", "AO"),
        "AP_BearishOutsideBar": ("bear", "AP"),
        "AQ_BearishPinBar": ("bear", "AQ"),
        "AR_BearishBreakdown": ("bear", "AR"),
        "AS_BearishFalseBreakdown": ("bear", "AS"),
        "AT_BearishRetest": ("bear", "AT"),
    }

    stats = {
        name: {"samples": 0, "up": 0, "down": 0, "correct": 0}
        for name in components
    }
    records: list[dict[str, object]] = []

    for i in range(len(result) - 1):
        next_close = float(rows[i + 1]["Close"])
        current_close = float(rows[i]["Close"])
        actual = 1 if next_close > current_close else -1 if next_close < current_close else 0
        if actual == 0:
            continue

        active: list[str] = []
        for name, (kind, key) in components.items():
            if kind == "trend":
                active_now = result[i]["AW_Trend"] == "صعودی"
            else:
                active_now = int(result[i][key]) == 1
            if not active_now:
                continue

            predicted = 1 if kind in ("trend", "bull") else -1
            stats[name]["samples"] += 1
            if actual == 1:
                stats[name]["up"] += 1
            else:
                stats[name]["down"] += 1
            if predicted == actual:
                stats[name]["correct"] += 1
            active.append(name)

        records.append(
            {
                "Timestamp": rows[i]["Timestamp"],
                "Current_Close": current_close,
                "Next_Close": next_close,
                "Actual_Next_Direction": actual,
                "Active_Components": "|".join(active),
            }
        )

    summary = []
    for name, s in stats.items():
        samples = s["samples"]
        accuracy = s["correct"] / samples * 100 if samples else 0.0
        summary.append(
            {
                "Component": name,
                "Samples": samples,
                "Next_Up": s["up"],
                "Next_Down": s["down"],
                "Correct": s["correct"],
                "Accuracy_%": f"{accuracy:.4f}",
            }
        )

    summary.sort(key=lambda x: float(x["Accuracy_%"]), reverse=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    summary_path = target
    details_path = target.with_name(target.stem + "_details.csv")

    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["Component", "Samples", "Next_Up", "Next_Down", "Correct", "Accuracy_%"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    with details_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["Timestamp", "Current_Close", "Next_Close", "Actual_Next_Direction", "Active_Components"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    print(f"input={source}")
    print(f"output={summary_path}")
    print(f"details={details_path}")
    print("TEST=INDIVIDUAL_BP_COMPONENTS_NEXT_CANDLE")
    print("Bullish component active -> predicts next direction +1")
    print("Bearish component active -> predicts next direction -1")
    print("Trend=صعودی -> predicts +1; Trend=نزولی is not used as a bearish component")
    print("zero-change next candles excluded")
    print(f"rows={len(rows)}")
    print()
    print("RESULT")
    for row in summary:
        print(
            f"{row['Component']}: samples={row['Samples']} "
            f"correct={row['Correct']} accuracy={row['Accuracy_%']}%"
        )


if __name__ == "__main__":
    main()
