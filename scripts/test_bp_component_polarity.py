#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test direct and inverse polarity of each Excel BP component against the next candle."
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    target = Path(args.output)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from backtest_exact_excel_formulas import load_rows, calculate
    except ImportError as exc:
        raise SystemExit("Could not import backtest_exact_excel_formulas.py") from exc

    rows = load_rows(source)
    if len(rows) < 2:
        raise SystemExit("Need at least 2 usable OHLC rows.")

    result = calculate(rows)

    components = {
        "Trend_Up": ("bull", None),
        "Trend_Down": ("bear", None),
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
        name: {"samples": 0, "direct_correct": 0, "up": 0, "down": 0}
        for name in components
    }

    for i in range(len(result) - 1):
        current_close = float(rows[i]["Close"])
        next_close = float(rows[i + 1]["Close"])
        actual = 1 if next_close > current_close else -1 if next_close < current_close else 0
        if actual == 0:
            continue

        for name, (kind, key) in components.items():
            if name == "Trend_Up":
                active = result[i]["AW_Trend"] == "صعودی"
            elif name == "Trend_Down":
                active = result[i]["AW_Trend"] == "نزولی"
            else:
                active = int(result[i][key]) == 1

            if not active:
                continue

            predicted = 1 if kind == "bull" else -1
            s = stats[name]
            s["samples"] += 1
            if actual == 1:
                s["up"] += 1
            else:
                s["down"] += 1
            if predicted == actual:
                s["direct_correct"] += 1

    summary = []
    for name, s in stats.items():
        n = s["samples"]
        direct = s["direct_correct"] / n * 100 if n else 0.0
        inverse = 100.0 - direct if n else 0.0
        effective = max(direct, inverse)
        polarity = "DIRECT" if direct >= inverse else "INVERSE"
        summary.append(
            {
                "Component": name,
                "Samples": n,
                "Next_Up": s["up"],
                "Next_Down": s["down"],
                "Direct_Correct": s["direct_correct"],
                "Direct_Accuracy_%": f"{direct:.4f}",
                "Inverse_Accuracy_%": f"{inverse:.4f}",
                "Effective_Accuracy_%": f"{effective:.4f}",
                "Suggested_Polarity": polarity,
                "Edge_From_50_%": f"{abs(direct - 50.0):.4f}",
            }
        )

    summary.sort(key=lambda x: (float(x["Effective_Accuracy_%"]), int(x["Samples"])), reverse=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "Component",
            "Samples",
            "Next_Up",
            "Next_Down",
            "Direct_Correct",
            "Direct_Accuracy_%",
            "Inverse_Accuracy_%",
            "Effective_Accuracy_%",
            "Suggested_Polarity",
            "Edge_From_50_%",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    print(f"input={source}")
    print(f"output={target}")
    print("TEST=BP_COMPONENT_DIRECT_VS_INVERSE_POLARITY")
    print("DIRECT=component direction is used as-is")
    print("INVERSE=component direction is flipped")
    print("zero-change next candles excluded")
    print(f"rows={len(rows)}")
    print()
    print("RESULT")
    for row in summary:
        print(
            f"{row['Component']}: samples={row['Samples']} "
            f"direct={row['Direct_Accuracy_%']}% "
            f"inverse={row['Inverse_Accuracy_%']}% "
            f"effective={row['Effective_Accuracy_%']}% "
            f"polarity={row['Suggested_Polarity']}"
        )


if __name__ == "__main__":
    main()
