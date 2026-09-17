#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

COMPONENTS = {
    "K_Hammer": ("bull", "K"), "L_InvertedHammer": ("bull", "L"), "M_DragonflyDoji": ("bull", "M"),
    "N_BullishMarubozu": ("bull", "N"), "O_BullishEngulfing": ("bull", "O"), "P_BullishHarami": ("bull", "P"),
    "Q_PiercingLine": ("bull", "Q"), "R_TweezerBottom": ("bull", "R"), "S_MorningStar": ("bull", "S"),
    "T_ThreeWhiteSoldiers": ("bull", "T"), "U_BullishOutsideBar": ("bull", "U"), "V_BullishPinBar": ("bull", "V"),
    "W_BullishBreakout": ("bull", "W"), "X_BullishFalseBreakout": ("bull", "X"), "Y_BullishRetest": ("bull", "Y"),
    "AF_HangingManLike": ("bear", "AF"), "AG_ShootingStar": ("bear", "AG"), "AH_GravestoneDoji": ("bear", "AH"),
    "AI_BearishMarubozu": ("bear", "AI"), "AJ_BearishEngulfing": ("bear", "AJ"), "AK_BearishHarami": ("bear", "AK"),
    "AL_DarkCloud": ("bear", "AL"), "AM_TweezerTop": ("bear", "AM"), "AN_EveningStar": ("bear", "AN"),
    "AO_ThreeBlackCrows": ("bear", "AO"), "AP_BearishOutsideBar": ("bear", "AP"), "AQ_BearishPinBar": ("bear", "AQ"),
    "AR_BearishBreakdown": ("bear", "AR"), "AS_BearishFalseBreakdown": ("bear", "AS"), "AT_BearishRetest": ("bear", "AT"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Learn BP component polarity on the first half and test on the second half.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-train-samples", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    target = Path(args.output)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from backtest_exact_excel_formulas import load_rows, calculate

    rows = load_rows(source)
    result = calculate(rows)
    if len(rows) < 4:
        raise SystemExit("Need at least 4 rows.")

    split = (len(result) - 1) // 2
    stats = {name: {"train_samples": 0, "train_direct": 0, "test_samples": 0, "test_correct": 0} for name in COMPONENTS}

    # Learn polarity only from first half.
    for name, (kind, key) in COMPONENTS.items():
        s = stats[name]
        for i in range(split):
            nxt = float(rows[i + 1]["Close"])
            cur = float(rows[i]["Close"])
            actual = 1 if nxt > cur else -1 if nxt < cur else 0
            if actual == 0 or int(result[i][key]) != 1:
                continue
            s["train_samples"] += 1
            pred = 1 if kind == "bull" else -1
            if pred == actual:
                s["train_direct"] += 1

        inv = s["train_samples"] - s["train_direct"]
        if s["train_samples"] >= args.min_train_samples and inv > s["train_direct"]:
            s["polarity"] = "INVERSE"
        else:
            s["polarity"] = "DIRECT"

    # Test learned polarity only on second half.
    records = []
    for i in range(split, len(result) - 1):
        nxt = float(rows[i + 1]["Close"])
        cur = float(rows[i]["Close"])
        actual = 1 if nxt > cur else -1 if nxt < cur else 0
        if actual == 0:
            continue
        for name, (kind, key) in COMPONENTS.items():
            if int(result[i][key]) != 1:
                continue
            s = stats[name]
            base_pred = 1 if kind == "bull" else -1
            pred = -base_pred if s["polarity"] == "INVERSE" else base_pred
            s["test_samples"] += 1
            if pred == actual:
                s["test_correct"] += 1

    summary = []
    for name, s in stats.items():
        train_n = s["train_samples"]
        train_direct = s["train_direct"]
        test_n = s["test_samples"]
        test_correct = s["test_correct"]
        train_direct_acc = train_direct / train_n * 100 if train_n else 0.0
        train_inverse_acc = (train_n - train_direct) / train_n * 100 if train_n else 0.0
        test_acc = test_correct / test_n * 100 if test_n else 0.0
        summary.append({
            "Component": name,
            "TrainSamples": train_n,
            "TrainDirect_%": f"{train_direct_acc:.4f}",
            "TrainInverse_%": f"{train_inverse_acc:.4f}",
            "LearnedPolarity": s["polarity"],
            "TestSamples": test_n,
            "TestCorrect": test_correct,
            "TestAccuracy_%": f"{test_acc:.4f}",
        })

    summary.sort(key=lambda x: float(x["TestAccuracy_%"]), reverse=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        fields = list(summary[0].keys()) if summary else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    print(f"input={source}")
    print(f"output={target}")
    print("TEST=BP_COMPONENT_POLARITY_WALKFORWARD")
    print("POLARITY learned on first half, evaluated only on second half")
    print(f"rows={len(rows)} split={split} min_train_samples={args.min_train_samples}")
    print()
    print("RESULT")
    for r in summary:
        print(
            f"{r['Component']}: train={r['TrainSamples']} "
            f"polarity={r['LearnedPolarity']} test={r['TestSamples']} "
            f"accuracy={r['TestAccuracy_%']}%"
        )


if __name__ == "__main__":
    main()
