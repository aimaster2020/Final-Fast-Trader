#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Walk-forward weighted BP with train-selected score threshold.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-train-samples", type=int, default=50)
    return p.parse_args()


COMPONENTS = {
    "Trend_Up": ("bull", "trend_up"),
    "Trend_Down": ("bear", "trend_down"),
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


def main():
    a = parse_args()
    source = Path(a.input)
    target = Path(a.output)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from backtest_exact_excel_formulas import load_rows, calculate

    rows = load_rows(source)
    result = calculate(rows)
    n = len(result)
    split = n // 2

    def active(i: int, kind: str, key: str) -> bool:
        if key == "trend_up":
            return result[i]["AW_Trend"] == "صعودی"
        if key == "trend_down":
            return result[i]["AW_Trend"] == "نزولی"
        return int(result[i][key]) == 1

    train = {name: [0, 0] for name in COMPONENTS}
    for i in range(split):
        c0 = float(rows[i]["Close"])
        c1 = float(rows[i + 1]["Close"])
        actual = 1 if c1 > c0 else -1 if c1 < c0 else 0
        if actual == 0:
            continue
        for name, (kind, key) in COMPONENTS.items():
            if not active(i, kind, key):
                continue
            train[name][0] += 1
            pred = 1 if kind == "bull" else -1
            if pred == actual:
                train[name][1] += 1

    weights = {}
    polarity = {}
    for name, (samples, correct) in train.items():
        if samples < a.min_train_samples:
            weights[name] = 0.0
            polarity[name] = 1
            continue
        acc = correct / samples
        polarity[name] = 1 if acc >= 0.5 else -1
        weights[name] = abs(acc - 0.5) * 2.0

    def score_at(i: int) -> float:
        score = 0.0
        for name, (kind, key) in COMPONENTS.items():
            w = weights.get(name, 0.0)
            if w <= 0 or not active(i, kind, key):
                continue
            base = 1 if kind == "bull" else -1
            score += w * base * polarity[name]
        return score

    train_samples = []
    for i in range(split):
        c0 = float(rows[i]["Close"])
        c1 = float(rows[i + 1]["Close"])
        actual = 1 if c1 > c0 else -1 if c1 < c0 else 0
        if actual == 0:
            continue
        s = score_at(i)
        pred = 1 if s > 0 else -1 if s < 0 else 0
        train_samples.append((s, actual, pred))

    # Threshold grid in score units. Select on train by highest accuracy,
    # breaking ties by more evaluated samples, then lower threshold.
    thresholds = [x / 100 for x in range(0, 101, 5)]
    candidates = []
    for threshold in thresholds:
        used = [(s, a) for s, a, _ in train_samples if abs(s) >= threshold and s != 0]
        if not used:
            continue
        correct = sum(1 for s, actual in used if (1 if s > 0 else -1) == actual)
        acc = correct / len(used)
        candidates.append((acc, len(used), -threshold, threshold))

    if not candidates:
        raise SystemExit("No train samples available for threshold selection.")

    best_acc, best_count, _, selected_threshold = max(candidates)

    def evaluate(start: int, threshold: float):
        total = correct = long_n = long_c = short_n = short_c = 0
        details = []
        for i in range(start, n - 1):
            c0 = float(rows[i]["Close"])
            c1 = float(rows[i + 1]["Close"])
            actual = 1 if c1 > c0 else -1 if c1 < c0 else 0
            if actual == 0:
                continue
            score = score_at(i)
            if score == 0 or abs(score) < threshold:
                continue
            pred = 1 if score > 0 else -1
            ok = int(pred == actual)
            total += 1
            correct += ok
            if pred == 1:
                long_n += 1
                long_c += ok
            else:
                short_n += 1
                short_c += ok
            details.append({
                "Timestamp": rows[i]["Timestamp"],
                "Current_Close": c0,
                "Next_Close": c1,
                "Weighted_Score": f"{score:.8f}",
                "Threshold": f"{threshold:.8f}",
                "Predicted": pred,
                "Actual": actual,
                "Correct": ok,
            })
        acc = correct / total * 100 if total else 0.0
        lacc = long_c / long_n * 100 if long_n else 0.0
        sacc = short_c / short_n * 100 if short_n else 0.0
        return total, correct, acc, long_n, long_c, lacc, short_n, short_c, sacc, details

    train_eval = evaluate(0, selected_threshold)
    test_eval = evaluate(split, selected_threshold)

    target.parent.mkdir(parents=True, exist_ok=True)
    detail_path = target.with_name(target.stem + "_details.csv")
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "SelectedThreshold",
            "TrainAccuracy_%",
            "TrainSamples",
            "TestAccuracy_%",
            "TestSamples",
            "TestCorrect",
            "TestLongAccuracy_%",
            "TestLongSamples",
            "TestShortAccuracy_%",
            "TestShortSamples",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({
            "SelectedThreshold": f"{selected_threshold:.8f}",
            "TrainAccuracy_%": f"{train_eval[2]:.4f}",
            "TrainSamples": train_eval[0],
            "TestAccuracy_%": f"{test_eval[2]:.4f}",
            "TestSamples": test_eval[0],
            "TestCorrect": test_eval[1],
            "TestLongAccuracy_%": f"{test_eval[5]:.4f}",
            "TestLongSamples": test_eval[3],
            "TestShortAccuracy_%": f"{test_eval[8]:.4f}",
            "TestShortSamples": test_eval[6],
        })

    with detail_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "Timestamp", "Current_Close", "Next_Close", "Weighted_Score",
            "Threshold", "Predicted", "Actual", "Correct",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(test_eval[9])

    print(f"input={source}")
    print(f"output={target}")
    print(f"details={detail_path}")
    print("TEST=WEIGHTED_BP_THRESHOLD_WALKFORWARD")
    print(f"rows={n} split={split} min_train_samples={a.min_train_samples}")
    print(f"selected_threshold={selected_threshold:.4f}")
    print(f"train_threshold_accuracy={train_eval[2]:.4f}% samples={train_eval[0]}")
    print("EVALUATION=second half only")
    print()
    print("RESULT")
    print(f"all={test_eval[1]}/{test_eval[0]} accuracy={test_eval[2]:.4f}%")
    print(f"long={test_eval[4]}/{test_eval[3]} accuracy={test_eval[5]:.4f}%")
    print(f"short={test_eval[7]}/{test_eval[6]} accuracy={test_eval[8]:.4f}%")


if __name__ == "__main__":
    main()
