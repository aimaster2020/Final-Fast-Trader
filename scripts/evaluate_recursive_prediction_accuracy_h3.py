from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=lambda c: str(c).strip().lower() in {"open", "high", "low", "close"},
    )
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


def score_direction(o: float, h: float, l: float, c: float) -> tuple[int, int]:
    body = c - o
    hc = h - c
    ho = h - o
    lc = l - c
    score = int(hc > body) + int(ho < hc) + int(lc > body)
    direction = -1 if score == 0 else 1 if score == 3 else 0
    return score, direction


def predict_next(o: float, h: float, l: float, c: float) -> tuple[float, float, float, float, int]:
    """Same recursive synthetic-frame predictor used by the H3 test."""
    _, direction = score_direction(o, h, l, c)
    if direction == 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    upper = abs(h - o)
    lower = abs(o - l)
    trade_magnitude = upper if direction == 1 else lower
    if not np.isfinite(trade_magnitude) or trade_magnitude <= 1e-12:
        return np.nan, np.nan, np.nan, np.nan, 0

    body_ratio = (c - o) / trade_magnitude
    upper_ratio = upper / trade_magnitude
    lower_ratio = lower / trade_magnitude

    po = c
    pc = po + body_ratio * trade_magnitude
    ph = po + upper_ratio * trade_magnitude
    pl = po - lower_ratio * trade_magnitude
    ph = max(po, pc, ph)
    pl = min(po, pc, pl)

    _, next_direction = score_direction(po, ph, pl, pc)
    return po, ph, pl, pc, next_direction


def evaluate_symbol(
    df: pd.DataFrame,
    mode: str,
) -> dict[str, int | float]:
    eligible = 0
    pred_available = [0, 0, 0]
    correct = [0, 0, 0]
    all_three = 0
    predicted_chain_agree = 0
    predicted_chain_total = 0

    n = len(df)
    for i in range(n - 3):
        o, h, l, c = map(float, df.iloc[i][["open", "high", "low", "close"]])
        _, current_dir = score_direction(o, h, l, c)

        if mode == "LONG" and current_dir != 1:
            continue
        if mode == "SHORT" and current_dir != -1:
            continue
        if current_dir == 0:
            continue

        eligible += 1

        p1 = predict_next(o, h, l, c)
        if p1[4] == 0 or not np.isfinite(p1[3]):
            continue

        p2 = predict_next(p1[0], p1[1], p1[2], p1[3])
        if p2[4] == 0 or not np.isfinite(p2[3]):
            continue

        p3 = predict_next(p2[0], p2[1], p2[2], p2[3])
        if p3[4] == 0 or not np.isfinite(p3[3]):
            continue

        predictions = [p1[4], p2[4], p3[4]]
        actuals = [
            score_direction(*map(float, df.iloc[i + 1][["open", "high", "low", "close"]]))[1],
            score_direction(*map(float, df.iloc[i + 2][["open", "high", "low", "close"]]))[1],
            score_direction(*map(float, df.iloc[i + 3][["open", "high", "low", "close"]]))[1],
        ]

        for k in range(3):
            if actuals[k] == 0:
                continue
            pred_available[k] += 1
            correct[k] += int(predictions[k] == actuals[k])

        if all(a != 0 for a in actuals):
            all_three += int(all(predictions[k] == actuals[k] for k in range(3)))

        predicted_chain_total += 1
        predicted_chain_agree += int(predictions[0] == predictions[1] == predictions[2])

    return {
        "eligible": eligible,
        "p1_n": pred_available[0],
        "p2_n": pred_available[1],
        "p3_n": pred_available[2],
        "p1_acc": correct[0] / pred_available[0] if pred_available[0] else 0.0,
        "p2_acc": correct[1] / pred_available[1] if pred_available[1] else 0.0,
        "p3_acc": correct[2] / pred_available[2] if pred_available[2] else 0.0,
        "all3_acc": all_three / min(pred_available) if min(pred_available) else 0.0,
        "chain_agree": predicted_chain_agree / predicted_chain_total if predicted_chain_total else 0.0,
        "chain_n": predicted_chain_total,
    }


def pooled(
    loaded: dict[str, pd.DataFrame],
    symbols: list[str],
    mode: str,
) -> dict[str, int | float]:
    rows = [evaluate_symbol(loaded[s], mode) for s in symbols]
    eligible = sum(int(r["eligible"]) for r in rows)
    p1_n = sum(int(r["p1_n"]) for r in rows)
    p2_n = sum(int(r["p2_n"]) for r in rows)
    p3_n = sum(int(r["p3_n"]) for r in rows)
    chain_n = sum(int(r["chain_n"]) for r in rows)
    return {
        "eligible": eligible,
        "p1_n": p1_n,
        "p2_n": p2_n,
        "p3_n": p3_n,
        "p1_acc": sum(float(r["p1_acc"]) * int(r["p1_n"]) for r in rows) / p1_n if p1_n else 0.0,
        "p2_acc": sum(float(r["p2_acc"]) * int(r["p2_n"]) for r in rows) / p2_n if p2_n else 0.0,
        "p3_acc": sum(float(r["p3_acc"]) * int(r["p3_n"]) for r in rows) / p3_n if p3_n else 0.0,
        "all3_acc": sum(int(r["all3_acc"] * min(int(r["p1_n"]), int(r["p2_n"]), int(r["p3_n"]))) for r in rows) / min(p1_n, p2_n, p3_n) if min(p1_n, p2_n, p3_n) else 0.0,
        "chain_agree": sum(float(r["chain_agree"]) * int(r["chain_n"]) for r in rows) / chain_n if chain_n else 0.0,
        "chain_n": chain_n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure true directional accuracy of the recursive H3 predictor."
    )
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    root = Path(args.data_dir)
    loaded = {s: load(root / f"{s}_1h.csv") for s in symbols}

    print("=" * 112)
    print("H3 RECURSIVE PREDICTION — TRUE DIRECTIONAL ACCURACY")
    print("=" * 112)
    print("Prediction chain: real t -> synthetic t+1 -> synthetic t+2 -> synthetic t+3")
    print("Accuracy: each predicted frame is compared with the ACTUAL corresponding future candle direction")
    print("Current signal: only real Score 0=SHORT / Score 3=LONG candles are evaluated")
    print("No PnL, fees, entry filter, or trade logic is used")
    print()

    for mode in ("SHORT", "LONG", "BOTH"):
        r = pooled(loaded, symbols, mode)
        print(mode)
        print(
            f"eligible={r['eligible']}  "
            f"P1_acc={float(r['p1_acc'])*100:.2f}% (N={int(r['p1_n'])})  "
            f"P2_acc={float(r['p2_acc'])*100:.2f}% (N={int(r['p2_n'])})  "
            f"P3_acc={float(r['p3_acc'])*100:.2f}% (N={int(r['p3_n'])})"
        )
        print(
            f"ALL_3_CORRECT={float(r['all3_acc'])*100:.2f}%  "
            f"prediction_chain_agreement={float(r['chain_agree'])*100:.2f}%  "
            f"chain_N={int(r['chain_n'])}"
        )
        print()

    print("Interpretation:")
    print("P1_acc = direction accuracy for t+1 prediction")
    print("P2_acc = direction accuracy for t+2 recursive prediction")
    print("P3_acc = direction accuracy for t+3 recursive prediction")
    print("ALL_3_CORRECT = percentage where all three recursive directions were correct")
    print("prediction_chain_agreement = self-agreement of predictions; NOT accuracy")
    print("=" * 112)


if __name__ == "__main__":
    main()
