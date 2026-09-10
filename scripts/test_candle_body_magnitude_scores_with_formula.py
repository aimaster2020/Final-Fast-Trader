from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS_DEFAULT = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in {"open", "high", "low", "close"})
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["J"] = df.close - df.open
    df["K"] = df.high - df.close
    df["L_checkpoint"] = df.close - df.low
    df["U"] = df.J.shift(-1) - df.J
    df["next_close"] = df.close.shift(-1)
    df = df.dropna().reset_index(drop=True)

    # CANDLE-BODY-MAGNITUDE-V1 score: three binary conditions, each worth 1 point.
    # HC<CO  => (H-C) < (C-O)
    # HC<HO  => (H-C) < (H-O)
    # LC<CO  => (C-L) < (C-O)
    df["HC_LT_CO"] = df.K < df.J
    df["HC_LT_HO"] = df.K < (df.high - df.open)
    df["LC_LT_CO"] = df.L_checkpoint < df.J
    df["score"] = (
        df["HC_LT_CO"].astype(int)
        + df["HC_LT_HO"].astype(int)
        + df["LC_LT_CO"].astype(int)
    )

    # New formula direction used in the recent experiment.
    df["formula_dir"] = np.where(
        (df.high - df.open) > (df.open - df.low), 1,
        np.where((df.open - df.low) > (df.high - df.open), -1, 0),
    )
    df["actual_dir"] = np.sign(df.U.to_numpy(float)).astype(int)
    return df


def acc(mask: np.ndarray, pred: int, actual: np.ndarray) -> tuple[int, float]:
    n = int(mask.sum())
    if n == 0:
        return 0, float("nan")
    return n, float(np.mean(actual[mask] == pred))


def print_score_section(df: pd.DataFrame, symbol: str) -> list[dict]:
    actual = df.actual_dir.to_numpy(int)
    formula = df.formula_dir.to_numpy(int)
    rows: list[dict] = []

    print(f"\n{symbol}")
    print("-" * 132)
    print("score  N     U+%     U-%    3-up_acc  2-up_acc  1-down_acc  0-down_acc  formula_dir_acc  anti_agreement_acc  anti_agreement_cov")

    # For each individual score, evaluate both the natural mapping candidates and the formula-veto version.
    for s in (0, 1, 2, 3):
        mask = df.score.to_numpy(int) == s
        n = int(mask.sum())
        up_pct = float(np.mean(actual[mask] > 0)) * 100 if n else float("nan")
        down_pct = float(np.mean(actual[mask] < 0)) * 100 if n else float("nan")

        # Natural direction mappings:
        # score 3 / 2 => UP, score 1 / 0 => DOWN.
        natural_pred = 1 if s >= 2 else -1
        natural_acc = float(np.mean(actual[mask] == natural_pred)) if n else float("nan")

        # Formula direction by itself within this score bucket.
        valid_formula = mask & (formula != 0)
        formula_acc = float(np.mean(actual[valid_formula] == formula[valid_formula])) if valid_formula.any() else float("nan")

        # Anti-agreement: retain natural score direction only when formula direction disagrees.
        # This is evaluated as a filter inside each score bucket.
        disagree = mask & (formula != 0) & (formula != natural_pred)
        veto_acc = float(np.mean(actual[disagree] == natural_pred)) if disagree.any() else float("nan")
        veto_cov = float(disagree.sum() / n) if n else float("nan")

        print(
            f"  {s}  {n:4d}  {up_pct:6.2f}%  {down_pct:6.2f}%"
            f"    {natural_acc:7.2%}  {natural_acc:7.2%}  {natural_acc:10.2%}  {natural_acc:10.2%}"
            f"      {formula_acc:7.2%}           {veto_acc:7.2%}          {veto_cov:7.2%}"
        )

        rows.append({
            "symbol": symbol,
            "score": s,
            "n": n,
            "up_pct": up_pct / 100,
            "down_pct": down_pct / 100,
            "natural_prediction": natural_pred,
            "natural_accuracy": natural_acc,
            "formula_only_accuracy": formula_acc,
            "anti_agreement_accuracy": veto_acc,
            "anti_agreement_coverage": veto_cov,
        })

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Test CANDLE-BODY-MAGNITUDE-V1 scores 0..3 with the new H-O/O-L direction rule")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=SYMBOLS_DEFAULT)
    ap.add_argument("--output-dir", default="reports/candle_body_magnitude_scores_formula")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    all_rows: list[dict] = []

    print("=" * 132)
    print("CANDLE-BODY-MAGNITUDE-V1 — SCORE 0..3 + NEW H-O/O-L DIRECTION RULE")
    print("=" * 132)
    print("conditions: HC<CO, HC<HO, LC<CO; each true condition = 1 point; score range = 0..3")
    print("natural mapping tested per score: 0/1 -> DOWN, 2/3 -> UP")
    print("formula direction: LONG if H-O > O-L; SHORT if O-L > H-O")
    print("anti-agreement: keep natural score direction only when formula direction disagrees")
    print("actual direction: sign(U), U=J_next-J_current, matching the existing direction checkpoint")

    for symbol in symbols:
        path = Path(args.data_dir) / f"{symbol}_1h.csv"
        df = load(str(path))
        rows = print_score_section(df, symbol)
        all_rows.extend(rows)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(out / "scores_formula_summary.csv", index=False)

    # Also provide a simpler pooled view across symbols.
    pooled: list[dict] = []
    for s in (0, 1, 2, 3):
        parts = summary[summary.score == s]
        n = int(parts.n.sum())
        if n:
            pooled.append({
                "score": s,
                "n": n,
                "weighted_natural_accuracy": float(np.average(parts.natural_accuracy, weights=parts.n)),
                "weighted_formula_only_accuracy": float(np.average(parts.formula_only_accuracy.fillna(0), weights=parts.n)),
                "weighted_veto_accuracy": float(np.average(parts.anti_agreement_accuracy.fillna(0), weights=parts.n)),
            })
    pd.DataFrame(pooled).to_csv(out / "scores_formula_pooled.csv", index=False)

    print("=" * 132)
    print(f"summary_csv={out / 'scores_formula_summary.csv'}")
    print(f"pooled_csv={out / 'scores_formula_pooled.csv'}")
    print("IMPORTANT: this pass evaluates each existing score bucket separately; it does not alter the original rule.")


if __name__ == "__main__":
    main()
