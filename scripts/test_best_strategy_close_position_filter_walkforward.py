from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_THRESHOLDS = "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in {"open", "high", "low", "close"})
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    body = df.close - df.open
    hc = df.high - df.close
    ho = df.high - df.open
    lc = df.low - df.close
    df["score"] = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)

    upper = (df.high - df.open).abs()
    lower = (df.open - df.low).abs()
    df["formula"] = np.where(lower > upper, 1, np.where(upper > lower, -1, 0))

    rng = df.high - df.low
    df["close_pos"] = np.where(rng > 0, (df.close - df.low) / rng, 0.5)

    j = body
    df["actual"] = np.sign(j.shift(-1) - j)
    return df.iloc[:-1].copy()


def base_pred(df: pd.DataFrame) -> np.ndarray:
    pred = np.where(df.score <= 1, -1, 1).astype(int)
    score = df.score.to_numpy(int)
    formula = df.formula.to_numpy(int)
    drop = ((score == 1) & (formula == 1)) | ((score == 2) & (formula == -1))
    pred[drop] = 0
    return pred


def evaluate(actual: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, float, float]:
    valid = actual != 0
    active = valid & (pred != 0)
    n = int(valid.sum())
    a = int(active.sum())
    c = int(np.sum(pred[active] == actual[active]))
    return n, a, c, c / a if a else float("nan"), a / n if n else float("nan")


def filtered_pred(df: pd.DataFrame, threshold: float, mode: str) -> np.ndarray:
    pred = base_pred(df)
    score = df.score.to_numpy(int)
    pos = df.close_pos.to_numpy(float)

    # Current best strategy only has active Score 1/2 trades when formula agrees
    # with the baseline direction. Add a close-position confirmation filter.
    if mode == "both":
        drop = ((score == 1) & (pos > threshold)) | ((score == 2) & (pos < (1.0 - threshold)))
    elif mode == "score1":
        drop = (score == 1) & (pos > threshold)
    else:
        drop = (score == 2) & (pos < (1.0 - threshold))
    pred[drop & (pred != 0)] = 0
    return pred


def walkforward(df: pd.DataFrame, threshold: float, mode: str, folds: int, train_ratio: float) -> tuple[int, int, int, float, float]:
    n = len(df)
    train0 = int(n * train_ratio)
    test_size = (n - train0) // folds
    total = active = correct = 0
    for k in range(folds):
        start = train0 + k * test_size
        end = train0 + (k + 1) * test_size if k < folds - 1 else n
        chunk = df.iloc[start:end]
        a = chunk.actual.to_numpy(int)
        p = filtered_pred(chunk, threshold, mode)
        nt, na, c, _, _ = evaluate(a, p)
        total += nt
        active += na
        correct += c
    acc = correct / active if active else float("nan")
    cov = active / total if total else float("nan")
    return total, active, correct, acc, cov


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward close-position filter on the current best Score x Formula strategy")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.50)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    thresholds = sorted({float(x) for x in args.thresholds.split(",") if x.strip()})
    frames = {s: load(Path(args.data_dir) / f"{s}_1h.csv") for s in symbols}

    rows = []
    for mode in ("both", "score1", "score2"):
        for t in thresholds:
            total = active = correct = 0
            for df in frames.values():
                n, a, c, _, _ = walkforward(df, t, mode, args.folds, args.train_ratio)
                total += n
                active += a
                correct += c
            acc = correct / active if active else float("nan")
            cov = active / total if total else float("nan")
            rows.append((acc, cov, mode, t, active, total))

    base_total = base_active = base_correct = 0
    for df in frames.values():
        n, a, c, _, _ = walkforward(df, 1.0, "both", args.folds, args.train_ratio)
        base_total += n
        base_active += a
        base_correct += c
    base_acc = base_correct / base_active

    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)

    print("=" * 108)
    print("WALK-FORWARD: BEST STRATEGY + CLOSE POSITION FILTER")
    print("=" * 108)
    print("Base: Score 0/1 SHORT, Score 2/3 LONG")
    print("Current best filter: drop S1+FormulaLong and S2+FormulaShort")
    print("Close position = (Close-Low)/(High-Low)")
    print(f"folds={args.folds} train_ratio={args.train_ratio:.2f}")
    print(f"BASE STRATEGY: acc={base_acc*100:.2f}% coverage=100.00% N={base_active}")
    print()
    print("TOP 10 POOLED FILTERS")
    print("rank mode       threshold   accuracy   coverage    delta")
    for i, (acc, cov, mode, t, active, total) in enumerate(rows[:10], 1):
        print(f"{i:>4} {mode:>9}      {t:7.2f}     {acc*100:7.2f}%   {cov*100:7.2f}%   {(acc-base_acc)*100:+6.2f}pp")

    print()
    print("SAME THRESHOLD / BOTH CELLS")
    print("threshold   accuracy   coverage    delta")
    for t in thresholds:
        acc, cov, mode, _, active, total = next(r for r in rows if r[2] == "both" and r[3] == t)
        print(f"{t:9.2f}   {acc*100:7.2f}%   {cov*100:7.2f}%   {(acc-base_acc)*100:+6.2f}pp")
    print("=" * 108)


if __name__ == "__main__":
    main()
