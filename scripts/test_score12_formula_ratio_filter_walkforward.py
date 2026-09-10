from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_THRESHOLDS = "1.0,1.05,1.1,1.2,1.3,1.5,1.75,2.0,2.5,3.0"


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
    df["J"] = body
    df["U"] = df.J.shift(-1) - df.J
    df["actual"] = np.sign(df.U)

    upper = (df.high - df.open).abs()
    lower = (df.open - df.low).abs()
    df["formula"] = np.where(lower > upper, 1, np.where(upper > lower, -1, 0))
    mn = np.minimum(upper, lower)
    mx = np.maximum(upper, lower)
    df["ratio"] = np.where(mn > 0, mx / mn, np.inf)
    return df.iloc[:-1].copy()


def baseline_pred(df: pd.DataFrame) -> np.ndarray:
    return np.where(df.score <= 1, -1, 1).astype(int)


def filtered_pred(df: pd.DataFrame, t1: float, t2: float) -> np.ndarray:
    pred = baseline_pred(df)
    score = df.score.to_numpy(int)
    formula = df.formula.to_numpy(int)
    ratio = df.ratio.to_numpy(float)

    drop = ((score == 1) & (formula == 1) & (ratio >= t1)) | ((score == 2) & (formula == -1) & (ratio >= t2))
    pred[drop] = 0
    return pred


def evaluate(actual: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, float, float]:
    valid = actual != 0
    active = valid & (pred != 0)
    n_total = int(valid.sum())
    n_active = int(active.sum())
    correct = int(np.sum(pred[active] == actual[active]))
    acc = correct / n_active if n_active else float("nan")
    cov = n_active / n_total if n_total else float("nan")
    return n_total, n_active, correct, acc, cov


def walkforward(df: pd.DataFrame, t1: float, t2: float, folds: int, train_ratio: float) -> tuple[int, int, int, float, float]:
    n = len(df)
    train0 = int(n * train_ratio)
    test_size = (n - train0) // folds
    total = active = correct = 0

    for k in range(folds):
        start = train0 + k * test_size
        end = train0 + (k + 1) * test_size if k < folds - 1 else n
        chunk = df.iloc[start:end]
        n_total, n_active, c, _, _ = evaluate(chunk.actual.to_numpy(int), filtered_pred(chunk, t1, t2))
        total += n_total
        active += n_active
        correct += c

    acc = correct / active if active else float("nan")
    cov = active / total if total else float("nan")
    return total, active, correct, acc, cov


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward ratio-strength filter scan for Score 1/2 formula disagreement")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.50)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    thresholds = sorted({float(x) for x in args.thresholds.split(",") if x.strip()})
    frames = {s: load(Path(args.data_dir) / f"{s}_1h.csv") for s in symbols}

    baseline_total = baseline_active = baseline_correct = 0
    for df in frames.values():
        n, a, c, _, _ = walkforward(df, float("inf"), float("inf"), args.folds, args.train_ratio)
        baseline_total += n
        baseline_active += a
        baseline_correct += c
    baseline_acc = baseline_correct / baseline_active

    results = []
    for t1, t2 in product(thresholds, repeat=2):
        total = active = correct = 0
        for df in frames.values():
            n, a, c, _, _ = walkforward(df, t1, t2, args.folds, args.train_ratio)
            total += n
            active += a
            correct += c
        acc = correct / active if active else float("nan")
        cov = active / total if total else float("nan")
        results.append((acc, cov, t1, t2, active, total))

    results.sort(key=lambda r: (r[0], r[1]), reverse=True)

    print("=" * 100)
    print("WALK-FORWARD: SCORE 1/2 FORMULA-DISAGREEMENT RATIO FILTER SCAN")
    print("=" * 100)
    print("Baseline: Score 0/1 = SHORT, Score 2/3 = LONG")
    print("Filter: S1+FormulaLong and S2+FormulaShort; ratio=max(H-O,O-L)/min(H-O,O-L)")
    print(f"folds={args.folds} train_ratio={args.train_ratio:.2f}")
    print(f"thresholds={','.join(f'{x:g}' for x in thresholds)}")
    print(f"BASELINE: acc={baseline_acc*100:.2f}% coverage=100.00% N={baseline_active}")
    print()

    print("TOP 10 POOLED FILTERS")
    print("rank   t1(S1)  t2(S2)    accuracy   coverage    delta")
    for i, (acc, cov, t1, t2, active, total) in enumerate(results[:10], 1):
        print(f"{i:>4}   {t1:7.2f} {t2:7.2f}     {acc*100:7.2f}%   {cov*100:7.2f}%   {(acc-baseline_acc)*100:+6.2f}pp")

    print()
    print("SAME-THRESHOLD SUMMARY")
    print(" t       accuracy   coverage    delta")
    for t in thresholds:
        row = next(r for r in results if r[2] == t and r[3] == t)
        acc, cov, *_ = row
        print(f"{t:5.2f}     {acc*100:7.2f}%   {cov*100:7.2f}%   {(acc-baseline_acc)*100:+6.2f}pp")

    print("=" * 100)


if __name__ == "__main__":
    main()
