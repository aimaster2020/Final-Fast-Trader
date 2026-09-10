from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_THRESHOLDS = "0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,0.80"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in {"open", "high", "low", "close"})
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    body = (df.close - df.open).abs()
    rng = df.high - df.low
    df["score"] = ((df.high - df.close) > (df.close - df.open)).astype(int) + ((df.high - df.open) < (df.high - df.close)).astype(int) + ((df.low - df.close) > (df.close - df.open)).astype(int)
    df["body_range"] = np.where(rng > 0, body / rng, np.nan)
    j = df.close - df.open
    df["u"] = j.shift(-1) - j
    df["actual"] = np.sign(df.u)
    return df.iloc[:-1].copy()


def baseline_pred(df: pd.DataFrame) -> np.ndarray:
    return np.where(df.score.to_numpy(int) <= 1, -1, 1).astype(int)


def filtered_pred(df: pd.DataFrame, threshold: float) -> np.ndarray:
    pred = baseline_pred(df)
    score = df.score.to_numpy(int)
    br = df.body_range.to_numpy(float)
    # Only filter the currently weaker middle buckets; retain Score 0/3 untouched.
    drop = ((score == 1) | (score == 2)) & np.isfinite(br) & (br <= threshold)
    pred[drop] = 0
    return pred


def evaluate(actual: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, float, float]:
    valid = actual != 0
    active = valid & (pred != 0)
    total = int(valid.sum())
    n_active = int(active.sum())
    correct = int(np.sum(pred[active] == actual[active]))
    acc = correct / n_active if n_active else float("nan")
    cov = n_active / total if total else float("nan")
    return total, n_active, correct, acc, cov


def walkforward(df: pd.DataFrame, threshold: float, folds: int, train_ratio: float) -> tuple[int, int, int, float, float]:
    n = len(df)
    train0 = int(n * train_ratio)
    test_size = (n - train0) // folds
    total = active = correct = 0
    for k in range(folds):
        start = train0 + k * test_size
        end = train0 + (k + 1) * test_size if k < folds - 1 else n
        chunk = df.iloc[start:end]
        a, b, c, _, _ = evaluate(chunk.actual.to_numpy(int), filtered_pred(chunk, threshold))
        total += a
        active += b
        correct += c
    acc = correct / active if active else float("nan")
    cov = active / total if total else float("nan")
    return total, active, correct, acc, cov


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward scan of body/range filters inside Score 1/2.")
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
        n, a, c, _, _ = walkforward(df, 1.0, args.folds, args.train_ratio)
        baseline_total += n
        baseline_active += a
        baseline_correct += c
    baseline_acc = baseline_correct / baseline_active

    results = []
    for t in thresholds:
        total = active = correct = 0
        for df in frames.values():
            n, a, c, _, _ = walkforward(df, t, args.folds, args.train_ratio)
            total += n
            active += a
            correct += c
        acc = correct / active if active else float("nan")
        cov = active / total if total else float("nan")
        results.append((acc, cov, t, active, total))
    results.sort(key=lambda r: (r[0], r[1]), reverse=True)

    print("=" * 100)
    print("WALK-FORWARD: SCORE 1/2 BODY-TO-RANGE FILTER SCAN")
    print("=" * 100)
    print("Baseline: Score 0/1 = SHORT, Score 2/3 = LONG")
    print("Filter: drop Score 1/2 when abs(C-O)/(H-L) <= threshold")
    print(f"folds={args.folds} train_ratio={args.train_ratio:.2f}")
    print(f"BASELINE: acc={baseline_acc*100:.2f}% coverage=100.00% N={baseline_active}")
    print()
    print("TOP 10 POOLED FILTERS")
    print("rank threshold   accuracy   coverage    delta")
    for i, (acc, cov, t, active, total) in enumerate(results[:10], 1):
        print(f"{i:>4}   {t:7.2f}     {acc*100:7.2f}%   {cov*100:7.2f}%   {(acc-baseline_acc)*100:+6.2f}pp")
    print()
    print("ALL THRESHOLDS")
    print(" threshold   accuracy   coverage    delta")
    for acc, cov, t, active, total in sorted(results, key=lambda r: r[2]):
        print(f" {t:8.2f}    {acc*100:7.2f}%   {cov*100:7.2f}%   {(acc-baseline_acc)*100:+6.2f}pp")
    print("=" * 100)


if __name__ == "__main__":
    main()
