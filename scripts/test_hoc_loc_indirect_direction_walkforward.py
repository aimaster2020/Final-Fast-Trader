from __future__ import annotations

import argparse
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_THRESHOLDS = "0.25,0.5,0.75,1,1.5,2,3,5,10"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in {"open", "high", "low", "close"})
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    o, h, l, c = df.open, df.high, df.low, df.close
    body = c - o
    rng = h - l

    # Exact project Score.
    df["score"] = ((h - c) > body).astype(int) + ((h - o) < (h - c)).astype(int) + ((l - c) > body).astype(int)

    # Requested quantities.
    A = h - o + c
    B = l - o + c
    df["A"] = A
    df["B"] = B

    # Algebraic decomposition:
    # A-C = H-O, C-B = O-L. These are the meaningful scale-free components.
    upper = h - o
    lower = o - l
    abs_body = body.abs()
    df["upper_body"] = np.where(abs_body > 0, upper / abs_body, np.nan)
    df["lower_body"] = np.where(abs_body > 0, lower / abs_body, np.nan)
    df["upper_range"] = np.where(rng > 0, upper / rng, np.nan)
    df["lower_range"] = np.where(rng > 0, lower / rng, np.nan)

    j = body
    df["actual"] = np.sign(j.shift(-1) - j)
    return df.iloc[:-1].copy()


def baseline_pred(df: pd.DataFrame) -> np.ndarray:
    return np.where(df.score.to_numpy(int) <= 1, -1, 1).astype(int)


def filtered_pred(df: pd.DataFrame, mode: str, threshold: float) -> np.ndarray:
    pred = baseline_pred(df)
    score = df.score.to_numpy(int)
    ub = df.upper_body.to_numpy(float)
    lb = df.lower_body.to_numpy(float)
    ur = df.upper_range.to_numpy(float)
    lr = df.lower_range.to_numpy(float)

    if mode == "upper_body":
        drop = ((score == 1) | (score == 2)) & np.isfinite(ub) & (ub <= threshold)
    elif mode == "lower_body":
        drop = ((score == 1) | (score == 2)) & np.isfinite(lb) & (lb <= threshold)
    elif mode == "both_body":
        drop = ((score == 1) | (score == 2)) & np.isfinite(ub) & np.isfinite(lb) & ((ub <= threshold) | (lb <= threshold))
    elif mode == "upper_range":
        drop = ((score == 1) | (score == 2)) & np.isfinite(ur) & (ur <= threshold)
    elif mode == "lower_range":
        drop = ((score == 1) | (score == 2)) & np.isfinite(lr) & (lr <= threshold)
    else:
        raise ValueError(mode)

    pred[drop] = 0
    return pred


def evaluate(df: pd.DataFrame, pred: np.ndarray) -> tuple[int, int, float, float]:
    actual = df.actual.to_numpy(int)
    valid = actual != 0
    active = valid & (pred != 0)
    total = int(valid.sum())
    n_active = int(active.sum())
    correct = int(np.sum(pred[active] == actual[active]))
    acc = correct / n_active if n_active else float("nan")
    cov = n_active / total if total else float("nan")
    return total, n_active, acc, cov


def walkforward(df: pd.DataFrame, mode: str, threshold: float, folds: int, train_ratio: float) -> tuple[int, int, int, float, float]:
    n = len(df)
    train0 = int(n * train_ratio)
    test_size = (n - train0) // folds
    total = active = correct = 0
    for k in range(folds):
        start = train0 + k * test_size
        end = n if k == folds - 1 else train0 + (k + 1) * test_size
        chunk = df.iloc[start:end]
        pred = filtered_pred(chunk, mode, threshold)
        actual = chunk.actual.to_numpy(int)
        valid = actual != 0
        active_mask = valid & (pred != 0)
        total += int(valid.sum())
        active += int(active_mask.sum())
        correct += int(np.sum(pred[active_mask] == actual[active_mask]))
    acc = correct / active if active else float("nan")
    cov = active / total if total else float("nan")
    return total, active, correct, acc, cov


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward scan of indirect H-O+C / L-O+C direction rules")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.50)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    modes = ["upper_body", "lower_body", "both_body", "upper_range", "lower_range"]
    frames = {s: load(Path(args.data_dir) / f"{s}_1h.csv") for s in symbols}

    baseline_total = baseline_active = baseline_correct = 0
    for df in frames.values():
        total, active, correct, _, _ = walkforward(df, "upper_body", float("inf"), args.folds, args.train_ratio)
        baseline_total += total
        baseline_active += active
        baseline_correct += correct
    baseline_acc = baseline_correct / baseline_active

    results = []
    for mode in modes:
        for t in thresholds:
            total = active = correct = 0
            for df in frames.values():
                n, a, c, _, _ = walkforward(df, mode, t, args.folds, args.train_ratio)
                total += n
                active += a
                correct += c
            acc = correct / active if active else float("nan")
            cov = active / total if total else float("nan")
            results.append((acc, cov, mode, t, active, total))

    results.sort(key=lambda x: (x[0], x[1]), reverse=True)

    print("=" * 110)
    print("WALK-FORWARD: INDIRECT H-O+C / L-O+C DIRECTION TEST")
    print("=" * 110)
    print("Baseline: Score 0/1 SHORT, Score 2/3 LONG")
    print("A=H-O+C, B=L-O+C; test uses their non-redundant decompositions:")
    print("A-C=H-O, C-B=O-L, normalized by |C-O| or H-L")
    print(f"folds={args.folds} train_ratio={args.train_ratio:.2f}")
    print(f"BASELINE: acc={baseline_acc*100:.2f}% coverage=100.00% N={baseline_active}")
    print()
    print("TOP 10 POOLED FILTERS")
    print("rank mode          threshold   accuracy   coverage    delta")
    for i, (acc, cov, mode, t, active, total) in enumerate(results[:10], 1):
        print(f"{i:>4} {mode:13} {t:9.2f}     {acc*100:7.2f}%   {cov*100:7.2f}%   {(acc-baseline_acc)*100:+6.2f}pp")

    print()
    print("BEST BY MODE")
    print("mode          threshold   accuracy   coverage    delta")
    for mode in modes:
        best = max((r for r in results if r[2] == mode), key=lambda r: (r[0], r[1]))
        acc, cov, _, t, _, _ = best
        print(f"{mode:13} {t:9.2f}     {acc*100:7.2f}%   {cov*100:7.2f}%   {(acc-baseline_acc)*100:+6.2f}pp")
    print("=" * 110)


if __name__ == "__main__":
    main()
