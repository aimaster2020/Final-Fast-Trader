from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

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

    body_signed = df.close - df.open
    body_abs = body_signed.abs()
    rng = df.high - df.low

    hc = df.high - df.close
    ho = df.high - df.open
    lc = df.low - df.close

    df["score"] = (hc > body_signed).astype(int) + (ho < hc).astype(int) + (lc > body_signed).astype(int)
    df["jabs"] = body_abs

    # Requested quantities:
    # A = H-O+C, B = L-O+C
    df["A"] = df.high - df.open + df.close
    df["B"] = df.low - df.open + df.close

    # Non-redundant decompositions / derived geometry.
    df["upper"] = df.A - df.close       # = H-O
    df["lower"] = df.close - df.B       # = O-L
    df["range"] = df.A - df.B            # = H-L
    df["mid_shift"] = (df.A + df.B) / 2.0 - df.close  # = (H+L)/2 - O

    df["u"] = body_signed.shift(-1) - body_signed
    df["actual"] = np.sign(df.u)
    return df.iloc[:-1].copy()


def baseline_pred(df: pd.DataFrame) -> np.ndarray:
    pred = np.where(df.score.to_numpy(int) <= 1, -1, 1).astype(int)
    # Current best strategy: remove the two weak disagreement cells.
    pred[(df.score.to_numpy(int) == 1) & (df.upper.to_numpy(float) < df.lower.to_numpy(float))] = pred[(df.score.to_numpy(int) == 1) & (df.upper.to_numpy(float) < df.lower.to_numpy(float))]
    # Formula: LONG when lower > upper; SHORT when upper > lower.
    # Drop S1+FormulaLong and S2+FormulaShort.
    formula = np.where(df.lower > df.upper, 1, np.where(df.upper > df.lower, -1, 0))
    pred[(df.score.to_numpy(int) == 1) & (formula == 1)] = 0
    pred[(df.score.to_numpy(int) == 2) & (formula == -1)] = 0
    return pred


def apply_mode(df: pd.DataFrame, pred: np.ndarray, mode: str, threshold: float) -> np.ndarray:
    score = df.score.to_numpy(int)
    upper = df.upper.to_numpy(float)
    lower = df.lower.to_numpy(float)
    body = df.jabs.to_numpy(float)
    rng = df.range.to_numpy(float)
    mid = df.mid_shift.to_numpy(float)

    mask = np.isfinite(upper) & np.isfinite(lower) & np.isfinite(body) & np.isfinite(rng) & np.isfinite(mid)
    weak = ((score == 1) | (score == 2)) & mask

    if mode == "upper_body":
        cond = weak & (upper / np.maximum(body, 1e-12) <= threshold)
    elif mode == "lower_body":
        cond = weak & (lower / np.maximum(body, 1e-12) <= threshold)
    elif mode == "upper_range":
        cond = weak & (upper / np.maximum(rng, 1e-12) <= threshold)
    elif mode == "lower_range":
        cond = weak & (lower / np.maximum(rng, 1e-12) <= threshold)
    elif mode == "mid_body":
        cond = weak & (np.abs(mid) / np.maximum(body, 1e-12) <= threshold)
    elif mode == "mid_range":
        cond = weak & (np.abs(mid) / np.maximum(rng, 1e-12) <= threshold)
    elif mode == "upper_lower_ratio":
        ratio = np.maximum(upper, lower) / np.maximum(np.minimum(upper, lower), 1e-12)
        cond = weak & (ratio >= threshold)
    elif mode == "upper_plus_lower_body":
        cond = weak & ((upper + lower) / np.maximum(body, 1e-12) <= threshold)
    else:
        raise ValueError(mode)

    out = pred.copy()
    # Filter only currently active trades; never reactivate a trade already removed by the current best rule.
    out[cond & (out != 0)] = 0
    return out


def evaluate(actual: np.ndarray, pred: np.ndarray) -> tuple[int, int, float, float]:
    valid = actual != 0
    active = valid & (pred != 0)
    total = int(valid.sum())
    n_active = int(active.sum())
    correct = int(np.sum(pred[active] == actual[active]))
    acc = correct / n_active if n_active else float("nan")
    cov = n_active / total if total else float("nan")
    return total, n_active, acc, cov


def walkforward(df: pd.DataFrame, mode: str | None, threshold: float, folds: int, train_ratio: float) -> tuple[int, int, float, float]:
    n = len(df)
    train0 = int(n * train_ratio)
    test_size = (n - train0) // folds
    total = active = correct = 0
    base = baseline_pred(df)

    for k in range(folds):
        start = train0 + k * test_size
        end = train0 + (k + 1) * test_size if k < folds - 1 else n
        chunk = df.iloc[start:end]
        pred = base[start:end].copy()
        if mode is not None:
            pred = apply_mode(chunk, pred, mode, threshold)
        actual = chunk.actual.to_numpy(int)
        valid = actual != 0
        active_mask = valid & (pred != 0)
        total += int(valid.sum())
        active += int(active_mask.sum())
        correct += int(np.sum(pred[active_mask] == actual[active_mask]))

    acc = correct / active if active else float("nan")
    cov = active / total if total else float("nan")
    return total, active, acc, cov


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward indirect A/B geometry rules")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.50)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    thresholds = sorted({float(x) for x in args.thresholds.split(",") if x.strip()})
    modes = ["upper_body", "lower_body", "upper_range", "lower_range", "mid_body", "mid_range", "upper_lower_ratio", "upper_plus_lower_body"]
    frames = {s: load(Path(args.data_dir) / f"{s}_1h.csv") for s in symbols}

    base_total = base_active = base_correct = 0
    for df in frames.values():
        n, a, acc, _ = walkforward(df, None, 0.0, args.folds, args.train_ratio)
        base_total += n
        base_active += a
        base_correct += int(round(acc * a))
    base_acc = base_correct / base_active

    results: list[tuple[float, float, float, str, int]] = []
    for mode, t in product(modes, thresholds):
        total = active = correct = 0
        for df in frames.values():
            n, a, acc, _ = walkforward(df, mode, t, args.folds, args.train_ratio)
            total += n
            active += a
            correct += int(round(acc * a))
        acc = correct / active if active else float("nan")
        cov = active / total if total else float("nan")
        results.append((acc, cov, t, mode, active))

    results.sort(key=lambda x: (x[0], x[1]), reverse=True)

    print("=" * 110)
    print("WALK-FORWARD: INDIRECT H-O+C / L-O+C GEOMETRY RULES")
    print("=" * 110)
    print("A=H-O+C, B=L-O+C")
    print("A-C=H-O, C-B=O-L, A-B=H-L; testing indirect normalized geometry inside current best strategy")
    print("Current best: Score 0/1 SHORT, Score 2/3 LONG + drop S1+FormulaLong + S2+FormulaShort")
    print(f"folds={args.folds} train_ratio={args.train_ratio:.2f}")
    print(f"BASELINE: acc={base_acc*100:.2f}% coverage=100.00% N={base_active}")
    print()
    print("TOP 15 POOLED RULES")
    print("rank mode                   threshold   accuracy   coverage    delta")
    for i, (acc, cov, t, mode, active) in enumerate(results[:15], 1):
        print(f"{i:>4} {mode:23} {t:8.2f}     {acc*100:7.2f}%   {cov*100:7.2f}%   {(acc-base_acc)*100:+6.2f}pp")

    print()
    print("BEST BY MODE")
    print("mode                     threshold   accuracy   coverage    delta")
    for mode in modes:
        best = max((r for r in results if r[3] == mode), key=lambda r: (r[0], r[1]))
        acc, cov, t, _, _ = best
        print(f"{mode:24} {t:8.2f}     {acc*100:7.2f}%   {cov*100:7.2f}%   {(acc-base_acc)*100:+6.2f}pp")

    print("=" * 110)


if __name__ == "__main__":
    main()
