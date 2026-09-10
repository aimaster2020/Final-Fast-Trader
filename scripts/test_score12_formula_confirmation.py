from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"


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
    df["actual_dir"] = np.sign(df.U).astype(int)
    df["upper"] = (df.high - df.open).abs()
    df["lower"] = (df.open - df.low).abs()
    df["formula_dir"] = np.where(df.lower > df.upper, 1, np.where(df.upper > df.lower, -1, 0))
    df["ratio"] = np.maximum(df.upper, df.lower) / np.maximum(np.minimum(df.upper, df.lower), 1e-12)
    df["base_dir"] = np.where(df.score >= 2, 1, -1)
    return df.iloc[:-1].copy()


def eval_mode(df: pd.DataFrame, require_score1: bool, require_score2: bool) -> tuple[int, float, float]:
    actual = df.actual_dir.to_numpy(int)
    pred = df.base_dir.to_numpy(int).copy()
    keep = np.ones(len(df), dtype=bool)

    # Score 0 and 3 are always kept with their baseline direction.
    # Score 1 baseline=SHORT: formula must also be SHORT when required.
    if require_score1:
        mask = df.score.to_numpy(int) == 1
        keep[mask] &= df.formula_dir.to_numpy(int)[mask] == -1
    # Score 2 baseline=LONG: formula must also be LONG when required.
    if require_score2:
        mask = df.score.to_numpy(int) == 2
        keep[mask] &= df.formula_dir.to_numpy(int)[mask] == 1

    valid = keep & (actual != 0)
    n = int(valid.sum())
    acc = float(np.mean(pred[valid] == actual[valid])) if n else float("nan")
    coverage = n / len(df) if len(df) else float("nan")
    return n, acc, coverage


def eval_ratio(df: pd.DataFrame, threshold: float, score: int) -> tuple[int, float, float]:
    actual = df.actual_dir.to_numpy(int)
    pred = df.base_dir.to_numpy(int)
    keep = np.ones(len(df), dtype=bool)
    s = df.score.to_numpy(int)
    ratio = df.ratio.to_numpy(float)
    fdir = df.formula_dir.to_numpy(int)
    mask = s == score
    desired = -1 if score == 1 else 1
    # Only require formula agreement when its magnitude dominance reaches threshold.
    keep[mask] = (~(ratio[mask] >= threshold)) | (fdir[mask] == desired)
    valid = keep & (actual != 0)
    n = int(valid.sum())
    acc = float(np.mean(pred[valid] == actual[valid])) if n else float("nan")
    coverage = n / len(df) if len(df) else float("nan")
    return n, acc, coverage


def main() -> None:
    ap = argparse.ArgumentParser(description="Test H-O/O-L as confirmation filter for Score 1/2 while freezing Score 0/3.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    args = ap.parse_args()

    all_frames = []
    print("=" * 108)
    print("SCORE 1/2 + H-O/O-L CONFIRMATION TEST")
    print("=" * 108)
    print("baseline: score 0/1=SHORT, score 2/3=LONG")
    print("confirmation: score1 requires formula SHORT; score2 requires formula LONG")

    for symbol in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        df = load(Path(args.data_dir) / f"{symbol}_1h.csv")
        df["symbol"] = symbol
        all_frames.append(df)

    pooled = pd.concat(all_frames, ignore_index=True)
    base_n, base_acc, base_cov = eval_mode(pooled, False, False)
    modes = [
        ("baseline", False, False),
        ("confirm_score1", True, False),
        ("confirm_score2", False, True),
        ("confirm_both_1_2", True, True),
    ]
    print("mode                         N       accuracy    coverage    delta_vs_baseline")
    for name, r1, r2 in modes:
        n, acc, cov = eval_mode(pooled, r1, r2)
        print(f"{name:28} {n:6d}     {acc*100:8.2f}%   {cov*100:8.2f}%      {(acc-base_acc)*100:+7.2f}pp")

    print()
    print("ratio-threshold confirmation (only score1 or score2 is filtered)")
    print("threshold   target     N       accuracy    coverage    delta")
    for score in (1, 2):
        for threshold in (1.00, 1.05, 1.10, 1.20, 1.30, 1.50, 2.00, 3.00):
            n, acc, cov = eval_ratio(pooled, threshold, score)
            print(f"  {threshold:5.2f}     score{score}   {n:6d}     {acc*100:8.2f}%   {cov*100:8.2f}%   {(acc-base_acc)*100:+7.2f}pp")
        print()

    print("=" * 108)


if __name__ == "__main__":
    main()
