from __future__ import annotations

import argparse
from itertools import product
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

    # Exact CANDLE-BODY-MAGNITUDE-V1 score definition used in the project.
    df["score"] = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)
    df["J"] = body
    df["U"] = df.J.shift(-1) - df.J
    df["actual"] = np.sign(df.U).astype(int)

    # Magnitude rules found earlier.
    df["upper"] = (df.high - df.open).abs()
    df["lower"] = (df.open - df.low).abs()
    df["formula"] = np.where(df.lower > df.upper, 1, np.where(df.upper > df.lower, -1, 0))
    return df.iloc[:-1].copy()


def evaluate(df: pd.DataFrame, s1: int, s2: int) -> tuple[int, int, float]:
    # score 0=fixed SHORT, score 3=fixed LONG.
    pred = np.where(df.score == 0, -1, np.where(df.score == 3, 1, 0)).astype(int)

    # score 1: choose s1 only when formula is non-tie; score 2: choose s2.
    mask1 = (df.score == 1) & (df.formula != 0)
    mask2 = (df.score == 2) & (df.formula != 0)
    pred[(df.score == 1) & (df.formula == 0)] = -1
    pred[(df.score == 2) & (df.formula == 0)] = 1
    pred[mask1] = s1
    pred[mask2] = s2

    valid = df.actual != 0
    n = int(valid.sum())
    correct = int(np.sum(pred[valid] == df.actual.to_numpy(int)[valid]))
    return n, correct, correct / n if n else float("nan")


def symbol_best(df: pd.DataFrame) -> tuple[int, int, float]:
    best = (-1, -1, -1.0)
    for s1, s2 in product((-1, 1), repeat=2):
        n, _, acc = evaluate(df, s1, s2)
        if acc > best[2]:
            best = (s1, s2, acc)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Exhaustive Score 1/2 + H-O/O-L confirmation mapping test")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    args = ap.parse_args()

    root = Path(args.data_dir)
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    frames: dict[str, pd.DataFrame] = {s: load(root / f"{s}_1h.csv") for s in symbols}

    # Standard expanded baseline: 0/1 SHORT, 2/3 LONG.
    baseline_n = 0
    baseline_correct = 0
    print("=" * 110)
    print("EXHAUSTIVE SCORE 1/2 + H-O/O-L CONFIRMATION TEST")
    print("=" * 110)
    print("fixed: score 0=SHORT, score 3=LONG")
    print("formula: LONG when O-L > H-O; SHORT when H-O > O-L")
    print("for score 1/2, every combination of formula/baseline agreement cells is tested")
    print()

    for symbol, df in frames.items():
        pred_base = np.where(df.score <= 1, -1, 1)
        valid = df.actual != 0
        n = int(valid.sum())
        correct = int(np.sum(pred_base[valid] == df.actual.to_numpy(int)[valid]))
        baseline_n += n
        baseline_correct += correct
        s1, s2, acc = symbol_best(df)
        print(f"{symbol}: baseline={correct/n*100:.2f}%  best_s1={'LONG' if s1>0 else 'SHORT'}  best_s2={'LONG' if s2>0 else 'SHORT'}  best={acc*100:.2f}%")
    print(f"baseline pooled={baseline_correct/baseline_n*100:.2f}% N={baseline_n}")
    print()

    results = []
    for s1, s2 in product((-1, 1), repeat=2):
        total_n = 0
        total_correct = 0
        counts = {k: [0, 0] for k in ((1, 1), (1, -1), (1, 0), (2, 1), (2, -1), (2, 0))}
        for df in frames.values():
            n, correct, _ = evaluate(df, s1, s2)
            total_n += n
            total_correct += correct
        acc = total_correct / total_n
        results.append((s1, s2, total_n, acc))

    results.sort(key=lambda x: x[3], reverse=True)
    print("POOLED MAPPING SEARCH (formula used only when score=1/2)")
    print("rank  score1  score2    N      accuracy   delta_vs_baseline")
    baseline_acc = baseline_correct / baseline_n
    for i, (s1, s2, n, acc) in enumerate(results, 1):
        print(f"{i:>4}  {'LONG' if s1>0 else 'SHORT':>6}  {'LONG' if s2>0 else 'SHORT':>6}  {n:6d}    {acc*100:8.2f}%    {(acc-baseline_acc)*100:+7.2f}pp")

    # Detailed interaction cell accuracy and coverage.
    print()
    print("INTERACTION CELLS — actual accuracy of each cell")
    print("score formula          N      accuracy")
    for score in (0, 1, 2, 3):
        for f, name in ((1, "FORMULA_LONG"), (-1, "FORMULA_SHORT"), (0, "TIE")):
            vals = []
            for df in frames.values():
                d = df[(df.score == score) & (df.formula == f) & (df.actual != 0)]
                if len(d):
                    vals.append(d)
            if not vals:
                continue
            d = pd.concat(vals, ignore_index=True)
            n = len(d)
            acc_long = float(np.mean(d.actual == 1))
            acc_short = float(np.mean(d.actual == -1))
            best = max(acc_long, acc_short)
            print(f"  {score}   {name:13} {n:6d}    {best*100:7.2f}%  ({'LONG' if acc_long>=acc_short else 'SHORT'})")

    print("=" * 110)


if __name__ == "__main__":
    main()
