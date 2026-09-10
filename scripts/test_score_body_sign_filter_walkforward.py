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

    # Exact project score.
    df["score"] = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)
    df["body_sign"] = np.where(body > 0, 1, np.where(body < 0, -1, 0))
    df["J"] = body
    df["U"] = df.J.shift(-1) - df.J
    df["actual"] = np.sign(df.U)
    return df.iloc[:-1].copy()


def baseline_pred(df: pd.DataFrame) -> np.ndarray:
    return np.where(df.score <= 1, -1, 1).astype(int)


def evaluate(df: pd.DataFrame, mapping: tuple[int, int, int, int]) -> tuple[int, int, int, float, float]:
    # Mapping gives prediction for score 1, split by body sign (+/-), then score 2.
    # score0 fixed SHORT; score3 fixed LONG.
    s1_pos, s1_neg, s2_pos, s2_neg = mapping
    score = df.score.to_numpy(int)
    bsign = df.body_sign.to_numpy(int)
    pred = np.where(score == 0, -1, np.where(score == 3, 1, 0)).astype(int)
    pred[(score == 1) & (bsign > 0)] = s1_pos
    pred[(score == 1) & (bsign < 0)] = s1_neg
    pred[(score == 2) & (bsign > 0)] = s2_pos
    pred[(score == 2) & (bsign < 0)] = s2_neg
    # Keep zero-body rows on baseline direction.
    pred[(score == 1) & (bsign == 0)] = -1
    pred[(score == 2) & (bsign == 0)] = 1

    valid = df.actual.to_numpy(int) != 0
    active = valid & (pred != 0)
    n = int(valid.sum())
    a = int(active.sum())
    c = int(np.sum(pred[active] == df.actual.to_numpy(int)[active]))
    acc = c / a if a else float("nan")
    cov = a / n if n else float("nan")
    return n, a, c, acc, cov


def walkforward(df: pd.DataFrame, mapping: tuple[int, int, int, int], folds: int, train_ratio: float) -> tuple[int, int, int, float, float]:
    n = len(df)
    train0 = int(n * train_ratio)
    test_size = (n - train0) // folds
    total = active = correct = 0
    for k in range(folds):
        start = train0 + k * test_size
        end = train0 + (k + 1) * test_size if k < folds - 1 else n
        n0, a0, c0, _, _ = evaluate(df.iloc[start:end], mapping)
        total += n0
        active += a0
        correct += c0
    return total, active, correct, correct / active if active else float("nan"), active / total if total else float("nan")


def label(m: tuple[int, int, int, int]) -> str:
    names = ["SHORT" if x < 0 else "LONG" for x in m]
    return f"S1(J+)={names[0]} S1(J-)={names[1]} S2(J+)={names[2]} S2(J-)={names[3]}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward exhaustive Score x current-body-sign mapping test")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.50)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    frames = {s: load(Path(args.data_dir) / f"{s}_1h.csv") for s in symbols}
    baseline_total = baseline_active = baseline_correct = 0
    base_map = (-1, -1, 1, 1)
    for df in frames.values():
        n, a, c, _, _ = walkforward(df, base_map, args.folds, args.train_ratio)
        baseline_total += n
        baseline_active += a
        baseline_correct += c
    baseline_acc = baseline_correct / baseline_active

    results = []
    for mapping in product((-1, 1), repeat=4):
        total = active = correct = 0
        for df in frames.values():
            n, a, c, _, _ = walkforward(df, mapping, args.folds, args.train_ratio)
            total += n
            active += a
            correct += c
        acc = correct / active
        cov = active / total
        results.append((acc, cov, mapping, active, total))
    results.sort(key=lambda x: (x[0], x[1]), reverse=True)

    print("=" * 118)
    print("WALK-FORWARD: SCORE × CURRENT BODY SIGN TEST")
    print("=" * 118)
    print("Baseline: score 0/1 SHORT, score 2/3 LONG")
    print("For score 1 and 2, independently choose LONG/SHORT by current body sign J=Close-Open")
    print(f"folds={args.folds} train_ratio={args.train_ratio:.2f}")
    print(f"BASELINE: acc={baseline_acc*100:.2f}% coverage=100.00% N={baseline_active}")
    print()
    print("TOP 10 POOLED MAPPINGS")
    print("rank  mapping                                               accuracy  coverage   delta")
    for i, (acc, cov, mapping, active, total) in enumerate(results[:10], 1):
        print(f"{i:>4}  {label(mapping):53} {acc*100:7.2f}%  {cov*100:7.2f}%  {(acc-baseline_acc)*100:+6.2f}pp")

    print("=" * 118)


if __name__ == "__main__":
    main()
