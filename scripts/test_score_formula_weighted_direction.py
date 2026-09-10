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

    # Exact CANDLE-BODY-MAGNITUDE-V1 score definition.
    df["score"] = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)
    df["J"] = body
    df["U"] = df.J.shift(-1) - df.J
    df["actual_dir"] = np.sign(df.U)

    # Previously found directional magnitude rules.
    long_mag = (df.high - df.open).abs()
    short_mag = (df.open - df.low).abs()

    # Inverse H-O/O-L direction: LONG when O-L > H-O.
    # Continuous signal is in [-1, +1]. Positive = LONG.
    denom = long_mag + short_mag
    df["formula_signal"] = np.where(
        denom > 0,
        (short_mag - long_mag) / denom,
        0.0,
    )
    return df.iloc[:-1].copy()


def accuracy(pred: np.ndarray, actual: np.ndarray) -> float:
    valid = (pred != 0) & (actual != 0)
    return float(np.mean(pred[valid] == actual[valid])) if valid.any() else float("nan")


def evaluate(df: pd.DataFrame, alpha: float) -> float:
    # Baseline score mapping:
    # score 0/1 => SHORT, score 2/3 => LONG.
    # Weighted formula shifts the score around its neutral midpoint 1.5.
    score_component = df.score.to_numpy(float) - 1.5
    formula_component = df.formula_signal.to_numpy(float)
    raw = score_component + alpha * formula_component
    pred = np.where(raw > 0, 1, np.where(raw < 0, -1, 0))
    return accuracy(pred, df.actual_dir.to_numpy(int))


def main() -> None:
    ap = argparse.ArgumentParser(description="Search weighted combinations of CANDLE-BODY score and inverse H-O/O-L direction signal")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--alpha-min", type=float, default=-10.0)
    ap.add_argument("--alpha-max", type=float, default=10.0)
    ap.add_argument("--alpha-step", type=float, default=0.05)
    args = ap.parse_args()

    root = Path(args.data_dir)
    frames = []
    print("=" * 96)
    print("CANDLE-BODY SCORE + H-O/O-L WEIGHTED DIRECTION SEARCH")
    print("=" * 96)
    print("fixed score meaning: 0/1=SHORT, 2/3=LONG")
    print("formula signal: +1 when O-L dominates, -1 when H-O dominates")
    print("prediction: sign((score-1.5) + alpha*formula_signal)")
    print()

    for symbol in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        df = load(root / f"{symbol}_1h.csv")
        df["symbol"] = symbol
        frames.append(df)

    pooled = pd.concat(frames, ignore_index=True)
    baseline = evaluate(pooled, 0.0)

    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step / 2, args.alpha_step)
    results = []
    for alpha in alphas:
        acc = evaluate(pooled, float(alpha))
        results.append((float(alpha), acc))

    results.sort(key=lambda x: (-x[1], abs(x[0])))

    print(f"baseline alpha=0 accuracy={baseline*100:.2f}% N={len(pooled)}")
    print("rank  alpha      accuracy   delta")
    for i, (alpha, acc) in enumerate(results[:25], 1):
        print(f"{i:>4}  {alpha:>6.2f}      {acc*100:>7.2f}%   {(acc-baseline)*100:+7.2f}pp")

    print("-" * 96)
    print("best alpha by symbol")
    print("symbol    alpha      accuracy   delta_vs_symbol_baseline")
    for symbol in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        d = pooled[pooled.symbol == symbol]
        b = evaluate(d, 0.0)
        best_a, best_acc = max(((float(a), evaluate(d, float(a))) for a in alphas), key=lambda x: x[1])
        print(f"{symbol:8} {best_a:>7.2f}      {best_acc*100:>7.2f}%   {(best_acc-b)*100:+7.2f}pp")

    print("=" * 96)


if __name__ == "__main__":
    main()
