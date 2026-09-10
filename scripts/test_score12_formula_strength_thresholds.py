from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
THRESHOLDS = (1.00, 1.05, 1.10, 1.20, 1.30, 1.50, 2.00)


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
    df["actual"] = np.sign(df.U)

    # Previously found magnitude directions.
    df["up_mag"] = (df.high - df.open).abs()
    df["down_mag"] = (df.open - df.low).abs()

    # Inverse standalone direction: larger O-L => LONG, larger H-O => SHORT.
    df["formula_dir"] = np.where(
        df.down_mag > df.up_mag, 1,
        np.where(df.up_mag > df.down_mag, -1, 0),
    )
    df["strength_ratio"] = np.maximum(df.up_mag, df.down_mag) / np.maximum(np.minimum(df.up_mag, df.down_mag), 1e-12)
    return df.iloc[:-1].copy()


def run_variant(df: pd.DataFrame, threshold: float, target_scores: set[int]) -> tuple[int, float]:
    # Baseline: score 0/1 => SHORT, score 2/3 => LONG.
    base = np.where(df.score.to_numpy(int) >= 2, 1, -1)
    actual = df.actual.to_numpy(int)
    formula = df.formula_dir.to_numpy(int)
    ratio = df.strength_ratio.to_numpy(float)
    score = df.score.to_numpy(int)

    pred = base.copy()
    eligible = np.isin(score, list(target_scores)) & (formula != 0) & (ratio >= threshold)
    pred[eligible] = formula[eligible]

    valid = actual != 0
    return int(valid.sum()), float(np.mean(pred[valid] == actual[valid]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Test H-O/O-L formula only when its magnitude advantage is strong, for score 1/2.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    args = ap.parse_args()

    root = Path(args.data_dir)
    all_data: dict[str, pd.DataFrame] = {}
    for symbol in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        all_data[symbol] = load(root / f"{symbol}_1h.csv")

    print("=" * 112)
    print("SCORE 1/2 + H-O/O-L STRENGTH THRESHOLD TEST")
    print("=" * 112)
    print("baseline: LONG=score 2/3 | SHORT=score 0/1")
    print("formula inverse: LONG when O-L > H-O | SHORT when H-O > O-L")
    print("formula replaces baseline only when max(H-O,O-L)/min(H-O,O-L) >= threshold")
    print("threshold  target   pooled_N   accuracy   baseline_delta")

    for target_name, target_scores in (("score1", {1}), ("score2", {2}), ("score1+2", {1, 2})):
        for threshold in THRESHOLDS:
            total_n = total_correct = baseline_correct = 0
            for df in all_data.values():
                n, acc = run_variant(df, threshold, target_scores)
                total_n += n
                total_correct += int(round(n * acc))
                base = np.where(df.score.to_numpy(int) >= 2, 1, -1)
                actual = df.actual.to_numpy(int)
                valid = actual != 0
                baseline_correct += int(np.sum(base[valid] == actual[valid]))
            pooled_acc = total_correct / total_n if total_n else float("nan")
            base_acc = baseline_correct / total_n if total_n else float("nan")
            print(f"{threshold:8.2f}  {target_name:8}   {total_n:8d}    {pooled_acc*100:8.2f}%    {(pooled_acc-base_acc)*100:+.2f}pp")
        print()

    print("score-by-score baseline accuracy:")
    for symbol, df in all_data.items():
        actual = df.actual.to_numpy(int)
        print(symbol, end="  ")
        for s in (0, 1, 2, 3):
            m = df.score.to_numpy(int) == s
            n = int(np.sum(m & (actual != 0)))
            pred = 1 if s >= 2 else -1
            acc = float(np.mean(actual[m] == pred)) if n else float("nan")
            print(f"S{s}={acc*100:.2f}%", end=" ")
        print()

    print("=" * 112)


if __name__ == "__main__":
    main()
