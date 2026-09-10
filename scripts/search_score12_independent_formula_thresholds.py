from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
THRESHOLDS = tuple(np.round(np.arange(1.00, 4.01, 0.05), 2))


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

    # Exact CANDLE-BODY-MAGNITUDE-V1 score.
    df["score"] = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)
    df["J"] = body
    df["U"] = df.J.shift(-1) - df.J
    df["actual"] = np.sign(df.U)

    # Useful magnitude-direction convention found previously:
    # LONG when O-L > H-O, SHORT when H-O > O-L.
    long_move = (df.open - df.low).abs()
    short_move = (df.high - df.open).abs()
    df["formula"] = np.where(long_move > short_move, 1, np.where(short_move > long_move, -1, 0))
    df["ratio"] = np.maximum(long_move, short_move) / np.maximum(np.minimum(long_move, short_move), 1e-12)
    return df.iloc[:-1].copy()


def predict(df: pd.DataFrame, t1: float, t2: float) -> np.ndarray:
    score = df.score.to_numpy(int)
    formula = df.formula.to_numpy(int)
    ratio = df.ratio.to_numpy(float)

    # Frozen: 0=SHORT, 3=LONG. Only 1 and 2 can be modified.
    pred = np.where(score >= 2, 1, -1).astype(int)
    m1 = (score == 1) & (ratio >= t1) & (formula != 0)
    m2 = (score == 2) & (ratio >= t2) & (formula != 0)
    pred[m1] = formula[m1]
    pred[m2] = formula[m2]
    return pred


def main() -> None:
    ap = argparse.ArgumentParser(description="Search independent H-O/O-L strength thresholds for score 1 and 2")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    args = ap.parse_args()

    dfs: dict[str, pd.DataFrame] = {}
    for symbol in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        dfs[symbol] = load(Path(args.data_dir) / f"{symbol}_1h.csv")

    baseline_n = 0
    baseline_correct = 0
    for df in dfs.values():
        actual = df.actual.to_numpy(int)
        pred = np.where(df.score.to_numpy(int) >= 2, 1, -1)
        valid = actual != 0
        baseline_n += int(valid.sum())
        baseline_correct += int((pred[valid] == actual[valid]).sum())
    baseline = baseline_correct / baseline_n

    results: list[tuple[float, float, int, int, float]] = []
    for t1 in THRESHOLDS:
        for t2 in THRESHOLDS:
            n = 0
            correct = 0
            for df in dfs.values():
                actual = df.actual.to_numpy(int)
                pred = predict(df, t1, t2)
                valid = actual != 0
                n += int(valid.sum())
                correct += int((pred[valid] == actual[valid]).sum())
            results.append((t1, t2, n, correct, correct / n))

    results.sort(key=lambda x: x[4], reverse=True)
    print("=" * 96)
    print("SCORE 1/2 INDEPENDENT H-O/O-L THRESHOLD SEARCH")
    print("=" * 96)
    print("frozen: score 0=SHORT, score 3=LONG; only scores 1 and 2 may switch to inverse H-O/O-L")
    print(f"baseline={baseline*100:.2f}% N={baseline_n}")
    print("rank  t_score1  t_score2  N      accuracy   delta")
    for rank, (t1, t2, n, correct, acc) in enumerate(results[:20], 1):
        print(f"{rank:>4}    {t1:>6.2f}    {t2:>6.2f}  {n:5d}    {acc*100:8.2f}%  {(acc-baseline)*100:+7.2f}pp")

    print("-" * 96)
    print("best-per-score probes")
    for target in ("score1_only", "score2_only"):
        rows = []
        for t in THRESHOLDS:
            t1 = t if target == "score1_only" else 99.0
            t2 = t if target == "score2_only" else 99.0
            n = 0
            correct = 0
            for df in dfs.values():
                actual = df.actual.to_numpy(int)
                pred = predict(df, t1, t2)
                valid = actual != 0
                n += int(valid.sum())
                correct += int((pred[valid] == actual[valid]).sum())
            rows.append((t, correct / n))
        rows.sort(key=lambda x: x[1], reverse=True)
        best_t, best_acc = rows[0]
        print(f"{target}: threshold={best_t:.2f} accuracy={best_acc*100:.2f}% delta={(best_acc-baseline)*100:+.2f}pp")

    print("=" * 96)


if __name__ == "__main__":
    main()
