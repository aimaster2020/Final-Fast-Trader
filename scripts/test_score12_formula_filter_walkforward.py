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

    # Exact project score: Score 0/1 -> SHORT, Score 2/3 -> LONG.
    df["score"] = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)
    df["J"] = body
    df["U"] = df.J.shift(-1) - df.J
    df["actual"] = np.sign(df.U)

    # Formula direction: LONG when O-L > H-O; SHORT when H-O > O-L.
    upper = (df.high - df.open).abs()
    lower = (df.open - df.low).abs()
    df["formula"] = np.where(lower > upper, 1, np.where(upper > lower, -1, 0))
    return df.iloc[:-1].copy()


def baseline_pred(df: pd.DataFrame) -> np.ndarray:
    return np.where(df.score <= 1, -1, 1).astype(int)


def filtered_pred(df: pd.DataFrame, drop_score1_formula_long: bool, drop_score2_formula_short: bool) -> np.ndarray:
    pred = baseline_pred(df)
    drop = np.zeros(len(df), dtype=bool)
    if drop_score1_formula_long:
        drop |= (df.score.to_numpy() == 1) & (df.formula.to_numpy() == 1)
    if drop_score2_formula_short:
        drop |= (df.score.to_numpy() == 2) & (df.formula.to_numpy() == -1)
    pred[drop] = 0
    return pred


def evaluate(actual: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, float, float]:
    valid = actual != 0
    active = valid & (pred != 0)
    n_total = int(valid.sum())
    n_active = int(active.sum())
    correct = int(np.sum(pred[active] == actual[active]))
    accuracy = correct / n_active if n_active else float("nan")
    coverage = n_active / n_total if n_total else float("nan")
    return n_total, n_active, correct, accuracy, coverage


def walkforward(df: pd.DataFrame, folds: int, train_ratio: float) -> dict[tuple[bool, bool], tuple[int, int, float, float, int]]:
    n = len(df)
    train0 = int(n * train_ratio)
    test_size = (n - train0) // folds
    out: dict[tuple[bool, bool], list[float]] = {}
    # Rule is fixed before walk-forward; training is used only to create chronological folds.
    for a in (False, True):
        for b in (False, True):
            total_n = total_active = correct = 0
            for k in range(folds):
                start = train0 + k * test_size
                end = train0 + (k + 1) * test_size if k < folds - 1 else n
                chunk = df.iloc[start:end]
                actual = chunk.actual.to_numpy(int)
                pred = filtered_pred(chunk, a, b)
                n_total, n_active, c, _, _ = evaluate(actual, pred)
                total_n += n_total
                total_active += n_active
                correct += c
            acc = correct / total_active if total_active else float("nan")
            cov = total_active / total_n if total_n else float("nan")
            out[(a, b)] = [total_n, total_active, correct, acc, cov]
    return {k: tuple(v) for k, v in out.items()}


def fmt_rule(a: bool, b: bool) -> str:
    if a and b:
        return "DROP S1+F_LONG & S2+F_SHORT"
    if a:
        return "DROP S1+F_LONG"
    if b:
        return "DROP S2+F_SHORT"
    return "BASELINE"


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward validation of Score 1/2 formula filters.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.50)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    print("=" * 100)
    print("WALK-FORWARD: SCORE × FORMULA FILTER TEST")
    print("=" * 100)
    print("Baseline: Score 0/1 = SHORT, Score 2/3 = LONG")
    print("Filter candidates: drop Score 1 + Formula LONG and/or Score 2 + Formula SHORT")
    print(f"folds={args.folds} train_ratio={args.train_ratio:.2f}")
    print()

    pooled: dict[str, list[int]] = {}
    for symbol in symbols:
        df = load(Path(args.data_dir) / f"{symbol}_1h.csv")
        results = walkforward(df, args.folds, args.train_ratio)
        print(symbol)
        for (a, b), (n_total, n_active, correct, acc, cov) in results.items():
            label = fmt_rule(a, b)
            pooled.setdefault(label, [0, 0, 0])
            pooled[label][0] += n_total
            pooled[label][1] += n_active
            pooled[label][2] += correct
            print(f"  {label:34} acc={acc*100:6.2f}%  coverage={cov*100:6.2f}%  active={n_active:5d}/{n_total:5d}")
        print()

    print("POOLED")
    base_acc = None
    for label in ("BASELINE", "DROP S1+F_LONG", "DROP S2+F_SHORT", "DROP S1+F_LONG & S2+F_SHORT"):
        n_total, n_active, correct = pooled[label]
        acc = correct / n_active if n_active else float("nan")
        cov = n_active / n_total if n_total else float("nan")
        if label == "BASELINE":
            base_acc = acc
        delta = (acc - base_acc) * 100 if base_acc is not None else 0.0
        print(f"  {label:34} acc={acc*100:6.2f}%  coverage={cov*100:6.2f}%  delta={delta:+6.2f}pp  active={n_active:5d}/{n_total:5d}")

    print("=" * 100)


if __name__ == "__main__":
    main()
