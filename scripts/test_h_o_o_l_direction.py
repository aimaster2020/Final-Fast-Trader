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

    df["J"] = df.close - df.open
    df["U"] = df.J.shift(-1) - df.J
    df["actual_dir"] = np.sign(df.U)

    # The two magnitude rules found previously:
    # LONG  -> next_close = C + |H-O|
    # SHORT -> next_close = C - |O-L|
    df["long_move"] = (df.high - df.open).abs()
    df["short_move"] = (df.open - df.low).abs()

    # Direction candidate: choose the side whose directional magnitude is larger.
    df["pred_dir"] = np.where(
        df.long_move > df.short_move,
        1,
        np.where(df.short_move > df.long_move, -1, 0),
    )
    return df.iloc[:-1].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Test H-O/O-L magnitude rules as a direction classifier")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    args = ap.parse_args()

    root = Path(args.data_dir)
    all_pred: list[np.ndarray] = []
    all_actual: list[np.ndarray] = []

    print("=" * 86)
    print("H-O / O-L MAGNITUDE RULES — DIRECTION TEST")
    print("=" * 86)
    print("LONG rule:  next_close = C + |H-O|")
    print("SHORT rule: next_close = C - |O-L|")
    print("direction: LONG when |H-O| > |O-L|, SHORT when |O-L| > |H-O|")
    print("actual: sign(U), U = J_next - J")
    print("symbol       N    pred_LONG%  pred_SHORT%   accuracy   baseline")

    for symbol in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        df = load(root / f"{symbol}_1h.csv")
        valid = (df.pred_dir != 0) & (df.actual_dir != 0)
        pred = df.loc[valid, "pred_dir"].to_numpy(int)
        actual = df.loc[valid, "actual_dir"].to_numpy(int)
        n = len(actual)
        acc = float(np.mean(pred == actual)) if n else float("nan")
        long_pct = float(np.mean(pred > 0)) if n else float("nan")
        short_pct = float(np.mean(pred < 0)) if n else float("nan")
        baseline = max(float(np.mean(actual > 0)), float(np.mean(actual < 0))) if n else float("nan")

        print(f"{symbol:8} {n:6d}    {long_pct*100:8.2f}%    {short_pct*100:9.2f}%   {acc*100:8.2f}%   {baseline*100:8.2f}%")
        all_pred.append(pred)
        all_actual.append(actual)

    if all_pred:
        pred = np.concatenate(all_pred)
        actual = np.concatenate(all_actual)
        acc = float(np.mean(pred == actual))
        print("-" * 86)
        print(f"POOLED     {len(actual):6d}    accuracy={acc*100:.2f}%")
    print("=" * 86)


if __name__ == "__main__":
    main()
