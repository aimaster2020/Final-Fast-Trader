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
    df["actual_dir"] = np.sign(df.U)
    df["upper"] = (df.high - df.open).abs()
    df["lower"] = (df.open - df.low).abs()
    df["formula_dir"] = np.where(df.lower > df.upper, 1, np.where(df.upper > df.lower, -1, 0))
    return df.iloc[:-1].copy()


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Score 0..3 x inverse H-O/O-L interaction cells.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    args = ap.parse_args()

    rows: list[dict] = []
    for symbol in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        df = load(Path(args.data_dir) / f"{symbol}_1h.csv")
        for score in range(4):
            for fdir, fname in ((1, "FORMULA_LONG"), (-1, "FORMULA_SHORT"), (0, "TIE")):
                d = df[(df.score == score) & (df.formula_dir == fdir) & (df.actual_dir != 0)]
                n = len(d)
                up = float(np.mean(d.actual_dir > 0)) if n else float("nan")
                down = float(np.mean(d.actual_dir < 0)) if n else float("nan")
                majority = 1 if n and up >= down else -1 if n else 0
                acc = max(up, down) if n else float("nan")
                rows.append({"symbol": symbol, "score": score, "formula": fname, "n": n, "up_pct": up, "down_pct": down, "majority_dir": majority, "majority_acc": acc})

    table = pd.DataFrame(rows)
    print("=" * 108)
    print("SCORE × H-O/O-L INTERACTION ANALYSIS")
    print("=" * 108)
    print("formula_dir: LONG when O-L > H-O; SHORT when H-O > O-L")
    print("For each cell, majority_acc shows the best constant direction for that cell (diagnostic only).")
    print()

    for symbol in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        print(symbol)
        print("score formula         N      UP%    DOWN%   majority")
        sub = table[table.symbol == symbol]
        for r in sub.itertuples(index=False):
            if r.n:
                md = "LONG" if r.majority_dir > 0 else "SHORT"
                print(f"  {r.score}   {r.formula:13} {r.n:5d}  {r.up_pct*100:7.2f}%  {r.down_pct*100:7.2f}%   {md:5} {r.majority_acc*100:7.2f}%")
            else:
                print(f"  {r.score}   {r.formula:13} {r.n:5d}      --       --      --       --")
        print()

    print("-" * 108)
    print("POOLED INTERACTION CELLS")
    print("score formula         N      UP%    DOWN%   majority")
    for score in range(4):
        for fdir, fname in ((1, "FORMULA_LONG"), (-1, "FORMULA_SHORT"), (0, "TIE")):
            d = table[(table.score == score) & (table.formula == fname)]
            n = int(d.n.sum())
            if not n:
                continue
            up_n = sum(int(round(r.n * r.up_pct)) for r in d.itertuples(index=False))
            down_n = n - up_n
            up = up_n / n
            down = down_n / n
            md = "LONG" if up >= down else "SHORT"
            acc = max(up, down)
            print(f"  {score}   {fname:13} {n:5d}  {up*100:7.2f}%  {down*100:7.2f}%   {md:5} {acc*100:7.2f}%")

    print("=" * 108)


if __name__ == "__main__":
    main()
