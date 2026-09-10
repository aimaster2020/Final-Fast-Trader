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
    df["actual"] = np.sign(df.U)

    up_move = (df.high - df.open).abs()
    down_move = (df.open - df.low).abs()
    # Previously useful inverse direction convention:
    # LONG when O-L > H-O, SHORT when H-O > O-L.
    df["formula_inv"] = np.where(down_move > up_move, 1, np.where(up_move > down_move, -1, 0))
    # Original direct convention, included as a control.
    df["formula_direct"] = np.where(up_move > down_move, 1, np.where(down_move > up_move, -1, 0))
    return df.iloc[:-1].copy()


def acc(d: pd.DataFrame, pred: pd.Series) -> tuple[int, float]:
    mask = (pred != 0) & (d.actual != 0)
    n = int(mask.sum())
    return n, float(np.mean(pred[mask].to_numpy(int) == d.actual[mask].to_numpy(int))) if n else float("nan")


def build_variants(d: pd.DataFrame) -> dict[str, pd.Series]:
    base = np.where(d.score.to_numpy(int) >= 2, 1, -1)
    out = {
        "baseline_23_long_01_short": pd.Series(base, index=d.index),
        "reverse_12_long_23_short": pd.Series(np.where(d.score.to_numpy(int) == 1, 1, np.where(d.score.to_numpy(int) == 2, -1, base)), index=d.index),
    }

    # Freeze strong buckets 0 and 3; for ambiguous 1/2 use formula direction.
    formula_inv = d.formula_inv.to_numpy(int)
    formula_direct = d.formula_direct.to_numpy(int)
    score = d.score.to_numpy(int)

    variants = {
        "0S_1_formulaInv_2_formulaInv_3L": np.where((score == 1) | (score == 2), formula_inv, base),
        "0S_1_formulaDirect_2_formulaDirect_3L": np.where((score == 1) | (score == 2), formula_direct, base),
        "0S_1_formulaInv_2_base_3L": np.where(score == 1, formula_inv, base),
        "0S_1_base_2_formulaInv_3L": np.where(score == 2, formula_inv, base),
        "0S_1_formulaDirect_2_base_3L": np.where(score == 1, formula_direct, base),
        "0S_1_base_2_formulaDirect_3L": np.where(score == 2, formula_direct, base),
    }
    for k, v in variants.items():
        out[k] = pd.Series(v, index=d.index)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Test expanded score direction, focusing on score 1/2 formula variants.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    args = ap.parse_args()

    all_records: list[dict] = []
    print("=" * 118)
    print("EXPANDED SCORE TEST — FREEZE SCORE 0/3, TEST SCORE 1/2 + H-O/O-L")
    print("=" * 118)
    print("baseline: LONG=2/3 | SHORT=0/1")
    print("formulaInv: LONG when O-L > H-O; SHORT when H-O > O-L")
    print("0 and 3 are frozen; score 1 and/or 2 may be replaced by formula direction.")

    for symbol in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        d = load(Path(args.data_dir) / f"{symbol}_1h.csv")
        variants = build_variants(d)
        print(f"\n{symbol}")
        print("variant                                      N      accuracy")
        for name, pred in variants.items():
            n, a = acc(d, pred)
            print(f"{name:42} {n:5d}     {a*100:7.2f}%")
            all_records.append({"symbol": symbol, "variant": name, "n": n, "accuracy": a})

        print("  score buckets:")
        for s in (0, 1, 2, 3):
            x = d[d.score == s]
            n = len(x)
            if n:
                up = float(np.mean(x.actual > 0)) * 100
                print(f"    score={s}: N={n:4d} UP={up:6.2f}% DOWN={100-up:6.2f}%")

    r = pd.DataFrame(all_records)
    print("\n" + "-" * 118)
    pooled = []
    for variant, g in r.groupby("variant"):
        pooled_n = int(g.n.sum())
        pooled_acc = float(np.average(g.accuracy, weights=g.n))
        pooled.append((pooled_acc, variant, pooled_n))
    for a, v, n in sorted(pooled, reverse=True):
        print(f"{v:42} pooled_N={n:5d} accuracy={a*100:7.2f}%")
    print("=" * 118)


if __name__ == "__main__":
    main()
