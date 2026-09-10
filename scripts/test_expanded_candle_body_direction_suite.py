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
    df["next_close"] = df.close.shift(-1)

    # Magnitude rules found previously.
    df["long_move"] = (df.high - df.open).abs()
    df["short_move"] = (df.open - df.low).abs()

    # Useful standalone convention found in prior test.
    df["formula_dir"] = np.where(
        df.short_move > df.long_move,
        1,
        np.where(df.long_move > df.short_move, -1, 0),
    )

    # Requested expanded checkpoint direction:
    # LONG = score 2 or 3, SHORT = score 0 or 1.
    df["expanded_dir"] = np.where(df.score >= 2, 1, -1)
    return df.iloc[:-1].copy()


def valid_accuracy(pred: np.ndarray, actual: np.ndarray) -> tuple[int, float]:
    valid = (pred != 0) & (actual != 0)
    n = int(valid.sum())
    return n, float(np.mean(pred[valid] == actual[valid])) if n else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Expanded CANDLE-BODY-MAGNITUDE direction + H-O/O-L formula suite")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    args = ap.parse_args()

    root = Path(args.data_dir)
    pooled_n = pooled_correct = 0
    pooled_formula_n = pooled_formula_correct = 0
    pooled_veto_n = pooled_veto_correct = 0

    print("=" * 108)
    print("EXPANDED CANDLE-BODY-MAGNITUDE-V1 + H-O/O-L TEST SUITE")
    print("=" * 108)
    print("checkpoint: LONG = score 2/3 | SHORT = score 0/1")
    print("magnitude: LONG C+|H-O| | SHORT C-|O-L|")
    print("formula-only direction: inverse rule => LONG if |O-L|>|H-O|")
    print("veto: keep checkpoint only when formula direction disagrees")
    print()

    for symbol in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        df = load(root / f"{symbol}_1h.csv")
        actual = df.actual_dir.to_numpy(int)
        expanded = df.expanded_dir.to_numpy(int)
        formula = df.formula_dir.to_numpy(int)

        print(symbol)
        print("  score  N     U+%     U-%    prediction")
        for score in (0, 1, 2, 3):
            m = df.score.to_numpy(int) == score
            n = int(m.sum())
            up = float(np.mean(actual[m] > 0)) if n else float("nan")
            down = float(np.mean(actual[m] < 0)) if n else float("nan")
            pred = "LONG" if score >= 2 else "SHORT"
            print(f"    {score}  {n:4d}  {up*100:7.2f}%  {down*100:7.2f}%   {pred}")

        valid = actual != 0
        n = int(valid.sum())
        expanded_acc = float(np.mean(expanded[valid] == actual[valid])) if n else float("nan")
        formula_n, formula_acc = valid_accuracy(formula, actual)

        disagree = valid & (formula != 0) & (formula != expanded)
        veto_n = int(disagree.sum())
        veto_acc = float(np.mean(expanded[disagree] == actual[disagree])) if veto_n else float("nan")
        veto_cov = veto_n / n if n else float("nan")

        # Apply the two magnitude formulas according to the expanded direction.
        actual_move = df.next_close.to_numpy(float) - df.close.to_numpy(float)
        ref_move = np.where(expanded > 0, df.long_move.to_numpy(float), -df.short_move.to_numpy(float))
        err = ref_move - actual_move
        mae = float(np.mean(np.abs(err)))
        mape = float(np.mean(np.abs(err) / np.maximum(np.abs(df.next_close.to_numpy(float)), 1e-12)))

        print(f"  expanded checkpoint: N={n} acc={expanded_acc*100:.2f}%")
        print(f"  formula-only inverse: N={formula_n} acc={formula_acc*100:.2f}%")
        print(f"  checkpoint + formula veto: N={veto_n} acc={veto_acc*100:.2f}% coverage={veto_cov*100:.2f}%")
        print(f"  magnitude formulas on expanded side: MAE_close={mae:.4f} MAPE_close={mape*100:.3f}%")
        print()

        pooled_n += n
        pooled_correct += int(np.sum(expanded[valid] == actual[valid]))
        pooled_formula_n += formula_n
        pooled_formula_correct += int(np.sum(formula[(formula != 0) & valid] == actual[(formula != 0) & valid]))
        pooled_veto_n += veto_n
        pooled_veto_correct += int(np.sum(expanded[disagree] == actual[disagree]))

    print("-" * 108)
    print(f"POOLED expanded checkpoint = {pooled_correct / pooled_n * 100:.2f}% (N={pooled_n})")
    print(f"POOLED formula-only inverse = {pooled_formula_correct / pooled_formula_n * 100:.2f}% (N={pooled_formula_n})")
    print(f"POOLED checkpoint+veto = {pooled_veto_correct / pooled_veto_n * 100:.2f}% (N={pooled_veto_n}, coverage={pooled_veto_n / pooled_n * 100:.2f}%)")
    print("=" * 108)


if __name__ == "__main__":
    main()
