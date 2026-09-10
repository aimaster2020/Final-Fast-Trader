from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BUCKETS = ("LONG_correct", "SHORT_correct", "LONG_wrong", "SHORT_wrong")


def load(path: str) -> pd.DataFrame:
    cols = {"open", "high", "low", "close"}
    df = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in cols)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["J"] = df.close - df.open
    df["K"] = df.high - df.close
    df["L_checkpoint"] = df.close - df.low
    df["U"] = df.J.shift(-1) - df.J
    df["next_close"] = df.close.shift(-1)
    df["next_move"] = df.next_close - df.close
    df["next_return"] = df.next_move / df.close
    df["up_ref"] = df.high - df.open
    df["down_ref"] = df.open - df.low
    df["up_ref_return"] = df.up_ref / df.close
    df["down_ref_return"] = -df.down_ref / df.close
    return df.dropna().reset_index(drop=True)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def r2(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0:
        return float("nan")
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def stats(actual_r: np.ndarray, ref_r: np.ndarray) -> dict[str, float]:
    if len(actual_r) == 0:
        return {"n": 0, "mae_bp": 0.0, "rmse_bp": 0.0, "bias_bp": 0.0, "corr": float("nan"), "r2": float("nan"), "w10bp": 0.0, "w25bp": 0.0, "w50bp": 0.0, "median_abs_bp": 0.0}
    err = ref_r - actual_r
    abs_bp = np.abs(err) * 10000.0
    return {
        "n": int(len(actual_r)),
        "mae_bp": float(np.mean(abs_bp)),
        "rmse_bp": float(np.sqrt(np.mean(err ** 2)) * 10000.0),
        "bias_bp": float(np.mean(err) * 10000.0),
        "corr": corr(ref_r, actual_r),
        "r2": r2(actual_r, ref_r),
        "w10bp": float(np.mean(abs_bp <= 10.0)),
        "w25bp": float(np.mean(abs_bp <= 25.0)),
        "w50bp": float(np.mean(abs_bp <= 50.0)),
        "median_abs_bp": float(np.median(abs_bp)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalized mechanism test for directional candle formulas")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.50)
    ap.add_argument("--output-dir", default="reports/formula_mechanism_v2")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    summary: list[dict[str, float | str | int]] = []

    for symbol in symbols:
        path = Path(args.data_dir) / f"{symbol}_1h.csv"
        df = load(str(path))
        n = len(df)
        train0 = int(n * args.train_ratio)
        rem = n - train0
        size = rem // args.folds
        rows = []

        for fold in range(args.folds):
            start = train0 + fold * size
            end = train0 + (fold + 1) * size if fold < args.folds - 1 else n
            train = df.iloc[:start]
            test = df.iloc[start:end]

            X = np.column_stack([np.ones(len(train)), train[["J", "K", "L_checkpoint"]].to_numpy(float)])
            y = train["U"].to_numpy(float)
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            Xt = np.column_stack([np.ones(len(test)), test[["J", "K", "L_checkpoint"]].to_numpy(float)])
            pred_dir = np.sign(Xt @ coef)
            actual_u_dir = np.sign(test.U.to_numpy(float))

            actual_r = test.next_return.to_numpy(float)
            ref_r = np.where(pred_dir > 0, test.up_ref_return.to_numpy(float), test.down_ref_return.to_numpy(float))
            masks = {
                "LONG_correct": (pred_dir > 0) & (actual_u_dir > 0),
                "SHORT_correct": (pred_dir < 0) & (actual_u_dir < 0),
                "LONG_wrong": (pred_dir > 0) & (actual_u_dir < 0),
                "SHORT_wrong": (pred_dir < 0) & (actual_u_dir > 0),
            }
            for bucket, mask in masks.items():
                s = stats(actual_r[mask], ref_r[mask])
                rows.append({"symbol": symbol, "fold": fold + 1, "bucket": bucket, **s})

        detail = pd.DataFrame(rows)
        detail.to_csv(out / f"{symbol}_mechanism.csv", index=False)

        for bucket in BUCKETS:
            actual_all: list[np.ndarray] = []
            ref_all: list[np.ndarray] = []
            for fold in range(args.folds):
                start = train0 + fold * size
                end = train0 + (fold + 1) * size if fold < args.folds - 1 else n
                train = df.iloc[:start]
                test = df.iloc[start:end]
                X = np.column_stack([np.ones(len(train)), train[["J", "K", "L_checkpoint"]].to_numpy(float)])
                coef = np.linalg.lstsq(X, train["U"].to_numpy(float), rcond=None)[0]
                Xt = np.column_stack([np.ones(len(test)), test[["J", "K", "L_checkpoint"]].to_numpy(float)])
                pred_dir = np.sign(Xt @ coef)
                actual_u_dir = np.sign(test.U.to_numpy(float))
                mask = {
                    "LONG_correct": (pred_dir > 0) & (actual_u_dir > 0),
                    "SHORT_correct": (pred_dir < 0) & (actual_u_dir < 0),
                    "LONG_wrong": (pred_dir > 0) & (actual_u_dir < 0),
                    "SHORT_wrong": (pred_dir < 0) & (actual_u_dir > 0),
                }[bucket]
                actual_all.append(test.next_return.to_numpy(float)[mask])
                ref_all.append(np.where(pred_dir[mask] > 0, test.up_ref_return.to_numpy(float)[mask], test.down_ref_return.to_numpy(float)[mask]))
            a = np.concatenate(actual_all) if actual_all else np.array([])
            r = np.concatenate(ref_all) if ref_all else np.array([])
            s = stats(a, r)
            summary.append({"symbol": symbol, "bucket": bucket, **s})

    pd.DataFrame(summary).to_csv(out / "mechanism_summary.csv", index=False)

    print("=" * 145)
    print("DIRECTIONAL FORMULA MECHANISM V2 — NORMALIZED NEXT-RETURN ANALYSIS")
    print("=" * 145)
    print("actual = (next_close-close)/close; reference = +(H-O)/C for LONG, -(O-L)/C for SHORT")
    print("MAPE is intentionally removed: near-zero next moves make percentage-of-move errors unstable.")
    print("bucket             N  MAE(bp) RMSE(bp) bias(bp) corr    R2    <=10bp <=25bp <=50bp median_abs_bp")
    for row in summary:
        print(f"{row['symbol']:8} {row['bucket']:14} {int(row['n']):4d} {row['mae_bp']:8.2f} {row['rmse_bp']:8.2f} {row['bias_bp']:8.2f} {row['corr']:6.3f} {row['r2']:6.3f} {row['w10bp']*100:7.2f}% {row['w25bp']*100:7.2f}% {row['w50bp']*100:7.2f}% {row['median_abs_bp']:12.2f}")
    print("-" * 145)
    print(f"summary_csv={out / 'mechanism_summary.csv'}")
    print(f"per_symbol_csv={out}\\<SYMBOL>_mechanism.csv")


if __name__ == "__main__":
    main()
