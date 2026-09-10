from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLDS = (0.001, 0.0025, 0.005, 0.01)
SYMBOLS_DEFAULT = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in {"open", "high", "low", "close"})
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["J"] = df.close - df.open
    df["K"] = df.high - df.close
    df["L_checkpoint"] = df.close - df.low
    df["U"] = df.J.shift(-1) - df.J
    df["next_close"] = df.close.shift(-1)
    df["next_move"] = df.next_close - df.close
    df["up_ref"] = df.high - df.open
    df["down_ref"] = df.open - df.low
    return df.dropna().reset_index(drop=True)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def metrics(actual_move: np.ndarray, ref: np.ndarray) -> tuple[int, float, float, float, float, dict[float, float]]:
    if len(actual_move) == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, {t: 0.0 for t in THRESHOLDS}
    err = ref - actual_move
    rel = np.abs(err) / np.maximum(np.abs(actual_move), 1e-12)
    within = {t: float(np.mean(rel <= t)) for t in THRESHOLDS}
    return len(actual_move), float(np.mean(np.abs(err))), float(np.mean(rel)), float(np.mean(err)), corr(ref, actual_move), within


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze whether directional candle ranges explain the next close move")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=SYMBOLS_DEFAULT)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.50)
    ap.add_argument("--output-dir", default="reports/formula_mechanism")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    summary = []

    for symbol in symbols:
        path = Path(args.data_dir) / f"{symbol}_1h.csv"
        df = load(str(path))
        n = len(df)
        train0 = int(n * args.train_ratio)
        rem = n - train0
        size = rem // args.folds
        rows = []

        # Use the original checkpoint in each OOS fold; only summarize mechanism after prediction.
        for fold in range(args.folds):
            start = train0 + fold * size
            end = train0 + (fold + 1) * size if fold < args.folds - 1 else n
            train = df.iloc[:start]
            test = df.iloc[start:end]
            X = np.column_stack([np.ones(len(train)), train[["J", "K", "L_checkpoint"]].to_numpy(float)])
            y = train["U"].to_numpy(float)
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            Xt = np.column_stack([np.ones(len(test)), test[["J", "K", "L_checkpoint"]].to_numpy(float)])
            pred_u = Xt @ coef
            pred_dir = np.sign(pred_u)
            actual_u_dir = np.sign(test.U.to_numpy(float))
            next_move = test.next_move.to_numpy(float)
            up_ref = test.up_ref.to_numpy(float)
            down_ref = test.down_ref.to_numpy(float)
            close = test.close.to_numpy(float)

            masks = {
                "LONG_correct": (pred_dir > 0) & (actual_u_dir > 0),
                "SHORT_correct": (pred_dir < 0) & (actual_u_dir < 0),
                "LONG_wrong": (pred_dir > 0) & (actual_u_dir < 0),
                "SHORT_wrong": (pred_dir < 0) & (actual_u_dir > 0),
            }
            for bucket, mask in masks.items():
                # Signed reference matching the predicted side.
                ref_move = np.where(pred_dir[mask] > 0, up_ref[mask], -down_ref[mask])
                actual = next_move[mask]
                N, mae, mape, bias, c, within = metrics(actual, ref_move)
                rows.append({"symbol": symbol, "fold": fold + 1, "bucket": bucket, "n": N, "mae_move": mae, "mape_move": mape, "bias_move": bias, "corr_ref_actual_move": c, "w025": within[0.0025], "w050": within[0.005]})

        detail = pd.DataFrame(rows)
        detail.to_csv(out / f"{symbol}_mechanism.csv", index=False)
        for bucket in ("LONG_correct", "SHORT_correct", "LONG_wrong", "SHORT_wrong"):
            d = detail[detail.bucket == bucket]
            nsum = int(d.n.sum())
            # Weighted aggregates for MAE/MAPE/correlation are computed from reconstructed records below.
            # Re-run concatenation to avoid averaging fold metrics incorrectly.
            actual_all = []
            ref_all = []
            for fold in range(args.folds):
                start = train0 + fold * size
                end = train0 + (fold + 1) * size if fold < args.folds - 1 else n
                train = df.iloc[:start]
                test = df.iloc[start:end]
                X = np.column_stack([np.ones(len(train)), train[["J", "K", "L_checkpoint"]].to_numpy(float)])
                y = train.U.to_numpy(float)
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                Xt = np.column_stack([np.ones(len(test)), test[["J", "K", "L_checkpoint"]].to_numpy(float)])
                pred_dir = np.sign(Xt @ coef)
                actual_u_dir = np.sign(test.U.to_numpy(float))
                mask = {"LONG_correct": (pred_dir > 0) & (actual_u_dir > 0), "SHORT_correct": (pred_dir < 0) & (actual_u_dir < 0), "LONG_wrong": (pred_dir > 0) & (actual_u_dir < 0), "SHORT_wrong": (pred_dir < 0) & (actual_u_dir > 0)}[bucket]
                actual_all.append(test.next_move.to_numpy(float)[mask])
                ref_all.append(np.where(pred_dir[mask] > 0, test.up_ref.to_numpy(float)[mask], -test.down_ref.to_numpy(float)[mask]))
            a = np.concatenate(actual_all) if actual_all else np.array([])
            r = np.concatenate(ref_all) if ref_all else np.array([])
            N, mae, mape, bias, c, within = metrics(a, r)
            summary.append({"symbol": symbol, "bucket": bucket, "n": N, "mae_move": mae, "mape_move": mape, "bias_move": bias, "corr_ref_actual_move": c, "w025": within[0.0025], "w050": within[0.005], "train0": train0, "folds": args.folds})

    result = pd.DataFrame(summary)
    result.to_csv(out / "mechanism_summary.csv", index=False)

    print("=" * 128)
    print("DIRECTIONAL FORMULA MECHANISM — DOES H-O / O-L EXPLAIN NEXT-CLOSE MOVE?")
    print("=" * 128)
    print("reference move: LONG=HIGH-OPEN   SHORT=-(OPEN-LOW)")
    print("groups: original OOS U-checkpoint prediction + actual U direction")
    print("metric: next_move = next_close-close")
    print("symbol bucket N MAE_move MAPE_move bias corr_ref_vs_actual <=0.25 <=0.50")
    for row in summary:
        print(f"{row['symbol']:8} {row['bucket']:13} {row['n']:4d} {row['mae_move']:10.6f} {row['mape_move']*100:9.3f}% {row['bias_move']:10.6f} {row['corr_ref_actual_move']:8.3f} {row['w025']*100:8.2f}% {row['w050']*100:8.2f}%")
    print("-" * 128)
    print(f"summary_csv={out / 'mechanism_summary.csv'}")
    print(f"per_symbol_csv={out}\\<SYMBOL>_mechanism.csv")


if __name__ == "__main__":
    main()
