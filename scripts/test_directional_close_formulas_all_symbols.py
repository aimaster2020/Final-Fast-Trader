from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close"]
THRESHOLDS = (0.001, 0.0025, 0.005, 0.01)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in REQUIRED)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED).copy().reset_index(drop=True)
    df["J"] = df["close"] - df["open"]
    df["K"] = df["high"] - df["close"]
    df["L_ckpt"] = df["close"] - df["low"]
    df["U"] = df["J"].shift(-1) - df["J"]
    df["next_close"] = df["close"].shift(-1)
    df["long_pred_close"] = df["close"] + np.abs(df["high"] - df["open"])
    df["short_pred_close"] = df["close"] - np.abs(df["low"] - df["open"])
    return df.dropna(subset=["U", "next_close"]).reset_index(drop=True)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = ["J", "K", "L_ckpt"]
    x_train = np.column_stack([np.ones(len(train)), train[features].to_numpy(float)])
    y_train = train["U"].to_numpy(float)
    coef = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
    x_test = np.column_stack([np.ones(len(test)), test[features].to_numpy(float)])
    return y_train, x_test @ coef


def stats(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if len(actual) == 0:
        return {
            "n": 0,
            "mae": 0.0,
            "mape": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p100": 0.0,
            "bias": 0.0,
        }
    err = predicted - actual
    rel = np.abs(err) / np.maximum(np.abs(actual), 1e-12)
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(rel)),
        "p10": float(np.mean(rel <= THRESHOLDS[0])),
        "p25": float(np.mean(rel <= THRESHOLDS[1])),
        "p50": float(np.mean(rel <= THRESHOLDS[2])),
        "p100": float(np.mean(rel <= THRESHOLDS[3])),
        "bias": float(np.mean(err)),
    }


def one_symbol(symbol: str, path: str, folds: int, train_ratio: float) -> tuple[list[dict], pd.DataFrame]:
    df = load_data(path)
    n = len(df)
    initial_train = int(n * train_ratio)
    remaining = n - initial_train
    if remaining < folds * 10:
        raise ValueError(f"{symbol}: not enough OOS candles: n={n}, folds={folds}")
    test_size = remaining // folds

    rows: list[dict] = []
    detail_rows: list[dict] = []
    print("=" * 132)
    print(f"{symbol} — DIRECTION CHECKPOINT + DIRECTIONAL CLOSE FORMULAS")
    print("=" * 132)
    print(f"candles={n} train0={initial_train} folds={folds} test/fold={test_size}")
    print("checkpoint: U=J_next-J_current, features=J, K, L_checkpoint, expanding walk-forward")
    print("LONG : predicted_close = C + |HIGH-OPEN|")
    print("SHORT: predicted_close = C - |LOW-OPEN|")
    print("LONG/SHORT groups are selected ONLY from OOS checkpoint prediction; no threshold tuning.")
    print("close_error = abs(predicted_close - actual_next_close) / actual_next_close")
    print("fold dir% longN longMAPE long<=.25 long<=.50 shortN shortMAPE short<=.25 short<=.50")

    all_groups = {"LONG": [], "SHORT": []}
    total_hits = 0
    total_dir = 0

    for fold in range(folds):
        start = initial_train + fold * test_size
        end = initial_train + (fold + 1) * test_size if fold < folds - 1 else n
        train = df.iloc[:start]
        test = df.iloc[start:end].copy()
        _, pred_u = fit_predict(train, test)
        actual_u = test["U"].to_numpy(float)
        pred_dir = np.sign(pred_u)
        actual_dir = np.sign(actual_u)
        valid = np.isfinite(pred_u) & np.isfinite(actual_u)
        pred_dir = pred_dir[valid]
        actual_dir = actual_dir[valid]
        test = test.iloc[np.flatnonzero(valid)].copy()

        mask_dir = (pred_dir != 0) & (actual_dir != 0)
        fold_hits = int(np.sum(pred_dir[mask_dir] == actual_dir[mask_dir]))
        fold_total = int(np.sum(mask_dir))
        total_hits += fold_hits
        total_dir += fold_total
        fold_acc = fold_hits / fold_total * 100 if fold_total else 0.0

        actual_close = test["next_close"].to_numpy(float)
        long_pred = test["long_pred_close"].to_numpy(float)
        short_pred = test["short_pred_close"].to_numpy(float)
        long_mask = pred_dir > 0
        short_mask = pred_dir < 0
        long_s = stats(actual_close[long_mask], long_pred[long_mask])
        short_s = stats(actual_close[short_mask], short_pred[short_mask])
        all_groups["LONG"].append((actual_close[long_mask], long_pred[long_mask]))
        all_groups["SHORT"].append((actual_close[short_mask], short_pred[short_mask]))

        print(
            f"{fold + 1:>4} {fold_acc:>5.2f}% {long_s['n']:>6} {long_s['mape']*100:>8.3f}% "
            f"{long_s['p25']*100:>9.2f}% {long_s['p50']*100:>9.2f}% "
            f"{short_s['n']:>7} {short_s['mape']*100:>9.3f}% {short_s['p25']*100:>10.2f}% {short_s['p50']*100:>10.2f}%"
        )

        for local_idx, (_, r) in enumerate(test.iterrows()):
            direction = "LONG" if pred_dir[local_idx] > 0 else "SHORT" if pred_dir[local_idx] < 0 else "ZERO"
            if direction == "ZERO":
                continue
            pred_close = float(long_pred[local_idx] if direction == "LONG" else short_pred[local_idx])
            actual = float(actual_close[local_idx])
            abs_err = abs(pred_close - actual)
            pct_err = abs_err / max(abs(actual), 1e-12)
            actual_u_value = float(actual_u[valid][local_idx])
            actual_direction = "LONG" if actual_u_value > 0 else "SHORT" if actual_u_value < 0 else "ZERO"
            detail_rows.append({
                "symbol": symbol,
                "fold": fold + 1,
                "row_in_oos_fold": local_idx,
                "direction_prediction": direction,
                "actual_U": actual_u_value,
                "actual_U_direction": actual_direction,
                "current_open": float(r["open"]),
                "current_high": float(r["high"]),
                "current_low": float(r["low"]),
                "current_close": float(r["close"]),
                "high_minus_open_abs": abs(float(r["high"]) - float(r["open"])),
                "low_minus_open_abs": abs(float(r["low"]) - float(r["open"])),
                "predicted_next_close": pred_close,
                "actual_next_close": actual,
                "absolute_error": abs_err,
                "percentage_error": pct_err,
                "within_0_10pct": pct_err <= 0.001,
                "within_0_25pct": pct_err <= 0.0025,
                "within_0_50pct": pct_err <= 0.005,
                "within_1pct": pct_err <= 0.01,
            })

    print("-" * 132)
    aggregate = {}
    for direction in ("LONG", "SHORT"):
        actual = np.concatenate([a for a, _ in all_groups[direction]])
        pred = np.concatenate([p for _, p in all_groups[direction]])
        s = stats(actual, pred)
        aggregate[direction] = s
        print(
            f"{direction:5} N={s['n']:>4} MAE={s['mae']:>10.4f} MAPE={s['mape']*100:>7.3f}% "
            f"<=0.10%={s['p10']*100:>6.2f}% <=0.25%={s['p25']*100:>6.2f}% "
            f"<=0.50%={s['p50']*100:>6.2f}% <=1.00%={s['p100']*100:>6.2f}% bias={s['bias']:+.4f}"
        )

    accuracy = total_hits / total_dir * 100 if total_dir else 0.0
    print(f"DIRECTION_ACCURACY={accuracy:.2f}% ({total_hits}/{total_dir})")
    print()

    summary_rows = [
        {
            "symbol": symbol,
            "direction_accuracy_pct": accuracy,
            "direction": direction,
            **aggregate[direction],
        }
        for direction in ("LONG", "SHORT")
    ]
    return summary_rows, pd.DataFrame(detail_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="All-symbol test for C+|H-O| LONG and C-|L-O| SHORT")
    ap.add_argument("--data-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.50)
    ap.add_argument("--output-dir", default="reports/formula_direction_test")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    details: list[pd.DataFrame] = []
    for symbol in symbols:
        path = Path(args.data_dir) / f"{symbol}_1h.csv"
        if not path.exists():
            print(f"SKIP {symbol}: missing {path}")
            continue
        rows, detail = one_symbol(symbol, str(path), args.folds, args.train_ratio)
        summary.extend(rows)
        details.append(detail)

    if not summary:
        raise SystemExit("No symbol files were found.")

    summary_df = pd.DataFrame(summary)
    detail_df = pd.concat(details, ignore_index=True)
    summary_path = out / "summary.csv"
    detail_path = out / "details.csv"
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    print("=" * 132)
    print("ALL SYMBOL SUMMARY")
    print("=" * 132)
    print("symbol direction N dirAcc% MAPE MAE <=0.25% <=0.50% <=1.00% bias")
    for _, r in summary_df.iterrows():
        print(
            f"{r['symbol']:7} {r['direction']:>7} {int(r['n']):>5} {r['direction_accuracy_pct']:>7.2f}% "
            f"{r['mape']*100:>7.3f}% {r['mae']:>10.4f} {r['p25']*100:>8.2f}% {r['p50']*100:>8.2f}% "
            f"{r['p100']*100:>8.2f}% {r['bias']:+.4f}"
        )
    print("-" * 132)
    print(f"summary_csv={summary_path}")
    print(f"details_csv={detail_path}")
    print("details columns: fold, predicted direction, actual U direction, OHLC, both formula components, predicted close, actual next close, absolute/% error, threshold flags")


if __name__ == "__main__":
    main()
