from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close"]
THRESHOLDS = (0.001, 0.0025, 0.005, 0.01)
BUCKETS = ("PL_AL", "PL_AS", "PS_AL", "PS_AS")


def load_data(path: str, limit: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in REQUIRED)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED).copy()
    if limit > 0:
        df = df.tail(limit + 1).copy()

    df["J"] = df["close"] - df["open"]
    df["U"] = df["J"].shift(-1) - df["J"]
    df["next_close"] = df["close"].shift(-1)
    df["actual_next_close_move"] = df["next_close"] - df["close"]
    df["long_pred_close"] = df["close"] + np.abs(df["high"] - df["open"])
    df["short_pred_close"] = df["close"] - np.abs(df["low"] - df["open"])
    return df.dropna(subset=["U", "next_close"]).reset_index(drop=True)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = ["J", "high", "close", "low"]
    # Rebuild the original checkpoint features exactly: J, K, L_checkpoint.
    train_k = train["high"] - train["close"]
    train_l = train["close"] - train["low"]
    test_k = test["high"] - test["close"]
    test_l = test["close"] - test["low"]

    x_train = np.column_stack([
        np.ones(len(train)),
        train["J"].to_numpy(float),
        train_k.to_numpy(float),
        train_l.to_numpy(float),
    ])
    y_train = train["U"].to_numpy(float)
    coef = np.linalg.lstsq(x_train, y_train, rcond=None)[0]

    x_test = np.column_stack([
        np.ones(len(test)),
        test["J"].to_numpy(float),
        test_k.to_numpy(float),
        test_l.to_numpy(float),
    ])
    return test["U"].to_numpy(float), x_test @ coef


def close_stats(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if len(actual) == 0:
        return {"n": 0.0, "mae": 0.0, "mape": 0.0, **{f"w{int(t*10000):04d}": 0.0 for t in THRESHOLDS}}
    rel = np.abs(predicted - actual) / np.maximum(np.abs(actual), 1e-12)
    return {
        "n": float(len(actual)),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "mape": float(np.mean(rel)),
        **{f"w{int(t*10000):04d}": float(np.mean(rel <= t)) for t in THRESHOLDS},
    }


def run_symbol(path: Path, folds: int, train_ratio: float, limit: int) -> tuple[list[dict], list[dict]]:
    symbol = path.stem.replace("_1h", "")
    df = load_data(str(path), limit)
    n = len(df)
    initial_train = int(n * train_ratio)
    remaining = n - initial_train
    if remaining < folds * 10:
        raise ValueError(f"{symbol}: not enough OOS candles n={n} folds={folds}")
    test_size = remaining // folds

    rows: list[dict] = []
    details: list[dict] = []
    for fold in range(folds):
        start = initial_train + fold * test_size
        end = initial_train + (fold + 1) * test_size if fold < folds - 1 else n
        train = df.iloc[:start]
        test = df.iloc[start:end].copy()
        _, pred_u = fit_predict(train, test)
        pred_dir = np.sign(pred_u)
        actual_dir = np.sign(test["U"].to_numpy(float))
        long_pred = test["long_pred_close"].to_numpy(float)
        short_pred = test["short_pred_close"].to_numpy(float)
        actual_close = test["next_close"].to_numpy(float)
        valid = np.isfinite(pred_u) & np.isfinite(actual_dir) & np.isfinite(actual_close)
        pred_dir = pred_dir[valid]
        actual_dir = actual_dir[valid]
        long_pred = long_pred[valid]
        short_pred = short_pred[valid]
        actual_close = actual_close[valid]
        testv = test.loc[valid]

        for key, mask, preds in (
            ("PL_AL", (pred_dir > 0) & (actual_dir > 0), long_pred),
            ("PL_AS", (pred_dir > 0) & (actual_dir < 0), long_pred),
            ("PS_AL", (pred_dir < 0) & (actual_dir > 0), short_pred),
            ("PS_AS", (pred_dir < 0) & (actual_dir < 0), short_pred),
        ):
            st = close_stats(actual_close[mask], preds[mask])
            rows.append({"symbol": symbol, "fold": fold + 1, "bucket": key, "dir_accuracy": float(np.mean(pred_dir == actual_dir)), **st})

        for i in range(len(testv)):
            if pred_dir[i] > 0:
                predicted_close = long_pred[i]
            elif pred_dir[i] < 0:
                predicted_close = short_pred[i]
            else:
                predicted_close = np.nan
            details.append({
                "symbol": symbol,
                "fold": fold + 1,
                "row_index": int(testv.index[i]),
                "open": float(testv.iloc[i]["open"]),
                "high": float(testv.iloc[i]["high"]),
                "low": float(testv.iloc[i]["low"]),
                "close": float(testv.iloc[i]["close"]),
                "J": float(testv.iloc[i]["J"]),
                "K_checkpoint": float(testv.iloc[i]["high"] - testv.iloc[i]["close"]),
                "L_checkpoint": float(testv.iloc[i]["close"] - testv.iloc[i]["low"]),
                "predicted_U": float(pred_u[valid][i]),
                "predicted_direction": int(pred_dir[i]),
                "actual_U": float(testv.iloc[i]["U"]),
                "actual_U_direction": int(actual_dir[i]),
                "long_formula_close": float(long_pred[i]),
                "short_formula_close": float(short_pred[i]),
                "selected_predicted_close": float(predicted_close),
                "actual_next_close": float(actual_close[i]),
                "selected_abs_error": float(abs(predicted_close - actual_close[i])),
                "selected_pct_error": float(abs(predicted_close - actual_close[i]) / max(abs(actual_close[i]), 1e-12)),
            })

    summary_rows: list[dict] = []
    for bucket in BUCKETS:
        group = [r for r in rows if r["bucket"] == bucket]
        actual_n = int(sum(r["n"] for r in group))
        if not actual_n:
            continue
        # Recompute aggregate using stored detailed rows to avoid fold-weighting artifacts.
        selected = []
        for d in details:
            if (bucket == "PL_AL" and d["predicted_direction"] > 0 and d["actual_U_direction"] > 0) or \
               (bucket == "PL_AS" and d["predicted_direction"] > 0 and d["actual_U_direction"] < 0) or \
               (bucket == "PS_AL" and d["predicted_direction"] < 0 and d["actual_U_direction"] > 0) or \
               (bucket == "PS_AS" and d["predicted_direction"] < 0 and d["actual_U_direction"] < 0):
                selected.append(d)
        errs = np.array([d["selected_pct_error"] for d in selected], dtype=float)
        abs_errs = np.array([d["selected_abs_error"] for d in selected], dtype=float)
        dir_acc = float(sum(d["predicted_direction"] == d["actual_U_direction"] for d in details) / len([d for d in details if d["predicted_direction"] != 0 and d["actual_U_direction"] != 0])) if details else 0.0
        summary_rows.append({
            "symbol": symbol,
            "bucket": bucket,
            "n": len(selected),
            "dir_accuracy": dir_acc,
            "mae": float(np.mean(abs_errs)),
            "mape": float(np.mean(errs)),
            "w0100": float(np.mean(errs <= 0.001)),
            "w0250": float(np.mean(errs <= 0.0025)),
            "w0500": float(np.mean(errs <= 0.005)),
            "w1000": float(np.mean(errs <= 0.01)),
        })
    return summary_rows, details


def main() -> None:
    ap = argparse.ArgumentParser(description="All-symbol LONG/SHORT quadrant test using the original U checkpoint")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output-dir", default="reports/formula_quadrants_all")
    args = ap.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict] = []
    all_details: list[dict] = []

    for symbol in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        path = Path(args.data_dir) / f"{symbol}_1h.csv"
        if not path.exists():
            print(f"SKIP {symbol}: missing {path}")
            continue
        print("=" * 126)
        print(f"{symbol} — ORIGINAL U CHECKPOINT + DIRECTIONAL CLOSE FORMULAS + QUADRANTS")
        print("=" * 126)
        print("LONG formula: C + |HIGH-OPEN|")
        print("SHORT formula: C - |LOW-OPEN|")
        print("PL_AL=pred LONG / actual U+   PL_AS=pred LONG / actual U-   PS_AL=pred SHORT / actual U+   PS_AS=pred SHORT / actual U-")
        summary, details = run_symbol(path, args.folds, args.train_ratio, args.limit)
        all_summary.extend(summary)
        all_details.extend(details)
        for r in summary:
            print(f"{r['bucket']:5} N={r['n']:>4} MAPE={r['mape']*100:>7.3f}% <=.10={r['w0100']*100:>6.2f}% <=.25={r['w0250']*100:>6.2f}% <=.50={r['w0500']*100:>6.2f}% <=1={r['w1000']*100:>6.2f}% MAE={r['mae']:>10.4f}")

    pd.DataFrame(all_summary).to_csv(output / "quadrant_summary.csv", index=False)
    pd.DataFrame(all_details).to_csv(output / "quadrant_details.csv", index=False)
    print("=" * 126)
    print("ALL-SYMBOL SUMMARY")
    print(pd.DataFrame(all_summary).to_string(index=False, formatters={"mape": "{:.4%}".format, "mae": "{:.6f}".format, "w0250": "{:.2%}".format, "w0500": "{:.2%}".format, "w1000": "{:.2%}".format}))
    print(f"summary_csv={output / 'quadrant_summary.csv'}")
    print(f"details_csv={output / 'quadrant_details.csv'}")


if __name__ == "__main__":
    main()
