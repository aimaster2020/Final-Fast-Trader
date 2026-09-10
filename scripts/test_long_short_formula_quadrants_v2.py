from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close"]
THRESHOLDS = (0.001, 0.0025, 0.005, 0.01)


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
    df["K"] = df["high"] - df["close"]
    df["L"] = df["close"] - df["low"]
    df["U"] = df["J"].shift(-1) - df["J"]
    df["next_close"] = df["close"].shift(-1)
    df["magnitude"] = np.abs(df["high"] - df["open"])
    return df.dropna(subset=["U", "next_close"]).reset_index(drop=True)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = ["J", "K", "L"]
    x_train = np.column_stack([np.ones(len(train)), train[features].to_numpy(float)])
    y_train = train["U"].to_numpy(float)
    coef = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
    x_test = np.column_stack([np.ones(len(test)), test[features].to_numpy(float)])
    return test["U"].to_numpy(float), x_test @ coef


def close_stats(actual: np.ndarray, predicted: np.ndarray) -> tuple[int, float, float, dict[float, float]]:
    if len(actual) == 0:
        return 0, 0.0, 0.0, {t: 0.0 for t in THRESHOLDS}
    rel = np.abs(predicted - actual) / np.maximum(np.abs(actual), 1e-12)
    mae = float(np.mean(np.abs(predicted - actual)))
    mape = float(np.mean(rel))
    within = {t: float(np.mean(rel <= t)) for t in THRESHOLDS}
    return len(actual), mae, mape, within


def main() -> None:
    ap = argparse.ArgumentParser(description="LONG/SHORT quadrants aligned with the original U checkpoint")
    ap.add_argument("--input", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.50)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = load_data(args.input, args.limit)
    n = len(df)
    initial_train = int(n * args.train_ratio)
    remaining = n - initial_train
    if remaining < args.folds * 10:
        raise ValueError(f"Not enough OOS candles: n={n} folds={args.folds}")
    test_size = remaining // args.folds

    buckets = {key: [] for key in ("PL_AL", "PL_AS", "PS_AL", "PS_AS")}
    direction_hits = 0
    direction_total = 0

    print("=" * 132)
    print("LONG/SHORT QUADRANT TEST V2 — ALIGNED WITH ORIGINAL U=J_next-J_current CHECKPOINT")
    print("=" * 132)
    print(f"candles={n} train0={initial_train} folds={args.folds} test/fold={test_size}")
    print("direction=WalkForward U=J_next-J_current features=J,K,L")
    print("actual LONG/SHORT classification = sign(U), exactly as the original checkpoint")
    print("prediction=LONG: C+|H-O|   SHORT: C-|H-O|")
    print("PL_AL=pred LONG/actual U+   PL_AS=pred LONG/actual U-   PS_AL=pred SHORT/actual U+   PS_AS=pred SHORT/actual U-")
    print("close_error=absolute percentage error versus ACTUAL next close")
    print("fold dir% PL_AL_n MAPE<=0.25<=0.50 PL_AS_n MAPE<=0.25<=0.50 PS_AL_n MAPE<=0.25<=0.50 PS_AS_n MAPE<=0.25<=0.50")

    for fold in range(args.folds):
        start = initial_train + fold * test_size
        end = initial_train + (fold + 1) * test_size if fold < args.folds - 1 else n
        train = df.iloc[:start]
        test = df.iloc[start:end].copy()

        y, p = fit_predict(train, test)
        predicted_direction = np.sign(p)
        actual_direction = np.sign(y)
        magnitude = test["magnitude"].to_numpy(float)
        current_close = test["close"].to_numpy(float)
        actual_close = test["next_close"].to_numpy(float)

        valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(magnitude) & np.isfinite(current_close) & np.isfinite(actual_close)
        y = y[valid]
        predicted_direction = predicted_direction[valid]
        actual_direction = actual_direction[valid]
        magnitude = magnitude[valid]
        current_close = current_close[valid]
        actual_close = actual_close[valid]

        direction_mask = (predicted_direction != 0) & (actual_direction != 0)
        fold_hits = int(np.sum(predicted_direction[direction_mask] == actual_direction[direction_mask]))
        fold_total = int(np.sum(direction_mask))
        direction_hits += fold_hits
        direction_total += fold_total

        preds = current_close + predicted_direction * magnitude
        masks = {
            "PL_AL": (predicted_direction > 0) & (actual_direction > 0),
            "PL_AS": (predicted_direction > 0) & (actual_direction < 0),
            "PS_AL": (predicted_direction < 0) & (actual_direction > 0),
            "PS_AS": (predicted_direction < 0) & (actual_direction < 0),
        }

        row = []
        for key, mask in masks.items():
            n_bucket, _, mape, within = close_stats(actual_close[mask], preds[mask])
            buckets[key].append((actual_close[mask], preds[mask]))
            row.append((n_bucket, mape, within[0.0025], within[0.005]))

        fold_acc = (fold_hits / fold_total * 100.0) if fold_total else 0.0
        print(
            f"{fold + 1:>4} {fold_acc:>5.2f}% "
            + " ".join(f"{n_:>6} {m * 100:>6.3f}% {w25 * 100:>6.2f}% {w50 * 100:>6.2f}%" for n_, m, w25, w50 in row)
        )

    print("-" * 132)
    for key in ("PL_AL", "PL_AS", "PS_AL", "PS_AS"):
        actual = np.concatenate([a for a, _ in buckets[key]]) if buckets[key] else np.array([])
        pred = np.concatenate([p for _, p in buckets[key]]) if buckets[key] else np.array([])
        n_bucket, mae, mape, within = close_stats(actual, pred)
        print(
            f"{key:5} N={n_bucket:>4} MAE={mae:>10.4f} MAPE={mape * 100:>7.3f}% "
            f"<=0.10%={within[0.001] * 100:>6.2f}% <=0.25%={within[0.0025] * 100:>6.2f}% "
            f"<=0.50%={within[0.005] * 100:>6.2f}% <=1.00%={within[0.01] * 100:>6.2f}%"
        )
    accuracy = direction_hits / direction_total * 100.0 if direction_total else 0.0
    print(f"DIRECTION_ACCURACY={accuracy:.2f}%")


if __name__ == "__main__":
    main()
