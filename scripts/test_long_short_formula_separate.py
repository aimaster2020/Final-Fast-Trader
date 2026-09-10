from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close"]
THRESHOLDS = (0.001, 0.0025, 0.005, 0.01)
FEATURES = ["J", "K", "L"]


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
    df["magnitude"] = (df["high"] - df["open"]).abs()
    return df.dropna(subset=["U", "next_close", "magnitude"]).reset_index(drop=True)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_train = np.column_stack([np.ones(len(train)), train[FEATURES].to_numpy(float)])
    y_train = train["U"].to_numpy(float)
    coef = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
    x_test = np.column_stack([np.ones(len(test)), test[FEATURES].to_numpy(float)])
    return test["U"].to_numpy(float), x_test @ coef


def metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, dict[float, float]]:
    if len(actual) == 0:
        return float("nan"), float("nan"), {t: float("nan") for t in THRESHOLDS}
    abs_error = np.abs(predicted - actual)
    rel_error = abs_error / np.maximum(np.abs(actual), 1e-12)
    return (
        float(np.mean(abs_error)),
        float(np.mean(rel_error)),
        {t: float(np.mean(rel_error <= t)) for t in THRESHOLDS},
    )


def print_side(name: str, mask: np.ndarray, actual: np.ndarray, predicted: np.ndarray) -> None:
    a = actual[mask]
    p = predicted[mask]
    mae, mape, within = metrics(a, p)
    print(
        f"{name:<5} N={len(a):>4} MAE={mae:>10.4f} MAPE={mape * 100:>7.3f}% "
        f"<=0.10%={within[0.001] * 100:>6.2f}% "
        f"<=0.25%={within[0.0025] * 100:>6.2f}% "
        f"<=0.50%={within[0.005] * 100:>6.2f}% "
        f"<=1.00%={within[0.01] * 100:>6.2f}%"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Separate LONG/SHORT test for checkpoint direction + H-O magnitude")
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

    all_actual = []
    all_pred = []
    all_dir = []
    all_true_u = []

    print("=" * 118)
    print("LONG vs SHORT SEPARATE TEST — DIRECTION CHECKPOINT + |H-O| MAGNITUDE")
    print("=" * 118)
    print(f"candles={n} train0={initial_train} folds={args.folds} test/fold={test_size}")
    print("direction=WalkForward U=J_next-J_current features=J,K,L")
    print("long_predicted_close=C+|H-O|    short_predicted_close=C-|H-O|")
    print("IMPORTANT: LONG/SHORT groups are selected ONLY from the OOS direction prediction.")
    print("error=absolute percentage error versus ACTUAL next close")
    print()
    print("fold train test dir% LONG_N LONG_MAPE LONG<=0.25% LONG<=0.50% SHORT_N SHORT_MAPE SHORT<=0.25% SHORT<=0.50%")

    for fold in range(args.folds):
        start = initial_train + fold * test_size
        end = initial_train + (fold + 1) * test_size if fold < args.folds - 1 else n
        train = df.iloc[:start]
        test = df.iloc[start:end]

        true_u, pred_u = fit_predict(train, test)
        direction = np.where(pred_u > 0, 1, np.where(pred_u < 0, -1, 0))
        actual_close = test["next_close"].to_numpy(float)
        magnitude = test["magnitude"].to_numpy(float)
        current_close = test["close"].to_numpy(float)

        long_pred = current_close + magnitude
        short_pred = current_close - magnitude
        valid = np.isfinite(true_u) & np.isfinite(pred_u) & np.isfinite(actual_close) & np.isfinite(magnitude)
        direction = direction[valid]
        actual_close = actual_close[valid]
        long_pred = long_pred[valid]
        short_pred = short_pred[valid]
        true_u = true_u[valid]

        dir_mask = np.sign(true_u) != 0
        dir_acc = float(np.mean(direction[dir_mask] == np.sign(true_u[dir_mask]))) if dir_mask.any() else 0.0
        long_mask = direction > 0
        short_mask = direction < 0

        def mape_only(mask: np.ndarray, pred: np.ndarray) -> float:
            if not mask.any():
                return float("nan")
            rel = np.abs(pred[mask] - actual_close[mask]) / np.maximum(np.abs(actual_close[mask]), 1e-12)
            return float(np.mean(rel) * 100)

        def within(mask: np.ndarray, pred: np.ndarray, threshold: float) -> float:
            if not mask.any():
                return float("nan")
            rel = np.abs(pred[mask] - actual_close[mask]) / np.maximum(np.abs(actual_close[mask]), 1e-12)
            return float(np.mean(rel <= threshold) * 100)

        print(
            f"{fold + 1:>4} {len(train):>5} {len(test):>4} {dir_acc * 100:>6.2f}% "
            f"{long_mask.sum():>7} {mape_only(long_mask, long_pred):>9.3f}% {within(long_mask, long_pred, .0025):>11.2f}% {within(long_mask, long_pred, .005):>11.2f}% "
            f"{short_mask.sum():>8} {mape_only(short_mask, short_pred):>10.3f}% {within(short_mask, short_pred, .0025):>12.2f}% {within(short_mask, short_pred, .005):>12.2f}%"
        )

        all_actual.append(actual_close)
        all_pred.append(np.where(direction > 0, long_pred, short_pred))
        all_dir.append(direction)
        all_true_u.append(true_u)

    actual = np.concatenate(all_actual)
    combined_pred = np.concatenate(all_pred)
    direction = np.concatenate(all_dir)
    true_u = np.concatenate(all_true_u)

    print("-" * 118)
    print("AGGREGATE")
    print_side("LONG", direction > 0, actual, np.concatenate([x for x in [combined_pred] if len(x)]))
    print_side("SHORT", direction < 0, actual, np.concatenate([x for x in [combined_pred] if len(x)]))

    # Correct aggregate side metrics separately, plus counts/direction quality.
    for name, side in (("LONG", direction > 0), ("SHORT", direction < 0)):
        mae, mape, within = metrics(actual[side], combined_pred[side])
        print(
            f"{name}_DETAIL N={side.sum()} MAE={mae:.4f} MAPE={mape * 100:.3f}% "
            f"<=0.10%={within[0.001] * 100:.2f}% <=0.25%={within[0.0025] * 100:.2f}% "
            f"<=0.50%={within[0.005] * 100:.2f}% <=1.00%={within[0.01] * 100:.2f}%"
        )

    direction_nonzero = np.sign(true_u) != 0
    print(f"DIRECTION_ACCURACY={np.mean(direction[direction_nonzero] == np.sign(true_u[direction_nonzero])) * 100:.2f}%")


if __name__ == "__main__":
    main()
