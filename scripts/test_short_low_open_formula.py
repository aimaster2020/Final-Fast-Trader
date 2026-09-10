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
    df["L_checkpoint"] = df["close"] - df["low"]
    df["U"] = df["J"].shift(-1) - df["J"]
    df["next_close"] = df["close"].shift(-1)
    df["magnitude"] = np.abs(df["low"] - df["open"])
    return df.dropna(subset=["U", "next_close"]).reset_index(drop=True)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = ["J", "K", "L_checkpoint"]
    x_train = np.column_stack([np.ones(len(train)), train[features].to_numpy(float)])
    y_train = train["U"].to_numpy(float)
    coef = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
    x_test = np.column_stack([np.ones(len(test)), test[features].to_numpy(float)])
    return test["U"].to_numpy(float), x_test @ coef


def stats(actual: np.ndarray, predicted: np.ndarray) -> tuple[int, float, float, dict[float, float]]:
    if len(actual) == 0:
        return 0, 0.0, 0.0, {t: 0.0 for t in THRESHOLDS}
    rel = np.abs(predicted - actual) / np.maximum(np.abs(actual), 1e-12)
    mae = float(np.mean(np.abs(predicted - actual)))
    mape = float(np.mean(rel))
    within = {t: float(np.mean(rel <= t)) for t in THRESHOLDS}
    return len(actual), mae, mape, within


def main() -> None:
    ap = argparse.ArgumentParser(description="SHORT formula C-|LOW-OPEN| using original walk-forward direction checkpoint")
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

    all_actual: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    direction_hits = 0
    direction_total = 0

    print("=" * 112)
    print("SHORT FORMULA TEST — C - |LOW-OPEN| ALIGNED WITH ORIGINAL U CHECKPOINT")
    print("=" * 112)
    print(f"candles={n} train0={initial_train} folds={args.folds} test/fold={test_size}")
    print("direction=WalkForward U=J_next-J_current features=J,K,L_checkpoint")
    print("SHORT predicted close = C - |LOW-OPEN|")
    print("SHORT samples = ONLY OOS cases where checkpoint predicts SHORT (predicted U < 0)")
    print("error=absolute percentage error versus ACTUAL next close")
    print("fold dir% SHORT_N MAPE <=0.10% <=0.25% <=0.50% <=1.00%")

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
        fold_acc = fold_hits / fold_total * 100.0 if fold_total else 0.0

        short_mask = predicted_direction < 0
        predicted_close = current_close - magnitude
        _, _, mape, within = stats(actual_close[short_mask], predicted_close[short_mask])
        short_n = int(np.sum(short_mask))

        all_actual.append(actual_close[short_mask])
        all_pred.append(predicted_close[short_mask])

        print(
            f"{fold + 1:>4} {fold_acc:>5.2f}% {short_n:>7} {mape * 100:>7.3f}% "
            f"{within[0.001] * 100:>8.2f}% {within[0.0025] * 100:>8.2f}% "
            f"{within[0.005] * 100:>8.2f}% {within[0.01] * 100:>8.2f}%"
        )

    actual = np.concatenate(all_actual) if all_actual else np.array([])
    pred = np.concatenate(all_pred) if all_pred else np.array([])
    n_short, mae, mape, within = stats(actual, pred)

    print("-" * 112)
    print(
        f"SHORT N={n_short:>4} MAE={mae:>10.4f} MAPE={mape * 100:>7.3f}% "
        f"<=0.10%={within[0.001] * 100:>6.2f}% <=0.25%={within[0.0025] * 100:>6.2f}% "
        f"<=0.50%={within[0.005] * 100:>6.2f}% <=1.00%={within[0.01] * 100:>6.2f}%"
    )
    bias = float(np.mean(pred - actual)) if n_short else 0.0
    print(f"SHORT_formula_bias_mean={bias:+.6f}")
    print(f"DIRECTION_ACCURACY={direction_hits / direction_total * 100.0:.2f}%")


if __name__ == "__main__":
    main()
