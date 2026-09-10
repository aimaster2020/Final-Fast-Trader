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
    df["formula_close"] = df["high"] - df["open"] + df["close"]
    return df.dropna(subset=["U", "next_close"]).reset_index(drop=True)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = ["J", "K", "L"]
    x_train = np.column_stack([np.ones(len(train)), train[features].to_numpy(float)])
    y_train = train["U"].to_numpy(float)
    coef = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
    x_test = np.column_stack([np.ones(len(test)), test[features].to_numpy(float)])
    return test["U"].to_numpy(float), x_test @ coef


def sign_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    mask = (np.sign(y) != 0) & (np.sign(p) != 0)
    return float(np.mean(np.sign(y[mask]) == np.sign(p[mask]))) if mask.any() else 0.0


def proximity(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, dict[float, float]]:
    rel_error = np.abs(predicted - actual) / np.maximum(np.abs(actual), 1e-12)
    mae = float(np.mean(np.abs(predicted - actual)))
    mape = float(np.mean(rel_error))
    within = {t: float(np.mean(rel_error <= t)) for t in THRESHOLDS}
    return mae, mape, within


def main() -> None:
    ap = argparse.ArgumentParser(description="Checkpoint direction + H-O+C next-close proximity")
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

    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    all_formula: list[np.ndarray] = []
    all_actual_close: list[np.ndarray] = []

    print("=" * 95)
    print("DIRECTION CHECKPOINT + H-O+C NEXT-CLOSE PROJECTION")
    print("=" * 95)
    print(f"candles={n} train0={initial_train} folds={args.folds} test/fold={test_size}")
    print("direction=WalkForward U=J_next-J_current features=J,K,L")
    print("formula=predicted_next_close = HIGH - OPEN + CLOSE")
    print("proximity=absolute percentage error versus ACTUAL next close")
    print("fold train test dir% MAE_close MAPE_close <=0.10% <=0.25% <=0.50% <=1.00%")

    for fold in range(args.folds):
        start = initial_train + fold * test_size
        end = initial_train + (fold + 1) * test_size if fold < args.folds - 1 else n
        train = df.iloc[:start]
        test = df.iloc[start:end]

        y, p = fit_predict(train, test)
        actual_close = test["next_close"].to_numpy(float)
        formula_close = test["formula_close"].to_numpy(float)

        mask = np.isfinite(y) & np.isfinite(p) & np.isfinite(actual_close) & np.isfinite(formula_close)
        y = y[mask]
        p = p[mask]
        actual_close = actual_close[mask]
        formula_close = formula_close[mask]

        acc = sign_accuracy(y, p)
        mae, mape, within = proximity(actual_close, formula_close)
        print(
            f"{fold + 1:>4} {len(train):>5} {len(test):>4} {acc * 100:>6.2f}% "
            f"{mae:>9.4f} {mape * 100:>9.3f}% "
            f"{within[0.001] * 100:>8.2f}% {within[0.0025] * 100:>8.2f}% "
            f"{within[0.005] * 100:>8.2f}% {within[0.01] * 100:>8.2f}%"
        )

        all_y.append(y)
        all_p.append(p)
        all_formula.append(formula_close)
        all_actual_close.append(actual_close)

    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    formula_close = np.concatenate(all_formula)
    actual_close = np.concatenate(all_actual_close)

    acc = sign_accuracy(y, p)
    mae, mape, within = proximity(actual_close, formula_close)
    bias = float(np.mean(formula_close - actual_close))

    print("-" * 95)
    print(
        f"ALL  {len(y):>5}       {acc * 100:>6.2f}% {mae:>9.4f} {mape * 100:>9.3f}% "
        f"{within[0.001] * 100:>8.2f}% {within[0.0025] * 100:>8.2f}% "
        f"{within[0.005] * 100:>8.2f}% {within[0.01] * 100:>8.2f}%"
    )
    print(f"formula_bias_mean={bias:+.6f}")


if __name__ == "__main__":
    main()
