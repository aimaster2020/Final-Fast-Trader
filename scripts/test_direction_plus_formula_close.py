from __future__ import annotations

import argparse
from pathlib import Path

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
        df = df.tail(limit).copy()

    df["J"] = df["close"] - df["open"]
    df["K"] = df["high"] - df["close"]
    df["L"] = df["close"] - df["low"]
    df["U"] = df["J"].shift(-1) - df["J"]
    df["formula_close"] = df["high"] - df["open"] + df["close"]
    return df.dropna().reset_index(drop=True)


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


def direction_and_formula_projection(test: pd.DataFrame, p_u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual_next_body_change = test["U"].to_numpy(float)
    direction_prediction = np.sign(p_u)
    formula_close = test["formula_close"].to_numpy(float)
    actual_next_close = test["close"].shift(-1).to_numpy(float)
    mask = np.isfinite(formula_close) & np.isfinite(actual_next_close)
    return actual_next_body_change[mask], np.column_stack([direction_prediction[mask], formula_close[mask], actual_next_close[mask]])


def run_symbol(df: pd.DataFrame, folds: int, train_ratio: float) -> None:
    n = len(df)
    initial_train = int(n * train_ratio)
    remaining = n - initial_train
    if remaining < folds * 10:
        raise ValueError(f"Not enough OOS candles: n={n} folds={folds}")
    test_size = remaining // folds

    all_y = []
    all_p = []
    all_formula = []
    all_actual_close = []

    print(f"candles={n} train0={initial_train} folds={folds} test/fold={test_size}")
    print("formula=HIGH-OPEN+CLOSE -> predicted_next_close")
    print("direction_model=U=J_next-J_current with features J,K,L")
    print("fold  train  test   dir%   MAE_close   MAPE_close  <=0.10%  <=0.25%  <=0.50%  <=1.00%")

    for i in range(folds):
        start = initial_train + i * test_size
        end = initial_train + (i + 1) * test_size if i < folds - 1 else n
        train = df.iloc[:start]
        test = df.iloc[start:end].copy()
        y, p = fit_predict(train, test)

        actual_close = test["close"].shift(-1).to_numpy(float)
        formula_close = test["formula_close"].to_numpy(float)
        mask = np.isfinite(actual_close) & np.isfinite(formula_close)
        y = y[mask]
        p = p[mask]
        formula_close = formula_close[mask]
        actual_close = actual_close[mask]

        acc = sign_accuracy(y, p)
        mae, mape, within = proximity(actual_close, formula_close)
        print(
            f"{i+1:>4} {len(train):>6} {len(test):>5} {acc*100:>7.2f} "
            f"{mae:>10.4f} {mape*100:>10.3f}% "
            f"{within[0.001]*100:>8.2f}% {within[0.0025]*100:>8.2f}% "
            f"{within[0.005]*100:>8.2f}% {within[0.01]*100:>8.2f}%"
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

    print("-" * 95)
    print(
        f"ALL  {len(y):>6}         {acc*100:>7.2f} {mae:>10.4f} {mape*100:>10.3f}% "
        f"{within[0.001]*100:>8.2f}% {within[0.0025]*100:>8.2f}% "
        f"{within[0.005]*100:>8.2f}% {within[0.01]*100:>8.2f}%"
    )

    residual = formula_close - actual_close
    bias = float(np.mean(residual))
    direction_formula = np.sign(formula_close - df.iloc[: len(formula_close)]["close"].to_numpy(float))
    print(f"formula_bias_mean={bias:+.6f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Checkpoint direction + H-O+C next-close proximity")
    ap.add_argument("--input", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.50)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = load_data(args.input, args.limit)
    print("=" * 95)
    print("DIRECTION CHECKPOINT + CANDLE FORMULA CLOSE PROJECTION")
    print("=" * 95)
    run_symbol(df, args.folds, args.train_ratio)


if __name__ == "__main__":
    main()
