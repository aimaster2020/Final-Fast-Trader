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
    df["move_mag"] = (df["high"] - df["open"]).abs()
    df["literal_formula_close"] = df["high"] - df["open"] + df["close"]
    return df.dropna(subset=["U", "next_close", "move_mag"]).reset_index(drop=True)


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
    ap = argparse.ArgumentParser(
        description="Old direction checkpoint + H-O magnitude applied with predicted direction"
    )
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
    all_combined: list[np.ndarray] = []
    all_literal: list[np.ndarray] = []
    all_actual: list[np.ndarray] = []

    print("=" * 118)
    print("DIRECTION CHECKPOINT + H-O MAGNITUDE (TRUE COMBINED TEST)")
    print("=" * 118)
    print(f"candles={n} train0={initial_train} folds={args.folds} test/fold={test_size}")
    print("direction=WalkForward U=J_next-J_current features=J,K,L")
    print("magnitude=abs(H-O)")
    print("combined_predicted_next_close = C + direction_sign * abs(H-O)")
    print("literal_formula = H-O+C (shown for comparison)")
    print("proximity=absolute percentage error versus ACTUAL next close")
    print("fold train test dir% combined_MAE combined_MAPE combined<=0.25% combined<=0.50% literal_MAE literal_MAPE")

    for fold in range(args.folds):
        start = initial_train + fold * test_size
        end = initial_train + (fold + 1) * test_size if fold < args.folds - 1 else n
        train = df.iloc[:start]
        test = df.iloc[start:end]

        y, p = fit_predict(train, test)
        direction = np.sign(p)
        actual = test["next_close"].to_numpy(float)
        magnitude = test["move_mag"].to_numpy(float)
        current_close = test["close"].to_numpy(float)
        literal = test["literal_formula_close"].to_numpy(float)
        combined = current_close + direction * magnitude

        mask = (
            np.isfinite(y)
            & np.isfinite(p)
            & np.isfinite(actual)
            & np.isfinite(magnitude)
            & np.isfinite(current_close)
            & np.isfinite(literal)
            & (direction != 0)
        )
        y = y[mask]
        p = p[mask]
        actual = actual[mask]
        combined = combined[mask]
        literal = literal[mask]

        acc = sign_accuracy(y, p)
        c_mae, c_mape, c_within = proximity(actual, combined)
        l_mae, l_mape, _ = proximity(actual, literal)
        print(
            f"{fold + 1:>4} {len(train):>5} {len(test):>4} {acc * 100:>6.2f}% "
            f"{c_mae:>12.4f} {c_mape * 100:>13.3f}% "
            f"{c_within[0.0025] * 100:>16.2f}% {c_within[0.005] * 100:>16.2f}% "
            f"{l_mae:>10.4f} {l_mape * 100:>10.3f}%"
        )

        all_y.append(y)
        all_p.append(p)
        all_combined.append(combined)
        all_literal.append(literal)
        all_actual.append(actual)

    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    combined = np.concatenate(all_combined)
    literal = np.concatenate(all_literal)
    actual = np.concatenate(all_actual)

    acc = sign_accuracy(y, p)
    c_mae, c_mape, c_within = proximity(actual, combined)
    l_mae, l_mape, l_within = proximity(actual, literal)
    c_bias = float(np.mean(combined - actual))
    l_bias = float(np.mean(literal - actual))

    print("-" * 118)
    print(
        f"ALL  {len(y):>5}       {acc * 100:>6.2f}% "
        f"{c_mae:>12.4f} {c_mape * 100:>13.3f}% "
        f"{c_within[0.0025] * 100:>16.2f}% {c_within[0.005] * 100:>16.2f}% "
        f"{l_mae:>10.4f} {l_mape * 100:>10.3f}%"
    )
    print(f"combined_bias_mean={c_bias:+.6f}")
    print(f"literal_formula_bias_mean={l_bias:+.6f}")
    print(f"combined_within_0.10%={c_within[0.001] * 100:.2f}%")
    print(f"combined_within_1.00%={c_within[0.01] * 100:.2f}%")
    print(f"literal_within_0.10%={l_within[0.001] * 100:.2f}%")
    print(f"literal_within_0.25%={l_within[0.0025] * 100:.2f}%")
    print(f"literal_within_0.50%={l_within[0.005] * 100:.2f}%")
    print(f"literal_within_1.00%={l_within[0.01] * 100:.2f}%")


if __name__ == "__main__":
    main()
