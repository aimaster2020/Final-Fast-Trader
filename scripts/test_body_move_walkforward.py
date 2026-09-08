from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close"]


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
    rng = (df["high"] - df["low"]).clip(lower=1e-12)
    df["M"] = rng
    df["N"] = df["J"] / rng
    df["O"] = (df["L"] - df["K"]) / rng
    df["U"] = df["J"].shift(-1) - df["J"]
    return df.dropna().reset_index(drop=True)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x_train = np.column_stack([np.ones(len(train)), train[features].to_numpy(float)])
    y_train = train["U"].to_numpy(float)
    coef = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
    x_test = np.column_stack([np.ones(len(test)), test[features].to_numpy(float)])
    return test["U"].to_numpy(float), x_test @ coef


def metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    mask = (np.sign(y) != 0) & (np.sign(p) != 0)
    acc = float(np.mean(np.sign(y[mask]) == np.sign(p[mask]))) if mask.any() else 0.0
    mae = float(np.mean(np.abs(y - p)))
    c = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0 else 0.0
    return acc, mae, c


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward OOS validation for U = J_next - J_current")
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.50)
    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError("--folds must be >= 2")
    if not 0.30 <= args.train_ratio <= 0.80:
        raise ValueError("--train-ratio must be between 0.30 and 0.80")

    df = load_data(args.input, args.limit)
    n = len(df)
    initial_train = int(n * args.train_ratio)
    remaining = n - initial_train
    if remaining < args.folds * 10:
        raise ValueError(f"Not enough OOS candles for {args.folds} folds: {n}")

    test_size = remaining // args.folds
    features = ["J", "K", "L"]
    all_y, all_p = [], []

    print("=" * 56)
    print("BODY MOVE WALK-FORWARD")
    print("=" * 56)
    print(f"candles={n} train0={initial_train} folds={args.folds} test/fold={test_size}")
    print("target=U=J_next-J_current leakage=NO")
    print("fold  train  test  dir%    MAE        corr")

    for i in range(args.folds):
        start = initial_train + i * test_size
        end = initial_train + (i + 1) * test_size if i < args.folds - 1 else n
        train = df.iloc[:start]
        test = df.iloc[start:end]
        y, p = fit_predict(train, test, features)
        acc, mae, c = metrics(y, p)
        all_y.append(y)
        all_p.append(p)
        print(f"{i+1:>4}  {len(train):>5}  {len(test):>4}  {acc*100:>6.2f}  {mae:>10.4f}  {c:>6.4f}")

    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    acc, mae, c = metrics(y, p)
    baseline_p = np.full(len(y), df.iloc[:initial_train]["U"].mean())
    bacc, bmae, bc = metrics(y, baseline_p)

    print("-" * 56)
    print(f"ALL   {len(y):>5}       {acc*100:>6.2f}  {mae:>10.4f}  {c:>6.4f}")
    print(f"BASE  {len(y):>5}       {bacc*100:>6.2f}  {bmae:>10.4f}  {bc:>6.4f}")
    print("model=J,K,L")


if __name__ == "__main__":
    main()
