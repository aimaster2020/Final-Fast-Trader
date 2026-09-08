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
    df["J_next"] = df["J"].shift(-1)
    df["U"] = df["J_next"] - df["J"]
    return df.dropna().reset_index(drop=True)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    features = ["J", "K", "L"]
    x_train = np.column_stack([np.ones(len(train)), train[features].to_numpy(float)])
    y_train = train["U"].to_numpy(float)
    coef = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
    x_test = np.column_stack([np.ones(len(test)), test[features].to_numpy(float)])
    return x_test @ coef


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward OOS estimation of next candle body magnitude")
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
    all_u, all_u_hat = [], []
    all_abs_j, all_abs_j_hat = [], []

    print("=" * 64)
    print("BODY MAGNITUDE WALK-FORWARD")
    print("=" * 64)
    print(f"candles={n} train0={initial_train} folds={args.folds} test/fold={test_size}")
    print("target=next body J_next and change U=J_next-J_current leakage=NO")
    print("fold  train  test  U_MAE     Jnext_MAE  |J|_MAE   corr")

    for i in range(args.folds):
        start = initial_train + i * test_size
        end = initial_train + (i + 1) * test_size if i < args.folds - 1 else n
        train = df.iloc[:start]
        test = df.iloc[start:end]

        u_hat = fit_predict(train, test)
        u = test["U"].to_numpy(float)
        j_next = test["J_next"].to_numpy(float)
        j_next_hat = test["J"].to_numpy(float) + u_hat
        abs_j = np.abs(j_next)
        abs_j_hat = np.abs(j_next_hat)

        u_mae = float(np.mean(np.abs(u - u_hat)))
        j_mae = float(np.mean(np.abs(j_next - j_next_hat)))
        abs_mae = float(np.mean(np.abs(abs_j - abs_j_hat)))
        corr = safe_corr(abs_j, abs_j_hat)

        all_u.append(u)
        all_u_hat.append(u_hat)
        all_abs_j.append(abs_j)
        all_abs_j_hat.append(abs_j_hat)
        print(f"{i+1:>4}  {len(train):>5}  {len(test):>4}  {u_mae:>8.3f}  {j_mae:>10.3f}  {abs_mae:>8.3f}  {corr:>6.4f}")

    u = np.concatenate(all_u)
    u_hat = np.concatenate(all_u_hat)
    abs_j = np.concatenate(all_abs_j)
    abs_j_hat = np.concatenate(all_abs_j_hat)

    u_mae = float(np.mean(np.abs(u - u_hat)))
    j_mae = float(np.mean(np.abs(np.abs(abs_j) - np.abs(abs_j_hat))))
    corr = safe_corr(abs_j, abs_j_hat)

    baseline = np.full(len(abs_j), np.mean(df.iloc[:initial_train]["J_next"].abs()))
    base_mae = float(np.mean(np.abs(abs_j - baseline)))

    print("-" * 64)
    print(f"ALL   {len(abs_j):>5}  U_MAE={u_mae:.3f}  |J_next|_MAE={j_mae:.3f}  corr={corr:.4f}")
    print(f"BASE  {len(abs_j):>5}  |J_next|_MAE={base_mae:.3f}")
    print("model=J,K,L -> U_hat -> J_next_hat -> abs(J_next_hat)")


if __name__ == "__main__":
    main()
