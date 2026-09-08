from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


REQUIRED = ["open", "high", "low", "close"]


def load_data(path: str, limit: int = 0) -> pd.DataFrame:
    # The project CSVs can contain many extra/formula columns. We only need OHLC.
    df = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in REQUIRED)
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED).copy()

    if limit > 0:
        df = df.tail(limit).copy()

    # Current candle geometry.
    df["J"] = df["close"] - df["open"]
    df["K"] = df["high"] - df["close"]
    df["L"] = df["close"] - df["low"]
    rng = (df["high"] - df["low"]).clip(lower=1e-12)
    df["M"] = rng
    df["N"] = df["J"] / rng
    df["O"] = (df["L"] - df["K"]) / rng

    # Target: U = J(next) - J(current). The next candle is used only as target.
    df["U"] = df["J"].shift(-1) - df["J"]
    return df.dropna().reset_index(drop=True)


def fit_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x1 = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(x1, y, rcond=None)[0]


def predict(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ coef


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def direction_accuracy(y: np.ndarray, p: np.ndarray) -> tuple[float, int]:
    ys, ps = np.sign(y), np.sign(p)
    mask = (ys != 0) & (ps != 0)
    if not mask.any():
        return 0.0, 0
    return float(np.mean(ys[mask] == ps[mask])), int(mask.sum())


def evaluate(name: str, features: list[str], train: pd.DataFrame, test: pd.DataFrame) -> dict:
    x_train = train[features].to_numpy(float)
    y_train = train["U"].to_numpy(float)
    x_test = test[features].to_numpy(float)
    y_test = test["U"].to_numpy(float)

    coef = fit_linear(x_train, y_train)
    pred = predict(x_test, coef)
    acc, n_dir = direction_accuracy(y_test, pred)
    mae = float(np.mean(np.abs(y_test - pred)))
    c = corr(y_test, pred)

    print(f"\n{name}")
    print(f"features       : {','.join(features)}")
    print(f"direction_acc  : {acc * 100:.2f}% ({n_dir})")
    print(f"MAE            : {mae:.8f}")
    print(f"correlation    : {c:.4f}")
    print("coef           : " + ", ".join(f"{v:.8f}" for v in coef))
    return {"name": name, "accuracy": acc, "mae": mae, "corr": c}


def main() -> None:
    parser = argparse.ArgumentParser(description="OOS test for next-candle body change U = J_next - J_current")
    parser.add_argument("--input", required=True, help="OHLC CSV path")
    parser.add_argument("--limit", type=int, default=0, help="Use last N rows; 0 = all")
    parser.add_argument("--test-ratio", type=float, default=0.30, help="Chronological OOS fraction")
    args = parser.parse_args()

    if not 0.05 <= args.test_ratio <= 0.50:
        raise ValueError("--test-ratio must be between 0.05 and 0.50")

    df = load_data(args.input, args.limit)
    if len(df) < 100:
        raise ValueError(f"Not enough usable candles: {len(df)}")

    split = int(len(df) * (1 - args.test_ratio))
    train, test = df.iloc[:split], df.iloc[split:]

    print("=" * 60)
    print("NEXT CANDLE BODY MOVE TEST")
    print("=" * 60)
    print(f"candles        : {len(df)}")
    print(f"train          : {len(train)}")
    print(f"test           : {len(test)}")
    print(f"target         : U = J_next - J_current")
    print("leakage        : NO")

    baseline_y = test["U"].to_numpy(float)
    baseline_pred = np.full(len(test), train["U"].mean())
    baseline_mae = float(np.mean(np.abs(baseline_y - baseline_pred)))
    baseline_acc, baseline_n = direction_accuracy(baseline_y, baseline_pred)
    print("\nBASELINE (train mean)")
    print(f"direction_acc  : {baseline_acc * 100:.2f}% ({baseline_n})")
    print(f"MAE            : {baseline_mae:.8f}")

    a = evaluate("MODEL A: J,K,L", ["J", "K", "L"], train, test)
    b = evaluate("MODEL B: J,K,L,M,N,O", ["J", "K", "L", "M", "N", "O"], train, test)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"A direction    : {a['accuracy'] * 100:.2f}%")
    print(f"B direction    : {b['accuracy'] * 100:.2f}%")
    print(f"A MAE          : {a['mae']:.8f}")
    print(f"B MAE          : {b['mae']:.8f}")
    print(f"A correlation  : {a['corr']:.4f}")
    print(f"B correlation  : {b['corr']:.4f}")


if __name__ == "__main__":
    main()
