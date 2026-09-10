from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
FEATURE_SETS = {
    "ABS_BODY_PCT": ["ABS_BODY_PCT"],
    "RANGE_PCT": ["RANGE_PCT"],
    "ABS_BODY_RANGE_PCT": ["ABS_BODY_PCT", "RANGE_PCT"],
    "WICKS_PCT": ["UPPER_PCT", "LOWER_PCT"],
    "FULL_GEOMETRY_PCT": ["ABS_BODY_PCT", "UPPER_PCT", "LOWER_PCT"],
}


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fit(train: pd.DataFrame, features: list[str]) -> np.ndarray:
    data = train[features + ["TARGET"]].apply(pd.to_numeric, errors="coerce").dropna()
    X = np.column_stack([np.ones(len(data)), data[features].to_numpy(float)])
    y = data["TARGET"].to_numpy(float)
    return np.linalg.lstsq(X, y, rcond=None)[0]


def predict(row: pd.Series, beta: np.ndarray, features: list[str]) -> float:
    x = np.array([1.0] + [float(row[f]) for f in features], dtype=float)
    return max(0.0, float(x @ beta))


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = actual - pred
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "corr": corr(actual, pred),
    }


def load_raw(path: Path, month: str, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"{path}: missing symbol")
    df = df[df["symbol"] == symbol].copy()

    required = ["J", "K", "L", "J_next", "open", "next_open"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if month:
            df = df.loc[ts.dt.strftime("%Y-%m") == month].copy()
        df = df.assign(_timestamp=ts).sort_values("_timestamp")

    base = df["open"].abs().replace(0, np.nan)
    df["ABS_BODY_PCT"] = df["J"].abs() / base
    df["RANGE_PCT"] = (df["K"] + df["L"]) / base
    df["UPPER_PCT"] = df["K"] / base
    df["LOWER_PCT"] = df["L"] / base
    df["TARGET"] = df["J_next"].abs() / df["next_open"].abs().replace(0, np.nan)

    return df.dropna(subset=["ABS_BODY_PCT", "RANGE_PCT", "UPPER_PCT", "LOWER_PCT", "TARGET"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanding walk-forward OOS prediction of next candle absolute body percentage")
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument("--input", default=None, help="Raw body magnitude CSV")
    parser.add_argument("--min-train", type=int, default=200)
    parser.add_argument("--output", default=None, help="TXT output path")
    args = parser.parse_args()

    input_path = Path(args.input or f"reports/body_magnitude_raw_{args.month}.csv")
    output_path = Path(args.output or f"reports/body_magnitude_pct_walkforward_{args.month}.txt")
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    lines = [
        "BODY MAGNITUDE PCT WALK-FORWARD",
        f"month={args.month}",
        f"input={input_path}",
        f"min_train={args.min_train}",
        "training=expanding_history_only",
        "target=abs(J_next)/next_open",
        "",
        "model: MAE RMSE corr",
    ]

    models = ["BASE_MEAN", "BASE_MEDIAN", "ABS_BODY_COPY", *FEATURE_SETS.keys()]
    pooled_actual = {name: [] for name in models}
    pooled_pred = {name: [] for name in models}

    for symbol in SYMBOLS:
        df = load_raw(input_path, args.month, symbol)
        if len(df) <= args.min_train:
            raise ValueError(f"{symbol}: need > {args.min_train} rows; got {len(df)}")

        actual = {name: [] for name in models}
        predictions = {name: [] for name in models}

        for i in range(args.min_train, len(df)):
            train = df.iloc[:i]
            row = df.iloc[i]
            y = float(row["TARGET"])

            actual["BASE_MEAN"].append(y)
            actual["BASE_MEDIAN"].append(y)
            actual["ABS_BODY_COPY"].append(y)
            predictions["BASE_MEAN"].append(float(train["TARGET"].mean()))
            predictions["BASE_MEDIAN"].append(float(train["TARGET"].median()))
            predictions["ABS_BODY_COPY"].append(float(row["ABS_BODY_PCT"]))

            for name, features in FEATURE_SETS.items():
                beta = fit(train, features)
                actual[name].append(y)
                predictions[name].append(predict(row, beta, features))

        lines.append(f"[{symbol}] n={len(df)} oos={len(df)-args.min_train}")
        for name in models:
            a = np.asarray(actual[name], dtype=float)
            p = np.asarray(predictions[name], dtype=float)
            m = metrics(a, p)
            lines.append(f"{name}: MAE={m['mae']:.8f} RMSE={m['rmse']:.8f} corr={m['corr']:.5f}")
            pooled_actual[name].extend(a.tolist())
            pooled_pred[name].extend(p.tolist())
        lines.append("")

    lines.append("[ALL_POOLED]")
    for name in models:
        a = np.asarray(pooled_actual[name], dtype=float)
        p = np.asarray(pooled_pred[name], dtype=float)
        m = metrics(a, p)
        lines.append(f"{name}: n={len(a)} MAE={m['mae']:.8f} RMSE={m['rmse']:.8f} corr={m['corr']:.5f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("output=" + str(output_path))
    print("-----")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
