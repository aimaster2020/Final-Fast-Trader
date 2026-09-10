from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
FEATURE_SETS = {
    "ABSJ": ["ABSJ"],
    "RANGE": ["RANGE"],
    "ABSJ_RANGE": ["ABSJ", "RANGE"],
    "ABSJ_K_L": ["ABSJ", "K", "L"],
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
    x = np.array([1.0] + [float(row[f]) for f in features])
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
    required = ["J", "K", "L", "J_next"]
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
    df["ABSJ"] = df["J"].abs()
    df["RANGE"] = df["K"] + df["L"]
    df["TARGET"] = df["J_next"].abs()
    return df.dropna(subset=["J", "K", "L", "J_next", "TARGET"]).reset_index(drop=True)


def run_symbol(df: pd.DataFrame, min_train: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if len(df) <= min_train:
        raise ValueError(f"Need > {min_train} rows; got {len(df)}")
    models = ["BASE_MEAN", "BASE_MEDIAN", "ABSJ_COPY", *FEATURE_SETS.keys()]
    actual = {name: [] for name in models}
    pred = {name: [] for name in models}

    for i in range(min_train, len(df)):
        train = df.iloc[:i]
        row = df.iloc[i]
        y = float(row["TARGET"])
        actual["BASE_MEAN"].append(y)
        actual["BASE_MEDIAN"].append(y)
        actual["ABSJ_COPY"].append(y)
        pred["BASE_MEAN"].append(float(train["TARGET"].mean()))
        pred["BASE_MEDIAN"].append(float(train["TARGET"].median()))
        pred["ABSJ_COPY"].append(float(row["ABSJ"]))
        for name, features in FEATURE_SETS.items():
            beta = fit(train, features)
            actual[name].append(y)
            pred[name].append(predict(row, beta, features))

    return {
        name: (np.asarray(actual[name], dtype=float), np.asarray(pred[name], dtype=float))
        for name in models
    }


def fmt(x: float, digits: int = 5) -> str:
    return "NA" if not np.isfinite(x) else f"{x:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanding walk-forward OOS prediction of absolute next candle body")
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument("--input", default=None, help="Raw body magnitude CSV")
    parser.add_argument("--min-train", type=int, default=200)
    parser.add_argument("--output", default=None, help="TXT output path")
    args = parser.parse_args()

    input_path = Path(args.input or f"reports/body_magnitude_raw_{args.month}.csv")
    output_path = Path(args.output or f"reports/body_magnitude_abs_walkforward_{args.month}.txt")
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    models = ["BASE_MEAN", "BASE_MEDIAN", "ABSJ_COPY", *FEATURE_SETS.keys()]
    pooled_actual = {name: [] for name in models}
    pooled_pred = {name: [] for name in models}

    lines = [
        "BODY MAGNITUDE ABS WALK-FORWARD",
        f"month={args.month}",
        f"input={input_path}",
        f"min_train={args.min_train}",
        "training=expanding_history_only",
        "target=abs(J_next)",
        "",
        "model: MAE RMSE corr",
    ]

    for symbol in SYMBOLS:
        df = load_raw(input_path, args.month, symbol)
        results = run_symbol(df, args.min_train)
        lines.append(f"[{symbol}] n={len(df)} oos={len(df)-args.min_train}")
        for name, (actual, pred) in results.items():
            m = metrics(actual, pred)
            lines.append(f"{name}: MAE={fmt(m['mae'])} RMSE={fmt(m['rmse'])} corr={fmt(m['corr'])}")
            pooled_actual[name].extend(actual.tolist())
            pooled_pred[name].extend(pred.tolist())
        lines.append("")

    lines.append("[ALL_POOLED]")
    for name in models:
        actual = np.asarray(pooled_actual[name], dtype=float)
        pred = np.asarray(pooled_pred[name], dtype=float)
        m = metrics(actual, pred)
        lines.append(f"{name}: n={len(actual)} MAE={fmt(m['mae'])} RMSE={fmt(m['rmse'])} corr={fmt(m['corr'])}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("output=" + str(output_path))
    print("-----")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
