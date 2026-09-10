from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_raw(path: Path, month: str, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"{path}: missing symbol")
    df = df[df["symbol"] == symbol].copy()

    required = ["J", "K", "L", "J_next", "next_open"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.assign(_timestamp=ts).sort_values("_timestamp")
        if month:
            df = df.loc[ts.dt.strftime("%Y-%m") == month].copy()

    # Current candle geometry and normalized magnitude target.
    df["abs_body_pct"] = df["J"].abs() / df["open"]
    df["range_pct"] = (df["K"] + df["L"]) / df["open"]
    df["wick_pct"] = (df["K"] + df["L"]) / df["open"]
    df["target"] = df["J_next"].abs() / df["next_open"]

    # Historical-only features: shift before rolling so the target candle is never included.
    for window in [3, 6, 12, 24]:
        df[f"hist_body_{window}"] = df["abs_body_pct"].shift(1).rolling(window).mean()
        df[f"hist_range_{window}"] = df["range_pct"].shift(1).rolling(window).mean()

    return df.dropna().reset_index(drop=True)


def fit_predict(train: pd.DataFrame, row: pd.Series, features: list[str]) -> float:
    data = train[features + ["target"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < len(features) + 10:
        return float(data["target"].median())
    X = np.column_stack([np.ones(len(data)), data[features].to_numpy(float)])
    y = data["target"].to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    x = np.array([1.0] + [float(row[f]) for f in features])
    return max(0.0, float(x @ beta))


def metric(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    err = actual - pred
    return (
        float(np.mean(np.abs(err))),
        float(np.sqrt(np.mean(err**2))),
        corr(actual, pred),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward historical-memory test for next-candle body magnitude")
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument("--input", default=None, help="Raw body magnitude CSV")
    parser.add_argument("--min-train", type=int, default=200)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = Path(args.input or f"reports/body_magnitude_raw_{args.month}.csv")
    output_path = Path(args.output or f"reports/body_magnitude_history_walkforward_{args.month}.txt")
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    lines = [
        "BODY MAGNITUDE HISTORY WALK-FORWARD",
        f"month={args.month}",
        f"input={input_path}",
        f"min_train={args.min_train}",
        "training=expanding_history_only",
        "target=abs(J_next)/next_open",
        "features=historical past-only rolling means",
        "",
    ]

    models = {
        "BASE_MEDIAN": None,
        "HIST_BODY_3": ["hist_body_3"],
        "HIST_BODY_6": ["hist_body_6"],
        "HIST_BODY_12": ["hist_body_12"],
        "HIST_BODY_24": ["hist_body_24"],
        "HIST_RANGE_3": ["hist_range_3"],
        "HIST_RANGE_6": ["hist_range_6"],
        "HIST_RANGE_12": ["hist_range_12"],
        "HIST_RANGE_24": ["hist_range_24"],
        "CURRENT_RANGE_HIST6": ["range_pct", "hist_range_6"],
        "CURRENT_RANGE_HIST12": ["range_pct", "hist_range_12"],
        "BODY_AND_RANGE_HIST12": ["abs_body_pct", "range_pct", "hist_body_12", "hist_range_12"],
    }

    pooled_actual = {name: [] for name in models}
    pooled_pred = {name: [] for name in models}

    for symbol in SYMBOLS:
        df = load_raw(input_path, args.month, symbol)
        if len(df) <= args.min_train:
            raise ValueError(f"{symbol}: need > {args.min_train} rows; got {len(df)}")

        actual = {name: [] for name in models}
        preds = {name: [] for name in models}

        for i in range(args.min_train, len(df)):
            train = df.iloc[:i]
            row = df.iloc[i]
            y = float(row["target"])

            for name, features in models.items():
                if name == "BASE_MEDIAN":
                    pred = float(train["target"].median())
                else:
                    pred = fit_predict(train, row, features)
                actual[name].append(y)
                preds[name].append(pred)

        lines.append(f"[{symbol}] n={len(df)} oos={len(df)-args.min_train}")
        for name in models:
            a = np.asarray(actual[name], dtype=float)
            p = np.asarray(preds[name], dtype=float)
            mae, rmse, c = metric(a, p)
            lines.append(f"{name}: MAE={mae:.8f} RMSE={rmse:.8f} corr={c:.5f}")
            pooled_actual[name].extend(a.tolist())
            pooled_pred[name].extend(p.tolist())
        lines.append("")

    lines.append("[ALL_POOLED]")
    for name in models:
        a = np.asarray(pooled_actual[name], dtype=float)
        p = np.asarray(pooled_pred[name], dtype=float)
        mae, rmse, c = metric(a, p)
        lines.append(f"{name}: n={len(a)} MAE={mae:.8f} RMSE={rmse:.8f} corr={c:.5f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("output=" + str(output_path))
    print("-----")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
