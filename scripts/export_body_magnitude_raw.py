from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


def load_data(symbol: str) -> pd.DataFrame:
    path = Path(f"reports/1h/{symbol}_1h.csv")
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], unit="s", utc=True, errors="coerce"
    )
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    # Current candle geometry.
    df["J"] = df["close"] - df["open"]
    df["K"] = df["high"] - df["close"]
    df["L"] = df["close"] - df["low"]

    # Next-candle body and the raw target used by the magnitude research.
    df["next_open"] = df["open"].shift(-1)
    df["next_close"] = df["close"].shift(-1)
    df["J_next"] = df["J"].shift(-1)
    df["U"] = df["J_next"] - df["J"]

    # Useful normalized targets/features for formula discovery.
    df["body_pct"] = df["J"] / df["open"]
    df["upper_wick_pct"] = df["K"] / df["open"]
    df["lower_wick_pct"] = df["L"] / df["open"]
    df["next_body_pct"] = df["J_next"] / df["next_open"]
    df["next_move_pct"] = (df["next_close"] - df["open"]) / df["open"]
    df["U_pct"] = df["U"] / df["open"]
    df["abs_U_pct"] = df["U_pct"].abs()
    df["direction"] = np.where(
        df["J_next"] > 0,
        1,
        np.where(df["J_next"] < 0, -1, 0),
    )

    # Keep only rows where the complete next-candle target is known.
    return df.dropna(subset=["next_open", "next_close", "J_next", "U"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path; defaults to reports/body_magnitude_raw_<month>.csv",
    )
    args = parser.parse_args()

    rows = []
    for symbol in SYMBOLS:
        df = load_data(symbol)
        df = df[df["timestamp"].dt.strftime("%Y-%m") == args.month].copy()
        if df.empty:
            continue
        if "symbol" in df.columns:
            df["symbol"] = symbol
        else:
            df.insert(0, "symbol", symbol)
        rows.append(df)

    if not rows:
        raise ValueError(f"No data found for month {args.month}")

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    output = Path(args.output or f"reports/body_magnitude_raw_{args.month}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, float_format="%.10f")

    print(f"month={args.month} rows={len(out)} symbols={','.join(SYMBOLS)}")
    print(f"output={output}")
    print("columns=" + ",".join(out.columns))


if __name__ == "__main__":
    main()
