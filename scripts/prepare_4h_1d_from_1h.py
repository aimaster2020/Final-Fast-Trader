from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


def convert_file(src: Path, dst: Path, hours: int) -> int:
    df = pd.read_csv(src)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{src}: missing columns: {missing}")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED).copy()
    df = df.sort_values(["symbol", "timestamp"]).drop_duplicates("timestamp", keep="first")
    if df.empty:
        raise ValueError(f"{src}: no usable rows")

    # The source is UTC 1h candles. Resample on UTC boundaries (4h or 1d).
    dt = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index(dt)

    rule = f"{hours}h"
    out = (
        df.resample(rule, label="left", closed="left")
        .agg(
            symbol=("symbol", "first"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            source_rows=("open", "size"),
        )
        .dropna(subset=["open", "high", "low", "close"])
    )

    # Keep only complete candles. A 4h candle must contain 4 source hours;
    # a 1d candle must contain 24 source hours.
    out = out[out["source_rows"] == hours].copy()
    out["timestamp"] = (out.index.view("int64") // 1_000_000_000).astype("int64")
    out["month"] = out.index.strftime("%Y-%m")
    out = out.reset_index(drop=True)
    out = out[["symbol", "month", "timestamp", "open", "high", "low", "close", "volume"]]

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return len(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build complete 4h/1d candles from existing 1h reports")
    parser.add_argument("--input-dir", default="reports/1h")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--hours", nargs="+", type=int, choices=[4, 24], default=[4, 24])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*_1h.csv"))
    if not files:
        raise FileNotFoundError(f"No *_1h.csv files found in {input_dir}")

    for hours in args.hours:
        tf = "4h" if hours == 4 else "1d"
        total = 0
        print(f"[{tf}]")
        for src in files:
            symbol = src.name.removesuffix("_1h.csv")
            dst = Path(args.output_dir) / tf / f"{symbol}_{tf}.csv"
            n = convert_file(src, dst, hours)
            total += n
            print(f"  {symbol}: {n} complete candles -> {dst}")
        print(f"  TOTAL: {total} candles")


if __name__ == "__main__":
    main()
