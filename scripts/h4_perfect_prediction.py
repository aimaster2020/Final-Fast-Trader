from pathlib import Path
import argparse
import pandas as pd


PATTERNS = {
    "Bullish Pin Bar": "LONG",
    "Hammer": "LONG",
    "Bearish Outside": "SHORT",
}


def is_hammer(row):
    o = row["open"]
    h = row["high"]
    l = row["low"]
    c = row["close"]

    rng = h - l
    if rng <= 0:
        return False

    body = abs(c - o)
    lower = min(o, c) - l
    upper = h - max(o, c)
    body_ratio = body / rng

    return (
        body_ratio <= 0.35
        and lower >= 2.0 * body
        and upper <= 0.35 * rng
    )


def is_bullish_pin(row):
    o = row["open"]
    h = row["high"]
    l = row["low"]
    c = row["close"]

    rng = h - l
    if rng <= 0:
        return False

    body = abs(c - o)
    lower = min(o, c) - l
    upper = h - max(o, c)

    return (
        lower / rng >= 0.60
        and upper / rng <= 0.20
        and c > o
    )


def is_bearish_outside(df, i):
    if i <= 0:
        return False

    row = df.iloc[i]
    prev = df.iloc[i - 1]

    return (
        row["high"] > prev["high"]
        and row["low"] < prev["low"]
        and row["close"] < row["open"]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Measure raw H4 directional market move assuming 100% prediction accuracy."
    )
    parser.add_argument(
        "--prepared-file",
        default=r".\reports\prepared_price_action_1h.csv",
    )
    parser.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT",
    )
    parser.add_argument("--horizon", type=int, default=4)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    df = pd.read_csv(args.prepared_file)
    df.columns = [c.strip().lower() for c in df.columns]

    symbol_col = next(
        (c for c in ("symbol", "ticker", "asset") if c in df.columns),
        None,
    )
    if symbol_col is None:
        raise ValueError(f"Symbol column not found. Columns: {list(df.columns)}")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")

    print()
    print("H4 PERFECT PREDICTION - RAW")
    print("-" * 78)
    print(
        f"{'SYMBOL':<10}"
        f"{'PATTERN':<22}"
        f"{'TRADES':>8}"
        f"{'AVG%':>10}"
        f"{'MEDIAN%':>10}"
        f"{'WIN%':>9}"
        f"{'TOTAL%':>10}"
    )
    print("-" * 78)

    for symbol in symbols:
        sdf = df[df[symbol_col].astype(str).str.upper() == symbol.upper()].copy()
        if sdf.empty:
            print(f"{symbol:<10} NO DATA")
            continue

        time_col = next(
            (c for c in ("timestamp", "time", "datetime", "date") if c in sdf.columns),
            None,
        )
        if time_col:
            sdf = sdf.sort_values(time_col)
        sdf = sdf.reset_index(drop=True)

        for pattern_name, direction in PATTERNS.items():
            returns = []

            for i in range(len(sdf) - args.horizon):
                row = sdf.iloc[i]

                if pattern_name == "Hammer":
                    matched = is_hammer(row)
                elif pattern_name == "Bullish Pin Bar":
                    matched = is_bullish_pin(row)
                else:
                    matched = is_bearish_outside(sdf, i)

                if not matched:
                    continue

                entry = float(row["close"])
                future = float(sdf.iloc[i + args.horizon]["close"])
                if entry <= 0 or future <= 0:
                    continue

                move = future / entry - 1.0
                if direction == "SHORT":
                    move = -move
                returns.append(move * 100.0)

            if not returns:
                print(f"{symbol:<10}{pattern_name:<22}{0:>8}{0:>10.3f}{0:>10.3f}{0:>9.1f}{0:>10.2f}")
                continue

            s = pd.Series(returns)
            avg_return = s.mean()
            median_return = s.median()
            win_rate = (s > 0).mean() * 100.0
            total_return = s.sum()

            print(
                f"{symbol:<10}"
                f"{pattern_name:<22}"
                f"{len(s):>8}"
                f"{avg_return:>10.3f}"
                f"{median_return:>10.3f}"
                f"{win_rate:>9.1f}"
                f"{total_return:>10.2f}"
            )

    print("-" * 78)
    print("Horizon: H4")
    print("Prediction accuracy assumption: 100%")
    print("Commission: 0%")
    print("Entry: pattern candle close")
    print("Exit: close after 4 candles")
    print("LONG/SHORT direction is assumed perfectly correct.")
    print("TOTAL% is the sum of raw directional moves; it is not compounded.")
    print()


if __name__ == "__main__":
    main()
