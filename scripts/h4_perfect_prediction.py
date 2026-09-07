from pathlib import Path
import argparse
import pandas as pd


PATTERNS = {
    "Bullish Pin Bar": "LONG",
    "Hammer": "LONG",
    "Bearish Outside": "SHORT",
}


THRESHOLDS = [0.26, 0.50, 1.00]


def is_hammer(row):
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)
    lower = min(o, c) - l
    upper = h - max(o, c)
    return (
        body / rng <= 0.35
        and lower >= 2.0 * body
        and upper <= 0.35 * rng
    )


def is_bullish_pin(row):
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)
    lower = min(o, c) - l
    upper = h - max(o, c)
    return lower / rng >= 0.60 and upper / rng <= 0.20 and c > o


def is_bearish_outside(df, i):
    if i <= 0:
        return False
    row, prev = df.iloc[i], df.iloc[i - 1]
    return (
        row["high"] > prev["high"]
        and row["low"] < prev["low"]
        and row["close"] < row["open"]
    )


def get_returns(sdf, pattern_name, direction, horizon):
    returns = []
    for i in range(len(sdf) - horizon):
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
        future = float(sdf.iloc[i + horizon]["close"])
        if entry <= 0 or future <= 0:
            continue

        move = future / entry - 1.0
        if direction == "SHORT":
            move = -move
        returns.append(move * 100.0)
    return pd.Series(returns, dtype=float)


def main():
    parser = argparse.ArgumentParser(
        description="Measure raw H4 directional move distribution assuming 100% prediction accuracy."
    )
    parser.add_argument("--prepared-file", default=r".\reports\prepared_price_action_1h.csv")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    parser.add_argument("--horizon", type=int, default=4)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    df = pd.read_csv(args.prepared_file)
    df.columns = [c.strip().lower() for c in df.columns]

    symbol_col = next((c for c in ("symbol", "ticker", "asset") if c in df.columns), None)
    if symbol_col is None:
        raise ValueError(f"Symbol column not found. Columns: {list(df.columns)}")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")

    print()
    print("H4 PERFECT PREDICTION - RAW DISTRIBUTION")
    print("-" * 128)
    print(
        f"{'SYMBOL':<10}{'PATTERN':<22}{'N':>6}{'AVG%':>9}{'MED%':>9}"
        f"{'P25%':>9}{'P75%':>9}{'P90%':>9}{'P95%':>9}{'WIN%':>8}"
        f"{'>0.26%':>9}{'>0.50%':>9}{'>1.00%':>9}"
    )
    print("-" * 128)

    for symbol in symbols:
        sdf = df[df[symbol_col].astype(str).str.upper() == symbol.upper()].copy()
        if sdf.empty:
            print(f"{symbol:<10} NO DATA")
            continue

        time_col = next((c for c in ("timestamp", "time", "datetime", "date") if c in sdf.columns), None)
        if time_col:
            sdf = sdf.sort_values(time_col)
        sdf = sdf.reset_index(drop=True)

        for pattern_name, direction in PATTERNS.items():
            s = get_returns(sdf, pattern_name, direction, args.horizon)
            if s.empty:
                print(f"{symbol:<10}{pattern_name:<22}{0:>6}")
                continue

            print(
                f"{symbol:<10}{pattern_name:<22}{len(s):>6}"
                f"{s.mean():>9.3f}{s.median():>9.3f}"
                f"{s.quantile(.25):>9.3f}{s.quantile(.75):>9.3f}"
                f"{s.quantile(.90):>9.3f}{s.quantile(.95):>9.3f}"
                f"{(s > 0).mean() * 100:>8.1f}"
                f"{(s > THRESHOLDS[0]).mean() * 100:>9.1f}"
                f"{(s > THRESHOLDS[1]).mean() * 100:>9.1f}"
                f"{(s > THRESHOLDS[2]).mean() * 100:>9.1f}"
            )

    print("-" * 128)
    print("Horizon: H4 | Prediction accuracy: 100% | Commission: 0%")
    print("Entry: pattern candle close | Exit: close after 4 candles")
    print(">0.26% = raw directional move large enough to cover 0.13% entry + 0.13% exit commission.")
    print("All percentages are raw, per-trade directional moves; no compounding.")
    print()


if __name__ == "__main__":
    main()
