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
    return body / rng <= 0.35 and lower >= 2.0 * body and upper <= 0.35 * rng


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
        description=(
            "H4 movement-capacity diagnostic. Uses realized future moves only "
            "to measure how many historical opportunities had enough movement; "
            "this is an oracle diagnostic, not a tradable filter."
        )
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
    print("H4 MOVEMENT CAPACITY - ORACLE DIAGNOSTIC")
    print("-" * 124)
    print(
        f"{'SYMBOL':<10}{'PATTERN':<22}{'N':>6}"
        f"{'AVG%':>9}{'>0.26%':>10}{'AVG@>0.26':>12}"
        f"{'>0.50%':>10}{'AVG@>0.50':>12}{'>1.00%':>10}{'AVG@>1.00':>12}"
    )
    print("-" * 124)

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

            values = [f"{symbol:<10}{pattern_name:<22}{len(s):>6}{s.mean():>9.3f}"]
            for threshold in THRESHOLDS:
                selected = s[s > threshold]
                pct = len(selected) / len(s) * 100.0
                avg_selected = selected.mean() if not selected.empty else 0.0
                values.append(f"{pct:>9.1f}%")
                values.append(f"{avg_selected:>11.3f}")
            print("".join(values))

    print("-" * 124)
    print("Horizon: H4 | Perfect directional prediction | Commission: 0%")
    print("Entry: pattern candle close | Exit: close after 4 candles")
    print("IMPORTANT: threshold selection uses the realized future move, so it is hindsight/oracle only.")
    print("Purpose: quantify historical movement capacity before designing an ex-ante filter.")
    print()


if __name__ == "__main__":
    main()
