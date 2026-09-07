from pathlib import Path
import argparse
import pandas as pd


PATTERNS = {
    "Bullish Pin Bar": "LONG",
    "Hammer": "LONG",
    "Bearish Outside": "SHORT",
}

FEE_ROUND_TRIP = 0.26


def candle_features(row, prev=None):
    o, h, l, c = map(float, (row["open"], row["high"], row["low"], row["close"]))
    rng = h - l
    if rng <= 0:
        return None
    body = abs(c - o)
    lower = min(o, c) - l
    upper = h - max(o, c)
    f = {
        "body_r": body / rng,
        "lower_r": lower / rng,
        "upper_r": upper / rng,
        "close_low_r": (h - c) / rng,
        "close_high_r": (c - l) / rng,
        "range": rng,
        "bull": c > o,
        "bear": c < o,
    }
    if prev is not None:
        prev_rng = float(prev["high"]) - float(prev["low"])
        f["range_exp"] = rng / prev_rng if prev_rng > 0 else 0.0
    else:
        f["range_exp"] = 0.0
    return f


def is_hammer(f):
    return f["body_r"] <= 0.35 and f["lower_r"] * 1.0 >= 2.0 * f["body_r"] and f["upper_r"] <= 0.35


def is_bullish_pin(f):
    return f["lower_r"] >= 0.60 and f["upper_r"] <= 0.20 and f["bull"]


def is_bearish_outside(row, prev, f):
    return (
        prev is not None
        and float(row["high"]) > float(prev["high"])
        and float(row["low"]) < float(prev["low"])
        and f["bear"]
    )


def collect(df, pattern_name, direction, horizon):
    rows = []
    for i in range(len(df) - horizon):
        row = df.iloc[i]
        prev = df.iloc[i - 1] if i > 0 else None
        f = candle_features(row, prev)
        if f is None:
            continue
        if pattern_name == "Hammer":
            matched = is_hammer(f)
        elif pattern_name == "Bullish Pin Bar":
            matched = is_bullish_pin(f)
        else:
            matched = is_bearish_outside(row, prev, f)
        if not matched:
            continue

        entry = float(row["close"])
        future = float(df.iloc[i + horizon]["close"])
        if entry <= 0 or future <= 0:
            continue
        move = (future / entry - 1.0) * 100.0
        if direction == "SHORT":
            move = -move
        f["move"] = move
        rows.append(f)
    return pd.DataFrame(rows)


def evaluate(s, mask):
    x = s.loc[mask, "move"]
    if x.empty:
        return None
    return {
        "n": len(x),
        "avg": x.mean(),
        "med": x.median(),
        "win": (x > 0).mean() * 100.0,
        "fee": (x > FEE_ROUND_TRIP).mean() * 100.0,
        "net_avg": x.mean() - FEE_ROUND_TRIP,
    }


def main():
    parser = argparse.ArgumentParser(
        description="H4 ex-ante candle-geometry filter scan. Filters use only the signal candle and previous candle."
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
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"Missing OHLC column: {c}")

    print()
    print("H4 EX-ANTE FILTER SCAN - RAW CAPACITY")
    print("Features use signal candle + previous candle only; future move is used only for evaluation.")
    print("Commission reference: 0.13% entry + 0.13% exit = 0.26% round trip")
    print()

    for symbol in symbols:
        sdf = df[df[symbol_col].astype(str).str.upper() == symbol.upper()].copy()
        if sdf.empty:
            continue
        time_col = next((c for c in ("timestamp", "time", "datetime", "date") if c in sdf.columns), None)
        if time_col:
            sdf = sdf.sort_values(time_col)
        sdf = sdf.reset_index(drop=True)

        for pattern_name, direction in PATTERNS.items():
            s = collect(sdf, pattern_name, direction, args.horizon)
            if s.empty:
                continue

            candidates = [("BASE", pd.Series(True, index=s.index))]
            if pattern_name == "Hammer":
                candidates += [
                    ("body<=0.25", s.body_r <= 0.25),
                    ("lower>=0.70", s.lower_r >= 0.70),
                    ("upper<=0.20", s.upper_r <= 0.20),
                    ("body<=0.25+lower>=0.70", (s.body_r <= 0.25) & (s.lower_r >= 0.70)),
                    ("body<=0.25+upper<=0.20", (s.body_r <= 0.25) & (s.upper_r <= 0.20)),
                    ("lower>=0.70+upper<=0.20", (s.lower_r >= 0.70) & (s.upper_r <= 0.20)),
                    ("body<=0.25+lower>=0.70+upper<=0.20", (s.body_r <= 0.25) & (s.lower_r >= 0.70) & (s.upper_r <= 0.20)),
                ]
            elif pattern_name == "Bullish Pin Bar":
                candidates += [
                    ("body<=0.25", s.body_r <= 0.25),
                    ("lower>=0.70", s.lower_r >= 0.70),
                    ("lower>=0.75", s.lower_r >= 0.75),
                    ("upper<=0.15", s.upper_r <= 0.15),
                    ("body<=0.25+lower>=0.70", (s.body_r <= 0.25) & (s.lower_r >= 0.70)),
                    ("lower>=0.70+upper<=0.15", (s.lower_r >= 0.70) & (s.upper_r <= 0.15)),
                    ("body<=0.25+lower>=0.75", (s.body_r <= 0.25) & (s.lower_r >= 0.75)),
                    ("body<=0.25+lower>=0.70+upper<=0.15", (s.body_r <= 0.25) & (s.lower_r >= 0.70) & (s.upper_r <= 0.15)),
                ]
            else:
                candidates += [
                    ("body>=0.50", s.body_r >= 0.50),
                    ("body>=0.60", s.body_r >= 0.60),
                    ("close_near_low<=0.25", s.close_low_r <= 0.25),
                    ("close_near_low<=0.20", s.close_low_r <= 0.20),
                    ("range_exp>=1.25", s.range_exp >= 1.25),
                    ("body>=0.50+close_near_low<=0.25", (s.body_r >= 0.50) & (s.close_low_r <= 0.25)),
                    ("body>=0.50+range_exp>=1.25", (s.body_r >= 0.50) & (s.range_exp >= 1.25)),
                    ("close_near_low<=0.25+range_exp>=1.25", (s.close_low_r <= 0.25) & (s.range_exp >= 1.25)),
                    ("body>=0.50+close_near_low<=0.25+range_exp>=1.25", (s.body_r >= 0.50) & (s.close_low_r <= 0.25) & (s.range_exp >= 1.25)),
                ]

            results = []
            for name, mask in candidates:
                r = evaluate(s, mask)
                if r is not None:
                    results.append((name, r))

            print(f"{symbol} | {pattern_name} | base_n={len(s)}")
            print(f"{'FILTER':<48}{'N':>6}{'AVG%':>9}{'MED%':>9}{'WIN%':>9}{'>0.26%':>10}{'AVG-FEE':>10}")
            for name, r in results:
                print(f"{name:<48}{r['n']:>6}{r['avg']:>9.3f}{r['med']:>9.3f}{r['win']:>9.1f}{r['fee']:>10.1f}{r['net_avg']:>10.3f}")
            print()

    print("NOTE: This scan is diagnostic. Choosing the best filter on the same sample is in-sample and can overfit.")
    print("Next validation should freeze 1-2 simple filters and test them on a separate time period.")


if __name__ == "__main__":
    main()
