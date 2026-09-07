import argparse
import pandas as pd


PATTERNS = {
    "Bullish Pin Bar": "LONG",
    "Hammer": "LONG",
    "Bearish Outside": "SHORT",
}

FEE_ROUND_TRIP = 0.26
MIN_TRAIN_SAMPLES = 20


def features(row, prev=None):
    o, h, l, c = map(float, (row["open"], row["high"], row["low"], row["close"]))
    rng = h - l
    if rng <= 0:
        return None
    body = abs(c - o)
    lower = min(o, c) - l
    upper = h - max(o, c)
    prev_rng = 0.0 if prev is None else float(prev["high"]) - float(prev["low"])
    return {
        "body_r": body / rng,
        "lower_r": lower / rng,
        "upper_r": upper / rng,
        "close_low_r": (h - c) / rng,
        "range_exp": rng / prev_rng if prev_rng > 0 else 0.0,
        "bull": c > o,
        "bear": c < o,
    }


def match(pattern, row, prev, f):
    if pattern == "Hammer":
        return f["body_r"] <= 0.35 and f["lower_r"] >= 2.0 * f["body_r"] and f["upper_r"] <= 0.35
    if pattern == "Bullish Pin Bar":
        return f["lower_r"] >= 0.60 and f["upper_r"] <= 0.20 and f["bull"]
    return (
        prev is not None
        and float(row["high"]) > float(prev["high"])
        and float(row["low"]) < float(prev["low"])
        and f["bear"]
    )


def collect(df, pattern, direction, horizon):
    out = []
    for i in range(len(df) - horizon):
        row = df.iloc[i]
        prev = df.iloc[i - 1] if i else None
        f = features(row, prev)
        if f is None or not match(pattern, row, prev, f):
            continue
        entry = float(row["close"])
        future = float(df.iloc[i + horizon]["close"])
        if entry <= 0 or future <= 0:
            continue
        move = (future / entry - 1.0) * 100.0
        if direction == "SHORT":
            move = -move
        f["move"] = move
        out.append(f)
    return pd.DataFrame(out)


def candidates(pattern, s):
    return [
        ("BASE", pd.Series(True, index=s.index)),
        ("body<=0.25", s.body_r <= 0.25),
        ("lower>=0.70", s.lower_r >= 0.70),
        ("lower>=0.75", s.lower_r >= 0.75),
        ("upper<=0.15", s.upper_r <= 0.15),
        ("upper<=0.20", s.upper_r <= 0.20),
        ("body>=0.50", s.body_r >= 0.50),
        ("body>=0.60", s.body_r >= 0.60),
        ("range_exp>=1.25", s.range_exp >= 1.25),
        ("body<=0.25+lower>=0.70", (s.body_r <= 0.25) & (s.lower_r >= 0.70)),
        ("body<=0.25+lower>=0.75", (s.body_r <= 0.25) & (s.lower_r >= 0.75)),
        ("body<=0.25+upper<=0.20", (s.body_r <= 0.25) & (s.upper_r <= 0.20)),
        ("lower>=0.70+upper<=0.15", (s.lower_r >= 0.70) & (s.upper_r <= 0.15)),
        ("lower>=0.70+upper<=0.20", (s.lower_r >= 0.70) & (s.upper_r <= 0.20)),
        ("body>=0.50+range_exp>=1.25", (s.body_r >= 0.50) & (s.range_exp >= 1.25)),
        ("body>=0.50+close_low<=0.25", (s.body_r >= 0.50) & (s.close_low_r <= 0.25)),
    ]


def score(s, mask):
    x = s.loc[mask, "move"]
    if len(x) == 0:
        return None
    return {
        "n": len(x),
        "avg": x.mean(),
        "med": x.median(),
        "win": (x > 0).mean() * 100.0,
        "over_fee": (x > FEE_ROUND_TRIP).mean() * 100.0,
        "net": x.mean() - FEE_ROUND_TRIP,
    }


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    symbol_col = next((c for c in ("symbol", "ticker", "asset") if c in df.columns), None)
    time_col = next((c for c in ("timestamp", "time", "datetime", "date") if c in df.columns), None)
    if symbol_col is None or time_col is None:
        raise ValueError("Need symbol and timestamp/time/datetime/date columns")
    df["_dt"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["_dt"]).copy()
    return df, symbol_col


def main():
    p = argparse.ArgumentParser(description="H4 walk-forward ex-ante filter validation")
    p.add_argument("--prepared-file", default=r".\reports\prepared_price_action_1h.csv")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--train-months", default="2026-05,2026-06")
    p.add_argument("--test-months", default="2026-07,2026-08")
    p.add_argument("--top-k", type=int, default=2)
    args = p.parse_args()

    train_months = set(x.strip() for x in args.train_months.split(",") if x.strip())
    test_months = set(x.strip() for x in args.test_months.split(",") if x.strip())
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    df, symbol_col = load(args.prepared_file)
    df["_month"] = df["_dt"].dt.strftime("%Y-%m")

    print()
    print("H4 WALK-FORWARD FILTER VALIDATION")
    print(f"TRAIN: {','.join(sorted(train_months))} | TEST: {','.join(sorted(test_months))}")
    print("Filter selection uses TRAIN only; TEST is frozen out-of-sample.")
    print("Commission: 0.13% entry + 0.13% exit = 0.26% round trip")
    print()

    for symbol in symbols:
        sdf = df[df[symbol_col].astype(str).str.upper() == symbol].copy()
        if sdf.empty:
            continue
        sdf = sdf.sort_values("_dt").reset_index(drop=True)
        train = sdf[sdf._month.isin(train_months)].copy().sort_values("_dt").reset_index(drop=True)
        test = sdf[sdf._month.isin(test_months)].copy().sort_values("_dt").reset_index(drop=True)
        if train.empty or test.empty:
            continue

        for pattern, direction in PATTERNS.items():
            tr = collect(train, pattern, direction, args.horizon)
            te = collect(test, pattern, direction, args.horizon)
            if tr.empty or te.empty:
                continue

            ranked = []
            for name, mask in candidates(pattern, tr):
                r = score(tr, mask)
                if r and r["n"] >= MIN_TRAIN_SAMPLES:
                    ranked.append((name, r))
            ranked.sort(key=lambda z: z[1]["net"], reverse=True)
            chosen = ranked[:args.top_k]

            print(f"{symbol} | {pattern} | TRAIN_N={len(tr)} TEST_N={len(te)}")
            print(f"{'FROZEN FILTER':<45}{'TR_N':>6}{'TR_NET':>9}{'TE_N':>6}{'TE_AVG':>9}{'TE_WIN':>9}{'TE>FEE':>9}{'TE_NET':>9}")
            for name, trr in chosen:
                test_mask = dict(candidates(pattern, te))[name]
                ter = score(te, test_mask)
                if ter is None:
                    continue
                print(f"{name:<45}{trr['n']:>6}{trr['net']:>9.3f}{ter['n']:>6}{ter['avg']:>9.3f}{ter['win']:>9.1f}{ter['over_fee']:>9.1f}{ter['net']:>9.3f}")
            print()

    print("Interpretation: TE_NET > 0 means the frozen filter's average H4 move exceeded the 0.26% round-trip fee reference.")
    print("This is still a raw-capacity test, not a capital-compounded execution backtest.")


if __name__ == "__main__":
    main()
