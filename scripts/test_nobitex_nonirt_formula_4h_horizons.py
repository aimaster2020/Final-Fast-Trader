from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MIN_CANDLES_4H = 100
NO_SIGNAL_MARKETS = {"DOSUSDT", "GRVTUSDT", "SLVONUSDT"}
TARGET_MARKETS = 233


def load_1h(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if "timestamp" not in df.columns:
        raise ValueError("missing timestamp column")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    return df.sort_values("timestamp").reset_index(drop=True)


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    x = df.copy()
    x.index = ts
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in x.columns:
        agg["volume"] = "sum"
    out = x.resample("4h", origin="epoch", label="left", closed="left").agg(agg)
    counts = x["close"].resample("4h", origin="epoch", label="left", closed="left").count()
    out["_n1h"] = counts
    out = out[out["_n1h"] == 4].copy()
    out = out.drop(columns=["_n1h"])
    out = out.dropna(subset=["open", "high", "low", "close"])
    out["timestamp"] = (out.index.view("int64") // 1_000_000_000).astype("int64")
    return out.reset_index(drop=True)


def apply_formula(df: pd.DataFrame) -> pd.DataFrame:
    body = df["close"] - df["open"]
    hc = df["high"] - df["close"]
    ho = df["high"] - df["open"]
    lc = df["low"] - df["close"]
    score = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)
    formula_dir = np.where(score <= 1, -1, np.where(score >= 2, 1, 0)).astype(int)

    upper = (df["high"] - df["open"]).abs()
    lower = (df["open"] - df["low"]).abs()
    side_formula = np.where(lower > upper, 1, np.where(upper > lower, -1, 0)).astype(int)

    signal_dir = formula_dir.copy()
    signal_dir[(score == 1) & (side_formula == 1)] = 0
    signal_dir[(score == 2) & (side_formula == -1)] = 0
    out = df.copy()
    out["signal_dir"] = signal_dir
    return out


def stats(df: pd.DataFrame, horizon: int, half_fee: float, round_fee: float) -> dict[str, float | int]:
    move_pct = (df["close"].shift(-horizon) / df["close"] - 1.0) * 100.0
    signed = df["signal_dir"] * move_pct
    valid = df["signal_dir"].ne(0) & move_pct.notna() & move_pct.ne(0)
    signed = signed[valid]
    if signed.empty:
        return {"n": 0, "win": np.nan, "mean": np.nan, "median": np.nan, "half": np.nan, "round": np.nan}
    return {
        "n": int(len(signed)),
        "win": float((signed > 0).mean() * 100.0),
        "mean": float(signed.mean()),
        "median": float(signed.median()),
        "half": float((signed >= half_fee).mean() * 100.0),
        "round": float((signed >= round_fee).mean() * 100.0),
    }


def established_universe(data_dir: Path) -> list[str]:
    report = (data_dir / "../nobitex_1h/nonirt_oos_direction_magnitude_stability.csv").resolve()
    if report.exists():
        try:
            prior = pd.read_csv(report)
            symbols = set(prior["symbol"].astype(str).str.upper())
            symbols.update(NO_SIGNAL_MARKETS)
            if len(symbols) == TARGET_MARKETS:
                return sorted(symbols)
        except Exception:
            pass

    files = sorted(data_dir.glob("*_1h.csv"))
    symbols = set()
    for p in files:
        symbol = p.stem[:-3].upper() if p.stem.lower().endswith("_1h") else p.stem.upper()
        if not symbol.endswith("IRT"):
            symbols.add(symbol)
    symbols.update(NO_SIGNAL_MARKETS)
    return sorted(symbols)[:TARGET_MARKETS]


def main() -> None:
    ap = argparse.ArgumentParser(description="Test fixed Nobitex formula on 4H candles with multiple holding horizons.")
    ap.add_argument("--data-dir", default="reports/nobitex_1h")
    ap.add_argument("--horizons", default="1,2,3,4")
    ap.add_argument("--half-fee", type=float, default=0.13)
    ap.add_argument("--round-fee", type=float, default=0.26)
    ap.add_argument("--output", default="reports/nobitex_4h/nonirt_formula_horizons.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    if not horizons or any(h < 1 for h in horizons):
        raise SystemExit("--horizons must contain positive integers")

    universe = established_universe(data_dir)
    if len(universe) != TARGET_MARKETS:
        raise SystemExit(f"Expected the established 233-market universe, found {len(universe)} symbols.")

    path_map = {p.stem[:-3].upper(): p for p in data_dir.glob("*_1h.csv")}
    records: list[dict[str, float | int | str]] = []

    for symbol in universe:
        path = path_map.get(symbol)
        if path is None:
            for h in horizons:
                records.append({"symbol": symbol, "horizon": h, "n_4h": 0, "n_trades": 0, "win": np.nan, "mean": np.nan, "median": np.nan, "half": np.nan, "round": np.nan})
            continue

        try:
            df4 = apply_formula(resample_4h(load_1h(path)))
            for h in horizons:
                if len(df4) < MIN_CANDLES_4H + h:
                    s = {"n": 0, "win": np.nan, "mean": np.nan, "median": np.nan, "half": np.nan, "round": np.nan}
                else:
                    s = stats(df4, h, args.half_fee, args.round_fee)
                records.append({
                    "symbol": symbol,
                    "horizon": h,
                    "n_4h": int(len(df4)),
                    "n_trades": s["n"],
                    "win": s["win"],
                    "mean": s["mean"],
                    "median": s["median"],
                    "half": s["half"],
                    "round": s["round"],
                })
        except Exception as exc:
            print(f"{symbol}: ERROR {exc}")
            for h in horizons:
                records.append({"symbol": symbol, "horizon": h, "n_4h": 0, "n_trades": 0, "win": np.nan, "mean": np.nan, "median": np.nan, "half": np.nan, "round": np.nan})

    out = pd.DataFrame(records)
    out.to_csv(output, index=False)

    print("=" * 150)
    print("NOBITEX NON-IRT | 4H DATA | H1-H4 HORIZON TEST | ALL 233 MARKETS")
    print("=" * 150)
    print(f"Established universe : {TARGET_MARKETS} markets")
    print(f"Markets with signal  : {int((out.n_trades > 0).groupby(out.horizon).sum().max()) if not out.empty else 0} max")
    print("Source               : existing 1H Nobitex data aggregated into complete 4H candles")
    print("Entry                : 4H candle close")
    print("Exit                 : close after H x 4H candles")
    print(f"Horizons             : {', '.join(f'H{h}' for h in horizons)}")
    print("Signal               : fixed formula + ambiguity filter")
    print("Fees                 : half 0.13% | round-trip 0.26%")
    print()

    print("SUMMARY — ALL 233")
    print("H   tested   mean_win   median   >=55   mean_move   half>=50   round>=50   mean_trades")
    for h in horizons:
        x = out[(out.horizon == h) & (out.n_trades > 0)]
        if x.empty:
            print(f"{h:<3d} {0:>7d}   --        --       --       --         --          --            --")
            continue
        print(
            f"{h:<3d} {len(x):>7d}   {x.win.mean():>7.2f}%   {x.win.median():>7.2f}%"
            f"  {int((x.win >= 55).sum()):>5d}   {x['mean'].mean():>+8.3f}%"
            f"   {int((x.half >= 50).sum()):>9d}   {int((x['round'] >= 50).sum()):>10d}"
            f"   {x.n_trades.mean():>9.1f}"
        )

    print()
    print("DETAIL — ALL 233 MARKETS")
    print("Each value = WIN% / mean signed move% / round-fee coverage%")
    header = "SYMBOL               | " + " | ".join(f"H{h:<18}" for h in horizons)
    print(header)
    print("-" * max(150, len(header)))
    for symbol in universe:
        chunks = []
        for h in horizons:
            r = out[(out.symbol == symbol) & (out.horizon == h)].iloc[0]
            if int(r.n_trades) == 0:
                chunks.append(f"H{h} -- / -- / --")
            else:
                chunks.append(f"H{h} {r.win:5.1f}% / {r['mean']:+6.3f}% / {r['round']:5.1f}%")
        print(f"{symbol:20s} | " + " | ".join(chunks))

    print()
    print(f"Detailed CSV: {output}")
    print("=" * 150)


if __name__ == "__main__":
    main()
