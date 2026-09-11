from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MIN_CANDLES = 100
NO_SIGNAL_MARKETS = {"DOSUSDT", "GRVTUSDT", "SLVONUSDT"}
TARGET_MARKETS = 233


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
    df = df.reset_index(drop=True)

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
    df["signal_dir"] = signal_dir
    return df


def horizon_stats(df: pd.DataFrame, horizon: int, half_fee: float, round_fee: float) -> dict[str, float | int]:
    future_close = df["close"].shift(-horizon)
    move_pct = (future_close / df["close"] - 1.0) * 100.0
    signed = df["signal_dir"] * move_pct
    valid = df["signal_dir"].ne(0) & move_pct.notna() & move_pct.ne(0)
    signed = signed[valid]
    if signed.empty:
        return {"n": 0, "win": np.nan, "mean": np.nan, "median": np.nan, "half": np.nan, "round": np.nan}
    return {
        "n": len(signed),
        "win": float((signed > 0).mean() * 100.0),
        "mean": float(signed.mean()),
        "median": float(signed.median()),
        "half": float((signed >= half_fee).mean() * 100.0),
        "round": float((signed >= round_fee).mean() * 100.0),
    }


def established_universe(root: Path) -> list[str]:
    report = root / "nonirt_oos_direction_magnitude_stability.csv"
    if report.exists():
        try:
            prior = pd.read_csv(report)
            symbols = set(prior["symbol"].astype(str).str.upper())
            symbols.update(NO_SIGNAL_MARKETS)
            if len(symbols) == TARGET_MARKETS:
                return sorted(symbols)
        except Exception:
            pass
    files = sorted(p for p in root.glob("*_1h.csv") if not p.stem[:-3].upper().endswith("IRT"))
    return sorted(set(p.stem[:-3].upper() for p in files) | NO_SIGNAL_MARKETS)[:TARGET_MARKETS]


def main() -> None:
    ap = argparse.ArgumentParser(description="Test the fixed Nobitex formula across multiple holding horizons.")
    ap.add_argument("--data-dir", default="reports/nobitex_1h")
    ap.add_argument("--horizons", default="1,2,3,4,5")
    ap.add_argument("--half-fee", type=float, default=0.13)
    ap.add_argument("--round-fee", type=float, default=0.26)
    ap.add_argument("--output", default="reports/nobitex_1h/nonirt_formula_horizons.csv")
    args = ap.parse_args()

    root = Path(args.data_dir)
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    universe = established_universe(root)
    if len(universe) != TARGET_MARKETS:
        raise SystemExit(f"Expected the established 233-market universe, found {len(universe)} symbols.")

    path_map = {p.stem[:-3].upper(): p for p in root.glob("*_1h.csv")}
    records = []
    for symbol in universe:
        path = path_map.get(symbol)
        if path is None:
            for h in horizons:
                records.append({"symbol": symbol, "horizon": h, "n": 0, "win": np.nan, "mean": np.nan, "median": np.nan, "half": np.nan, "round": np.nan})
            continue
        try:
            df = load(path)
            for h in horizons:
                if len(df) < MIN_CANDLES:
                    s = {"n": 0, "win": np.nan, "mean": np.nan, "median": np.nan, "half": np.nan, "round": np.nan}
                else:
                    s = horizon_stats(df, h, args.half_fee, args.round_fee)
                records.append({"symbol": symbol, "horizon": h, **s})
        except Exception:
            for h in horizons:
                records.append({"symbol": symbol, "horizon": h, "n": 0, "win": np.nan, "mean": np.nan, "median": np.nan, "half": np.nan, "round": np.nan})

    out = pd.DataFrame(records)
    out.to_csv(args.output, index=False)

    print("=" * 150)
    print("NOBITEX NON-IRT | FORMULA HORIZON TEST | ALL 233 MARKETS")
    print("=" * 150)
    print(f"Established universe : {TARGET_MARKETS} markets")
    print(f"Markets printed      : {out.symbol.nunique()} / {TARGET_MARKETS}")
    print("Entry                : current candle close")
    print("Exit                 : close after H candles")
    print("Horizons             : H1 H2 H3 H4 H5 (1H candles)")
    print("Signal               : fixed formula + ambiguity filter")
    print("Fees                 : half 0.13% | round-trip 0.26%")
    print()

    summary = []
    for h in horizons:
        x = out[out.horizon == h]
        valid = x[x.n > 0]
        summary.append((h, len(valid), valid.win.mean(), valid.win.median(), int((valid.win >= 55).sum()), valid['mean'].mean(), int((valid.half >= 50).sum()), int((valid['round'] >= 50).sum()), valid.n.mean()))
    print("SUMMARY — ALL 233")
    print("H   tested   mean_win   median   >=55   mean_move   half>=50   round>=50   mean_n")
    for h, tested, mw, med, ge55, mm, half, rnd, mn in summary:
        print(f"{h:<3d} {tested:>7d}   {mw:>7.2f}%   {med:>7.2f}%  {ge55:>5d}   {mm:>+8.3f}%   {half:>9d}   {rnd:>10d}   {mn:>7.1f}")

    print()
    print("DETAIL — ALL 233 MARKETS")
    print("Each value = WIN% / mean signed move% / round-fee coverage%")
    print("SYMBOL               | H1                  | H2                  | H3                  | H4                  | H5")
    print("-" * 150)
    for symbol in universe:
        chunks = []
        for h in horizons:
            r = out[(out.symbol == symbol) & (out.horizon == h)].iloc[0]
            if int(r.n) == 0:
                chunks.append(f"H{h} -- / -- / --")
            else:
                chunks.append(f"H{h} {r.win:5.1f}% / {r['mean']:+6.3f}% / {r['round']:5.1f}%")
        print(f"{symbol:20s} | " + " | ".join(chunks))

    print()
    print(f"Detailed CSV: {args.output}")
    print("=" * 150)


if __name__ == "__main__":
    main()
