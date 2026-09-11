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

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
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
    df = df.copy()
    df["signal_dir"] = signal_dir
    return df


def stats(df: pd.DataFrame, half_fee: float, round_fee: float) -> dict[str, float | int]:
    move_pct = (df["close"].shift(-1) / df["close"] - 1.0) * 100.0
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
    report = data_dir / "../nobitex_1h/nonirt_oos_direction_magnitude_stability.csv"
    report = report.resolve()
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
    ap = argparse.ArgumentParser(description="Test fixed Nobitex formula on 4H candles with H1 horizon.")
    ap.add_argument("--data-dir", default="reports/nobitex_1h")
    ap.add_argument("--half-fee", type=float, default=0.13)
    ap.add_argument("--round-fee", type=float, default=0.26)
    ap.add_argument("--output", default="reports/nobitex_4h/nonirt_formula_h1.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    universe = established_universe(data_dir)
    if len(universe) != TARGET_MARKETS:
        raise SystemExit(f"Expected the established 233-market universe, found {len(universe)} symbols.")

    path_map = {p.stem[:-3].upper(): p for p in data_dir.glob("*_1h.csv")}
    records: list[dict[str, float | int | str]] = []

    for symbol in universe:
        path = path_map.get(symbol)
        if path is None:
            records.append({"symbol": symbol, "n_4h": 0, "n_trades": 0, "win": np.nan, "mean": np.nan, "median": np.nan, "half": np.nan, "round": np.nan})
            continue
        try:
            df4 = apply_formula(resample_4h(load_1h(path)))
            if len(df4) < MIN_CANDLES_4H:
                s = {"n": 0, "win": np.nan, "mean": np.nan, "median": np.nan, "half": np.nan, "round": np.nan}
            else:
                s = stats(df4, args.half_fee, args.round_fee)
            records.append({
                "symbol": symbol,
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
            records.append({"symbol": symbol, "n_4h": 0, "n_trades": 0, "win": np.nan, "mean": np.nan, "median": np.nan, "half": np.nan, "round": np.nan})

    out = pd.DataFrame(records)
    out.to_csv(output, index=False)
    valid = out[out.n_trades > 0]

    print("=" * 120)
    print("NOBITEX NON-IRT | 4H DATA | H1 HORIZON | ALL 233 MARKETS")
    print("=" * 120)
    print(f"Established universe : {TARGET_MARKETS} markets")
    print(f"Markets tested       : {len(valid)} / {TARGET_MARKETS}")
    print("Source               : existing 1H Nobitex data aggregated into complete 4H candles")
    print("Entry                : 4H candle close")
    print("Exit                 : close after 1 x 4H candle")
    print("Signal               : fixed formula + ambiguity filter")
    print("Fees                 : half 0.13% | round-trip 0.26%")
    print()
    print(f"Mean 4H candles      : {valid.n_4h.mean():.1f}")
    print(f"Mean trades/market   : {valid.n_trades.mean():.1f}")
    print(f"Mean win             : {valid.win.mean():.2f}%")
    print(f"Median win           : {valid.win.median():.2f}%")
    print(f">=55% win            : {int((valid.win >= 55).sum())} / {len(valid)}")
    print(f"Mean signed move     : {valid['mean'].mean():+.3f}%")
    print(f"Half-fee >=50%       : {int((valid.half >= 50).sum())} / {len(valid)}")
    print(f"Round-fee >=50%      : {int((valid['round'] >= 50).sum())} / {len(valid)}")
    print()
    print("DETAIL — ALL 233 MARKETS")
    print("SYMBOL               | 4H candles | trades | win    | mean    | median  | half   | round")
    print("-" * 120)
    for r in out.itertuples(index=False):
        if r.n_trades == 0:
            print(f"{r.symbol:20s} | {r.n_4h:10d} | {r.n_trades:6d} | --     | --      | --      | --     | --")
        else:
            print(f"{r.symbol:20s} | {r.n_4h:10d} | {r.n_trades:6d} | {r.win:6.2f}% | {r.mean:+7.3f}% | {r.median:+7.3f}% | {r.half:6.2f}% | {r.round:6.2f}%")
    print()
    print(f"Detailed CSV: {output}")
    print("=" * 120)


if __name__ == "__main__":
    main()
