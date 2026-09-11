from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NO_SIGNAL_MARKETS = {"DOSUSDT", "GRVTUSDT", "SLVONUSDT"}
TARGET_MARKETS = 233
MIN_CANDLES = 20


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df.reset_index(drop=True)


def signal(df: pd.DataFrame) -> pd.DataFrame:
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


def universe(data_dir: Path) -> list[str]:
    names = []
    for p in data_dir.glob("*_4h.csv"):
        symbol = p.stem[:-3].upper() if p.stem.lower().endswith("_4h") else p.stem.upper()
        if symbol and not symbol.endswith("IRT"):
            names.append(symbol)
    symbols = sorted(set(names) | NO_SIGNAL_MARKETS)
    return symbols[:TARGET_MARKETS]


def main() -> None:
    ap = argparse.ArgumentParser(description="Test fixed formula H1 on native Nobitex 4H candles.")
    ap.add_argument("--data-dir", default="reports/nobitex_4h")
    ap.add_argument("--half-fee", type=float, default=0.13)
    ap.add_argument("--round-fee", type=float, default=0.26)
    ap.add_argument("--output", default="reports/nobitex_4h/native_formula_h1.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    symbols = universe(data_dir)
    path_map = {p.stem[:-3].upper(): p for p in data_dir.glob("*_4h.csv")}
    records = []
    for symbol in symbols:
        path = path_map.get(symbol)
        if path is None:
            records.append({"symbol": symbol, "n_4h": 0, "n_trades": 0, "win": np.nan, "mean": np.nan, "median": np.nan, "half": np.nan, "round": np.nan})
            continue
        try:
            df = signal(load(path))
            if len(df) < MIN_CANDLES:
                s = {"n": 0, "win": np.nan, "mean": np.nan, "median": np.nan, "half": np.nan, "round": np.nan}
            else:
                s = stats(df, args.half_fee, args.round_fee)
            records.append({"symbol": symbol, "n_4h": int(len(df)), "n_trades": s["n"], "win": s["win"], "mean": s["mean"], "median": s["median"], "half": s["half"], "round": s["round"]})
        except Exception as exc:
            print(f"{symbol}: ERROR {exc}")
            records.append({"symbol": symbol, "n_4h": 0, "n_trades": 0, "win": np.nan, "mean": np.nan, "median": np.nan, "half": np.nan, "round": np.nan})

    out = pd.DataFrame(records)
    out.to_csv(output, index=False)
    valid = out[out.n_trades > 0]

    print("=" * 120)
    print("NOBITEX NON-IRT | NATIVE 4H DATA | H1 HORIZON | ALL 233 MARKETS")
    print("=" * 120)
    print(f"Universe requested   : {TARGET_MARKETS}")
    print(f"Markets with signals : {len(valid)}")
    print("Source               : native Nobitex UDF resolution=240")
    print("Entry                : 4H candle close")
    print("Exit                 : close after 1 x 4H candle")
    print("Signal               : fixed formula + ambiguity filter")
    print("Fees                 : half 0.13% | round-trip 0.26%")
    print()
    if not valid.empty:
        print(f"Mean 4H candles      : {valid.n_4h.mean():.1f}")
        print(f"Mean trades/market   : {valid.n_trades.mean():.1f}")
        print(f"Mean win             : {valid.win.mean():.2f}%")
        print(f"Median win           : {valid.win.median():.2f}%")
        print(f">=55% win            : {int((valid.win >= 55).sum())} / {len(valid)}")
        print(f"Mean signed move     : {valid['mean'].mean():+.3f}%")
        print(f"Half-fee >=50%       : {int((valid.half >= 50).sum())} / {len(valid)}")
        print(f"Round-fee >=50%      : {int((valid['round'] >= 50).sum())} / {len(valid)}")
    print()
    print("DETAIL — ALL MARKETS FOUND")
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
