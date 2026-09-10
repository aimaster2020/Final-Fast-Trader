from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_ENTRY_MIN = 0.008


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in {"open", "high", "low", "close"})
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


def score_direction(o: float, h: float, l: float, c: float) -> tuple[int, int]:
    body = c - o
    hc = h - c
    ho = h - o
    lc = l - c
    score = int(hc > body) + int(ho < hc) + int(lc > body)
    direction = -1 if score == 0 else 1 if score == 3 else 0
    return score, direction


def predict_frame(o: float, h: float, l: float, c: float) -> tuple[float, float, float, float, int, float]:
    """Predict one next OHLC frame using only the supplied frame."""
    _score, direction = score_direction(o, h, l, c)
    if direction == 0:
        return np.nan, np.nan, np.nan, np.nan, 0, np.nan

    upper = abs(h - o)
    lower = abs(o - l)
    magnitude = upper if direction == 1 else lower
    opposite = lower if direction == 1 else upper
    if not np.isfinite(magnitude) or magnitude <= 1e-12:
        return np.nan, np.nan, np.nan, np.nan, direction, np.nan

    wick_ratio = opposite / max(magnitude, 1e-12)
    predicted_open = c

    if direction == 1:
        predicted_close = predicted_open + magnitude
        predicted_high = predicted_close
        predicted_low = predicted_open - magnitude * wick_ratio
    else:
        predicted_close = predicted_open - magnitude
        predicted_low = predicted_close
        predicted_high = predicted_open + magnitude * wick_ratio

    predicted_high = max(predicted_open, predicted_close, predicted_high)
    predicted_low = min(predicted_open, predicted_close, predicted_low)
    predicted_pct = magnitude / max(abs(predicted_open), 1e-12)
    return predicted_open, predicted_high, predicted_low, predicted_close, direction, predicted_pct


def two_frame_prediction(df: pd.DataFrame, i: int) -> tuple[int, int, float, float, bool]:
    """Predict t+1, then predict t+2 from synthetic t+1 and report agreement."""
    o1, h1, l1, c1 = map(float, df.iloc[i][["open", "high", "low", "close"]])
    p1 = predict_frame(o1, h1, l1, c1)
    if not np.isfinite(p1[3]) or p1[4] == 0:
        return 0, 0, np.nan, np.nan, False

    p2 = predict_frame(p1[0], p1[1], p1[2], p1[3])
    if not np.isfinite(p2[3]) or p2[4] == 0:
        return p1[4], 0, p1[5], np.nan, False

    return p1[4], p2[4], p1[5], p2[5], p1[4] == p2[4]


def run_symbol(df: pd.DataFrame, initial_capital: float, fee: float, entry_min: float, mode: str) -> dict[str, float | int]:
    capital = float(initial_capital)
    fee_factor = (1.0 - fee) ** 2
    trades = wins = eligible = agree = 0
    gross_sum = 0.0

    n = len(df)
    i = 0
    while i < n - 2:
        o, h, l, c = map(float, df.iloc[i][["open", "high", "low", "close"]])
        _score, entry_dir = score_direction(o, h, l, c)
        entry_mag = abs(h - o) if entry_dir == 1 else abs(o - l) if entry_dir == -1 else np.nan
        entry_pct = entry_mag / max(abs(c), 1e-12) if np.isfinite(entry_mag) else np.nan

        if entry_dir == 0 or not np.isfinite(entry_pct) or entry_pct < entry_min:
            i += 1
            continue
        if mode == "LONG" and entry_dir != 1:
            i += 1
            continue
        if mode == "SHORT" and entry_dir != -1:
            i += 1
            continue

        eligible += 1
        p1_dir, p2_dir, _p1_pct, _p2_pct, agreement = two_frame_prediction(df, i)
        if agreement:
            agree += 1
        if not agreement or p1_dir != entry_dir or p2_dir != entry_dir:
            i += 1
            continue

        entry = c
        actual_exit = float(df.iloc[i + 2].close)
        gross_ret = ((actual_exit - entry) / entry) if entry_dir == 1 else ((entry - actual_exit) / entry)
        capital *= max((1.0 + gross_ret) * fee_factor, 0.0)
        trades += 1
        wins += int(gross_ret > 0)
        gross_sum += gross_ret
        i += 3

    return {
        "final": capital,
        "return": capital / initial_capital - 1.0,
        "eligible": eligible,
        "agreement": agree,
        "agreement_rate": agree / eligible if eligible else 0.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "avg_gross_trade": gross_sum / trades if trades else 0.0,
    }


def pooled(loaded: dict[str, pd.DataFrame], symbols: list[str], initial_capital: float, fee: float, entry_min: float, mode: str) -> dict[str, float | int]:
    rs = [run_symbol(loaded[s], initial_capital, fee, entry_min, mode) for s in symbols]
    final = sum(float(r["final"]) for r in rs)
    eligible = sum(int(r["eligible"]) for r in rs)
    agreement = sum(int(r["agreement"]) for r in rs)
    trades = sum(int(r["trades"]) for r in rs)
    wins = sum(int(round(float(r["win_rate"]) * int(r["trades"]))) for r in rs)
    gross = sum(float(r["avg_gross_trade"]) * int(r["trades"]) for r in rs)
    initial = initial_capital * len(symbols)
    return {
        "final": final,
        "return": final / initial - 1.0,
        "eligible": eligible,
        "agreement": agreement,
        "agreement_rate": agreement / eligible if eligible else 0.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "avg_gross_trade": gross / trades if trades else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict H2 recursive prediction-entry test.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=0.0013)
    ap.add_argument("--entry-min", type=float, default=DEFAULT_ENTRY_MIN)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    root = Path(args.data_dir)
    loaded = {s: load(root / f"{s}_1h.csv") for s in symbols}

    print("=" * 112)
    print("STRICT H2 RECURSIVE PREDICTION ENTRY TEST")
    print("=" * 112)
    print("Entry signal: real current candle SCORE 0=SHORT / SCORE 3=LONG")
    print(f"Entry magnitude filter: >= {args.entry_min * 100:.2f}%")
    print("Prediction 1: synthetic t+1 from real t")
    print("Prediction 2: synthetic t+2 from predicted t+1")
    print("Entry requires: pred(t+1) == pred(t+2) == current signal")
    print("Exit: actual close at t+2 | signal exits disabled | one active trade per symbol")
    print()

    for mode in ("SHORT", "LONG", "BOTH"):
        print(mode)
        print("fee      final       return      eligible agree%  trades   win%   avg_gross")
        for fee in (0.0, args.fee):
            r = pooled(loaded, symbols, args.initial_capital, fee, args.entry_min, mode)
            print(
                f"{fee*100:5.2f}%  {float(r['final']):10.2f}  {float(r['return'])*100:+9.2f}%  "
                f"{int(r['eligible']):8d} {float(r['agreement_rate'])*100:6.1f}%  "
                f"{int(r['trades']):6d}  {float(r['win_rate'])*100:5.1f}%  "
                f"{float(r['avg_gross_trade'])*100:+9.4f}%"
            )
        print()

    print("=" * 112)


if __name__ == "__main__":
    main()
