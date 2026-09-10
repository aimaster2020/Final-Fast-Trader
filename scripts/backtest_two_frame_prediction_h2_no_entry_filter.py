from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"


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
    _score, direction = score_direction(o, h, l, c)
    if direction == 0:
        return np.nan, np.nan, np.nan, np.nan, 0, np.nan

    upper = abs(h - o)
    lower = abs(o - l)
    magnitude = upper if direction == 1 else lower
    opposite = lower if direction == 1 else upper
    if not np.isfinite(magnitude) or magnitude <= 1e-12:
        return np.nan, np.nan, np.nan, np.nan, 0, np.nan

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
    o, h, l, c = map(float, df.iloc[i][["open", "high", "low", "close"]])
    p1 = predict_frame(o, h, l, c)
    if p1[4] == 0 or not np.isfinite(p1[3]):
        return 0, 0, np.nan, np.nan, False

    p2 = predict_frame(p1[0], p1[1], p1[2], p1[3])
    if p2[4] == 0 or not np.isfinite(p2[3]):
        return p1[4], 0, p1[5], np.nan, False

    return p1[4], p2[4], p1[5], p2[5], p1[4] == p2[4]


def run_symbol(df: pd.DataFrame, initial_capital: float, fee: float, mode: str) -> dict[str, float | int]:
    capital = float(initial_capital)
    fee_factor = (1.0 - fee) ** 2
    eligible = agree = trades = wins = 0
    gross_sum = 0.0

    n = len(df)
    i = 0
    while i < n - 2:
        o, h, l, c = map(float, df.iloc[i][["open", "high", "low", "close"]])
        _score, entry_dir = score_direction(o, h, l, c)

        if entry_dir == 0:
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
        if p1_dir != entry_dir or p2_dir != entry_dir:
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
        "agreement_rate": agree / eligible if eligible else 0.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "avg_gross_trade": gross_sum / trades if trades else 0.0,
    }


def pooled(loaded: dict[str, pd.DataFrame], symbols: list[str], initial_capital: float, fee: float, mode: str) -> dict[str, float | int]:
    rs = [run_symbol(loaded[s], initial_capital, fee, mode) for s in symbols]
    final = sum(float(r["final"]) for r in rs)
    eligible = sum(int(r["eligible"]) for r in rs)
    trades = sum(int(r["trades"]) for r in rs)
    wins = sum(int(round(float(r["win_rate"]) * int(r["trades"]))) for r in rs)
    gross = sum(float(r["avg_gross_trade"]) * int(r["trades"]) for r in rs)
    agreement = sum(float(r["agreement_rate"]) * int(r["eligible"]) for r in rs)
    initial = initial_capital * len(symbols)
    return {
        "final": final,
        "return": final / initial - 1.0,
        "eligible": eligible,
        "agreement_rate": agreement / eligible if eligible else 0.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "avg_gross_trade": gross / trades if trades else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict H2 recursive prediction agreement test without entry magnitude filter.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=0.0013)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    root = Path(args.data_dir)
    loaded = {s: load(root / f"{s}_1h.csv") for s in symbols}

    print("=" * 112)
    print("H2 RECURSIVE PREDICTION AGREEMENT — NO ENTRY FILTER")
    print("=" * 112)
    print("Current signal: SCORE 0=SHORT / SCORE 3=LONG")
    print("Entry magnitude filter: DISABLED")
    print("Prediction 1: synthetic t+1 from real t")
    print("Prediction 2: synthetic t+2 from predicted t+1")
    print("Entry requires: pred(t+1) == pred(t+2) == current signal")
    print("Entry: current close | Exit: actual close at t+2")
    print("Signal exits: disabled | one active trade per symbol")
    print()

    for mode in ("SHORT", "LONG", "BOTH"):
        print(mode)
        print("fee      final       return      eligible  agree%   trades   win%   avg_gross")
        for fee in (0.0, args.fee):
            r = pooled(loaded, symbols, args.initial_capital, fee, mode)
            print(f"{fee*100:5.2f}%  {float(r['final']):10.2f}  {float(r['return'])*100:+9.2f}%  {int(r['eligible']):8d}  {float(r['agreement_rate'])*100:6.1f}%  {int(r['trades']):6d}  {float(r['win_rate'])*100:5.1f}%  {float(r['avg_gross_trade'])*100:+9.4f}%")
        print()

    print("=" * 112)


if __name__ == "__main__":
    main()
