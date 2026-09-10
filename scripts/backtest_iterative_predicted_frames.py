from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_HORIZONS = "1,2,3,4,5"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    for c in ("open", "high", "low", "close"):
        if c not in cols:
            raise ValueError(f"Missing column {c} in {path}")
    df = df.rename(columns={cols[c]: c for c in ("open", "high", "low", "close")})
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open", "high", "low", "close"]].dropna().reset_index(drop=True)


def score_direction(o: float, h: float, l: float, c: float) -> tuple[int, int]:
    body = c - o
    hc = h - c
    ho = h - o
    lc = l - c
    score = int(hc > body) + int(ho < hc) + int(lc > body)
    direction = -1 if score == 0 else 1 if score == 3 else 0
    return score, direction


def predict_next_frame(o: float, h: float, l: float, c: float, direction: int) -> tuple[float, float, float, float]:
    if direction == 1:
        magnitude = abs(h - o)
        next_o = c
        next_c = next_o + magnitude
    else:
        magnitude = abs(o - l)
        next_o = c
        next_c = next_o - magnitude
    next_h = max(next_o, next_c)
    next_l = min(next_o, next_c)
    return next_o, next_h, next_l, next_c


def iterative_prediction(df: pd.DataFrame, i: int, horizon: int) -> tuple[int, float]:
    o = float(df.iloc[i].open)
    h = float(df.iloc[i].high)
    l = float(df.iloc[i].low)
    c = float(df.iloc[i].close)

    initial_score, initial_direction = score_direction(o, h, l, c)
    if initial_score not in (0, 3):
        return 0, np.nan

    direction = initial_direction
    predicted_close = c
    for _ in range(horizon):
        o, h, l, c = predict_next_frame(o, h, l, c, direction)
        predicted_close = c
        _, direction = score_direction(o, h, l, c)
        if direction == 0:
            return 0, np.nan

    return initial_direction, predicted_close


def run_symbol(df: pd.DataFrame, initial_capital: float, fee: float, horizon: int) -> dict[str, float | int]:
    capital = float(initial_capital)
    fee_factor = (1.0 - fee) ** 2
    trades = wins = 0
    gross_sum = 0.0
    n = len(df)

    i = 0
    while i < n - horizon:
        score, direction = score_direction(
            float(df.iloc[i].open), float(df.iloc[i].high), float(df.iloc[i].low), float(df.iloc[i].close)
        )
        if score not in (0, 3):
            i += 1
            continue

        pred_dir, predicted_close = iterative_prediction(df, i, horizon)
        if pred_dir == 0 or not np.isfinite(predicted_close):
            i += 1
            continue

        entry = float(df.iloc[i].close)
        actual_exit = float(df.iloc[i + horizon].close)
        gross_ret = (actual_exit - entry) / entry if direction == 1 else (entry - actual_exit) / entry
        capital *= max((1.0 + gross_ret) * fee_factor, 0.0)
        trades += 1
        wins += int(gross_ret > 0)
        gross_sum += gross_ret
        i += horizon + 1

    return {
        "final": capital,
        "return": capital / initial_capital - 1.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "avg_gross": gross_sum / trades if trades else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Recursive synthetic-frame prediction backtest.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=0.0013)
    ap.add_argument("--horizons", default=DEFAULT_HORIZONS)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    loaded = {s: load(Path(args.data_dir) / f"{s}_1h.csv") for s in symbols}

    print("=" * 104)
    print("ITERATIVE PREDICTED-FRAME BACKTEST")
    print("=" * 104)
    print("Direction: SCORE 0 = SHORT, SCORE 3 = LONG; SCORES 1/2 = NO TRADE")
    print("Entry: current candle close")
    print("H=1: direct next-frame prediction")
    print("H>1: each synthetic predicted frame becomes the input for the next prediction")
    print("No future observed candle is used to construct intermediate predicted frames")
    print("Actual result: compared with observed close at +H")
    print("Commission: 0% and configured fee per side")
    print()

    for h in horizons:
        print(f"HORIZON = {h}")
        for fee in (0.0, args.fee):
            pooled_final = 0.0
            trades = wins = 0
            gross_sum = 0.0
            for symbol in symbols:
                r = run_symbol(loaded[symbol], args.initial_capital, fee, h)
                pooled_final += float(r["final"])
                trades += int(r["trades"])
                wins += int(round(float(r["win_rate"]) * int(r["trades"])))
                gross_sum += float(r["avg_gross"]) * int(r["trades"])
            pooled_initial = args.initial_capital * len(symbols)
            pooled_return = pooled_final / pooled_initial - 1.0
            win_rate = wins / trades if trades else 0.0
            avg_gross = gross_sum / trades if trades else 0.0
            print(f"fee={fee*100:.2f}% final={pooled_final:.2f} return={pooled_return*100:+.2f}% trades={trades} win={win_rate*100:.1f}% avg_gross={avg_gross*100:+.4f}%")
        print()

    print("=" * 104)


if __name__ == "__main__":
    main()
