from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_HORIZONS = "1,2,3,4,5"
DEFAULT_ENTRY_MIN = 0.0080


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


def predict_frame(o: float, h: float, l: float, c: float, direction: int) -> tuple[float, float, float, float]:
    magnitude = abs(h - o) if direction == 1 else abs(o - l)
    next_o = c
    next_c = next_o + magnitude if direction == 1 else next_o - magnitude
    return next_o, max(next_o, next_c), min(next_o, next_c), next_c


def recursive_prediction(df: pd.DataFrame, i: int, horizon: int) -> tuple[int, float]:
    o, h, l, c = (float(df.iloc[i][x]) for x in ("open", "high", "low", "close"))
    score, initial_direction = score_direction(o, h, l, c)
    if score not in (0, 3):
        return 0, np.nan
    direction = initial_direction
    predicted_close = c
    for _ in range(horizon):
        o, h, l, c = predict_frame(o, h, l, c, direction)
        predicted_close = c
        _, direction = score_direction(o, h, l, c)
        if direction == 0:
            return 0, np.nan
    return initial_direction, predicted_close


def run_symbol(df: pd.DataFrame, initial_capital: float, fee: float, horizon: int, entry_min: float) -> dict[str, float | int]:
    capital = float(initial_capital)
    fee_factor = (1.0 - fee) ** 2
    trades = wins = 0
    gross_sum = 0.0
    n = len(df)
    i = 0
    while i < n - horizon:
        row = df.iloc[i]
        score, direction = score_direction(float(row.open), float(row.high), float(row.low), float(row.close))
        if score not in (0, 3):
            i += 1
            continue
        expected_move = abs(float(row.high) - float(row.open)) if direction == 1 else abs(float(row.open) - float(row.low))
        expected_pct = expected_move / max(abs(float(row.close)), 1e-12)
        if expected_pct < entry_min:
            i += 1
            continue
        pred_dir, predicted_close = recursive_prediction(df, i, horizon)
        if pred_dir == 0 or not np.isfinite(predicted_close):
            i += 1
            continue
        entry = float(row.close)
        actual_exit = float(df.iloc[i + horizon].close)
        gross_ret = (actual_exit - entry) / entry if direction == 1 else (entry - actual_exit) / entry
        capital *= max((1.0 + gross_ret) * fee_factor, 0.0)
        trades += 1
        wins += int(gross_ret > 0)
        gross_sum += gross_ret
        i += horizon + 1
    return {"final": capital, "return": capital / initial_capital - 1.0, "trades": trades, "win_rate": wins / trades if trades else 0.0, "avg_gross": gross_sum / trades if trades else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description="Recursive predicted-frame backtest with 0.8% entry filter.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=0.0013)
    ap.add_argument("--entry-min", type=float, default=DEFAULT_ENTRY_MIN)
    ap.add_argument("--horizons", default=DEFAULT_HORIZONS)
    args = ap.parse_args()
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    if args.entry_min < 0:
        raise ValueError("entry-min must be >= 0")
    loaded = {s: load(Path(args.data_dir) / f"{s}_1h.csv") for s in symbols}
    print("=" * 104)
    print("ITERATIVE PREDICTED-FRAME BACKTEST")
    print("=" * 104)
    print("Direction: SCORE 0 = SHORT, SCORE 3 = LONG; scores 1/2 ignored")
    print(f"Entry magnitude filter: >= {args.entry_min * 100:.2f}%")
    print("Entry: current close | Exit: actual close at +H")
    print("H>1: synthetic predicted frame -> next synthetic predicted frame")
    print("No observed future OHLC is used to construct intermediate predictions")
    for h in horizons:
        print(f"HORIZON = {h}")
        for fee in (0.0, args.fee):
            pooled_final = 0.0
            trades = wins = 0
            gross_sum = 0.0
            for s in symbols:
                r = run_symbol(loaded[s], args.initial_capital, fee, h, args.entry_min)
                pooled_final += float(r["final"])
                trades += int(r["trades"])
                wins += int(round(float(r["win_rate"]) * int(r["trades"])))
                gross_sum += float(r["avg_gross"]) * int(r["trades"])
            pooled_initial = args.initial_capital * len(symbols)
            ret = pooled_final / pooled_initial - 1.0
            win = wins / trades if trades else 0.0
            avg = gross_sum / trades if trades else 0.0
            print(f"fee={fee*100:.2f}% final={pooled_final:.2f} return={ret*100:+.2f}% trades={trades} win={win*100:.1f}% avg_gross={avg*100:+.4f}%")
        print()
    print("=" * 104)


if __name__ == "__main__":
    main()
