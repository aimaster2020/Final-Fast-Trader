from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_ENTRY_MIN = 0.008


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=lambda c: str(c).strip().lower() in {"open", "high", "low", "close"},
    )
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


def predict_next_frame(o: float, h: float, l: float, c: float) -> tuple[float, float, float, float, int, float]:
    """Create one synthetic next candle from the current frame.

    The predicted trade direction comes from SCORE 0/3. The predicted movement is the
    trade-side excursion: LONG=|H-O|, SHORT=|O-L|. The next synthetic candle keeps the
    source candle's normalized geometry, scaled to that predicted move.
    """
    _score, direction = score_direction(o, h, l, c)
    if direction == 0:
        return np.nan, np.nan, np.nan, np.nan, 0, np.nan

    upper = abs(h - o)
    lower = abs(o - l)
    magnitude = upper if direction == 1 else lower
    if not np.isfinite(magnitude) or magnitude <= 1e-12:
        return np.nan, np.nan, np.nan, np.nan, 0, np.nan

    body = c - o
    # Keep normalized shape, but orient it to the predicted next-frame direction.
    body_ratio = abs(body) / magnitude
    upper_ratio = upper / magnitude
    lower_ratio = lower / magnitude

    predicted_open = c
    predicted_body = -body_ratio * magnitude if direction == 1 else body_ratio * magnitude
    predicted_close = predicted_open + predicted_body
    predicted_high = predicted_open + upper_ratio * magnitude
    predicted_low = predicted_open - lower_ratio * magnitude

    predicted_high = max(predicted_open, predicted_close, predicted_high)
    predicted_low = min(predicted_open, predicted_close, predicted_low)
    predicted_pct = magnitude / max(abs(predicted_open), 1e-12)

    _synthetic_score, synthetic_direction = score_direction(
        predicted_open, predicted_high, predicted_low, predicted_close
    )
    return (
        predicted_open,
        predicted_high,
        predicted_low,
        predicted_close,
        synthetic_direction,
        predicted_pct,
    )


def two_frame_prediction(df: pd.DataFrame, i: int) -> tuple[int, int, float]:
    """Predict t+1 and then t+2, using the predicted frame as the next input.

    Returns (direction_1, direction_2, predicted_move_pct_to_t2).
    """
    o = float(df.iloc[i].open)
    h = float(df.iloc[i].high)
    l = float(df.iloc[i].low)
    c = float(df.iloc[i].close)

    p1o, p1h, p1l, p1c, d1, p1pct = predict_next_frame(o, h, l, c)
    if d1 == 0 or not np.isfinite(p1c):
        return 0, 0, np.nan

    p2o, p2h, p2l, p2c, d2, p2pct = predict_next_frame(p1o, p1h, p1l, p1c)
    if d2 == 0 or not np.isfinite(p2c):
        return d1, 0, np.nan

    return d1, d2, p2pct


def run_symbol(
    df: pd.DataFrame,
    initial_capital: float,
    fee: float,
    entry_min: float,
    mode: str,
) -> dict[str, float | int]:
    capital = float(initial_capital)
    fee_factor = (1.0 - fee) ** 2
    trades = wins = 0
    gross_sum = 0.0

    n = len(df)
    i = 0
    while i < n - 2:
        o = float(df.iloc[i].open)
        h = float(df.iloc[i].high)
        l = float(df.iloc[i].low)
        c = float(df.iloc[i].close)
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

        d1, d2, _pred2_pct = two_frame_prediction(df, i)
        if d1 == 0 or d2 == 0:
            i += 1
            continue

        # Entry is allowed only when BOTH predicted frames agree with the current signal.
        if d1 != entry_dir or d2 != entry_dir:
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
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "avg_gross_trade": gross_sum / trades if trades else 0.0,
    }


def pooled(
    loaded: dict[str, pd.DataFrame],
    symbols: list[str],
    initial_capital: float,
    fee: float,
    entry_min: float,
    mode: str,
) -> dict[str, float | int]:
    rs = [run_symbol(loaded[s], initial_capital, fee, entry_min, mode) for s in symbols]
    final = sum(float(r["final"]) for r in rs)
    trades = sum(int(r["trades"]) for r in rs)
    wins = sum(int(round(float(r["win_rate"]) * int(r["trades"]))) for r in rs)
    gross = sum(float(r["avg_gross_trade"]) * int(r["trades"]) for r in rs)
    initial = initial_capital * len(symbols)
    return {
        "final": final,
        "return": final / initial - 1.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "avg_gross_trade": gross / trades if trades else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="H2 backtest requiring two recursive predicted frames to agree before entry."
    )
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=0.0013)
    ap.add_argument("--entry-min", type=float, default=DEFAULT_ENTRY_MIN)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    root = Path(args.data_dir)
    loaded = {s: load(root / f"{s}_1h.csv") for s in symbols}

    print("=" * 104)
    print("H2 BACKTEST — ENTER ONLY WHEN TWO PREDICTED FRAMES AGREE")
    print("=" * 104)
    print("Current signal: SCORE 0=SHORT / SCORE 3=LONG")
    print(f"Entry filter: expected current move >= {args.entry_min * 100:.2f}%")
    print("Prediction: t+1 is predicted from real t; t+2 is predicted from synthetic t+1")
    print("Entry condition: predicted t+1 direction == predicted t+2 direction == current signal")
    print("Entry: current close | Exit: actual close at t+2")
    print("Signal exits: disabled | one active trade per symbol")
    print()

    for mode in ("SHORT", "LONG", "BOTH"):
        print(mode)
        print("fee      final       return      trades   win%   avg_gross")
        for fee in (0.0, args.fee):
            r = pooled(loaded, symbols, args.initial_capital, fee, args.entry_min, mode)
            print(
                f"{fee * 100:5.2f}%  {float(r['final']):10.2f}  "
                f"{float(r['return']) * 100:+9.2f}%  "
                f"{int(r['trades']):6d}  "
                f"{float(r['win_rate']) * 100:5.1f}%  "
                f"{float(r['avg_gross_trade']) * 100:+9.4f}%"
            )
        print()

    print("=" * 104)


if __name__ == "__main__":
    main()
