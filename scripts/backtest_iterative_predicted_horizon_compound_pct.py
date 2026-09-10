from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_HORIZONS = "1,2,3,4,5"
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


def score_direction(open_: float, high: float, low: float, close: float) -> tuple[int, int]:
    body = close - open_
    hc = high - close
    ho = high - open_
    lc = low - close
    score = int(hc > body) + int(ho < hc) + int(lc > body)
    direction = -1 if score == 0 else 1 if score == 3 else 0
    return score, direction


def first_prediction(o: float, h: float, l: float, c: float) -> tuple[int, float, float, float, float]:
    """Return direction, predicted move, predicted move pct, and normalized OHLC shape."""
    _score, direction = score_direction(o, h, l, c)
    if direction == 0:
        return 0, np.nan, np.nan, np.nan, np.nan

    upper = abs(h - o)
    lower = abs(o - l)
    body = c - o
    magnitude = upper if direction == 1 else lower
    pct = magnitude / max(abs(c), 1e-12)
    shape_scale = magnitude
    if not np.isfinite(magnitude) or magnitude <= 1e-12:
        return 0, np.nan, np.nan, np.nan, np.nan

    body_ratio = body / shape_scale
    upper_ratio = upper / shape_scale
    lower_ratio = lower / shape_scale
    return direction, magnitude, pct, body_ratio, upper_ratio


def recursive_prediction(df: pd.DataFrame, start: int, horizon: int) -> tuple[int, float, float]:
    """Recursively predict H candles using predicted percentage movement.

    Step 1 is computed from the real current candle. The resulting predicted move percentage
    is then applied to each subsequent synthetic candle's current open. The synthetic OHLC
    keeps the previous frame's normalized shape, so the predicted frame itself is the next
    model input. No future real OHLC is used in the recursive chain.
    """
    o = float(df.iloc[start].open)
    h = float(df.iloc[start].high)
    l = float(df.iloc[start].low)
    c = float(df.iloc[start].close)

    first_dir, magnitude, pct, body_ratio, upper_ratio = first_prediction(o, h, l, c)
    if first_dir == 0 or not np.isfinite(pct):
        return 0, np.nan, np.nan

    lower = abs(o - l)
    shape_mag = magnitude
    lower_ratio = lower / max(shape_mag, 1e-12)

    current_open = c
    current_dir = first_dir
    predicted_close = c

    for step in range(horizon):
        step_mag = pct * abs(current_open)
        if not np.isfinite(step_mag) or step_mag <= 1e-12:
            return 0, np.nan, np.nan

        # Preserve the extreme-score candle shape while applying the previous prediction's
        # movement percentage to the new predicted open.
        if current_dir == 1:
            # LONG prediction is represented by the score-3 (bearish) candle shape.
            predicted_body = -abs(body_ratio) * step_mag
            predicted_open = current_open
            predicted_close = predicted_open + predicted_body
            predicted_high = predicted_open + upper_ratio * step_mag
            predicted_low = predicted_open - lower_ratio * step_mag
        else:
            # SHORT prediction is represented by the score-0 (bullish) candle shape.
            predicted_body = abs(body_ratio) * step_mag
            predicted_open = current_open
            predicted_close = predicted_open + predicted_body
            predicted_high = predicted_open + upper_ratio * step_mag
            predicted_low = predicted_open - lower_ratio * step_mag

        predicted_high = max(predicted_open, predicted_close, predicted_high)
        predicted_low = min(predicted_open, predicted_close, predicted_low)

        _score, next_dir = score_direction(predicted_open, predicted_high, predicted_low, predicted_close)
        if next_dir == 0:
            return 0, np.nan, np.nan

        current_dir = next_dir
        current_open = predicted_close

    terminal_move = predicted_close - c
    terminal_pct = terminal_move / max(abs(c), 1e-12)
    return current_dir, predicted_close, terminal_pct


def run_symbol(
    df: pd.DataFrame,
    initial_capital: float,
    fee: float,
    horizon: int,
    entry_min: float,
    mode: str,
) -> dict[str, float | int]:
    capital = float(initial_capital)
    fee_factor = (1.0 - fee) ** 2
    trades = wins = 0
    gross_sum = 0.0

    n = len(df)
    i = 0
    while i < n - horizon:
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

        pred_dir, pred_close, pred_terminal_pct = recursive_prediction(df, i, horizon)
        if pred_dir == 0 or not np.isfinite(pred_close):
            i += 1
            continue

        # Require the recursive terminal direction to agree with the entry direction.
        if pred_dir != entry_dir:
            i += 1
            continue

        entry = c
        actual_exit = float(df.iloc[i + horizon].close)
        gross_ret = ((actual_exit - entry) / entry) if entry_dir == 1 else ((entry - actual_exit) / entry)

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
        "avg_gross_trade": gross_sum / trades if trades else 0.0,
    }


def pooled(
    loaded: dict[str, pd.DataFrame],
    symbols: list[str],
    initial_capital: float,
    fee: float,
    horizon: int,
    entry_min: float,
    mode: str,
) -> dict[str, float | int]:
    rs = [
        run_symbol(loaded[s], initial_capital, fee, horizon, entry_min, mode)
        for s in symbols
    ]
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
    ap = argparse.ArgumentParser(description="Recursive predicted-candle horizon backtest using compounded predicted movement percentages.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=0.0013)
    ap.add_argument("--entry-min", type=float, default=DEFAULT_ENTRY_MIN)
    ap.add_argument("--horizons", default=DEFAULT_HORIZONS)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    horizons = sorted({int(x) for x in args.horizons.split(",") if x.strip()})
    if any(h < 1 for h in horizons):
        raise ValueError("Horizons must be >= 1.")

    root = Path(args.data_dir)
    loaded = {s: load(root / f"{s}_1h.csv") for s in symbols}

    print("=" * 104)
    print("ITERATIVE PREDICTED-CANDLE BACKTEST — COMPOUNDED PREDICTED %")
    print("=" * 104)
    print("Entry direction: real current candle SCORE 0=SHORT / SCORE 3=LONG")
    print(f"Entry magnitude filter: >= {args.entry_min * 100:.2f}%")
    print("Step 1: predict next-frame movement from real candle")
    print("Step 2+: use prior predicted movement percentage on the predicted open")
    print("Recursive input: each predicted OHLC frame becomes the next input")
    print("Decision: terminal recursive direction must agree with entry direction")
    print("Entry: current close | Exit/P&L: actual close at +H")
    print("Signal exits: disabled | one active trade per symbol")
    print()

    for h in horizons:
        print(f"HORIZON = {h}")
        for mode in ("SHORT", "LONG", "BOTH"):
            print(mode)
            print("fee      final       return      trades   win%   avg_gross")
            for fee in (0.0, args.fee):
                r = pooled(loaded, symbols, args.initial_capital, fee, h, args.entry_min, mode)
                print(
                    f"{fee*100:5.2f}%  {float(r['final']):10.2f}  "
                    f"{float(r['return'])*100:+9.2f}%  "
                    f"{int(r['trades']):6d}  "
                    f"{float(r['win_rate'])*100:5.1f}%  "
                    f"{float(r['avg_gross_trade'])*100:+9.4f}%"
                )
            print()

    print("=" * 104)


if __name__ == "__main__":
    main()
