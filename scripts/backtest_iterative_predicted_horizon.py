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


def predict_next_frame(
    open_: float,
    high: float,
    low: float,
    close: float,
) -> tuple[float, float, float, float, int, int, float]:
    """Build one predicted OHLC frame from the current frame only.

    The current score determines the trade direction. The current candle's normalized
    OHLC shape is preserved, while its trade-side excursion becomes the predicted
    movement. The resulting synthetic candle is then fed into the next prediction.
    """
    score, direction = score_direction(open_, high, low, close)
    if direction == 0:
        return np.nan, np.nan, np.nan, np.nan, score, direction, np.nan

    upper = abs(high - open_)
    lower = abs(open_ - low)
    body = close - open_

    # Baseline magnitude law:
    # score 3 / LONG  -> |H-O|
    # score 0 / SHORT -> |O-L|
    magnitude = upper if direction == 1 else lower
    reference_side = upper if direction == 1 else lower
    if not np.isfinite(magnitude) or reference_side <= 1e-12:
        return np.nan, np.nan, np.nan, np.nan, score, direction, np.nan

    # Preserve the source candle's normalized geometry. This is important because
    # the score itself depends on the OHLC shape, not only on the close movement.
    scale = magnitude / reference_side
    predicted_open = close
    predicted_body = body * scale
    predicted_close = predicted_open + predicted_body
    predicted_high = predicted_open + upper * scale
    predicted_low = predicted_open - lower * scale

    predicted_high = max(predicted_open, predicted_close, predicted_high)
    predicted_low = min(predicted_open, predicted_close, predicted_low)
    predicted_pct = magnitude / max(abs(predicted_open), 1e-12)

    return (
        predicted_open,
        predicted_high,
        predicted_low,
        predicted_close,
        score,
        direction,
        predicted_pct,
    )


def iterative_prediction(
    df: pd.DataFrame,
    start: int,
    horizon: int,
) -> tuple[int, float, int]:
    """Predict H future frames recursively without using future real OHLC."""
    o = float(df.iloc[start].open)
    h = float(df.iloc[start].high)
    l = float(df.iloc[start].low)
    c = float(df.iloc[start].close)
    first_direction = 0
    final_direction = 0
    predicted_close = c

    for step in range(horizon):
        po, ph, pl, pc, _score, direction, _pct = predict_next_frame(o, h, l, c)
        if direction == 0 or not np.isfinite(pc):
            return 0, np.nan, first_direction
        if step == 0:
            first_direction = direction
        final_direction = direction
        predicted_close = pc
        o, h, l, c = po, ph, pl, pc

    return final_direction, predicted_close, first_direction


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
        _score, first_dir = score_direction(
            float(df.iloc[i].open),
            float(df.iloc[i].high),
            float(df.iloc[i].low),
            float(df.iloc[i].close),
        )

        first_mag = (
            abs(float(df.iloc[i].high) - float(df.iloc[i].open))
            if first_dir == 1
            else abs(float(df.iloc[i].open) - float(df.iloc[i].low))
            if first_dir == -1
            else np.nan
        )
        first_pct = (
            first_mag / max(abs(float(df.iloc[i].close)), 1e-12)
            if np.isfinite(first_mag)
            else np.nan
        )

        if first_dir == 0 or not np.isfinite(first_pct) or first_pct < entry_min:
            i += 1
            continue
        if mode == "LONG" and first_dir != 1:
            i += 1
            continue
        if mode == "SHORT" and first_dir != -1:
            i += 1
            continue

        pred_dir, _pred_close, _ = iterative_prediction(df, i, horizon)
        if pred_dir == 0 or not np.isfinite(_pred_close):
            i += 1
            continue

        entry = float(df.iloc[i].close)
        actual_exit = float(df.iloc[i + horizon].close)
        trade_dir = first_dir

        # Only enter when the recursive prediction keeps the original signal direction.
        if pred_dir != trade_dir:
            i += 1
            continue

        gross_ret = (
            (actual_exit - entry) / entry
            if trade_dir == 1
            else (entry - actual_exit) / entry
        )
        capital *= max((1.0 + gross_ret) * fee_factor, 0.0)
        trades += 1
        wins += int(gross_ret > 0)
        gross_sum += gross_ret

        # One active trade per symbol; ignore signals inside the holding window.
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
        run_symbol(
            loaded[s],
            initial_capital,
            fee,
            horizon,
            entry_min,
            mode,
        )
        for s in symbols
    ]

    final = sum(float(r["final"]) for r in rs)
    trades = sum(int(r["trades"]) for r in rs)
    wins = sum(
        int(round(float(r["win_rate"]) * int(r["trades"])))
        for r in rs
    )
    gross = sum(
        float(r["avg_gross_trade"]) * int(r["trades"])
        for r in rs
    )
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
        description="Recursive predicted-candle horizon backtest with 0.8% entry filter."
    )
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
    print("ITERATIVE PREDICTED-CANDLE BACKTEST")
    print("=" * 104)
    print("Entry direction: real current candle SCORE 0=SHORT / SCORE 3=LONG")
    print(f"Entry magnitude filter: >= {args.entry_min * 100:.2f}%")
    print("Prediction: each predicted candle is the only input to the next prediction")
    print("Synthetic candle: preserve normalized OHLC shape and rescale trade-side magnitude")
    print("Decision filter: final recursive predicted direction must agree with entry direction")
    print("Entry: current close | Exit/P&L: actual close at +H")
    print("Signal exits: disabled | one active trade per symbol")
    print()

    for h in horizons:
        print(f"HORIZON = {h}")
        for mode in ("SHORT", "LONG", "BOTH"):
            print(mode)
            print("fee      final       return      trades   win%   avg_gross")
            for fee in (0.0, args.fee):
                r = pooled(
                    loaded,
                    symbols,
                    args.initial_capital,
                    fee,
                    h,
                    args.entry_min,
                    mode,
                )
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
