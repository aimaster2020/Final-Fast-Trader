from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_HORIZONS = "1,2,3,4,5"
DEFAULT_ENTRY_MIN = 0.0080


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in {"open", "high", "low", "close"})
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    body = df.close - df.open
    hc = df.high - df.close
    ho = df.high - df.open
    lc = df.low - df.close
    score = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)
    direction = np.where(score == 0, -1, np.where(score == 3, 1, 0)).astype(int)

    long_magnitude = (df.high - df.open).abs()
    short_magnitude = (df.open - df.low).abs()
    magnitude = np.where(direction == 1, long_magnitude, np.where(direction == -1, short_magnitude, np.nan))
    magnitude_pct = magnitude / df.close

    df["body"] = body
    df["score"] = score
    df["direction"] = direction
    df["magnitude_pct"] = magnitude_pct
    return df


def run_symbol(df: pd.DataFrame, initial_capital: float, fee_each_side: float, horizon: int, entry_min_fraction: float, direction_filter: int | None = None) -> dict[str, float | int]:
    capital = float(initial_capital)
    fee_factor = (1.0 - fee_each_side) ** 2
    trades = wins = 0
    gross_sum = 0.0

    n = len(df)
    i = 0
    while i < n - horizon:
        row = df.iloc[i]
        trade_dir = int(row.direction)
        if direction_filter is not None and trade_dir != direction_filter:
            i += 1
            continue
        if trade_dir == 0:
            i += 1
            continue

        expected_pct = float(row.magnitude_pct)
        if not np.isfinite(expected_pct) or expected_pct < entry_min_fraction:
            i += 1
            continue

        entry = float(row.close)
        exit_price = float(df.iloc[i + horizon].close)
        gross_ret = ((exit_price - entry) / entry) if trade_dir == 1 else ((entry - exit_price) / entry)
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


def pooled_result(loaded: dict[str, pd.DataFrame], symbols: list[str], initial_capital: float, fee: float, horizon: int, entry_min_fraction: float, direction_filter: int | None) -> dict[str, float | int]:
    pooled_final = 0.0
    pooled_trades = pooled_wins = 0
    pooled_gross_sum = 0.0
    for symbol in symbols:
        r = run_symbol(loaded[symbol], initial_capital, fee, horizon, entry_min_fraction, direction_filter=direction_filter)
        pooled_final += float(r["final"])
        pooled_trades += int(r["trades"])
        pooled_wins += int(round(float(r["win_rate"]) * int(r["trades"])))
        pooled_gross_sum += float(r["avg_gross_trade"]) * int(r["trades"])
    pooled_initial = initial_capital * len(symbols)
    return {
        "final": pooled_final,
        "return": pooled_final / pooled_initial - 1.0,
        "trades": pooled_trades,
        "win_rate": pooled_wins / pooled_trades if pooled_trades else 0.0,
        "avg_gross_trade": pooled_gross_sum / pooled_trades if pooled_trades else 0.0,
    }


def print_mode(label: str, loaded: dict[str, pd.DataFrame], symbols: list[str], initial_capital: float, args_fee: float, horizon: int, entry_min_fraction: float, direction_filter: int | None) -> None:
    print(label)
    print("fee      final       return      trades   win%   avg_gross")
    for fee in (0.0, args_fee):
        r = pooled_result(loaded, symbols, initial_capital, fee, horizon, entry_min_fraction, direction_filter)
        print(f"{fee * 100:5.2f}%  {float(r['final']):10.2f}  {float(r['return']) * 100:+9.2f}%  {int(r['trades']):6d}  {float(r['win_rate']) * 100:5.1f}%  {float(r['avg_gross_trade']) * 100:+9.4f}%")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fixed-horizon backtest with 0.8% entry magnitude filter and no signal-based exit rules.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=0.0013, help="Commission per side; 0.0013 = 0.13%.")
    ap.add_argument("--horizons", default=DEFAULT_HORIZONS)
    ap.add_argument("--entry-min", type=float, default=DEFAULT_ENTRY_MIN, help="Minimum expected move fraction; 0.008 = 0.8%.")
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    horizons = sorted({int(x) for x in args.horizons.split(",") if x.strip()})
    if any(h < 1 for h in horizons):
        raise ValueError("Horizons must be >= 1.")
    if args.entry_min < 0:
        raise ValueError("entry-min must be >= 0.")

    root = Path(args.data_dir)
    loaded = {symbol: load(root / f"{symbol}_1h.csv") for symbol in symbols}

    print("=" * 104)
    print("FIXED-HORIZON BACKTEST: ENTRY MAGNITUDE FILTER = 0.8% + NO SIGNAL EXIT")
    print("=" * 104)
    print("Direction: SCORE 0 = SHORT, SCORE 3 = LONG; SCORES 1/2 = NO TRADE")
    print("Modes: SHORT ONLY | LONG ONLY | SCORE 0/3 ONLY")
    print("Entry: current candle close")
    print("Exit: close of candle +H")
    print(f"Entry magnitude filter: expected move >= {args.entry_min * 100:.2f}% of current close")
    print("Magnitude: LONG=|H-O| ; SHORT=|O-L|")
    print("Progressive opposite-signal exit: DISABLED")
    print("Open-position rule: one active trade per symbol; signals during a trade are ignored")
    print("Commission: reported at 0% and configured fee per side")
    print(f"Initial capital per symbol: {args.initial_capital:.2f}")
    print()

    for horizon in horizons:
        print(f"HORIZON = {horizon}")
        print_mode("SHORT ONLY", loaded, symbols, args.initial_capital, args.fee, horizon, args.entry_min, -1)
        print_mode("LONG ONLY", loaded, symbols, args.initial_capital, args.fee, horizon, args.entry_min, 1)
        print_mode("SCORE 0/3 ONLY", loaded, symbols, args.initial_capital, args.fee, horizon, args.entry_min, None)

    print("=" * 104)


if __name__ == "__main__":
    main()
