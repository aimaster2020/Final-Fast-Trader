from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_ENTRY_FRACTIONS = "0.0075,0.0076,0.0077,0.0078,0.0079,0.0080,0.0081,0.0082,0.0083,0.0084,0.0085,0.0086,0.0087,0.0088,0.0089,0.0090"
DEFAULT_NOISE_FRACTIONS = "0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.75,1.00"
ROOM_THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.035]
FEE_DEFAULT = 0.0013
LOOKBACK = 1


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

    upper = (df.high - df.open).abs()
    lower = (df.open - df.low).abs()
    formula = np.where(lower > upper, 1, np.where(upper > lower, -1, 0))

    direction = np.where(score <= 1, -1, 1).astype(int)
    direction[(score == 1) & (formula == 1)] = 0
    direction[(score == 2) & (formula == -1)] = 0

    magnitude = np.where(direction == 1, upper, np.where(direction == -1, lower, np.nan))
    magnitude_pct = magnitude / df.close

    df["body"] = body
    df["direction"] = direction
    df["magnitude"] = magnitude
    df["magnitude_pct"] = magnitude_pct
    return df


def room_ok(df: pd.DataFrame, i: int, direction: int, threshold: float) -> bool:
    if threshold <= 0:
        return True
    c = float(df.iloc[i].close)
    if c <= 0:
        return False
    prev_high = float(df.iloc[i - LOOKBACK:i].high.max())
    prev_low = float(df.iloc[i - LOOKBACK:i].low.min())
    room = (prev_high - c) / c if direction == 1 else (c - prev_low) / c
    return room >= threshold


def run_symbol(
    df: pd.DataFrame,
    initial_capital: float,
    fee_each_side: float,
    entry_min_fraction: float,
    noise_fraction: float,
    room_threshold: float,
) -> dict[str, float | int]:
    capital = float(initial_capital)
    fee_factor = (1.0 - fee_each_side) ** 2
    trades = wins = third_exits = 0
    hold_sum = 0
    i = 0
    n = len(df)

    while i < n - 1:
        row = df.iloc[i]
        trade_dir = int(row.direction)
        expected_pct = float(row.magnitude_pct)
        signal_body = abs(float(row.body))

        if (
            trade_dir == 0
            or not np.isfinite(expected_pct)
            or expected_pct < entry_min_fraction
            or signal_body <= 0
            or not room_ok(df, i, trade_dir, room_threshold)
        ):
            i += 1
            continue

        entry = float(row.close)
        opposite_count = 0
        first_opposite_strength = 0.0
        exit_i = None

        j = i + 1
        while j < n:
            current = df.iloc[j]
            pred_dir = int(current.direction)
            pred_mag = float(current.magnitude) if np.isfinite(current.magnitude) else np.nan
            if pred_dir == 0 or not np.isfinite(pred_mag):
                j += 1
                continue

            current_move_pct = abs(float(current.close) - entry) / entry
            reference_pct = max(current_move_pct, signal_body / max(abs(entry), 1e-12), 1e-12)

            if pred_dir == trade_dir:
                opposite_count = 0
                first_opposite_strength = 0.0
                j += 1
                continue

            opposite_count += 1
            opposite_strength_pct = pred_mag / max(abs(float(current.close)), 1e-12)

            if opposite_count == 1:
                first_opposite_strength = opposite_strength_pct
                if opposite_strength_pct <= noise_fraction * reference_pct:
                    j += 1
                    continue
                exit_i = j
                break

            if opposite_count == 2:
                if (
                    opposite_strength_pct < 2.0 * first_opposite_strength
                    or opposite_strength_pct <= 2.0 * noise_fraction * reference_pct
                ):
                    j += 1
                    continue
                exit_i = j
                break

            third_exits += 1
            exit_i = j
            break

        if exit_i is None:
            break

        exit_price = float(df.iloc[exit_i].close)
        gross_ret = ((exit_price - entry) / entry) if trade_dir == 1 else ((entry - exit_price) / entry)
        capital *= max((1.0 + gross_ret) * fee_factor, 0.0)
        trades += 1
        wins += int(gross_ret > 0)
        hold_sum += exit_i - i
        i = exit_i + 1

    return {
        "final": capital,
        "return": capital / initial_capital - 1.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "avg_hold": hold_sum / trades if trades else 0.0,
        "third_exit_pct": third_exits / trades if trades else 0.0,
    }


def pooled(
    loaded: dict[str, pd.DataFrame],
    symbols: list[str],
    initial_capital: float,
    fee: float,
    entry_min: float,
    noise: float,
    room: float,
) -> tuple[float, float, int, float]:
    final = 0.0
    trades = 0
    wins = 0.0
    for symbol in symbols:
        r = run_symbol(loaded[symbol], initial_capital, fee, entry_min, noise, room)
        final += float(r["final"])
        trades += int(r["trades"])
        wins += int(r["trades"]) * float(r["win_rate"])
    initial = initial_capital * len(symbols)
    return final, final / initial - 1.0, trades, wins / trades if trades else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare canonical baseline vs same baseline plus previous-extreme room entry filter.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=FEE_DEFAULT)
    ap.add_argument("--entry-fractions", default=DEFAULT_ENTRY_FRACTIONS)
    ap.add_argument("--noise-fractions", default=DEFAULT_NOISE_FRACTIONS)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    entry_fractions = sorted({float(x) for x in args.entry_fractions.split(",") if x.strip()})
    noise_fractions = sorted({float(x) for x in args.noise_fractions.split(",") if x.strip()})
    root = Path(args.data_dir)
    loaded = {s: load(root / f"{s}_1h.csv") for s in symbols}

    best: tuple[float, float, float, int, float, float] | None = None
    for entry_min in entry_fractions:
        for noise in noise_fractions:
            final, ret, trades, win = pooled(loaded, symbols, args.initial_capital, args.fee, entry_min, noise, 0.0)
            candidate = (ret, entry_min, noise, trades, final, win)
            if best is None or candidate[0] > best[0]:
                best = candidate

    assert best is not None
    best_ret, best_entry, best_noise, best_trades, best_final, best_win = best

    print("=" * 112)
    print("BASELINE VS PREVIOUS-EXTREME ROOM FILTER")
    print("Baseline execution logic preserved; only the entry-room condition is added")
    print("Room = previous candle high/low vs current close; Lookback=1; current candle excluded")
    print(f"Fee={args.fee*100:.2f}%/side | Initial capital=${args.initial_capital:.2f}/symbol")
    print("=" * 112)
    print()
    print("BEST BASELINE CONFIG FOUND ON THE SAME DATASET")
    print(f"Entry minimum = {best_entry*100:.3f}% | Noise = {best_noise:.2f}")
    print(f"Baseline: final=${best_final:,.2f} return={best_ret*100:+.2f}% trades={best_trades} win={best_win*100:.1f}%")
    print()
    print("SAME FIXED CONFIG + ROOM FILTER")
    print(" ROOM>=      FINAL       RETURN     TRADES    WIN")
    for room in ROOM_THRESHOLDS:
        final, ret, trades, win = pooled(
            loaded, symbols, args.initial_capital, args.fee, best_entry, best_noise, room
        )
        print(f" {room*100:5.1f}%   ${final:10.2f}   {ret*100:+8.2f}%   {trades:6d}   {win*100:5.1f}%")
    print("=" * 112)


if __name__ == "__main__":
    main()
