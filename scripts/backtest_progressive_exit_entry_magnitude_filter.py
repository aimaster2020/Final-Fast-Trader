from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_ENTRY_FRACTIONS = "0,0.00065,0.0013"
DEFAULT_NOISE = "0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.75,1.00"


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

    # Magnitude law: expected price movement in the predicted direction.
    magnitude = np.where(direction == 1, upper, np.where(direction == -1, lower, np.nan))
    magnitude_pct = magnitude / df.close

    df["body"] = body
    df["direction"] = direction
    df["magnitude"] = magnitude
    df["magnitude_pct"] = magnitude_pct
    return df


def run_symbol(
    df: pd.DataFrame,
    initial_capital: float,
    fee_each_side: float,
    entry_min_fraction: float,
    noise_fraction: float,
    direction_filter: int | None = None,
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

        # Optional direction filter for isolated LONG or SHORT evaluation.
        if direction_filter is not None and trade_dir != direction_filter:
            i += 1
            continue

        # Entry filter: predicted movement must cover at least the requested fraction of price.
        # 0.00065 = 0.065%, 0.0013 = 0.13%.
        if trade_dir == 0 or not np.isfinite(expected_pct) or expected_pct < entry_min_fraction or signal_body <= 0:
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

            # Track actual movement from entry; this is state, not the entry filter.
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
                # First opposing signal is noise when it is small relative to the move so far.
                if opposite_strength_pct <= noise_fraction * reference_pct:
                    j += 1
                    continue
                exit_i = j
                break

            if opposite_count == 2:
                # Second opposing signal: require at least 2x first-opposite strength and
                # a second noise threshold. Otherwise the trade continues.
                if opposite_strength_pct < 2.0 * first_opposite_strength or opposite_strength_pct <= 2.0 * noise_fraction * reference_pct:
                    j += 1
                    continue
                exit_i = j
                break

            # Third opposing signal: mandatory exit.
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


def pooled_result(
    loaded: dict[str, pd.DataFrame],
    symbols: list[str],
    initial_capital: float,
    fee: float,
    entry_min: float,
    noise: float,
    direction_filter: int | None,
) -> tuple[float, float, int]:
    pooled_final = 0.0
    pooled_trades = 0
    pooled_initial = initial_capital * len(symbols)

    for symbol in symbols:
        r = run_symbol(
            loaded[symbol],
            initial_capital,
            fee,
            entry_min,
            noise,
            direction_filter=direction_filter,
        )
        pooled_final += float(r["final"])
        pooled_trades += int(r["trades"])

    pooled_return = pooled_final / pooled_initial - 1.0
    return pooled_final, pooled_return, pooled_trades


def print_direction_result(
    label: str,
    loaded: dict[str, pd.DataFrame],
    symbols: list[str],
    initial_capital: float,
    args_fee: float,
    entry_min: float,
    noise: float,
    direction_filter: int | None,
) -> None:
    print(label)
    print("fee      pooled_final   pooled_return   trades")
    for fee in (0.0, args_fee):
        pooled_final, pooled_return, pooled_trades = pooled_result(
            loaded,
            symbols,
            initial_capital,
            fee,
            entry_min,
            noise,
            direction_filter,
        )
        print(f"{fee*100:5.2f}%    {pooled_final:12.2f}   {pooled_return*100:+10.2f}%   {pooled_trades:6d}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest entry magnitude filter + progressive opposite-signal exit, split by direction.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=0.0013, help="Commission per side; 0.0013 = 0.13%.")
    ap.add_argument("--entry-fractions", default=DEFAULT_ENTRY_FRACTIONS, help="Minimum predicted move as fraction of price.")
    ap.add_argument("--noise-fractions", default=DEFAULT_NOISE)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    entry_fractions = sorted({float(x) for x in args.entry_fractions.split(",") if x.strip()})
    noise_fractions = sorted({float(x) for x in args.noise_fractions.split(",") if x.strip()})
    root = Path(args.data_dir)
    loaded = {symbol: load(root / f"{symbol}_1h.csv") for symbol in symbols}

    print("=" * 112)
    print("REAL CAPITAL BACKTEST: ENTRY MAGNITUDE FILTER + PROGRESSIVE OPPOSITE EXIT")
    print("=" * 112)
    print("Direction: Score 0/1 SHORT, Score 2/3 LONG; drop S1+FormulaLong and S2+FormulaShort")
    print("Magnitude: LONG=|H-O| ; SHORT=|O-L|")
    print("Entry filter: predicted move / entry price must be >= threshold")
    print("0.00065 = 0.065% ; 0.0013 = 0.13%")
    print("Exit: same direction continues; 1st opposite=noise test; 2nd requires 2x first; 3rd mandatory")
    print("Direction split: LONG and SHORT are evaluated independently with the same per-symbol initial capital")
    print(f"Initial capital per symbol: {args.initial_capital:.2f}")
    print()

    for entry_min in entry_fractions:
        for noise in noise_fractions:
            print(f"ENTRY MIN = {entry_min * 100:.3f}% | NOISE = {noise:.2f}")
            print_direction_result(
                "LONG",
                loaded,
                symbols,
                args.initial_capital,
                args.fee,
                entry_min,
                noise,
                direction_filter=1,
            )
            print_direction_result(
                "SHORT",
                loaded,
                symbols,
                args.initial_capital,
                args.fee,
                entry_min,
                noise,
                direction_filter=-1,
            )
            print_direction_result(
                "ALL",
                loaded,
                symbols,
                args.initial_capital,
                args.fee,
                entry_min,
                noise,
                direction_filter=None,
            )

    print("=" * 112)


if __name__ == "__main__":
    main()
