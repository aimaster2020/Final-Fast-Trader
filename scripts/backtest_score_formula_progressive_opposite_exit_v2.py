from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_NOISE = "0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.75,1.00"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=lambda c: str(c).strip().lower() in {"open", "high", "low", "close"},
    )
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

    # Current direction rule.
    direction = np.where(score <= 1, -1, 1).astype(int)
    direction[(score == 1) & (formula == 1)] = 0
    direction[(score == 2) & (formula == -1)] = 0

    # Magnitude law for the predicted next close move.
    # LONG prediction  -> |H-O|
    # SHORT prediction -> |O-L|
    magnitude = np.where(direction == 1, upper, np.where(direction == -1, lower, np.nan))

    df["body"] = body
    df["direction"] = direction
    df["magnitude"] = magnitude
    return df


def run_symbol(
    df: pd.DataFrame,
    initial_capital: float,
    fee_each_side: float,
    noise_fraction: float,
) -> dict[str, float | int]:
    capital = float(initial_capital)
    fee_factor = (1.0 - fee_each_side) ** 2

    trades = wins = forced_third = first_ignored = 0
    hold_sum = 0
    exit_count = 0

    i = 0
    n = len(df)
    while i < n - 1:
        signal = df.iloc[i]
        trade_dir = int(signal.direction)
        signal_body = abs(float(signal.body))

        if trade_dir == 0 or not np.isfinite(signal.magnitude) or signal_body <= 0:
            i += 1
            continue

        entry = float(signal.close)
        first_opposite_magnitude: float | None = None
        opposite_count = 0
        last_move_from_entry = signal_body
        exit_i: int | None = None
        exit_reason = ""

        j = i + 1
        while j < n:
            current = df.iloc[j]
            pred_dir = int(current.direction)
            pred_mag = float(current.magnitude) if np.isfinite(current.magnitude) else np.nan

            # Update the movement state from the beginning of the trade.
            # For LONG, positive movement is favorable; for SHORT, negative close movement is favorable.
            if trade_dir == 1:
                move_from_entry = float(current.close) - entry
            else:
                move_from_entry = entry - float(current.close)
            last_move_from_entry = max(abs(move_from_entry), signal_body)

            if pred_dir == 0 or not np.isfinite(pred_mag):
                j += 1
                continue

            if pred_dir == trade_dir:
                # Direction agrees with the open trade: keep the trade alive and clear the opposite streak.
                opposite_count = 0
                first_opposite_magnitude = None
                j += 1
                continue

            # Opposite prediction: its magnitude is the evidence for a possible reversal.
            opposite_count += 1

            if opposite_count == 1:
                first_opposite_magnitude = pred_mag
                # First opposite signal is treated as noise when its predicted move is below
                # the noise fraction of the movement accumulated since entry.
                if pred_mag <= noise_fraction * last_move_from_entry:
                    first_ignored += 1
                j += 1
                continue

            if opposite_count == 2:
                first = max(float(first_opposite_magnitude or 0.0), 1e-12)
                # Second consecutive opposite signal must overcome BOTH:
                #   1) two times the first opposite signal magnitude
                #   2) two times the configured noise fraction of the current movement
                second_threshold = max(2.0 * first, 2.0 * noise_fraction * last_move_from_entry)
                if pred_mag >= second_threshold:
                    exit_i = j
                    exit_reason = "second_opposite_2x"
                    break
                j += 1
                continue

            # Third consecutive opposite prediction is a mandatory exit.
            exit_i = j
            exit_reason = "third_opposite"
            forced_third += 1
            break

        # No exit signal before the end of data: do not manufacture an exit.
        if exit_i is None:
            break

        exit_price = float(df.iloc[exit_i].close)
        if trade_dir == 1:
            gross_ret = (exit_price - entry) / entry
        else:
            gross_ret = (entry - exit_price) / entry

        capital *= max((1.0 + gross_ret) * fee_factor, 0.0)
        trades += 1
        wins += int(gross_ret > 0)
        hold_sum += exit_i - i
        exit_count += 1

        # Resume only after the exit candle; one position at a time.
        i = exit_i + 1

    return {
        "final": capital,
        "return": capital / initial_capital - 1.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "avg_hold": hold_sum / exit_count if exit_count else 0.0,
        "third_exit_pct": forced_third / exit_count if exit_count else 0.0,
        "first_ignored": first_ignored,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Correct progressive opposite-signal exit using the magnitude law."
    )
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=0.0013, help="Commission per side; 0.0013 = 0.13%.")
    ap.add_argument("--noise-pcts", default=DEFAULT_NOISE)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    noise_pcts = sorted({float(x) for x in args.noise_pcts.split(",") if x.strip()})
    root = Path(args.data_dir)

    print("=" * 116)
    print("REAL CAPITAL BACKTEST: PROGRESSIVE OPPOSITE EXIT V2")
    print("=" * 116)
    print("Direction: Score 0/1 SHORT, Score 2/3 LONG; drop S1+FormulaLong and S2+FormulaShort")
    print("Magnitude: LONG opposite strength=|H-O| ; SHORT opposite strength=|O-L|")
    print("State: movement-from-entry is updated on every candle")
    print("Exit: 1st opposite=store/check noise; 2nd consecutive=needs 2x first + noise; 3rd=mandatory")
    print(f"Initial capital per symbol: {args.initial_capital:.2f}")
    print(f"Noise fractions: {','.join(f'{x:g}' for x in noise_pcts)}")
    print()

    pooled_initial = args.initial_capital * len(symbols)

    for noise in noise_pcts:
        print(f"NOISE FRACTION = {noise:.2f}")
        print("symbol   final       return    trades   win%   avg_hold  third_exit%")

        for fee in (0.0, args.fee):
            print(f"  FEE EACH SIDE = {fee * 100:.2f}% | ROUND TRIP = {fee * 200:.2f}%")
            pooled_final = 0.0
            pooled_trades = 0
            pooled_wins = 0

            for symbol in symbols:
                result = run_symbol(
                    load(root / f"{symbol}_1h.csv"),
                    args.initial_capital,
                    fee,
                    noise,
                )
                print(
                    f"  {symbol:8} {result['final']:10.2f}  {result['return']*100:+8.2f}%"
                    f"  {result['trades']:6d}  {result['win_rate']*100:6.2f}%"
                    f"  {result['avg_hold']:9.2f}  {result['third_exit_pct']*100:10.2f}%"
                )
                pooled_final += float(result["final"])
                pooled_trades += int(result["trades"])
                pooled_wins += int(round(float(result["win_rate"]) * int(result["trades"])))

            pooled_return = pooled_final / pooled_initial - 1.0
            pooled_win = pooled_wins / pooled_trades if pooled_trades else 0.0
            print(
                f"  POOLED   {pooled_final:10.2f}  {pooled_return*100:+8.2f}%"
                f"  {pooled_trades:6d}  {pooled_win*100:6.2f}%"
            )
        print()

    print("=" * 116)


if __name__ == "__main__":
    main()
