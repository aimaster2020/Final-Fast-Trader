from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
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

    # Magnitude law: predicted next body move magnitude by direction.
    magnitude = np.where(direction == 1, upper, np.where(direction == -1, lower, np.nan))

    df["body"] = body
    df["direction"] = direction
    df["magnitude"] = magnitude
    return df


def run_symbol(df: pd.DataFrame, initial_capital: float, fee_each_side: float, noise_pct: float) -> dict[str, float | int]:
    capital = float(initial_capital)
    fee_factor = (1.0 - fee_each_side) ** 2
    trades = wins = exits = forced_third = 0
    hold_sum = 0

    i = 0
    n = len(df)
    while i < n - 1:
        row = df.iloc[i]
        trade_dir = int(row.direction)
        if trade_dir == 0 or not np.isfinite(row.magnitude) or abs(float(row.body)) <= 0:
            i += 1
            continue

        entry = float(row.close)
        signal_body = abs(float(row.body))
        opposite_count = 0
        first_opposite_strength = None
        exit_i = None
        exit_reason = ""

        j = i + 1
        while j < n:
            current = df.iloc[j]
            pred_dir = int(current.direction)
            pred_mag = float(current.magnitude) if np.isfinite(current.magnitude) else np.nan

            if pred_dir == 0 or not np.isfinite(pred_mag):
                j += 1
                continue

            if pred_dir == trade_dir:
                # Same-direction prediction: trade continues and the movement state is refreshed.
                opposite_count = 0
                first_opposite_strength = None
                j += 1
                continue

            # Opposite prediction. Compare predicted opposite movement to the actual movement from entry.
            favorable_move = abs(float(current.close) - entry)
            reference = max(favorable_move, signal_body, 1e-12)
            ratio = pred_mag / reference

            if opposite_count == 0:
                first_opposite_strength = pred_mag
                opposite_count = 1
                # First opposite signal is noise when it is small relative to movement so far.
                if ratio <= noise_pct:
                    j += 1
                    continue
                exit_i = j
                exit_reason = "first_strong_opposite"
                break

            if opposite_count == 1:
                opposite_count = 2
                first = max(float(first_opposite_strength or 0.0), 1e-12)
                # Second consecutive opposite signal must be at least 2x the first opposite signal,
                # after applying the noise percentage. Otherwise it is still treated as noise.
                threshold_amount = 2.0 * first
                if pred_mag <= threshold_amount or pred_mag / reference <= 2.0 * noise_pct:
                    j += 1
                    continue
                exit_i = j
                exit_reason = "second_strong_opposite"
                break

            # Third consecutive opposite signal => mandatory exit.
            exit_i = j
            exit_reason = "third_opposite"
            forced_third += 1
            break

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
        exits += 1
        hold_sum += exit_i - i
        i = exit_i + 1

    return {
        "final": capital,
        "return": capital / initial_capital - 1.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "avg_hold": hold_sum / exits if exits else 0.0,
        "third_exit_pct": forced_third / exits if exits else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest progressive opposite-signal exit rule.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=0.0013)
    ap.add_argument("--noise-pcts", default=DEFAULT_NOISE)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    noise_pcts = sorted({float(x) for x in args.noise_pcts.split(",") if x.strip()})
    root = Path(args.data_dir)

    print("=" * 110)
    print("REAL CAPITAL BACKTEST: PROGRESSIVE OPPOSITE-SIGNAL EXIT")
    print("=" * 110)
    print("Direction: Score 0/1 SHORT, Score 2/3 LONG; drop S1+FormulaLong and S2+FormulaShort")
    print("Magnitude: opposite signal strength = |H-O| for LONG prediction, |O-L| for SHORT prediction")
    print("Exit logic: same direction => continue; first opposite => noise test; second consecutive => 2x first; third => mandatory exit")
    print("Noise threshold is tested as fraction of movement from entry (with signal body as a minimum reference).")
    print(f"Initial capital per symbol: {args.initial_capital:.2f}")
    print(f"Noise fractions: {','.join(f'{x:g}' for x in noise_pcts)}")
    print()

    for noise in noise_pcts:
        print(f"NOISE FRACTION = {noise:.2f}")
        print("symbol   final       return    trades   win%   avg_hold   third_exit%")
        for fee in (0.0, args.fee):
            print(f"  FEE EACH SIDE = {fee * 100:.2f}% | ROUND TRIP = {fee * 200:.2f}%")
            pooled_final = 0.0
            pooled_initial = args.initial_capital * len(symbols)
            pooled_trades = 0
            pooled_wins = 0
            for symbol in symbols:
                r = run_symbol(load(root / f"{symbol}_1h.csv"), args.initial_capital, fee, noise)
                print(f"  {symbol:8} {r['final']:10.2f}  {r['return']*100:+8.2f}%  {r['trades']:6d}  {r['win_rate']*100:6.2f}%  {r['avg_hold']:9.2f}  {r['third_exit_pct']*100:10.2f}%")
                pooled_final += float(r["final"])
                pooled_trades += int(r["trades"])
                pooled_wins += int(round(float(r["win_rate"]) * int(r["trades"])))
            pooled_return = pooled_final / pooled_initial - 1.0
            pooled_win = pooled_wins / pooled_trades if pooled_trades else 0.0
            print(f"  POOLED   {pooled_final:10.2f}  {pooled_return*100:+8.2f}%  {pooled_trades:6d}  {pooled_win*100:6.2f}%")
        print()

    print("=" * 110)


if __name__ == "__main__":
    main()
