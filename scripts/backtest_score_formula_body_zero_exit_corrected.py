from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_THRESHOLDS = "0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50"


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

    # Exact project Score.
    df["score"] = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)

    # Formula direction: LONG when O-L > H-O; SHORT when H-O > O-L.
    upper = (df.high - df.open).abs()
    lower = (df.open - df.low).abs()
    formula = np.where(lower > upper, 1, np.where(upper > lower, -1, 0))

    # Current best direction rule.
    direction = np.where(df.score <= 1, -1, 1).astype(int)
    direction[(df.score == 1) & (formula == 1)] = 0
    direction[(df.score == 2) & (formula == -1)] = 0

    # Current magnitude law, retained as the expected next-close level.
    target = np.where(
        direction == 1,
        df.close + upper,
        np.where(direction == -1, df.close - lower, np.nan),
    )

    df["body"] = body
    df["body_abs"] = body.abs()
    df["direction"] = direction
    df["formula"] = formula
    df["magnitude_target"] = target
    return df


def run_symbol(
    df: pd.DataFrame,
    initial_capital: float,
    fee_each_side: float,
    threshold: float,
) -> dict[str, float | int]:
    capital = float(initial_capital)
    fee_factor = (1.0 - fee_each_side) ** 2

    trades = wins = 0
    hold_sum = 0
    exit_count = 0
    gross_pnl_sum = 0.0

    i = 0
    n = len(df)
    while i < n - 1:
        row = df.iloc[i]
        direction = int(row.direction)
        signal_body = abs(float(row.body))

        if direction == 0 or not np.isfinite(row.magnitude_target) or signal_body <= 0:
            i += 1
            continue

        entry = float(row.close)
        # Magnitude law is recorded for this trade but is NOT an exit condition.
        _magnitude_target = float(row.magnitude_target)

        exit_i = None
        for j in range(i + 1, n):
            current_body = abs(float(df.iloc[j].body))
            if current_body <= threshold * signal_body:
                exit_i = j
                break

        # No body-near-zero event before the end of data: do not force an artificial exit.
        if exit_i is None:
            break

        exit_price = float(df.iloc[exit_i].close)
        if direction == 1:
            gross_ret = (exit_price - entry) / entry
        else:
            gross_ret = (entry - exit_price) / entry

        capital *= max((1.0 + gross_ret) * fee_factor, 0.0)
        trades += 1
        wins += int(gross_ret > 0)
        hold_sum += exit_i - i
        exit_count += 1
        gross_pnl_sum += gross_ret

        # Only one position at a time; resume scanning after the exit candle.
        i = exit_i + 1

    return {
        "final": capital,
        "return": capital / initial_capital - 1.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "avg_hold": hold_sum / exit_count if exit_count else 0.0,
        "gross_avg_trade": gross_pnl_sum / trades if trades else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Corrected real-capital backtest: current direction rule + body-near-zero exit."
    )
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=0.0013, help="Commission per side; 0.0013 = 0.13%.")
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    thresholds = sorted({float(x) for x in args.thresholds.split(",") if x.strip()})
    root = Path(args.data_dir)

    print("=" * 110)
    print("CORRECTED REAL CAPITAL BACKTEST: DIRECTION + MAGNITUDE + BODY-NEAR-ZERO EXIT")
    print("=" * 110)
    print("Direction: Score 0/1 SHORT, Score 2/3 LONG; drop S1+FormulaLong and S2+FormulaShort")
    print("Magnitude: LONG expected next close=C+|H-O| ; SHORT=C-|O-L|")
    print("Exit: ONLY when |current body| <= threshold × |signal body|")
    print("No price-target exit. No forced end-of-data exit. One position at a time.")
    print(f"Initial capital per symbol: {args.initial_capital:.2f}")
    print(f"Thresholds: {','.join(f'{x:g}' for x in thresholds)}")
    print()

    for threshold in thresholds:
        print(f"THRESHOLD = {threshold:.2f}")
        print("symbol   final       return    trades   win%   avg_hold")

        pooled_final_no_fee = 0.0
        pooled_final_fee = 0.0
        pooled_trades = 0
        pooled_wins = 0

        for fee in (0.0, args.fee):
            print(f"  FEE EACH SIDE = {fee * 100:.2f}% | ROUND TRIP = {fee * 200:.2f}%")
            for symbol in symbols:
                result = run_symbol(load(root / f"{symbol}_1h.csv"), args.initial_capital, fee, threshold)
                print(
                    f"  {symbol:8} {result['final']:10.2f}  {result['return']*100:+8.2f}%"
                    f"  {result['trades']:6d}  {result['win_rate']*100:6.2f}%  {result['avg_hold']:9.2f}"
                )
                if fee == 0.0:
                    pooled_final_no_fee += float(result["final"])
                else:
                    pooled_final_fee += float(result["final"])
                pooled_trades += int(result["trades"])
                pooled_wins += int(round(float(result["win_rate"]) * int(result["trades"])))

            pooled_initial = args.initial_capital * len(symbols)
            pooled_final = pooled_final_no_fee if fee == 0.0 else pooled_final_fee
            pooled_return = pooled_final / pooled_initial - 1.0
            print(f"  POOLED   {pooled_final:10.2f}  {pooled_return*100:+8.2f}%")
        print()

    print("=" * 110)


if __name__ == "__main__":
    main()
