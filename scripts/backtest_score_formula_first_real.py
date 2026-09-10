from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"


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

    df["score"] = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)

    upper = (df.high - df.open).abs()
    lower = (df.open - df.low).abs()
    df["formula"] = np.where(lower > upper, 1, np.where(upper > lower, -1, 0))

    df["direction"] = np.where(df.score <= 1, -1, 1)
    df.loc[(df.score == 1) & (df.formula == 1), "direction"] = 0
    df.loc[(df.score == 2) & (df.formula == -1), "direction"] = 0

    df["target"] = np.where(
        df.direction == 1,
        df.close + upper,
        np.where(df.direction == -1, df.close - lower, np.nan),
    )
    return df


def one_symbol(df: pd.DataFrame, initial_capital: float, fee_each_side: float) -> dict[str, float | int]:
    capital = float(initial_capital)
    trades = wins = target_hits = 0
    fee_factor = (1.0 - fee_each_side) ** 2

    for i in range(len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]
        direction = int(row.direction)
        if direction == 0 or not np.isfinite(row.target):
            continue

        entry = float(row.close)
        target = float(row.target)
        next_high = float(nxt.high)
        next_low = float(nxt.low)
        next_close = float(nxt.close)

        if direction == 1:
            hit = next_high >= target
            exit_price = target if hit else next_close
            gross_ret = (exit_price - entry) / entry
        else:
            hit = next_low <= target
            exit_price = target if hit else next_close
            gross_ret = (entry - exit_price) / entry

        capital *= max((1.0 + gross_ret) * fee_factor, 0.0)
        trades += 1
        wins += int(gross_ret > 0)
        target_hits += int(hit)

    return {
        "final": capital,
        "return": capital / initial_capital - 1.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "target_hits": target_hits,
        "target_hit_rate": target_hits / trades if trades else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="First real capital backtest: current direction rule + magnitude target rule.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=0.0013, help="Commission per side; 0.0013 = 0.13%.")
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    root = Path(args.data_dir)

    print("=" * 110)
    print("FIRST REAL BACKTEST: DIRECTION + MAGNITUDE")
    print("=" * 110)
    print("Direction: Score 0/1 SHORT, Score 2/3 LONG; drop S1+FormulaLong and S2+FormulaShort")
    print("Magnitude: LONG target=C+|H-O| ; SHORT target=C-|O-L|")
    print("Execution: entry at signal candle Close; if next candle hits target, exit at target; otherwise exit at next Close")
    print(f"Initial capital per symbol: {args.initial_capital:.2f}")
    print()

    for fee in (0.0, args.fee):
        print(f"FEE EACH SIDE = {fee * 100:.2f}% | ROUND TRIP = {fee * 200:.2f}%")
        print("symbol   final       return    trades   win%   target_hit%")
        pooled_final = 0.0
        pooled_initial = args.initial_capital * len(symbols)
        pooled_trades = pooled_wins = pooled_hits = 0

        for symbol in symbols:
            r = one_symbol(load(root / f"{symbol}_1h.csv"), args.initial_capital, fee)
            print(
                f"{symbol:8} {r['final']:10.2f}  {r['return']*100:+8.2f}%"
                f"  {r['trades']:6d}  {r['win_rate']*100:6.2f}%  {r['target_hit_rate']*100:10.2f}%"
            )
            pooled_final += float(r["final"])
            pooled_trades += int(r["trades"])
            pooled_wins += int(round(float(r["win_rate"]) * int(r["trades"])))
            pooled_hits += int(r["target_hits"])

        pooled_return = pooled_final / pooled_initial - 1.0
        pooled_win = pooled_wins / pooled_trades if pooled_trades else 0.0
        pooled_hit = pooled_hits / pooled_trades if pooled_trades else 0.0
        print(
            f"POOLED   {pooled_final:10.2f}  {pooled_return*100:+8.2f}%"
            f"  {pooled_trades:6d}  {pooled_win*100:6.2f}%  {pooled_hit*100:10.2f}%"
        )
        print()

    print("=" * 110)


if __name__ == "__main__":
    main()
