from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_THRESHOLDS = "0.00,0.05,0.10,0.15,0.20,0.25,0.30"


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

    # Exact direction score used throughout the current strategy.
    df["score"] = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)

    upper = (df.high - df.open).abs()
    lower = (df.open - df.low).abs()
    formula = np.where(lower > upper, 1, np.where(upper > lower, -1, 0))

    # Current best direction rule.
    direction = np.where(df.score <= 1, -1, 1).astype(int)
    direction[(df.score == 1) & (formula == 1)] = 0
    direction[(df.score == 2) & (formula == -1)] = 0
    df["direction"] = direction

    df["body_abs"] = body.abs()
    return df


def simulate(df: pd.DataFrame, initial_capital: float, fee_each_side: float, zero_threshold: float, exit_on_sign_flip: bool) -> dict[str, float | int]:
    capital = float(initial_capital)
    trades = wins = losses = exits_zero = exits_flip = 0
    holding_bars_total = 0
    fee_factor = (1.0 - fee_each_side) ** 2

    i = 0
    n = len(df)
    while i < n - 1:
        row = df.iloc[i]
        direction = int(row.direction)
        if direction == 0:
            i += 1
            continue

        entry = float(row.close)
        entry_direction = direction
        exit_i = None
        exit_price = None
        exit_reason = None

        j = i + 1
        while j < n:
            cur = df.iloc[j]
            body_abs = float(cur.body_abs)
            cur_body = float(cur.close - cur.open)

            near_zero = body_abs <= zero_threshold * max(abs(float(row.close - row.open)), 1e-12)
            sign_flip = (entry_direction == 1 and cur_body < 0) or (entry_direction == -1 and cur_body > 0)

            if near_zero:
                exit_i = j
                exit_price = float(cur.close)
                exit_reason = "zero"
                break
            if exit_on_sign_flip and sign_flip:
                exit_i = j
                exit_price = float(cur.close)
                exit_reason = "flip"
                break
            j += 1

        if exit_i is None:
            # Dataset end is the only forced exit; no fixed take-profit is used.
            exit_i = n - 1
            exit_price = float(df.iloc[-1].close)
            exit_reason = "end"

        if entry_direction == 1:
            gross_ret = (exit_price - entry) / entry
        else:
            gross_ret = (entry - exit_price) / entry

        capital *= max((1.0 + gross_ret) * fee_factor, 0.0)
        trades += 1
        wins += int(gross_ret > 0)
        losses += int(gross_ret < 0)
        exits_zero += int(exit_reason == "zero")
        exits_flip += int(exit_reason == "flip")
        holding_bars_total += exit_i - i

        # No overlapping positions.
        i = exit_i + 1

    return {
        "final": capital,
        "return": capital / initial_capital - 1.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "loss_rate": losses / trades if trades else 0.0,
        "zero_exits": exits_zero,
        "flip_exits": exits_flip,
        "avg_hold": holding_bars_total / trades if trades else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Capital backtest with body-near-zero exits; no price target exit.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=0.0013)
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--exit-on-sign-flip", action="store_true")
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    thresholds = sorted({float(x) for x in args.thresholds.split(",") if x.strip()})
    root = Path(args.data_dir)
    frames = {s: load(root / f"{s}_1h.csv") for s in symbols}

    print("=" * 116)
    print("REAL CAPITAL BACKTEST: SCORE + FORMULA DIRECTION + BODY-NEAR-ZERO EXIT")
    print("=" * 116)
    print("Direction: Score 0/1 SHORT, Score 2/3 LONG; drop S1+FormulaLong and S2+FormulaShort")
    print("Exit: body size approaches zero; NO price-target exit")
    print("Threshold = |current candle body| <= threshold × |signal candle body|")
    print(f"Sign-flip exit: {'ON' if args.exit_on_sign_flip else 'OFF'}")
    print(f"Initial capital per symbol: {args.initial_capital:.2f}")
    print()

    for threshold in thresholds:
        print(f"THRESHOLD = {threshold:.2f}")
        print("symbol   final       return    trades   win%   avg_hold")
        pooled_final = 0.0
        pooled_initial = args.initial_capital * len(symbols)
        pooled_trades = pooled_wins = 0
        pooled_hold = 0.0

        for fee in (0.0, args.fee):
            print(f"  FEE EACH SIDE = {fee * 100:.2f}% | ROUND TRIP = {fee * 200:.2f}%")
            pooled_final = pooled_wins = pooled_trades = 0
            pooled_hold = 0.0
            for symbol in symbols:
                r = simulate(frames[symbol], args.initial_capital, fee, threshold, args.exit_on_sign_flip)
                print(f"  {symbol:8} {r['final']:10.2f}  {r['return']*100:+8.2f}%  {r['trades']:6d}  {r['win_rate']*100:6.2f}%  {r['avg_hold']:8.2f}")
                pooled_final += float(r["final"])
                pooled_trades += int(r["trades"])
                pooled_wins += int(round(float(r["win_rate"]) * int(r["trades"])))
                pooled_hold += float(r["avg_hold"]) * int(r["trades"])
            pooled_return = pooled_final / pooled_initial - 1.0
            pooled_win = pooled_wins / pooled_trades if pooled_trades else 0.0
            pooled_avg_hold = pooled_hold / pooled_trades if pooled_trades else 0.0
            print(f"  POOLED   {pooled_final:10.2f}  {pooled_return*100:+8.2f}%  {pooled_trades:6d}  {pooled_win*100:6.2f}%  {pooled_avg_hold:8.2f}")
        print()

    print("=" * 116)


if __name__ == "__main__":
    main()
