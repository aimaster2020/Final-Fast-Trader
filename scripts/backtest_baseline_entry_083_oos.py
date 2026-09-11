from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
ENTRY_MIN = 0.0083
NOISE = 1.0
FEE = 0.0013


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"{path}: missing {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

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
    df["body"] = body
    df["direction"] = direction
    df["magnitude_pct"] = magnitude / df.close
    return df


def run_oos(df: pd.DataFrame, initial_capital: float, split: float) -> tuple[float, int, float, float, int, int, int, int]:
    start = max(1, int(len(df) * (1.0 - split)))
    capital = initial_capital
    peak = capital
    max_dd = 0.0
    trades = wins = third_exits = 0
    hold_sum = 0
    i = start

    while i < len(df) - 1:
        row = df.iloc[i]
        trade_dir = int(row.direction)
        expected_pct = float(row.magnitude_pct)
        signal_body = abs(float(row.body))
        if trade_dir == 0 or not np.isfinite(expected_pct) or expected_pct < ENTRY_MIN or signal_body <= 0:
            i += 1
            continue

        entry = float(row.close)
        opposite_count = 0
        first_opposite_strength = 0.0
        exit_i = None
        j = i + 1

        while j < len(df):
            current = df.iloc[j]
            pred_dir = int(current.direction)
            pred_mag = float(current.magnitude_pct) * float(current.close) if np.isfinite(current.magnitude_pct) else np.nan
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
                if opposite_strength_pct <= NOISE * reference_pct:
                    j += 1
                    continue
                exit_i = j
                break
            if opposite_count == 2:
                if opposite_strength_pct < 2.0 * first_opposite_strength or opposite_strength_pct <= 2.0 * NOISE * reference_pct:
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
        gross = (exit_price - entry) / entry if trade_dir == 1 else (entry - exit_price) / entry
        ret = (1.0 + gross) * ((1.0 - FEE) ** 2) - 1.0
        capital *= max(1.0 + ret, 0.0)
        trades += 1
        wins += int(gross > 0)
        hold_sum += exit_i - i
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak if peak > 0 else 0.0)
        i = exit_i + 1

    return capital, trades, (wins / trades if trades else 0.0), max_dd, start, len(df) - start, third_exits, (hold_sum / trades if trades else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--oos-fraction", type=float, default=0.30)
    args = ap.parse_args()
    if not 0 < args.oos_fraction < 1:
        raise SystemExit("--oos-fraction must be between 0 and 1")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    root = Path(args.data_dir)
    pooled_initial = args.initial_capital * len(symbols)
    pooled_final = 0.0
    pooled_trades = pooled_wins = 0
    pooled_dd = 0.0

    print("=" * 112)
    print("FIXED 0.83% BASELINE OUT-OF-SAMPLE BACKTEST")
    print("Entry min=0.83% | Noise=1.00 | Fee=0.13%/side | Progressive exit preserved")
    print(f"OOS = last {args.oos_fraction*100:.0f}% of each symbol; no parameter re-selection")
    print("=" * 112)

    for symbol in symbols:
        df = load(root / f"{symbol}_1h.csv")
        final, trades, win, dd, start, oos_n, third, avg_hold = run_oos(df, args.initial_capital, args.oos_fraction)
        ret = (final / args.initial_capital - 1.0) * 100
        pooled_final += final
        pooled_trades += trades
        pooled_wins += int(round(trades * win))
        pooled_dd = max(pooled_dd, dd)
        print(f"{symbol}: OOS rows={oos_n:4d} capital=${final:,.2f} return={ret:7.2f}% trades={trades:4d} win={win*100:5.1f}% DD={dd*100:6.2f}% avg_hold={avg_hold:5.1f} third_exit={third:3d}")

    pooled_ret = (pooled_final / pooled_initial - 1.0) * 100
    pooled_win = pooled_wins / pooled_trades * 100 if pooled_trades else 0.0
    print("-" * 112)
    print(f"POOLED OOS: ${pooled_final:,.2f} from ${pooled_initial:,.2f} return={pooled_ret:7.2f}% trades={pooled_trades:4d} win={pooled_win:5.1f}% max_symbol_DD={pooled_dd*100:6.2f}%")


if __name__ == "__main__":
    main()
