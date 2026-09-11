from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
THRESHOLDS = [0.00065, 0.0013, 0.00195, 0.0026, 0.0039, 0.0052, 0.0065, 0.0075, 0.0080, 0.0083, 0.0085]
NOISE = 1.0
FEE = 0.0013


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
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


def run(df: pd.DataFrame, initial_capital: float, threshold: float) -> tuple[float, int, float]:
    capital = initial_capital
    factor = (1.0 - FEE) ** 2
    trades = wins = 0
    i = 0
    while i < len(df) - 1:
        row = df.iloc[i]
        direction = int(row.direction)
        expected = float(row.magnitude_pct) if np.isfinite(row.magnitude_pct) else np.nan
        body = abs(float(row.body))
        if direction == 0 or not np.isfinite(expected) or expected < threshold or body <= 0:
            i += 1
            continue

        entry = float(row.close)
        first_strength = 0.0
        opposite_count = 0
        exit_i = None
        j = i + 1
        while j < len(df):
            cur = df.iloc[j]
            pred_dir = int(cur.direction)
            pred_mag = float(cur.magnitude_pct) if np.isfinite(cur.magnitude_pct) else np.nan
            if pred_dir == 0 or not np.isfinite(pred_mag):
                j += 1
                continue
            move_pct = abs(float(cur.close) - entry) / entry
            reference = max(move_pct, body / max(abs(entry), 1e-12), 1e-12)
            if pred_dir == direction:
                opposite_count = 0
                first_strength = 0.0
                j += 1
                continue
            opposite_count += 1
            if opposite_count == 1:
                first_strength = pred_mag
                if pred_mag <= NOISE * reference:
                    j += 1
                    continue
                exit_i = j
                break
            if opposite_count == 2:
                if pred_mag < 2.0 * first_strength or pred_mag <= 2.0 * NOISE * reference:
                    j += 1
                    continue
                exit_i = j
                break
            exit_i = j
            break
        if exit_i is None:
            break
        exit_price = float(df.iloc[exit_i].close)
        gross = ((exit_price - entry) / entry) if direction == 1 else ((entry - exit_price) / entry)
        capital *= max((1.0 + gross) * factor, 0.0)
        trades += 1
        wins += int(gross > 0)
        i = exit_i + 1
    return capital, trades, wins / trades * 100 if trades else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    args = ap.parse_args()
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    root = Path(args.data_dir)
    loaded = {s: load(root / f"{s}_1h.csv") for s in symbols}

    pooled_initial = args.initial_capital * len(symbols)
    print("=" * 100)
    print("BASELINE ENTRY THRESHOLDS TIED TO FEE COVERAGE")
    print("Noise=1.00 | Fee=0.13%/side | Progressive exit preserved | Entry threshold is expected movement / Close")
    print("0.065%=half-side fee | 0.13%=one-side fee | 0.26%=round-trip fee")
    print("=" * 100)
    print(" THRESHOLD    FINAL      RETURN    TRADES    WIN")
    for th in THRESHOLDS:
        finals = []
        trades = 0
        wins_weighted = 0.0
        for s in symbols:
            final, n, win = run(loaded[s], args.initial_capital, th)
            finals.append(final)
            trades += n
            wins_weighted += n * win / 100.0
        pooled_final = sum(finals)
        pooled_return = pooled_final / pooled_initial - 1.0
        pooled_win = wins_weighted / trades * 100 if trades else 0.0
        print(f"   {th*100:6.3f}%   ${pooled_final:8.2f}   {pooled_return*100:+7.2f}%   {trades:7d}   {pooled_win:6.1f}%")


if __name__ == "__main__":
    main()
