from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
ENTRY_MIN = 0.0083
NOISE = 1.0
FEE = 0.0013
HOLD_MONTHS = ["2026-05", "2026-06", "2026-07", "2026-08"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().lstrip("\ufeff") for c in df.columns]
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"{path}: missing {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "month" in df.columns:
        month = df["month"].astype(str).str.strip()
    elif "timestamp" in df.columns:
        raw_ts = pd.to_numeric(df["timestamp"], errors="coerce")
        unit = "ms" if raw_ts.dropna().median() > 10**11 else "s"
        ts = pd.to_datetime(raw_ts, errors="coerce", unit=unit, utc=True)
        month = ts.dt.strftime("%Y-%m").fillna("")
    elif "datetime" in df.columns:
        ts = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        month = ts.dt.strftime("%Y-%m").fillna("")
    else:
        raise ValueError(f"{path}: need month, timestamp, or datetime column")

    df = df.dropna(subset=["open", "high", "low", "close"])
    month = month.loc[df.index].reset_index(drop=True)
    df = df.reset_index(drop=True)

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

    out = df.copy()
    out["direction"] = direction
    out["magnitude_pct"] = magnitude / out.close
    out["body"] = body
    out["month"] = month
    return out


def run_range(df: pd.DataFrame, month: str, initial: float = 1000.0) -> tuple[float, int, float]:
    fee_factor = (1.0 - FEE) ** 2
    capital = initial
    trades = wins = 0
    i = 0
    n = len(df)
    while i < n - 1:
        row = df.iloc[i]
        if row["month"] != month:
            i += 1
            continue
        direction = int(row.direction)
        expected = float(row.magnitude_pct)
        body = abs(float(row.body))
        if direction == 0 or not np.isfinite(expected) or expected < ENTRY_MIN or body <= 0:
            i += 1
            continue

        entry = float(row.close)
        first_opposite = 0.0
        opposite_count = 0
        exit_i = None
        j = i + 1
        while j < n:
            current = df.iloc[j]
            pred_dir = int(current.direction)
            pred_mag = float(current.magnitude_pct) if np.isfinite(current.magnitude_pct) else np.nan
            if pred_dir == 0 or not np.isfinite(pred_mag):
                j += 1
                continue
            move_pct = abs(float(current.close) - entry) / max(abs(entry), 1e-12)
            reference = max(move_pct, body / max(abs(entry), 1e-12), 1e-12)
            if pred_dir == direction:
                opposite_count = 0
                first_opposite = 0.0
                j += 1
                continue
            opposite_count += 1
            if opposite_count == 1:
                first_opposite = pred_mag
                if pred_mag <= NOISE * reference:
                    j += 1
                    continue
                exit_i = j
                break
            if opposite_count == 2:
                if pred_mag < 2.0 * first_opposite or pred_mag <= 2.0 * NOISE * reference:
                    j += 1
                    continue
                exit_i = j
                break
            exit_i = j
            break

        if exit_i is None:
            i += 1
            continue
        exit_price = float(df.iloc[exit_i].close)
        gross = (exit_price - entry) / entry if direction == 1 else (entry - exit_price) / entry
        capital *= max((1.0 + gross) * fee_factor, 0.0)
        trades += 1
        wins += gross > 0
        i = exit_i + 1
    return capital, trades, wins / trades * 100 if trades else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    args = ap.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    root = Path(args.data_dir)
    loaded = {s: load(root / f"{s}_1h.csv") for s in symbols}

    print("=" * 88)
    print("FIXED 0.83% BASELINE MONTHLY BREAKDOWN")
    print("Entry=0.83% | Noise=1.00 | Fee=0.13%/side | No re-selection")
    print("=" * 88)
    for month in HOLD_MONTHS:
        finals = []
        trades = 0
        wins_weight = 0
        print(f"\n{month}")
        print("SYMBOL   FINAL    RETURN   TRADES  WIN")
        for s in symbols:
            final, t, w = run_range(loaded[s], month, args.initial_capital)
            finals.append(final)
            trades += t
            wins_weight += t * w / 100.0
            print(f"{s:7s} ${final:8.2f} {final/args.initial_capital-1:+8.2%} {t:7d} {w:6.1f}%")
        pooled_initial = args.initial_capital * len(symbols)
        pooled_final = sum(finals)
        pooled_win = wins_weight / trades * 100 if trades else 0.0
        print(f"POOLED   ${pooled_final:8.2f} {pooled_final/pooled_initial-1:+8.2%} {trades:7d} {pooled_win:6.1f}%")


if __name__ == "__main__":
    main()
