from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
FEE = 0.0013
NOISE = 1.0
CANDIDATE_ENTRIES = [0.0075, 0.0077, 0.0079, 0.0081, 0.0083, 0.0085, 0.0087, 0.0090]
MONTHS = ["2026-05", "2026-06", "2026-07", "2026-08"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"{path}: missing {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "month" in df.columns:
        month = df["month"].astype(str)
    elif "timestamp" in df.columns:
        month = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", utc=True).dt.strftime("%Y-%m")
    else:
        raise ValueError(f"{path}: missing month/timestamp")

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    month = month.loc[df.index].reset_index(drop=True)

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
    df["direction"] = direction
    df["magnitude_pct"] = magnitude / df.close
    df["body_pct"] = body.abs() / df.close
    df["month"] = month.values
    return df


def run_window(df: pd.DataFrame, entry_min: float, start_month: str, end_month: str, initial: float = 1000.0) -> tuple[float, int, float]:
    fee_factor = (1.0 - FEE) ** 2
    cap = initial
    trades = wins = 0
    i = 0
    while i < len(df) - 1:
        month = str(df.iloc[i].month)
        if month < start_month or month > end_month:
            i += 1
            continue

        row = df.iloc[i]
        direction = int(row.direction)
        expected = float(row.magnitude_pct)
        body = float(row.body_pct)
        if direction == 0 or not np.isfinite(expected) or expected < entry_min or body <= 0:
            i += 1
            continue

        entry = float(row.close)
        opposite_count = 0
        first_opposite = 0.0
        exit_i = None
        j = i + 1
        while j < len(df):
            current = df.iloc[j]
            pred_dir = int(current.direction)
            pred_mag = float(current.magnitude_pct) if np.isfinite(current.magnitude_pct) else np.nan
            if pred_dir == 0 or not np.isfinite(pred_mag):
                j += 1
                continue
            move_pct = abs(float(current.close) - entry) / max(abs(entry), 1e-12)
            reference = max(move_pct, body, 1e-12)
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
        cap *= max((1.0 + gross) * fee_factor, 0.0)
        trades += 1
        wins += gross > 0
        i = exit_i + 1
    return cap, trades, (wins / trades * 100 if trades else 0.0)


def pooled(loaded: dict[str, pd.DataFrame], symbols: list[str], entry: float, start: str, end: str, initial: float = 1000.0) -> tuple[float, int, float]:
    finals = []
    total_trades = 0
    total_wins = 0.0
    for s in symbols:
        final, trades, win = run_window(loaded[s], entry, start, end, initial)
        finals.append(final)
        total_trades += trades
        total_wins += trades * win / 100.0
    pooled_final = sum(finals)
    pooled_initial = initial * len(symbols)
    pooled_win = total_wins / total_trades * 100 if total_trades else 0.0
    return pooled_final, total_trades, pooled_win


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    root = Path(args.data_dir)
    loaded = {s: load(root / f"{s}_1h.csv") for s in symbols}

    print("=" * 96)
    print("MONTHLY WALK-FORWARD TEST — BASELINE ENTRY THRESHOLD")
    print("Noise=1.00 | Fee=0.13%/side | Candidate entry thresholds selected only from prior months")
    print("Train -> next-month OOS; no parameter selection using the OOS month")
    print("=" * 96)

    for test_month in MONTHS[1:]:
        idx = MONTHS.index(test_month)
        train_start = MONTHS[0]
        train_end = MONTHS[idx - 1]
        best_entry = None
        best_return = -float("inf")
        best_trades = 0

        for entry in CANDIDATE_ENTRIES:
            final, trades, _ = pooled(loaded, symbols, entry, train_start, train_end, args.initial_capital)
            ret = final / (args.initial_capital * len(symbols)) - 1.0
            if ret > best_return:
                best_return = ret
                best_entry = entry
                best_trades = trades

        oos_final, oos_trades, oos_win = pooled(loaded, symbols, best_entry, test_month, test_month, args.initial_capital)
        oos_ret = oos_final / (args.initial_capital * len(symbols)) - 1.0

        print(f"\nTRAIN {train_start}..{train_end} -> OOS {test_month}")
        print(f"selected entry={best_entry*100:.2f}% | train return={best_return*100:+.2f}% | train trades={best_trades}")
        print(f"OOS return={oos_ret*100:+.2f}% | OOS trades={oos_trades} | OOS win={oos_win:.1f}% | OOS final=${oos_final:.2f}")

    print("\nNOTE: With only four months, this is a stability check, not a statistically strong validation.")


if __name__ == "__main__":
    main()
