from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
NOISE = 1.0
FEE = 0.0013
CANDIDATES = [0.0075, 0.0080, 0.0083, 0.0085, 0.0090]
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
    out["month"] = month.values
    out["direction"] = direction
    out["magnitude_pct"] = magnitude / out.close
    out["body"] = body
    return out


def backtest(df: pd.DataFrame, start_idx: int, end_idx: int, entry_min: float, initial: float) -> tuple[float, int, int]:
    fee_factor = (1.0 - FEE) ** 2
    capital = initial
    trades = wins = 0
    i = start_idx
    n = min(end_idx, len(df) - 1)
    while i < n:
        row = df.iloc[i]
        direction = int(row.direction)
        expected = float(row.magnitude_pct)
        body = abs(float(row.body))
        if direction == 0 or not np.isfinite(expected) or expected < entry_min or body <= 0:
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
            break
        exit_price = float(df.iloc[exit_i].close)
        gross = (exit_price - entry) / entry if direction == 1 else (entry - exit_price) / entry
        capital *= max((1.0 + gross) * fee_factor, 0.0)
        trades += 1
        wins += int(gross > 0)
        i = exit_i + 1
    return capital, trades, wins


def pooled_train(loaded: dict[str, pd.DataFrame], symbols: list[str], end_month_idx: int, entry_min: float, initial: float) -> tuple[float, int, int]:
    months = MONTHS[: end_month_idx + 1]
    finals = []
    trades = wins = 0
    for s in symbols:
        df = loaded[s]
        idx = df.index[df["month"].isin(months)]
        if len(idx) == 0:
            continue
        f, t, w = backtest(df, int(idx.min()), int(idx.max() + 1), entry_min, initial)
        finals.append(f)
        trades += t
        wins += w
    return sum(finals), trades, wins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    args = ap.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    root = Path(args.data_dir)
    loaded = {s: load(root / f"{s}_1h.csv") for s in symbols}
    capital = args.initial_capital * len(symbols)

    print("=" * 96)
    print("COMPOUNDED MONTHLY WALK-FORWARD — BASELINE ENTRY THRESHOLD")
    print("Train -> next month OOS | candidate selected only from prior months | capital compounds")
    print("Noise=1.00 | Fee=0.13%/side | Candidates=0.75,0.80,0.83,0.85,0.90%")
    print("=" * 96)

    cumulative_trades = cumulative_wins = 0
    for oos_pos in range(1, len(MONTHS)):
        train_end_idx = oos_pos - 1
        oos_month = MONTHS[oos_pos]
        pooled_scores = []
        for candidate in CANDIDATES:
            final, trades, wins = pooled_train(loaded, symbols, train_end_idx, candidate, capital)
            ret = final / capital - 1.0 if capital else 0.0
            pooled_scores.append((ret, candidate, trades, wins))
        _, selected, train_trades, _ = max(pooled_scores, key=lambda x: x[0])

        month_finals = []
        month_trades = month_wins = 0
        for s in symbols:
            df = loaded[s]
            idx = df.index[df["month"] == oos_month]
            if len(idx) == 0:
                continue
            start, end = int(idx.min()), int(idx.max() + 1)
            initial_symbol = capital / len(symbols)
            f, t, w = backtest(df, start, end, selected, initial_symbol)
            month_finals.append(f)
            month_trades += t
            month_wins += w

        new_capital = sum(month_finals)
        oos_return = new_capital / capital - 1.0 if capital else 0.0
        win_rate = month_wins / month_trades * 100 if month_trades else 0.0
        print(f"TRAIN through {MONTHS[train_end_idx]} -> OOS {oos_month}")
        print(f" selected={selected*100:.2f}% | train trades={train_trades}")
        print(f" OOS capital=${new_capital:.2f} return={oos_return*100:+.2f}% trades={month_trades} win={win_rate:.1f}%")
        print(f" cumulative ${capital:.2f} -> ${new_capital:.2f}")
        print()
        capital = new_capital
        cumulative_trades += month_trades
        cumulative_wins += month_wins

    total_return = capital / (args.initial_capital * len(symbols)) - 1.0
    total_win = cumulative_wins / cumulative_trades * 100 if cumulative_trades else 0.0
    print("=" * 96)
    print(f"FINAL COMPOUNDED: ${capital:.2f} from ${args.initial_capital*len(symbols):.2f} return={total_return*100:+.2f}% trades={cumulative_trades} win={total_win:.1f}%")


if __name__ == "__main__":
    main()
