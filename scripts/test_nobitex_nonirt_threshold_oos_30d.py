from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEE = 0.0013
NOISE = 1.0
THRESHOLDS = [0.0200, 0.0225, 0.0250]
MIN_CANDLES = 48


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"timestamp", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"missing columns: {sorted(required - set(df.columns))}")
    for c in ("timestamp", "open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    body = df["close"] - df["open"]
    hc = df["high"] - df["close"]
    ho = df["high"] - df["open"]
    lc = df["low"] - df["close"]
    score = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)

    upper = (df["high"] - df["open"]).abs()
    lower = (df["open"] - df["low"]).abs()
    formula = np.where(lower > upper, 1, np.where(upper > lower, -1, 0))

    direction = np.where(score <= 1, -1, 1).astype(int)
    direction[(score == 1) & (formula == 1)] = 0
    direction[(score == 2) & (formula == -1)] = 0

    magnitude = np.where(direction == 1, upper, np.where(direction == -1, lower, np.nan))

    out = df.copy()
    out["body"] = body
    out["direction"] = direction
    out["magnitude_pct"] = magnitude / out["close"]
    out["datetime"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    return out


def backtest_segment(
    df: pd.DataFrame,
    initial_capital: float,
    entry_min: float,
) -> dict[str, float | int]:
    fee_factor = (1.0 - FEE) ** 2
    capital = float(initial_capital)
    trades = wins = 0
    i = 0
    n = len(df)

    while i < n - 1:
        row = df.iloc[i]
        trade_dir = int(row.direction)
        expected_pct = float(row.magnitude_pct)
        signal_body = abs(float(row.body))

        if trade_dir == 0 or not np.isfinite(expected_pct) or expected_pct < entry_min or signal_body <= 0:
            i += 1
            continue

        entry = float(row.close)
        opposite_count = 0
        first_opposite_strength = 0.0
        exit_i = None
        j = i + 1

        while j < n:
            current = df.iloc[j]
            pred_dir = int(current.direction)
            pred_mag = float(current.magnitude_pct) if np.isfinite(current.magnitude_pct) else np.nan
            if pred_dir == 0 or not np.isfinite(pred_mag):
                j += 1
                continue

            current_move_pct = abs(float(current.close) - entry) / max(abs(entry), 1e-12)
            reference_pct = max(current_move_pct, signal_body / max(abs(entry), 1e-12), 1e-12)

            if pred_dir == trade_dir:
                opposite_count = 0
                first_opposite_strength = 0.0
                j += 1
                continue

            opposite_count += 1
            if opposite_count == 1:
                first_opposite_strength = pred_mag
                if pred_mag <= NOISE * reference_pct:
                    j += 1
                    continue
                exit_i = j
                break

            if opposite_count == 2:
                if pred_mag < 2.0 * first_opposite_strength or pred_mag <= 2.0 * NOISE * reference_pct:
                    j += 1
                    continue
                exit_i = j
                break

            exit_i = j
            break

        if exit_i is None:
            break

        exit_price = float(df.iloc[exit_i].close)
        gross_ret = ((exit_price - entry) / entry) if trade_dir == 1 else ((entry - exit_price) / entry)
        capital *= max((1.0 + gross_ret) * fee_factor, 0.0)
        trades += 1
        wins += int(gross_ret > 0)
        i = exit_i + 1

    return {
        "final": capital,
        "return": capital / initial_capital - 1.0,
        "trades": trades,
        "wins": wins,
    }


def pooled(
    loaded: dict[str, pd.DataFrame],
    symbols: list[str],
    start_ts: int,
    end_ts: int,
    threshold: float,
    initial_capital: float,
) -> dict[str, float | int]:
    total_final = 0.0
    total_trades = total_wins = 0
    tested = 0

    for s in symbols:
        df = loaded[s]
        seg = df[(df.timestamp >= start_ts) & (df.timestamp <= end_ts)].reset_index(drop=True)
        if len(seg) < MIN_CANDLES:
            continue
        r = backtest_segment(seg, initial_capital, threshold)
        total_final += float(r["final"])
        total_trades += int(r["trades"])
        total_wins += int(r["wins"])
        tested += 1

    initial = initial_capital * tested
    return {
        "tested": tested,
        "initial": initial,
        "final": total_final,
        "return": total_final / initial - 1.0 if initial else 0.0,
        "trades": total_trades,
        "win_rate": total_wins / total_trades if total_trades else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/select threshold on early 30d data and test on later OOS data for non-IRT Nobitex markets.")
    ap.add_argument("--data-dir", default="reports/nobitex_1h")
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--train-hours", type=int, default=528, help="Training window in hours; default 22 days.")
    ap.add_argument("--oos-hours", type=int, default=168, help="OOS window in hours; default 7 days.")
    args = ap.parse_args()

    root = Path(args.data_dir)
    files = sorted(p for p in root.glob("*_1h.csv") if not p.name.upper().endswith("IRT_1H.CSV"))
    files = [p for p in files if "_IRT_" not in p.name.upper()]
    if not files:
        raise SystemExit(f"No non-IRT *_1h.csv files found in {root}")

    loaded: dict[str, pd.DataFrame] = {}
    for path in files:
        symbol = path.stem[:-3] if path.stem.endswith("_1h") else path.stem
        if symbol.upper().endswith("IRT"):
            continue
        try:
            df = load(path)
            if len(df) >= MIN_CANDLES:
                loaded[symbol] = df
        except Exception as exc:
            print(f"SKIP {symbol}: {exc}")

    if not loaded:
        raise SystemExit("No valid non-IRT markets available.")

    global_max = min(int(df.timestamp.max()) for df in loaded.values())
    oos_end = global_max
    oos_start = oos_end - args.oos_hours * 3600 + 3600
    train_end = oos_start - 3600
    train_start = train_end - args.train_hours * 3600 + 3600

    print("=" * 112)
    print("NOBITEX NON-IRT | THRESHOLD SELECTION + OOS TEST")
    print("=" * 112)
    print(f"Markets loaded : {len(loaded)}")
    print(f"Commission    : {FEE*100:.2f}% per side ({FEE*200:.2f}% round trip)")
    print(f"Training      : {pd.to_datetime(train_start, unit='s', utc=True)} -> {pd.to_datetime(train_end, unit='s', utc=True)}")
    print(f"OOS           : {pd.to_datetime(oos_start, unit='s', utc=True)} -> {pd.to_datetime(oos_end, unit='s', utc=True)}")
    print()

    print("TRAIN")
    train_rows = []
    for threshold in THRESHOLDS:
        r = pooled(loaded, list(loaded), train_start, train_end, threshold, args.initial_capital)
        train_rows.append((r["return"], threshold, r))
        print(f"threshold={threshold*100:.2f}% tested={r['tested']:3d} return={r['return']*100:+.2f}% trades={r['trades']:4d} win={r['win_rate']*100:.1f}%")

    _, selected, selected_train = max(train_rows, key=lambda x: x[0])
    print()
    print(f"SELECTED THRESHOLD: {selected*100:.2f}% (train return {selected_train['return']*100:+.2f}%)")
    print()

    print("OOS")
    for threshold in THRESHOLDS:
        r = pooled(loaded, list(loaded), oos_start, oos_end, threshold, args.initial_capital)
        marker = "  <- SELECTED" if threshold == selected else ""
        print(f"threshold={threshold*100:.2f}% tested={r['tested']:3d} return={r['return']*100:+.2f}% trades={r['trades']:4d} win={r['win_rate']*100:.1f}%{marker}")

    print("=" * 112)


if __name__ == "__main__":
    main()
