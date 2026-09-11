from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NO_SIGNAL_MARKETS = {"DOSUSDT", "GRVTUSDT", "SLVONUSDT"}
DEFAULT_DATA_DIR = "reports/nobitex_1h"
DEFAULT_FEE = 0.0013


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def direction_series(df: pd.DataFrame) -> np.ndarray:
    body = df["close"] - df["open"]
    hc = df["high"] - df["close"]
    ho = df["high"] - df["open"]
    lc = df["low"] - df["close"]
    score = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)
    # User-requested entry signal: direction of the frame, WITHOUT the ambiguity filter.
    return np.where(score <= 1, -1, np.where(score >= 2, 1, 0)).astype(int)


def trade_return(entry: float, exit_: float, direction: int) -> float:
    if direction == 1:
        return (exit_ - entry) / entry
    return (entry - exit_) / entry


def simulate_symbol(
    df: pd.DataFrame,
    fee: float,
    opposite_count: int,
    mode: str,
    initial_capital: float,
) -> dict[str, float | int]:
    dirs = direction_series(df)
    n = len(df)
    capital = float(initial_capital)
    trades = 0
    wins = 0
    gross_sum = 0.0
    hold_sum = 0

    i = 0
    while i < n - 1:
        entry_dir = int(dirs[i])
        if entry_dir == 0:
            i += 1
            continue

        entry = float(df.iloc[i].close)
        opposite_seen = 0
        consecutive = 0
        exit_idx = None

        j = i + 1
        while j < n:
            d = int(dirs[j])

            if d == -entry_dir:
                opposite_seen += 1
                consecutive += 1
            elif d == entry_dir or d == 0:
                if mode == "consecutive":
                    consecutive = 0

            if mode == "consecutive" and consecutive >= opposite_count:
                exit_idx = j
                break
            if mode == "discrete" and opposite_seen >= opposite_count:
                exit_idx = j
                break

            j += 1

        if exit_idx is None:
            exit_idx = n - 1

        exit_price = float(df.iloc[exit_idx].close)
        gross = trade_return(entry, exit_price, entry_dir)
        net_factor = max((1.0 + gross) * (1.0 - fee) ** 2, 0.0)
        capital *= net_factor

        trades += 1
        wins += int(gross > 0)
        gross_sum += gross
        hold_sum += exit_idx - i

        # Reverse immediately on the exit frame when its direction is opposite.
        i = exit_idx

    return {
        "final": capital,
        "return": capital / initial_capital - 1.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "avg_gross": gross_sum / trades if trades else 0.0,
        "avg_hold": hold_sum / trades if trades else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="1H sequential trading backtest with opposite-frame exits, continuous vs discrete."
    )
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=DEFAULT_FEE)
    args = ap.parse_args()

    root = Path(args.data_dir)
    files = sorted(root.glob("*_1h.csv"))
    symbols = []
    loaded: dict[str, pd.DataFrame] = {}

    for path in files:
        stem = path.stem
        symbol = stem[:-3] if stem.endswith("_1h") else stem
        symbol = symbol.upper()
        if symbol.endswith("IRT") or symbol in NO_SIGNAL_MARKETS:
            continue
        try:
            df = load(path)
            if len(df) >= 2:
                symbols.append(symbol)
                loaded[symbol] = df
        except Exception as exc:
            print(f"SKIP {symbol}: {exc}")

    if not loaded:
        raise SystemExit(f"No valid 1H non-IRT files found under {root}")

    print("=" * 118)
    print("NOBITEX NON-IRT | 1H SEQUENTIAL TRADING | OPPOSITE-FRAME EXIT")
    print("=" * 118)
    print("Entry direction : current frame formula direction, NO ambiguity filter")
    print("Entry           : current candle close")
    print("Exit            : opposite-direction frames")
    print("Continuous      : N opposite frames consecutively")
    print("Discrete        : N opposite frames accumulated during the trade")
    print(f"Fee             : {args.fee * 100:.2f}% per side ({args.fee * 200:.2f}% round trip)")
    print(f"Markets         : {len(symbols)}")
    print()

    rows = []
    for mode in ("consecutive", "discrete"):
        print(f"MODEL = {mode.upper()}")
        print("N_OPPOSITE  final_total  return%  trades  win%  avg_gross%  avg_hold_h")
        total_initial = args.initial_capital * len(symbols)
        total_final = 0.0
        total_trades = 0
        total_wins = 0
        total_gross = 0.0
        total_hold = 0.0

        for k in (1, 2, 3):
            per_symbol = [
                simulate_symbol(loaded[s], args.fee, k, mode, args.initial_capital)
                for s in symbols
            ]
            final_total = sum(float(r["final"]) for r in per_symbol)
            trades = sum(int(r["trades"]) for r in per_symbol)
            wins = sum(int(round(float(r["win_rate"]) * int(r["trades"]))) for r in per_symbol)
            gross_sum = sum(float(r["avg_gross"]) * int(r["trades"]) for r in per_symbol)
            hold_sum = sum(float(r["avg_hold"]) * int(r["trades"]) for r in per_symbol)
            result = {
                "model": mode,
                "opposite_count": k,
                "markets": len(symbols),
                "initial_total": total_initial,
                "final_total": final_total,
                "return": final_total / total_initial - 1.0,
                "trades": trades,
                "win_rate": wins / trades if trades else 0.0,
                "avg_gross": gross_sum / trades if trades else 0.0,
                "avg_hold": hold_sum / trades if trades else 0.0,
            }
            rows.append(result)
            print(
                f"{k:10d}  {final_total:11.2f}  {result['return']*100:+8.2f}%"
                f"  {trades:6d}  {result['win_rate']*100:5.1f}%"
                f"  {result['avg_gross']*100:+10.4f}%  {result['avg_hold']:10.2f}"
            )
        print()

    out = pd.DataFrame(rows)
    output = root / "backtest_opposite_frame_exit_1h.csv"
    out.to_csv(output, index=False)

    print("COMPARISON")
    print("model        N   return%   trades   win%   avg_gross%   avg_hold_h")
    for _, r in out.sort_values("return", ascending=False).iterrows():
        print(
            f"{r['model']:12s} {int(r['opposite_count']):1d} "
            f"{r['return']*100:+8.2f}% {int(r['trades']):8d} "
            f"{r['win_rate']*100:6.2f}% {r['avg_gross']*100:+11.4f}% "
            f"{r['avg_hold']:11.2f}"
        )

    print()
    print(f"Detailed CSV: {output}")
    print("=" * 118)


if __name__ == "__main__":
    main()
