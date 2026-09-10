from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_HORIZONS = "1,2,3,4,5"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=lambda c: str(c).strip().lower() in {"open", "high", "low", "close"},
    )
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    body = df.close - df.open
    hc = df.high - df.close
    ho = df.high - df.open
    lc = df.low - df.close
    score = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)

    # Direction in this experiment is determined ONLY by the score extremes:
    # score 0 = SHORT, score 3 = LONG. Scores 1 and 2 are ignored.
    direction = np.where(score == 0, -1, np.where(score == 3, 1, 0)).astype(int)

    df["body"] = body
    df["score"] = score
    df["direction"] = direction
    return df


def run_symbol(
    df: pd.DataFrame,
    initial_capital: float,
    fee_each_side: float,
    horizon: int,
    direction_filter: int | None = None,
) -> dict[str, float | int]:
    capital = float(initial_capital)
    fee_factor = (1.0 - fee_each_side) ** 2

    trades = wins = 0
    gross_sum = 0.0
    hold_sum = 0

    n = len(df)
    for i in range(n - horizon):
        row = df.iloc[i]
        trade_dir = int(row.direction)

        # Optional isolated LONG/SHORT tests. The score 0/3 mode passes None.
        if direction_filter is not None and trade_dir != direction_filter:
            continue
        if trade_dir == 0:
            continue

        entry = float(row.close)
        exit_price = float(df.iloc[i + horizon].close)
        gross_ret = (
            (exit_price - entry) / entry
            if trade_dir == 1
            else (entry - exit_price) / entry
        )

        capital *= max((1.0 + gross_ret) * fee_factor, 0.0)
        trades += 1
        wins += int(gross_ret > 0)
        gross_sum += gross_ret
        hold_sum += horizon

    return {
        "final": capital,
        "return": capital / initial_capital - 1.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "avg_gross_trade": gross_sum / trades if trades else 0.0,
        "avg_hold": hold_sum / trades if trades else 0.0,
    }


def pooled_result(
    loaded: dict[str, pd.DataFrame],
    symbols: list[str],
    initial_capital: float,
    fee: float,
    horizon: int,
    direction_filter: int | None,
) -> dict[str, float | int]:
    pooled_final = 0.0
    pooled_trades = 0
    pooled_wins = 0
    pooled_gross_sum = 0.0

    for symbol in symbols:
        r = run_symbol(
            loaded[symbol],
            initial_capital,
            fee,
            horizon,
            direction_filter=direction_filter,
        )
        pooled_final += float(r["final"])
        pooled_trades += int(r["trades"])
        pooled_wins += int(round(float(r["win_rate"]) * int(r["trades"])))
        pooled_gross_sum += float(r["avg_gross_trade"]) * int(r["trades"])

    pooled_initial = initial_capital * len(symbols)
    return {
        "final": pooled_final,
        "return": pooled_final / pooled_initial - 1.0,
        "trades": pooled_trades,
        "win_rate": pooled_wins / pooled_trades if pooled_trades else 0.0,
        "avg_gross_trade": pooled_gross_sum / pooled_trades if pooled_trades else 0.0,
    }


def print_mode(
    label: str,
    loaded: dict[str, pd.DataFrame],
    symbols: list[str],
    initial_capital: float,
    args_fee: float,
    horizon: int,
    direction_filter: int | None,
) -> None:
    print(label)
    print("fee      final       return      trades   win%   avg_gross")
    for fee in (0.0, args_fee):
        r = pooled_result(
            loaded,
            symbols,
            initial_capital,
            fee,
            horizon,
            direction_filter,
        )
        print(
            f"{fee * 100:5.2f}%  "
            f"{float(r['final']):10.2f}  "
            f"{float(r['return']) * 100:+9.2f}%  "
            f"{int(r['trades']):6d}  "
            f"{float(r['win_rate']) * 100:5.1f}%  "
            f"{float(r['avg_gross_trade']) * 100:+9.4f}%"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fixed-horizon backtest with no entry filter and no signal-based exit rules."
    )
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=0.0013, help="Commission per side; 0.0013 = 0.13%.")
    ap.add_argument("--horizons", default=DEFAULT_HORIZONS, help="Comma-separated candle horizons.")
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    horizons = sorted({int(x) for x in args.horizons.split(",") if x.strip()})
    if any(h < 1 for h in horizons):
        raise ValueError("Horizons must be >= 1.")

    root = Path(args.data_dir)
    loaded = {symbol: load(root / f"{symbol}_1h.csv") for symbol in symbols}

    print("=" * 104)
    print("FIXED-HORIZON BACKTEST: NO ENTRY FILTER + NO SIGNAL EXIT")
    print("=" * 104)
    print("Direction: SCORE 0 = SHORT, SCORE 3 = LONG; SCORES 1/2 = NO TRADE")
    print("Entry: current candle close")
    print("Exit: close of candle +H")
    print("Entry magnitude filter: DISABLED")
    print("Progressive opposite-signal exit: DISABLED")
    print("Commission: reported at 0% and configured fee per side")
    print(f"Initial capital per symbol: {args.initial_capital:.2f}")
    print()

    for horizon in horizons:
        print(f"HORIZON = {horizon}")
        print_mode("SHORT ONLY", loaded, symbols, args.initial_capital, args.fee, horizon, -1)
        print_mode("LONG ONLY", loaded, symbols, args.initial_capital, args.fee, horizon, 1)
        print_mode("SCORE 0/3 ONLY", loaded, symbols, args.initial_capital, args.fee, horizon, None)

    print("=" * 104)


if __name__ == "__main__":
    main()
