from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NO_SIGNAL_MARKETS = {"DOSUSDT", "GRVTUSDT", "SLVONUSDT"}
DEFAULT_DATA_DIR = "reports/nobitex_intraday_native/1h"
DEFAULT_INITIAL_CAPITAL = 1000.0
DEFAULT_FEE = 0.0013


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
    return df.reset_index(drop=True)


def formula_direction(df: pd.DataFrame) -> np.ndarray:
    """Original formula direction, deliberately WITHOUT the ambiguity filter."""
    body = df["close"] - df["open"]
    hc = df["high"] - df["close"]
    ho = df["high"] - df["open"]
    lc = df["low"] - df["close"]
    score = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)
    return np.where(score <= 1, -1, np.where(score >= 2, 1, 0)).astype(int)


def net_factor(entry: float, exit_: float, direction: int, fee: float) -> float:
    gross_factor = exit_ / entry if direction == 1 else entry / exit_
    return max(gross_factor * (1.0 - fee) ** 2, 0.0)


def backtest(
    df: pd.DataFrame,
    opposite_count: int,
    mode: str,
    initial_capital: float,
    fee: float,
) -> dict[str, float | int]:
    """Run one symbol as an always-available sequential strategy.

    Entry: every non-zero formula direction while flat.
    Exit: N opposite frames, either consecutive or cumulative/discrete.
    Reversal: when an exit occurs, the same closing frame can immediately open
    a new trade in its direction.
    Final open trade: force-close on the final candle.
    """
    dirs = formula_direction(df)
    closes = df["close"].to_numpy(float)
    n = len(df)

    capital = float(initial_capital)
    peak = capital
    max_dd = 0.0

    entry_idx: int | None = None
    entry_price = np.nan
    position = 0
    opposite_seen = 0
    consecutive_opposite = 0

    trades = 0
    wins = 0
    gross_returns: list[float] = []
    net_returns: list[float] = []

    def close_trade(i: int, new_direction: int | None) -> None:
        nonlocal capital, peak, max_dd
        nonlocal entry_idx, entry_price, position
        nonlocal opposite_seen, consecutive_opposite
        nonlocal trades, wins

        if entry_idx is None or position == 0:
            return

        exit_price = float(closes[i])
        gross_factor = exit_price / entry_price if position == 1 else entry_price / exit_price
        gross_ret = gross_factor - 1.0
        factor = max(gross_factor * (1.0 - fee) ** 2, 0.0)
        net_ret = factor - 1.0

        capital *= factor
        trades += 1
        wins += int(net_ret > 0.0)
        gross_returns.append(gross_ret)
        net_returns.append(net_ret)

        peak = max(peak, capital)
        dd = 1.0 - capital / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

        entry_idx = None
        entry_price = np.nan
        position = 0
        opposite_seen = 0
        consecutive_opposite = 0

        if new_direction in (-1, 1):
            entry_idx = i
            entry_price = exit_price
            position = int(new_direction)

    for i in range(n):
        signal = int(dirs[i])

        # Flat: enter immediately on the current candle close.
        if position == 0:
            if signal != 0:
                entry_idx = i
                entry_price = float(closes[i])
                position = signal
                opposite_seen = 0
                consecutive_opposite = 0
            continue

        # We do not count the entry candle itself as an opposite frame.
        if i == entry_idx:
            continue

        is_opposite = signal != 0 and signal == -position

        if mode == "consecutive":
            if is_opposite:
                consecutive_opposite += 1
            else:
                consecutive_opposite = 0
            should_exit = consecutive_opposite >= opposite_count
        elif mode == "discrete":
            if is_opposite:
                opposite_seen += 1
            should_exit = opposite_seen >= opposite_count
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if should_exit:
            # Reversal is allowed on the exit candle itself.
            close_trade(i, signal if signal != 0 else None)

    # Force-close any remaining position at the last available close.
    if position != 0 and entry_idx is not None and n > 0:
        close_trade(n - 1, None)

    return {
        "final": capital,
        "return": capital / initial_capital - 1.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "max_dd": max_dd,
        "avg_net": float(np.mean(net_returns)) if net_returns else 0.0,
        "median_net": float(np.median(net_returns)) if net_returns else 0.0,
        "avg_gross": float(np.mean(gross_returns)) if gross_returns else 0.0,
    }


def summarize(records: list[dict[str, object]], initial_capital: float, label: str) -> None:
    out = pd.DataFrame(records)
    if out.empty:
        print(f"{label}: no valid markets")
        return

    profitable = int((out["return"] > 0).sum())
    loss = int((out["return"] < 0).sum())
    zero = int((out["return"] == 0).sum())

    # Equal-weight portfolio view: each tested market starts with the same capital.
    portfolio_final = float(out["final"].sum())
    portfolio_initial = initial_capital * len(out)
    portfolio_return = portfolio_final / portfolio_initial - 1.0

    print(f"{label}")
    print(
        f"markets={len(out)}  portfolio_return={portfolio_return * 100:+.2f}%  "
        f"mean_market_return={out['return'].mean() * 100:+.2f}%  "
        f"median_market_return={out['return'].median() * 100:+.2f}%"
    )
    print(
        f"profitable={profitable}  loss={loss}  zero={zero}  "
        f"mean_trades={out['trades'].mean():.1f}  mean_win={out['win_rate'].mean() * 100:.2f}%  "
        f"median_win={out['win_rate'].median() * 100:.2f}%  mean_maxDD={out['max_dd'].mean() * 100:.2f}%"
    )

    rank = out.sort_values("return", ascending=False).head(10)
    print("top10: " + " | ".join(
        f"{r.symbol} {r.return * 100:+.1f}%/{int(r.trades)}t/{r.win_rate * 100:.1f}%"
        for _, r in rank.iterrows()
    ))
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sequential 1H backtest: enter on formula direction, exit after 1/2/3 opposite frames."
    )
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    ap.add_argument("--fee", type=float, default=DEFAULT_FEE)
    ap.add_argument("--symbols", default="", help="Optional comma-separated symbol list")
    ap.add_argument("--output", default="reports/nobitex_1h_opposite_frame_exit.csv")
    args = ap.parse_args()

    root = Path(args.data_dir)
    if not root.exists():
        raise SystemExit(f"Data directory not found: {root}")

    requested = {x.strip().upper() for x in args.symbols.split(",") if x.strip()}
    files = sorted(root.glob("*_1h.csv"))
    symbols = []
    for path in files:
        symbol = path.stem[: -len("_1h")].upper()
        if symbol.endswith("IRT") or symbol in NO_SIGNAL_MARKETS:
            continue
        if requested and symbol not in requested:
            continue
        symbols.append(symbol)

    if not symbols:
        raise SystemExit(f"No non-IRT 1H data found under {root}")

    loaded: dict[str, pd.DataFrame] = {}
    skipped = 0
    for symbol in symbols:
        try:
            df = load(root / f"{symbol}_1h.csv")
            if len(df) < 10:
                skipped += 1
                continue
            loaded[symbol] = df
        except Exception as exc:
            skipped += 1
            print(f"SKIP {symbol}: {exc}")

    fee_pct = args.fee * 100.0
    print("=" * 130)
    print("NOBITEX NON-IRT | NATIVE 1H | SEQUENTIAL TRADING | OPPOSITE-FRAME EXITS")
    print("=" * 130)
    print(f"Markets found/tested : {len(symbols)}/{len(loaded)}  skipped={skipped}")
    print("Entry                : every non-zero formula direction, NO ambiguity filter")
    print("Formula              : score 0/1 SHORT, score 2/3 LONG")
    print("Entry price          : current candle close")
    print("Exit price           : close of the frame that completes the exit rule")
    print("Re-entry             : immediate reversal on the same exit candle")
    print(f"Fee                  : {fee_pct:.2f}% per side ({fee_pct * 2:.2f}% round trip)")
    print("Consecutive          : N opposite signals must be consecutive")
    print("Discrete             : N opposite signals accumulate; same-direction/zero does not reset")
    print("Open final trade     : forced close on last available 1H close")
    print()

    all_records: list[dict[str, object]] = []
    for mode in ("consecutive", "discrete"):
        for n_opposite in (1, 2, 3):
            records = []
            for symbol, df in loaded.items():
                r = backtest(df, n_opposite, mode, args.initial_capital, args.fee)
                records.append({"symbol": symbol, "mode": mode, "opposite_frames": n_opposite, **r})
            all_records.extend(records)
            summarize(records, args.initial_capital, f"{mode.upper()} | EXIT AFTER {n_opposite} OPPOSITE FRAME(S)")

    out = pd.DataFrame(all_records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    print("COMPARISON")
    print("mode         N   portfolio_return   mean_market   median_market   profitable_markets   mean_trades   mean_win   mean_maxDD")
    for mode in ("consecutive", "discrete"):
        for n_opposite in (1, 2, 3):
            x = out[(out.mode == mode) & (out.opposite_frames == n_opposite)].copy()
            p_ret = x.final.sum() / (args.initial_capital * len(x)) - 1.0
            print(
                f"{mode:12s} {n_opposite:1d}  {p_ret * 100:+16.2f}%  "
                f"{x['return'].mean() * 100:+11.2f}%  {x['return'].median() * 100:+13.2f}%  "
                f"{int((x['return'] > 0).sum()):18d}  {x.trades.mean():11.1f}  "
                f"{x.win_rate.mean() * 100:8.2f}%  {x.max_dd.mean() * 100:9.2f}%"
            )

    best_idx = None
    best_value = -np.inf
    for mode in ("consecutive", "discrete"):
        for n_opposite in (1, 2, 3):
            x = out[(out.mode == mode) & (out.opposite_frames == n_opposite)]
            value = float(x.final.sum() / (args.initial_capital * len(x)) - 1.0)
            if value > best_value:
                best_value = value
                best_idx = (mode, n_opposite)
    if best_idx:
        print(f"BEST BY EQUAL-WEIGHT PORTFOLIO RETURN: {best_idx[0]} N={best_idx[1]} -> {best_value * 100:+.2f}%")

    print(f"Detailed CSV: {output}")
    print("=" * 130)


if __name__ == "__main__":
    main()
