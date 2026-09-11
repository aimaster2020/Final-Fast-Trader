from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEE = 0.0013
ENTRY_MIN = 0.0083
NOISE = 1.0
MIN_CANDLES = 48


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"missing columns: {sorted(required - set(df.columns))}")
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

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
    return out


def backtest(
    df: pd.DataFrame,
    initial_capital: float,
    fee_each_side: float,
    entry_min_fraction: float,
    noise_fraction: float,
    direction_filter: int | None = None,
) -> dict[str, float | int]:
    capital = float(initial_capital)
    fee_factor = (1.0 - fee_each_side) ** 2
    trades = wins = 0
    i = 0
    n = len(df)

    while i < n - 1:
        row = df.iloc[i]
        trade_dir = int(row.direction)
        expected_pct = float(row.magnitude_pct)
        signal_body = abs(float(row.body))

        if direction_filter is not None and trade_dir != direction_filter:
            i += 1
            continue

        if trade_dir == 0 or not np.isfinite(expected_pct) or expected_pct < entry_min_fraction or signal_body <= 0:
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
                if pred_mag <= noise_fraction * reference_pct:
                    j += 1
                    continue
                exit_i = j
                break

            if opposite_count == 2:
                if pred_mag < 2.0 * first_opposite_strength or pred_mag <= 2.0 * noise_fraction * reference_pct:
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
        "win_rate": wins / trades if trades else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the fixed 0.83% Nobitex strategy test across every downloaded 1h market.")
    ap.add_argument("--data-dir", default="reports/nobitex_1h")
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=FEE, help="Commission per side; default 0.0013 = 0.13%.")
    ap.add_argument("--entry-min", type=float, default=ENTRY_MIN, help="Fixed predicted-move threshold; default 0.0083 = 0.83%.")
    ap.add_argument("--noise", type=float, default=NOISE)
    ap.add_argument("--min-candles", type=int, default=MIN_CANDLES)
    ap.add_argument("--direction", choices=["ALL", "LONG", "SHORT"], default="ALL")
    args = ap.parse_args()

    root = Path(args.data_dir)
    files = sorted(root.glob("*_1h.csv"))
    if not files:
        raise SystemExit(f"No *_1h.csv files found in {root}")

    direction_filter = {"ALL": None, "LONG": 1, "SHORT": -1}[args.direction]
    results: list[dict[str, float | int | str]] = []
    skipped = 0

    for path in files:
        symbol = path.stem[:-3] if path.stem.endswith("_1h") else path.stem
        try:
            df = load(path)
            if len(df) < args.min_candles:
                skipped += 1
                continue
            r = backtest(df, args.initial_capital, args.fee, args.entry_min, args.noise, direction_filter)
            results.append({"symbol": symbol, "candles": len(df), **r})
        except Exception as exc:
            skipped += 1
            print(f"SKIP {symbol}: {exc}")

    if not results:
        raise SystemExit("No valid markets remained after filtering.")

    out = pd.DataFrame(results).sort_values("return", ascending=False).reset_index(drop=True)
    out_path = root / f"all_market_test_{args.direction.lower()}.csv"
    out.to_csv(out_path, index=False)

    profitable = int((out["return"] > 0).sum())
    losing = int((out["return"] < 0).sum())
    flat = int((out["return"] == 0).sum())
    pooled_initial = args.initial_capital * len(out)
    pooled_final = float(out["final"].sum())
    pooled_return = pooled_final / pooled_initial - 1.0
    total_trades = int(out["trades"].sum())
    total_wins = int(out["wins"].sum())
    median_return = float(out["return"].median())
    mean_return = float(out["return"].mean())

    print("=" * 110)
    print("NOBITEX ALL-MARKET BACKTEST | FIXED 0.83% ENTRY FILTER")
    print("=" * 110)
    print(f"Markets discovered from downloaded files : {len(files)}")
    print(f"Markets tested                          : {len(out)}")
    print(f"Markets skipped                         : {skipped}")
    print(f"Candles minimum                         : {args.min_candles}")
    print(f"Entry threshold                         : {args.entry_min * 100:.2f}%")
    print(f"Noise                                   : {args.noise:.2f}")
    print(f"Commission                              : {args.fee * 100:.2f}% per side ({args.fee * 200:.2f}% round trip)")
    print(f"Direction                               : {args.direction}")
    print()
    print(f"Profitable markets                      : {profitable} ({profitable / len(out) * 100:.1f}%)")
    print(f"Losing markets                          : {losing} ({losing / len(out) * 100:.1f}%)")
    print(f"Flat markets                            : {flat} ({flat / len(out) * 100:.1f}%)")
    print(f"Mean market return                      : {mean_return * 100:+.2f}%")
    print(f"Median market return                    : {median_return * 100:+.2f}%")
    print(f"Total trades                            : {total_trades}")
    print(f"Aggregate trade win rate                : {total_wins / total_trades * 100 if total_trades else 0:.1f}%")
    print(f"Pooled independent capital              : ${pooled_initial:.2f} -> ${pooled_final:.2f} ({pooled_return * 100:+.2f}%)")
    print()
    print("TOP 10")
    for _, r in out.head(10).iterrows():
        print(f"{r.symbol:20s} return={r['return'] * 100:+7.2f}% trades={int(r['trades']):4d} win={r['win_rate'] * 100:5.1f}%")
    print("BOTTOM 10")
    for _, r in out.tail(10).sort_values("return").iterrows():
        print(f"{r.symbol:20s} return={r['return'] * 100:+7.2f}% trades={int(r['trades']):4d} win={r['win_rate'] * 100:5.1f}%")
    print()
    print(f"Detailed CSV: {out_path}")
    print("=" * 110)


if __name__ == "__main__":
    main()
