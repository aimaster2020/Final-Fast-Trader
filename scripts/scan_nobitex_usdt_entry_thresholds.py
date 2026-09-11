from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEE = 0.0013
NOISE = 1.0
MIN_CANDLES = 48
DEFAULT_THRESHOLDS = "0.83,1.00,1.25,1.50,1.75,2.00,2.50,3.00"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
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
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan stricter entry thresholds on Nobitex USDT 1h markets only.")
    ap.add_argument("--data-dir", default="reports/nobitex_1h")
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=FEE)
    ap.add_argument("--noise", type=float, default=NOISE)
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, help="Percent thresholds, e.g. 0.83,1,1.5,2")
    ap.add_argument("--min-candles", type=int, default=MIN_CANDLES)
    args = ap.parse_args()

    root = Path(args.data_dir)
    files = sorted(p for p in root.glob("*_1h.csv") if not p.stem[:-3].upper().endswith("IRT"))
    thresholds = sorted({float(x.strip()) for x in args.thresholds.split(",") if x.strip()})
    loaded: list[tuple[str, pd.DataFrame]] = []

    skipped = 0
    for path in files:
        symbol = path.stem[:-3] if path.stem.endswith("_1h") else path.stem
        try:
            df = load(path)
            if len(df) < args.min_candles:
                skipped += 1
                continue
            loaded.append((symbol, df))
        except Exception as exc:
            skipped += 1
            print(f"SKIP {symbol}: {exc}")

    if not loaded:
        raise SystemExit("No valid non-IRT markets found.")

    rows: list[dict[str, float | int]] = []
    n_markets = len(loaded)
    initial_total = args.initial_capital * n_markets

    print("=" * 112)
    print("NOBITEX USDT/TRADEABLE NON-IRT MARKETS | ENTRY THRESHOLD SCAN")
    print("=" * 112)
    print(f"Candidate files (non-IRT) : {len(files)}")
    print(f"Markets tested            : {n_markets}")
    print(f"Skipped                   : {skipped}")
    print(f"Commission                : {args.fee * 100:.2f}% per side ({args.fee * 200:.2f}% round trip)")
    print(f"Noise                     : {args.noise:.2f}")
    print("Thresholds are percentage of predicted move / entry price.")
    print()

    for threshold_pct in thresholds:
        entry_min = threshold_pct / 100.0
        pooled_final = 0.0
        total_trades = total_wins = 0
        market_returns: list[float] = []
        profitable = losing = flat = 0

        for symbol, df in loaded:
            r = backtest(df, args.initial_capital, args.fee, entry_min, args.noise)
            pooled_final += float(r["final"])
            total_trades += int(r["trades"])
            total_wins += int(r["wins"])
            market_returns.append(float(r["return"]))
            if r["return"] > 0:
                profitable += 1
            elif r["return"] < 0:
                losing += 1
            else:
                flat += 1

        pooled_return = pooled_final / initial_total - 1.0
        win_rate = total_wins / total_trades if total_trades else 0.0
        mean_return = float(np.mean(market_returns))
        median_return = float(np.median(market_returns))
        rows.append(
            {
                "threshold_pct": threshold_pct,
                "markets": n_markets,
                "profitable_pct": profitable / n_markets,
                "losing_pct": losing / n_markets,
                "flat_pct": flat / n_markets,
                "mean_return": mean_return,
                "median_return": median_return,
                "pooled_return": pooled_return,
                "trades": total_trades,
                "win_rate": win_rate,
                "pooled_final": pooled_final,
            }
        )

    result = pd.DataFrame(rows).sort_values("threshold_pct").reset_index(drop=True)
    out_path = root / "usdt_entry_threshold_scan.csv"
    result.to_csv(out_path, index=False)

    print("THRESHOLD     POOLED RETURN    MEAN     MEDIAN   PROFITABLE   WIN RATE   TRADES")
    for _, r in result.iterrows():
        print(
            f"{r.threshold_pct:>8.2f}%     {r.pooled_return * 100:>+9.2f}%   "
            f"{r.mean_return * 100:>+7.2f}%  {r.median_return * 100:>+7.2f}%   "
            f"{r.profitable_pct * 100:>7.1f}%   {r.win_rate * 100:>7.1f}%   {int(r.trades):>6d}"
        )

    best = result.loc[result["pooled_return"].idxmax()]
    print()
    print(
        f"BEST POOLED: threshold={best.threshold_pct:.2f}% "
        f"return={best.pooled_return * 100:+.2f}% "
        f"profitable_markets={best.profitable_pct * 100:.1f}% "
        f"win={best.win_rate * 100:.1f}% trades={int(best.trades)}"
    )
    print(f"Detailed CSV: {out_path}")
    print("=" * 112)


if __name__ == "__main__":
    main()
