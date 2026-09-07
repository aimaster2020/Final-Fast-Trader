from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal

DEFAULT_VALUES = (-200.0, -150.0, -100.0, -75.0, -50.0, 50.0, 75.0, 100.0, 150.0, 200.0)
DEFAULT_MONTHS = ("2026-05", "2026-07")
TIMEFRAMES = ("5m", "15m", "1h")


def load_month(path: Path, month: str, symbol: str) -> list[Candle]:
    candles: list[Candle] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("month") != month or row.get("symbol") != symbol:
                continue
            try:
                candles.append(
                    Candle(
                        int(float(row["timestamp"])),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(candles, key=lambda c: c.timestamp)


def signal_side(candle: Candle) -> int:
    signal = decide(candle).signal
    if signal == Signal.BUY:
        return 1
    if signal == Signal.SELL:
        return -1
    return 0


def pnl_pct(side: int, entry: float, exit_price: float) -> float:
    return side * (exit_price - entry) / entry if entry > 0 else 0.0


def run(
    candles: list[Candle],
    lower: float,
    upper: float,
    commission: float,
    initial_capital: float = 1000.0,
) -> dict[str, float | int]:
    equity = initial_capital
    position: tuple[int, float, float] | None = None
    trades = wins = 0
    gross_profit = gross_loss = 0.0

    for candle in candles:
        sig = signal_side(candle)
        body = candle.close - candle.open

        if position is None:
            if sig != 0 and not (lower <= body <= upper):
                position = (sig, candle.close, equity)
        else:
            side, entry, capital = position
            opposite = sig != 0 and sig != side
            exit_triggered = (
                side > 0 and opposite and body <= upper
            ) or (
                side < 0 and opposite and body >= lower
            )
            if exit_triggered:
                gross = capital * pnl_pct(side, entry, candle.close)
                fees = capital * commission * 2.0
                net = gross - fees
                equity += net
                trades += 1
                if net > 0:
                    wins += 1
                    gross_profit += max(gross, 0.0)
                elif net < 0:
                    gross_loss += min(gross, 0.0)
                position = None

    if position is not None:
        side, entry, capital = position
        gross = capital * pnl_pct(side, entry, candles[-1].close)
        fees = capital * commission * 2.0
        net = gross - fees
        equity += net
        trades += 1
        if net > 0:
            wins += 1
            gross_profit += max(gross, 0.0)
        elif net < 0:
            gross_loss += min(gross, 0.0)

    return {
        "return_pct": (equity / initial_capital - 1.0) * 100.0,
        "trades": trades,
        "win_rate": wins / trades * 100.0 if trades else 0.0,
        "profit_factor": gross_profit / abs(gross_loss) if gross_loss < 0 else (float("inf") if gross_profit else 0.0),
    }


def parse_values(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(x.strip()) for x in raw.split(",") if x.strip()}))
    if len(values) < 2:
        raise SystemExit("Need at least two threshold values")
    return values


def compounded_return(returns: list[float]) -> float:
    factor = 1.0
    for ret in returns:
        factor *= 1.0 + ret / 100.0
    return (factor - 1.0) * 100.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rank one common asymmetric BODY bias by combined performance of each timeframe across months."
    )
    ap.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--commission", type=float, default=0.0)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--values", default=",".join(str(v) for v in DEFAULT_VALUES))
    ap.add_argument("--input-5m", default="reports/prepared_price_action_5m.csv")
    ap.add_argument("--input-15m", default="reports/prepared_price_action_15m.csv")
    ap.add_argument("--input-1h", default="reports/prepared_price_action_1h.csv")
    args = ap.parse_args()

    months = tuple(x.strip() for x in args.months.split(",") if x.strip())
    if len(months) < 2:
        raise SystemExit("Need at least two months")
    values = parse_values(args.values)
    paths = {
        "5m": Path(args.input_5m),
        "15m": Path(args.input_15m),
        "1h": Path(args.input_1h),
    }

    data: dict[tuple[str, str], list[Candle]] = {}
    for timeframe in TIMEFRAMES:
        for month in months:
            candles = load_month(paths[timeframe], month, args.symbol)
            if not candles:
                raise SystemExit(f"No usable {timeframe} candles for {args.symbol} {month}: {paths[timeframe]}")
            data[(timeframe, month)] = candles

    rows = []
    for lower in values:
        for upper in values:
            if lower >= upper:
                continue

            tf_results = {}
            all_returns: list[float] = []
            total_trades = 0
            for timeframe in TIMEFRAMES:
                month_results = [
                    run(data[(timeframe, month)], lower, upper, args.commission, args.initial_capital)
                    for month in months
                ]
                month_returns = [float(r["return_pct"]) for r in month_results]
                combined = compounded_return(month_returns)
                total_tf_trades = sum(int(r["trades"]) for r in month_results)
                tf_results[timeframe] = {
                    "combined": combined,
                    "avg_month": sum(month_returns) / len(month_returns),
                    "worst_month": min(month_returns),
                    "trades": total_tf_trades,
                    "months": month_returns,
                }
                all_returns.extend(month_returns)
                total_trades += total_tf_trades

            combined_returns = [float(tf_results[tf]["combined"]) for tf in TIMEFRAMES]
            worst_tf = min(combined_returns)
            avg_tf = sum(combined_returns) / len(combined_returns)
            positive_tf = sum(x > 0 for x in combined_returns)
            positive_months = sum(x > 0 for x in all_returns)
            rows.append((lower, upper, avg_tf, worst_tf, positive_tf, positive_months, total_trades, tf_results))

    rows.sort(key=lambda x: (x[4], x[3], x[2], x[5], x[6]), reverse=True)

    print(
        f"COMMON_TIMEFRAME_BIAS | symbol={args.symbol} | months={','.join(months)} | "
        f"commission={args.commission * 100:.2f}% per side"
    )
    print("LOWER | UPPER | AVG_TF | WORST_TF | POS_TF | POS_MONTHS | TRADES | 5m_COMB | 15m_COMB | 1h_COMB")
    for lower, upper, avg_tf, worst_tf, positive_tf, positive_months, total_trades, tf_results in rows[:15]:
        print(
            f"{lower:>6.0f} | {upper:>5.0f} | {avg_tf:>+7.2f}% | {worst_tf:>+8.2f}% | "
            f"{positive_tf:>5}/3 | {positive_months:>9}/{len(months) * len(TIMEFRAMES)} | {total_trades:>6} | "
            f"{float(tf_results['5m']['combined']):>+7.2f}% | "
            f"{float(tf_results['15m']['combined']):>8.2f}% | "
            f"{float(tf_results['1h']['combined']):>7.2f}%"
        )

    baseline = next((r for r in rows if r[0] == -100.0 and r[1] == 100.0), None)
    if baseline:
        _, _, avg_tf, worst_tf, positive_tf, positive_months, total_trades, tf_results = baseline
        print(
            f"BASELINE[-100,+100] | avg_tf={avg_tf:+.2f}% worst_tf={worst_tf:+.2f}% "
            f"positive_tf={positive_tf}/3 positive_months={positive_months}/{len(months) * len(TIMEFRAMES)} total_trades={total_trades} "
            f"5m={float(tf_results['5m']['combined']):+.2f}% 15m={float(tf_results['15m']['combined']):+.2f}% 1h={float(tf_results['1h']['combined']):+.2f}%"
        )

    robust = [r for r in rows if r[4] == len(TIMEFRAMES) and r[6] >= 50]
    if robust:
        best = robust[0]
        print(
            f"COMMON_TIMEFRAME_ROBUST_BEST | lower={best[0]:.0f} upper={best[1]:.0f} "
            f"avg_tf={best[2]:+.2f}% worst_tf={best[3]:+.2f}% positive_tf={best[4]}/3 "
            f"positive_months={best[5]}/{len(months) * len(TIMEFRAMES)} total_trades={best[6]}"
        )
    else:
        print("COMMON_TIMEFRAME_ROBUST_BEST | none (requires all 3 timeframe combined returns positive and >=50 trades)")


if __name__ == "__main__":
    main()
