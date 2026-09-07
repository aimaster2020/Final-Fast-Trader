from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal

TIMEFRAMES = ("5m", "15m", "1h")
DEFAULT_VALUES = tuple(float(x) for x in range(-200, 201, 25))
DEFAULT_MONTHS = "2026-05,2026-06,2026-07,2026-08"
DEFAULT_COMMISSION = 0.0013


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


def observed_days(candles: list[Candle]) -> int:
    days = {
        datetime.fromtimestamp(c.timestamp, tz=timezone.utc).date()
        for c in candles
    }
    return len(days)


def run(
    candles: list[Candle],
    lower: float,
    upper: float,
    commission: float,
    initial_capital: float,
) -> dict[str, float | int]:
    equity = initial_capital
    position: tuple[int, float, float] | None = None
    trades = wins = 0
    fees_total = 0.0

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
                fees_total += fees
                net = gross - fees
                equity += net
                trades += 1
                wins += int(net > 0)
                position = None

    if position is not None:
        side, entry, capital = position
        gross = capital * pnl_pct(side, entry, candles[-1].close)
        fees = capital * commission * 2.0
        fees_total += fees
        net = gross - fees
        equity += net
        trades += 1
        wins += int(net > 0)

    return {
        "return_pct": (equity / initial_capital - 1.0) * 100.0,
        "trades": trades,
        "win_rate": wins / trades * 100.0 if trades else 0.0,
        "fees": fees_total,
        "days": observed_days(candles),
    }


def compounded(returns: list[float]) -> float:
    factor = 1.0
    for r in returns:
        factor *= 1.0 + r / 100.0
    return (factor - 1.0) * 100.0


def parse_values(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(x.strip()) for x in raw.split(",") if x.strip()}))
    if len(values) < 2:
        raise SystemExit("Need at least two threshold values")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Commission-aware BODY bias optimizer with average completed trades per observed day."
    )
    ap.add_argument("--months", default=DEFAULT_MONTHS)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--values", default=','.join(str(int(v)) for v in DEFAULT_VALUES))
    ap.add_argument("--input-5m", default="reports/prepared_price_action_5m.csv")
    ap.add_argument("--input-15m", default="reports/prepared_price_action_15m.csv")
    ap.add_argument("--input-1h", default="reports/prepared_price_action_1h.csv")
    args = ap.parse_args()

    months = tuple(x.strip() for x in args.months.split(",") if x.strip())
    values = parse_values(args.values)
    paths = {"5m": Path(args.input_5m), "15m": Path(args.input_15m), "1h": Path(args.input_1h)}

    data: dict[tuple[str, str], list[Candle]] = {}
    for tf, path in paths.items():
        for month in months:
            candles = load_month(path, month, args.symbol)
            if not candles:
                raise SystemExit(f"No usable {tf} candles for {args.symbol} {month}: {path}")
            data[(tf, month)] = candles

    rows = []
    for lower in values:
        for upper in values:
            if lower >= upper:
                continue

            tf_results: dict[str, dict[str, float | int | list[dict[str, float | int]]]] = {}
            combined_returns: list[float] = []
            total_trades = 0
            total_days = 0

            for tf in TIMEFRAMES:
                month_results = []
                for month in months:
                    result = run(data[(tf, month)], lower, upper, args.commission, args.initial_capital)
                    result["month"] = month
                    month_results.append(result)
                returns = [float(x["return_pct"]) for x in month_results]
                tf_combined = compounded(returns)
                tf_trades = sum(int(x["trades"]) for x in month_results)
                tf_days = sum(int(x["days"]) for x in month_results)
                tf_avg_daily = tf_trades / tf_days if tf_days else 0.0
                tf_results[tf] = {
                    "combined": tf_combined,
                    "trades": tf_trades,
                    "days": tf_days,
                    "trades_per_day": tf_avg_daily,
                    "months": month_results,
                }
                combined_returns.append(tf_combined)
                total_trades += tf_trades
                total_days += tf_days

            worst_tf = min(combined_returns)
            avg_tf = sum(combined_returns) / len(combined_returns)
            positive_tf = sum(x > 0 for x in combined_returns)
            total_tpd = total_trades / (total_days / len(TIMEFRAMES)) if total_days else 0.0
            rows.append((lower, upper, avg_tf, worst_tf, positive_tf, total_trades, total_tpd, tf_results))

    # First require all three timeframe combined returns positive.
    # Then maximize the weakest timeframe, average return, and lower trading frequency.
    rows.sort(key=lambda x: (x[4], x[3], x[2], -x[6], x[5]), reverse=True)

    print(
        f"COMMISSION_OPT | symbol={args.symbol} | months={','.join(months)} | "
        f"commission={args.commission * 100:.2f}% per side | roundtrip={args.commission * 200:.2f}%"
    )
    print("LOWER | UPPER | AVG_TF | WORST_TF | POS_TF | TRADES | 5m_T/D | 15m_T/D | 1h_T/D | 5m_COMB | 15m_COMB | 1h_COMB")

    candidates = [r for r in rows if r[4] == 3 and r[5] >= 50]
    display = candidates[:15] if candidates else rows[:15]
    for lower, upper, avg_tf, worst_tf, positive_tf, total_trades, total_tpd, tf_results in display:
        print(
            f"{lower:>6.0f} | {upper:>5.0f} | {avg_tf:>+7.2f}% | {worst_tf:>+8.2f}% | {positive_tf}/3 | "
            f"{total_trades:>6} | {float(tf_results['5m']['trades_per_day']):>7.2f} | "
            f"{float(tf_results['15m']['trades_per_day']):>8.2f} | {float(tf_results['1h']['trades_per_day']):>7.2f} | "
            f"{float(tf_results['5m']['combined']):>+8.2f}% | {float(tf_results['15m']['combined']):>+9.2f}% | "
            f"{float(tf_results['1h']['combined']):>+8.2f}%"
        )

    baseline = next((r for r in rows if r[0] == -100.0 and r[1] == 100.0), None)
    if baseline:
        _, _, avg_tf, worst_tf, positive_tf, total_trades, _, tf_results = baseline
        print(
            f"BASELINE[-100,+100] | avg_tf={avg_tf:+.2f}% worst_tf={worst_tf:+.2f}% positive_tf={positive_tf}/3 "
            f"trades={total_trades} | 5m={float(tf_results['5m']['combined']):+.2f}%/{float(tf_results['5m']['trades_per_day']):.2f}T/D "
            f"15m={float(tf_results['15m']['combined']):+.2f}%/{float(tf_results['15m']['trades_per_day']):.2f}T/D "
            f"1h={float(tf_results['1h']['combined']):+.2f}%/{float(tf_results['1h']['trades_per_day']):.2f}T/D"
        )

    if candidates:
        best = candidates[0]
        print(
            f"OPTIMIZED_BEST | lower={best[0]:.0f} upper={best[1]:.0f} avg_tf={best[2]:+.2f}% "
            f"worst_tf={best[3]:+.2f}% positive_tf={best[4]}/3 trades={best[5]} "
            f"5m_T/D={float(best[7]['5m']['trades_per_day']):.2f} "
            f"15m_T/D={float(best[7]['15m']['trades_per_day']):.2f} "
            f"1h_T/D={float(best[7]['1h']['trades_per_day']):.2f}"
        )
    else:
        print("OPTIMIZED_BEST | none (requires all 3 timeframe combined returns positive and >=50 trades)")


if __name__ == "__main__":
    main()
