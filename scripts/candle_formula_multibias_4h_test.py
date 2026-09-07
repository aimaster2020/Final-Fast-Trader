from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal

TIMEFRAMES_FAST = ("5m", "15m", "1h")
TIMEFRAMES_ALL = ("5m", "15m", "1h", "4h")
DEFAULT_VALUES = tuple(float(x) for x in range(-200, 201, 25))
DEFAULT_MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")


def read_rows(path: Path, symbol: str) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return [row for row in csv.DictReader(f) if row.get("symbol") == symbol]


def row_to_candle(row: dict[str, str]) -> Candle | None:
    try:
        return Candle(
            int(float(row["timestamp"])),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_month(path: Path, month: str, symbol: str) -> list[Candle]:
    candles: list[Candle] = []
    for row in read_rows(path, symbol):
        if row.get("month") != month:
            continue
        candle = row_to_candle(row)
        if candle is not None:
            candles.append(candle)
    return sorted(candles, key=lambda c: c.timestamp)


def month_from_timestamp(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m")


def resample_1h_to_4h(path: Path, symbol: str, month: str) -> list[Candle]:
    hourly: list[Candle] = []
    for row in read_rows(path, symbol):
        candle = row_to_candle(row)
        if candle is not None:
            hourly.append(candle)
    hourly.sort(key=lambda c: c.timestamp)

    buckets: dict[int, list[Candle]] = {}
    for candle in hourly:
        bucket = candle.timestamp - (candle.timestamp % 14_400)
        buckets.setdefault(bucket, []).append(candle)

    out: list[Candle] = []
    for bucket_ts in sorted(buckets):
        group = sorted(buckets[bucket_ts], key=lambda c: c.timestamp)
        if len(group) != 4:
            continue
        expected = [bucket_ts + 3_600 * i for i in range(4)]
        if [c.timestamp for c in group] != expected:
            continue
        if month_from_timestamp(bucket_ts) != month:
            continue
        out.append(
            Candle(
                bucket_ts,
                group[0].open,
                max(c.high for c in group),
                min(c.low for c in group),
                group[-1].close,
            )
        )
    return out


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
    candles: list[Candle], lower: float, upper: float, commission: float, initial_capital: float
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
            continue

        side, entry, capital = position
        opposite = sig != 0 and sig != side
        exit_triggered = (side > 0 and opposite and body <= upper) or (
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

    if position is not None and candles:
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
        "profit_factor": gross_profit / abs(gross_loss)
        if gross_loss < 0
        else (float("inf") if gross_profit else 0.0),
    }


def compounded(returns: list[float]) -> float:
    factor = 1.0
    for value in returns:
        factor *= 1.0 + value / 100.0
    return (factor - 1.0) * 100.0


def parse_values(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(x.strip()) for x in raw.split(",") if x.strip()}))
    if len(values) < 2:
        raise SystemExit("Need at least two threshold values")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Test one common BODY bias for 5m/15m/1h and an independent bias for 4h. "
            "4h is resampled from 1h."
        )
    )
    ap.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--commission", type=float, default=0.0013)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--values", default=",".join(str(v) for v in DEFAULT_VALUES))
    ap.add_argument("--input-5m", default="reports/prepared_price_action_5m.csv")
    ap.add_argument("--input-15m", default="reports/prepared_price_action_15m.csv")
    ap.add_argument("--input-1h", default="reports/prepared_price_action_1h.csv")
    args = ap.parse_args()

    months = tuple(x.strip() for x in args.months.split(",") if x.strip())
    values = parse_values(args.values)
    paths = {
        "5m": Path(args.input_5m),
        "15m": Path(args.input_15m),
        "1h": Path(args.input_1h),
    }

    data: dict[tuple[str, str], list[Candle]] = {}
    for timeframe in TIMEFRAMES_FAST:
        for month in months:
            candles = load_month(paths[timeframe], month, args.symbol)
            if not candles:
                raise SystemExit(f"No usable {timeframe} candles for {args.symbol} {month}")
            data[(timeframe, month)] = candles
    for month in months:
        candles = resample_1h_to_4h(paths["1h"], args.symbol, month)
        if not candles:
            raise SystemExit(f"No usable 4h candles built from 1h for {args.symbol} {month}")
        data[("4h", month)] = candles

    fast_pairs = [(lower, upper) for lower in values for upper in values if lower < upper]
    cache: dict[tuple[str, str, float, float], dict[str, float | int]] = {}
    for tf in TIMEFRAMES_FAST:
        for month in months:
            for lower, upper in fast_pairs:
                cache[(tf, month, lower, upper)] = run(
                    data[(tf, month)], lower, upper, args.commission, args.initial_capital
                )

    four_hour_pairs = fast_pairs
    results = []
    for fast_lower, fast_upper in fast_pairs:
        fast_tf = {}
        for tf in TIMEFRAMES_FAST:
            monthly = [cache[(tf, month, fast_lower, fast_upper)] for month in months]
            returns = [float(r["return_pct"]) for r in monthly]
            trades = sum(int(r["trades"]) for r in monthly)
            days = sum(len(data[(tf, month)]) for month in months)
            fast_tf[tf] = {
                "combined": compounded(returns),
                "returns": returns,
                "trades": trades,
                "trades_per_day": trades / days if days else 0.0,
            }

        for h4_lower, h4_upper in four_hour_pairs:
            monthly = [
                run(data[("4h", month)], h4_lower, h4_upper, args.commission, args.initial_capital)
                for month in months
            ]
            returns = [float(r["return_pct"]) for r in monthly]
            trades = sum(int(r["trades"]) for r in monthly)
            days = sum(len(data[("4h", month)]) for month in months)
            four = {
                "combined": compounded(returns),
                "returns": returns,
                "trades": trades,
                "trades_per_day": trades / days if days else 0.0,
            }
            tf_combined = [float(fast_tf[tf]["combined"]) for tf in TIMEFRAMES_FAST] + [four["combined"]]
            positive_tf = sum(x > 0 for x in tf_combined)
            worst_tf = min(tf_combined)
            avg_tf = sum(tf_combined) / 4.0
            results.append((
                fast_lower, fast_upper, h4_lower, h4_upper,
                avg_tf, worst_tf, positive_tf,
                sum(x > 0 for x in fast_tf["5m"]["returns"])
                + sum(x > 0 for x in fast_tf["15m"]["returns"])
                + sum(x > 0 for x in fast_tf["1h"]["returns"])
                + sum(x > 0 for x in four["returns"]),
                sum(int(fast_tf[tf]["trades"]) for tf in TIMEFRAMES_FAST) + trades,
                fast_tf,
                four,
            ))

    results.sort(key=lambda x: (x[6], x[5], x[4], x[7], x[8]), reverse=True)

    print(
        f"MULTIBIAS_4H | symbol={args.symbol} | months={','.join(months)} | "
        f"commission={args.commission * 100:.2f}% per side | roundtrip={args.commission * 200:.2f}%"
    )
    print(
        "FAST_BIAS(5m/15m/1h) | 4H_BIAS | AVG_TF | WORST_TF | POS_TF | "
        "POS_MONTHS | TRADES | 5m_T/D | 15m_T/D | 1h_T/D | 4h_T/D | 5m | 15m | 1h | 4h"
    )
    for row in results[:20]:
        fast_lower, fast_upper, h4_lower, h4_upper, avg_tf, worst_tf, positive_tf, positive_months, total_trades, fast_tf, four = row
        print(
            f"[{fast_lower:.0f},{fast_upper:.0f}] | [{h4_lower:.0f},{h4_upper:.0f}] | "
            f"{avg_tf:+6.2f}% | {worst_tf:+7.2f}% | {positive_tf}/4 | "
            f"{positive_months}/{len(months) * 4} | {total_trades:6} | "
            f"{float(fast_tf['5m']['trades_per_day']):6.2f} | "
            f"{float(fast_tf['15m']['trades_per_day']):7.2f} | "
            f"{float(fast_tf['1h']['trades_per_day']):6.2f} | "
            f"{float(four['trades_per_day']):6.2f} | "
            f"{float(fast_tf['5m']['combined']):+6.2f}% | "
            f"{float(fast_tf['15m']['combined']):+7.2f}% | "
            f"{float(fast_tf['1h']['combined']):+6.2f}% | "
            f"{float(four['combined']):+6.2f}%"
        )

    baseline_fast = None
    for row in results:
        if row[0] == -100.0 and row[1] == 100.0 and row[2] == -100.0 and row[3] == 100.0:
            baseline_fast = row
            break
    if baseline_fast:
        _, _, _, _, avg_tf, worst_tf, positive_tf, positive_months, total_trades, fast_tf, four = baseline_fast
        print(
            f"BASELINE[-100,+100] | avg_tf={avg_tf:+.2f}% worst_tf={worst_tf:+.2f}% positive_tf={positive_tf}/4 "
            f"positive_months={positive_months}/{len(months)*4} total_trades={total_trades} "
            f"5m={float(fast_tf['5m']['combined']):+.2f}% 15m={float(fast_tf['15m']['combined']):+.2f}% "
            f"1h={float(fast_tf['1h']['combined']):+.2f}% 4h={float(four['combined']):+.2f}%"
        )

    robust = [r for r in results if r[6] == 4 and r[8] >= 100]
    if robust:
        best = robust[0]
        print(
            f"MULTIBIAS_ROBUST_BEST | fast=[{best[0]:.0f},{best[1]:.0f}] 4h=[{best[2]:.0f},{best[3]:.0f}] "
            f"avg_tf={best[4]:+.2f}% worst_tf={best[5]:+.2f}% positive_tf={best[6]}/4 "
            f"positive_months={best[7]}/{len(months)*4} total_trades={best[8]}"
        )
    else:
        print("MULTIBIAS_ROBUST_BEST | none (requires all 4 timeframe combined returns positive and >=100 trades)")


if __name__ == "__main__":
    main()
