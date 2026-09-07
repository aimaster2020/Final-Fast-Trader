from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal

DEFAULT_VALUES = tuple(float(x) for x in range(-200, 201, 25))
DEFAULT_MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
TIMEFRAMES = ("5m", "15m", "1h", "4h")


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
    candles = []
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
        expected = [bucket_ts + 3600 * i for i in range(4)]
        if [c.timestamp for c in group] != expected:
            continue
        if month_from_timestamp(bucket_ts) != month:
            continue
        out.append(Candle(bucket_ts, group[0].open, max(c.high for c in group), min(c.low for c in group), group[-1].close))
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


def run(candles: list[Candle], lower: float, upper: float, commission: float, initial_capital: float = 1000.0) -> dict[str, float | int]:
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
            exit_triggered = (side > 0 and opposite and body <= upper) or (side < 0 and opposite and body >= lower)
            if exit_triggered:
                gross = capital * pnl_pct(side, entry, candle.close)
                net = gross - capital * commission * 2.0
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
        net = gross - capital * commission * 2.0
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
    ap = argparse.ArgumentParser(description="Optimize one common asymmetric BODY bias across 5m/15m/1h/4h with commission. 4h is built from 1h.")
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
    paths = {"5m": Path(args.input_5m), "15m": Path(args.input_15m), "1h": Path(args.input_1h)}

    data: dict[tuple[str, str], list[Candle]] = {}
    for timeframe in ("5m", "15m", "1h"):
        for month in months:
            candles = load_month(paths[timeframe], month, args.symbol)
            if not candles:
                raise SystemExit(f"No usable {timeframe} candles for {args.symbol} {month}: {paths[timeframe]}")
            data[(timeframe, month)] = candles
    for month in months:
        candles = resample_1h_to_4h(paths["1h"], args.symbol, month)
        if not candles:
            raise SystemExit(f"No usable 4h candles built from 1h for {args.symbol} {month}: {paths['1h']}")
        data[("4h", month)] = candles

    rows = []
    for lower in values:
        for upper in values:
            if lower >= upper:
                continue
            tf_results = {}
            all_returns: list[float] = []
            total_trades = 0
            for timeframe in TIMEFRAMES:
                month_results = [run(data[(timeframe, month)], lower, upper, args.commission, args.initial_capital) for month in months]
                month_returns = [float(r["return_pct"]) for r in month_results]
                combined = compounded_return(month_returns)
                total_tf_trades = sum(int(r["trades"]) for r in month_results)
                total_days = sum(len(data[(timeframe, month)]) for month in months)
                tf_results[timeframe] = {"combined": combined, "avg_month": sum(month_returns) / len(month_returns), "worst_month": min(month_returns), "trades": total_tf_trades, "days": total_days, "trades_per_day": total_tf_trades / total_days if total_days else 0.0, "months": month_returns}
                all_returns.extend(month_returns)
                total_trades += total_tf_trades

            combined_returns = [float(tf_results[tf]["combined"]) for tf in TIMEFRAMES]
            worst_tf = min(combined_returns)
            avg_tf = sum(combined_returns) / len(combined_returns)
            positive_tf = sum(x > 0 for x in combined_returns)
            positive_months = sum(x > 0 for x in all_returns)
            rows.append((lower, upper, avg_tf, worst_tf, positive_tf, positive_months, total_trades, tf_results))

    rows.sort(key=lambda x: (x[4], x[3], x[2], x[5], x[6]), reverse=True)
    print(f"COMMISSION_OPT | symbol={args.symbol} | months={','.join(months)} | commission={args.commission * 100:.2f}% per side | roundtrip={args.commission * 200:.2f}% | 4h=RESAMPLED_FROM_1H")
    print("LOWER | UPPER | AVG_TF | WORST_TF | POS_TF | POS_MONTHS | TRADES | 5m_T/D | 15m_T/D | 1h_T/D | 4h_T/D | 5m_COMB | 15m_COMB | 1h_COMB | 4h_COMB")
    for lower, upper, avg_tf, worst_tf, positive_tf, positive_months, total_trades, tf_results in rows[:15]:
        print(f"{lower:>6.0f} | {upper:>5.0f} | {avg_tf:>+7.2f}% | {worst_tf:>+8.2f}% | {positive_tf:>5}/4 | {positive_months:>9}/{len(months) * len(TIMEFRAMES)} | {total_trades:>6} | {float(tf_results['5m']['trades_per_day']):>7.2f} | {float(tf_results['15m']['trades_per_day']):>8.2f} | {float(tf_results['1h']['trades_per_day']):>7.2f} | {float(tf_results['4h']['trades_per_day']):>7.2f} | {float(tf_results['5m']['combined']):>+8.2f}% | {float(tf_results['15m']['combined']):>+9.2f}% | {float(tf_results['1h']['combined']):>+8.2f}% | {float(tf_results['4h']['combined']):>+8.2f}%")

    baseline = next((r for r in rows if r[0] == -100.0 and r[1] == 100.0), None)
    if baseline:
        _, _, avg_tf, worst_tf, positive_tf, positive_months, total_trades, tf_results = baseline
        print(f"BASELINE[-100,+100] | avg_tf={avg_tf:+.2f}% worst_tf={worst_tf:+.2f}% positive_tf={positive_tf}/4 positive_months={positive_months}/{len(months) * len(TIMEFRAMES)} total_trades={total_trades} 5m={float(tf_results['5m']['combined']):+.2f}%/{float(tf_results['5m']['trades_per_day']):.2f}T/D 15m={float(tf_results['15m']['combined']):+.2f}%/{float(tf_results['15m']['trades_per_day']):.2f}T/D 1h={float(tf_results['1h']['combined']):+.2f}%/{float(tf_results['1h']['trades_per_day']):.2f}T/D 4h={float(tf_results['4h']['combined']):+.2f}%/{float(tf_results['4h']['trades_per_day']):.2f}T/D")

    robust = [r for r in rows if r[4] == len(TIMEFRAMES) and r[6] >= 100]
    if robust:
        best = robust[0]
        print(f"OPTIMIZED_BEST | lower={best[0]:.0f} upper={best[1]:.0f} avg_tf={best[2]:+.2f}% worst_tf={best[3]:+.2f}% positive_tf={best[4]}/4 positive_months={best[5]}/{len(months) * len(TIMEFRAMES)} total_trades={best[6]} 5m={float(best[7]['5m']['combined']):+.2f}%/{float(best[7]['5m']['trades_per_day']):.2f}T/D 15m={float(best[7]['15m']['combined']):+.2f}%/{float(best[7]['15m']['trades_per_day']):.2f}T/D 1h={float(best[7]['1h']['combined']):+.2f}%/{float(best[7]['1h']['trades_per_day']):.2f}T/D 4h={float(best[7]['4h']['combined']):+.2f}%/{float(best[7]['4h']['trades_per_day']):.2f}T/D")
    else:
        print("OPTIMIZED_BEST | none (requires all 4 timeframe combined returns positive and >=100 trades)")


if __name__ == "__main__":
    main()
