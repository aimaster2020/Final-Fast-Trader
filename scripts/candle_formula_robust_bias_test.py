from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal

DEFAULT_VALUES = (-200.0, -150.0, -100.0, -75.0, -50.0, 50.0, 75.0, 100.0, 150.0, 200.0)


def load_month(path: Path, month: str, symbol: str) -> list[Candle]:
    candles: list[Candle] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("month") != month or row.get("symbol") != symbol:
                continue
            try:
                candles.append(Candle(
                    int(float(row["timestamp"])),
                    float(row["open"]), float(row["high"]),
                    float(row["low"]), float(row["close"])),
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


def run(candles: list[Candle], lower: float, upper: float) -> dict[str, float | int]:
    equity = 1000.0
    position: tuple[int, float, float] | None = None
    trades = wins = 0
    gross_profit = gross_loss = 0.0
    long_entries = short_entries = ignored = 0
    peak = equity
    max_dd = 0.0

    for candle in candles:
        sig = signal_side(candle)
        body = candle.close - candle.open

        if position is None:
            if sig != 0 and not (lower <= body <= upper):
                position = (sig, candle.close, equity)
                if sig > 0:
                    long_entries += 1
                else:
                    short_entries += 1
            elif sig != 0:
                ignored += 1
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
                equity += gross
                trades += 1
                if gross > 0:
                    wins += 1
                    gross_profit += gross
                elif gross < 0:
                    gross_loss += gross
                position = None

        mtm = equity
        if position is not None:
            side, entry, capital = position
            mtm += capital * pnl_pct(side, entry, candle.close)
        peak = max(peak, mtm)
        max_dd = max(max_dd, (peak - mtm) / peak if peak else 0.0)

    if position is not None:
        side, entry, capital = position
        gross = capital * pnl_pct(side, entry, candles[-1].close)
        equity += gross
        trades += 1
        if gross > 0:
            wins += 1
            gross_profit += gross
        elif gross < 0:
            gross_loss += gross

    win_rate = wins / trades * 100.0 if trades else 0.0
    pf = gross_profit / abs(gross_loss) if gross_loss < 0 else (float("inf") if gross_profit else 0.0)
    return {
        "final": equity,
        "return_pct": (equity / 1000.0 - 1.0) * 100.0,
        "trades": trades,
        "win_rate": win_rate,
        "profit_factor": pf,
        "max_dd": max_dd * 100.0,
        "long_entries": long_entries,
        "short_entries": short_entries,
        "ignored": ignored,
    }


def parse_values(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(x.strip()) for x in raw.split(",") if x.strip()}))
    if len(values) < 2:
        raise SystemExit("Need at least two threshold values")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description="Robust multi-month asymmetric BODY range bias sweep.")
    ap.add_argument("--months", default="2026-05,2026-07")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--commission", type=float, default=0.0)
    ap.add_argument("--allocation", type=float, default=1.0)
    ap.add_argument("--values", default=",".join(str(v) for v in DEFAULT_VALUES))
    ap.add_argument("--input-5m", default="reports/prepared_price_action_5m.csv")
    ap.add_argument("--input-15m", default="reports/prepared_price_action_15m.csv")
    ap.add_argument("--input-1h", default="reports/prepared_price_action_1h.csv")
    args = ap.parse_args()

    months = tuple(x.strip() for x in args.months.split(",") if x.strip())
    values = parse_values(args.values)
    jobs = (("5m", Path(args.input_5m)), ("15m", Path(args.input_15m)), ("1h", Path(args.input_1h)))

    for timeframe, path in jobs:
        month_candles = {m: load_month(path, m, args.symbol) for m in months}
        if any(not month_candles[m] for m in months):
            missing = [m for m in months if not month_candles[m]]
            raise SystemExit(f"No usable {timeframe} candles for: {', '.join(missing)}")

        rows = []
        for lower in values:
            for upper in values:
                if lower >= upper:
                    continue
                per_month = [run(month_candles[m], lower, upper) for m in months]
                returns = [float(r["return_pct"]) for r in per_month]
                trades = [int(r["trades"]) for r in per_month]
                avg_return = sum(returns) / len(returns)
                worst_return = min(returns)
                positive_months = sum(r > 0 for r in returns)
                total_trades = sum(trades)
                rows.append((lower, upper, avg_return, worst_return, positive_months, total_trades, per_month))

        # Robust ranking: positive months first, then worst month, average return, and total trades.
        rows.sort(key=lambda x: (x[4], x[3], x[2], x[5]), reverse=True)

        print(f"\n{timeframe} | months={','.join(months)} | commission={args.commission * 100:.2f}%")
        header = "LOWER | UPPER | AVG | WORST | POS_MONTHS | TOTAL_TRADES"
        for idx, month in enumerate(months):
            header += f" | {month}"
        print(header)

        for lower, upper, avg_return, worst_return, positive_months, total_trades, per_month in rows:
            monthly = " | ".join(f"{float(r['return_pct']):+.2f}%/{int(r['trades'])}" for r in per_month)
            print(f"{lower:>6.0f} | {upper:>5.0f} | {avg_return:>+6.2f}% | {worst_return:>+6.2f}% | {positive_months:>10}/{len(months)} | {total_trades:>12} | {monthly}")

        baseline = next((r for r in rows if r[0] == -100.0 and r[1] == 100.0), None)
        if baseline:
            print(f"BASELINE[-100,+100] {timeframe} | avg={baseline[2]:+.2f}% worst={baseline[3]:+.2f}% positive={baseline[4]}/{len(months)} total_trades={baseline[5]}")

        # Pick the best robust result requiring every month to be positive and at least 10 total trades.
        robust = [r for r in rows if r[4] == len(months) and r[5] >= 10]
        if robust:
            best = robust[0]
            print(f"ROBUST_BEST {timeframe} | lower={best[0]:.0f} upper={best[1]:.0f} avg={best[2]:+.2f}% worst={best[3]:+.2f}% months={best[4]}/{len(months)} total_trades={best[5]}")
        else:
            best = rows[0]
            print(f"BEST_ANY {timeframe} | lower={best[0]:.0f} upper={best[1]:.0f} avg={best[2]:+.2f}% worst={best[3]:+.2f}% positive={best[4]}/{len(months)} total_trades={best[5]}")


if __name__ == "__main__":
    main()
