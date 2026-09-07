from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal

DEFAULT_VALUES = (-200.0, -150.0, -100.0, -75.0, -50.0, 50.0, 75.0, 100.0, 150.0, 200.0)
DEFAULT_MONTHS = ("2026-05", "2026-07")


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


def run(candles: list[Candle], lower: float, upper: float) -> dict[str, float | int]:
    equity = 1000.0
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
                equity += gross
                trades += 1
                if gross > 0:
                    wins += 1
                    gross_profit += gross
                elif gross < 0:
                    gross_loss += gross
                position = None

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

    return {
        "return_pct": (equity / 1000.0 - 1.0) * 100.0,
        "trades": trades,
        "win_rate": wins / trades * 100.0 if trades else 0.0,
        "profit_factor": gross_profit / abs(gross_loss) if gross_loss < 0 else (float("inf") if gross_profit else 0.0),
    }


def parse_values(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(x.strip()) for x in raw.split(",") if x.strip()}))
    if len(values) < 2:
        raise SystemExit("Need at least two threshold values")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description="Find one common asymmetric BODY bias for 5m/15m/1h across multiple months.")
    ap.add_argument("--months", default=",".join(DEFAULT_MONTHS))
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
    paths = {
        "5m": Path(args.input_5m),
        "15m": Path(args.input_15m),
        "1h": Path(args.input_1h),
    }

    data: dict[tuple[str, str], list[Candle]] = {}
    for timeframe, path in paths.items():
        for month in months:
            candles = load_month(path, month, args.symbol)
            if not candles:
                raise SystemExit(f"No usable {timeframe} candles for {args.symbol} {month}: {path}")
            data[(timeframe, month)] = candles

    rows = []
    for lower in values:
        for upper in values:
            if lower >= upper:
                continue

            cells: list[tuple[str, str, dict[str, float | int]]] = []
            for timeframe in ("5m", "15m", "1h"):
                for month in months:
                    result = run(data[(timeframe, month)], lower, upper)
                    cells.append((timeframe, month, result))

            returns = [float(r["return_pct"]) for _, _, r in cells]
            trades = [int(r["trades"]) for _, _, r in cells]
            avg_return = sum(returns) / len(returns)
            worst_return = min(returns)
            positive_cells = sum(x > 0 for x in returns)
            total_trades = sum(trades)
            rows.append((lower, upper, avg_return, worst_return, positive_cells, total_trades, cells))

    # Primary ranking: all timeframe/month cells positive, then strongest worst cell,
    # then average return, then total trade count. This avoids choosing a pair that
    # wins only because one timeframe/month is unusually strong.
    rows.sort(key=lambda x: (x[4], x[3], x[2], x[5]), reverse=True)

    print(
        f"COMMON_BIAS | symbol={args.symbol} | months={','.join(months)} | "
        f"commission={args.commission * 100:.2f}%"
    )
    print("LOWER | UPPER | AVG_6 | WORST_6 | POS_6 | TOTAL_TRADES | 5m_MAY | 5m_JUL | 15m_MAY | 15m_JUL | 1h_MAY | 1h_JUL")

    for lower, upper, avg_return, worst_return, positive_cells, total_trades, cells in rows[:15]:
        by_key = {(tf, mo): r for tf, mo, r in cells}
        values_text = " | ".join(
            f"{float(by_key[(tf, mo)]['return_pct']):+.2f}%/{int(by_key[(tf, mo)]['trades'])}"
            for tf, mo in (("5m", months[0]), ("5m", months[-1]), ("15m", months[0]), ("15m", months[-1]), ("1h", months[0]), ("1h", months[-1]))
        )
        print(
            f"{lower:>6.0f} | {upper:>5.0f} | {avg_return:>+6.2f}% | {worst_return:>+7.2f}% | "
            f"{positive_cells:>4}/6 | {total_trades:>12} | {values_text}"
        )

    baseline = next((r for r in rows if r[0] == -100.0 and r[1] == 100.0), None)
    if baseline:
        print(
            f"BASELINE[-100,+100] | avg={baseline[2]:+.2f}% worst={baseline[3]:+.2f}% "
            f"positive={baseline[4]}/6 total_trades={baseline[5]}"
        )

    robust = [r for r in rows if r[4] == len(data) and r[5] >= 20]
    if robust:
        best = robust[0]
        print(
            f"COMMON_ROBUST_BEST | lower={best[0]:.0f} upper={best[1]:.0f} "
            f"avg={best[2]:+.2f}% worst={best[3]:+.2f}% positive={best[4]}/6 total_trades={best[5]}"
        )
    else:
        print("COMMON_ROBUST_BEST | none (requires all timeframe/month cells positive and >=20 total trades)")


if __name__ == "__main__":
    main()
