from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal

DEFAULT_COMMISSION = 0.0
DEFAULT_VALUES = (-200.0, -150.0, -100.0, -75.0, -50.0, 50.0, 75.0, 100.0, 150.0, 200.0)


@dataclass
class Position:
    side: int
    entry_price: float
    capital: float


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
    if entry <= 0:
        return 0.0
    return side * (exit_price - entry) / entry


def run_backtest(
    candles: list[Candle],
    lower: float,
    upper: float,
    initial_capital: float,
    commission: float,
    allocation: float,
) -> dict[str, float | int]:
    equity = initial_capital
    position: Position | None = None
    trades = wins = losses = 0
    gross_profit = gross_loss = 0.0
    long_entries = short_entries = 0
    ignored_in_range = 0
    peak = initial_capital
    max_dd = 0.0

    for candle in candles:
        sig = signal_side(candle)
        body = candle.close - candle.open

        if position is None:
            if sig != 0 and not (lower <= body <= upper):
                position = Position(sig, candle.close, equity * allocation)
                if sig > 0:
                    long_entries += 1
                else:
                    short_entries += 1
            elif sig != 0:
                ignored_in_range += 1
        else:
            opposite = sig != 0 and sig != position.side
            if opposite:
                exit_triggered = (
                    position.side > 0 and body <= upper
                ) or (
                    position.side < 0 and body >= lower
                )
                if exit_triggered:
                    gross = position.capital * pnl_pct(
                        position.side, position.entry_price, candle.close
                    )
                    fees = position.capital * commission * 2.0
                    net = gross - fees
                    equity += net
                    trades += 1
                    if net > 0:
                        wins += 1
                    elif net < 0:
                        losses += 1
                    gross_profit += max(gross, 0.0)
                    gross_loss += min(gross, 0.0)
                    position = None

        mtm = equity
        if position is not None:
            mtm += position.capital * pnl_pct(
                position.side, position.entry_price, candle.close
            )
        peak = max(peak, mtm)
        max_dd = max(max_dd, (peak - mtm) / peak if peak else 0.0)

    if position is not None:
        candle = candles[-1]
        gross = position.capital * pnl_pct(
            position.side, position.entry_price, candle.close
        )
        fees = position.capital * commission * 2.0
        net = gross - fees
        equity += net
        trades += 1
        if net > 0:
            wins += 1
        elif net < 0:
            losses += 1
        gross_profit += max(gross, 0.0)
        gross_loss += min(gross, 0.0)

    win_rate = wins / trades * 100.0 if trades else 0.0
    pf = gross_profit / abs(gross_loss) if gross_loss < 0 else (float("inf") if gross_profit else 0.0)
    return {
        "final": equity,
        "return_pct": (equity / initial_capital - 1.0) * 100.0,
        "trades": trades,
        "win_rate": win_rate,
        "profit_factor": pf,
        "max_dd": max_dd * 100.0,
        "long_entries": long_entries,
        "short_entries": short_entries,
        "ignored_in_range": ignored_in_range,
    }


def parse_values(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(x.strip()) for x in raw.split(",") if x.strip()}))
    if len(values) < 2:
        raise SystemExit("Need at least two range values")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep asymmetric BODY range bias: lower threshold for SHORT, upper threshold for LONG."
    )
    ap.add_argument("--month", default="2026-05")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    ap.add_argument("--allocation", type=float, default=1.0)
    ap.add_argument("--values", default=",".join(str(v) for v in DEFAULT_VALUES))
    ap.add_argument("--input-5m", default="reports/prepared_price_action_5m.csv")
    ap.add_argument("--input-15m", default="reports/prepared_price_action_15m.csv")
    ap.add_argument("--input-1h", default="reports/prepared_price_action_1h.csv")
    args = ap.parse_args()

    values = parse_values(args.values)
    jobs = (
        ("5m", Path(args.input_5m)),
        ("15m", Path(args.input_15m)),
        ("1h", Path(args.input_1h)),
    )

    for timeframe, path in jobs:
        candles = load_month(path, args.month, args.symbol)
        if not candles:
            raise SystemExit(f"No usable {timeframe} candles for {args.symbol} {args.month}: {path}")

        results: list[tuple[float, float, dict[str, float | int]]] = []
        for lower in values:
            for upper in values:
                if lower >= upper:
                    continue
                r = run_backtest(candles, lower, upper, args.initial_capital, args.commission, args.allocation)
                results.append((lower, upper, r))

        results.sort(key=lambda x: (float(x[2]["return_pct"]), float(x[2]["profit_factor"])), reverse=True)
        print(f"\n{timeframe} | candles={len(candles)} | commission={args.commission * 100:.2f}%")
        print("LOWER(SHORT) | UPPER(LONG) | RETURN | FINAL | TRADES | WIN | PF | DD | LONG | SHORT | IGNORED")
        for lower, upper, r in results:
            pf = r["profit_factor"]
            pf_text = "INF" if pf == float("inf") else f"{float(pf):.2f}"
            print(
                f"{lower:>6.0f} | {upper:>6.0f} | {float(r['return_pct']):>7.2f}% | "
                f"{float(r['final']):>8.2f} | {int(r['trades']):>6} | {float(r['win_rate']):>5.2f}% | "
                f"{pf_text:>4} | {float(r['max_dd']):>5.2f}% | {int(r['long_entries']):>4} | "
                f"{int(r['short_entries']):>5} | {int(r['ignored_in_range']):>7}"
            )

        baseline = next((x for x in results if x[0] == -100.0 and x[1] == 100.0), None)
        if baseline:
            _, _, r = baseline
            print(
                f"BASELINE[-100,+100] {timeframe} | return={float(r['return_pct']):.2f}% "
                f"trades={int(r['trades'])} win={float(r['win_rate']):.2f}%"
            )

        best = results[0]
        print(
            f"BEST {timeframe} | lower={best[0]:.0f} upper={best[1]:.0f} "
            f"return={float(best[2]['return_pct']):.2f}% trades={int(best[2]['trades'])} "
            f"win={float(best[2]['win_rate']):.2f}%"
        )


if __name__ == "__main__":
    main()
