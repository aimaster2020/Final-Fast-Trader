from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from fast_pattern_trader.models import Candle

MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
CAPITAL = 1000.0


@dataclass
class Position:
    side: int
    entry: float
    capital: float


def load(path: Path, month: str, symbol: str) -> list[Candle]:
    rows: list[Candle] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("month") != month or row.get("symbol") != symbol:
                continue
            try:
                rows.append(
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
    return sorted(rows, key=lambda c: c.timestamp)


def scores(c: Candle) -> tuple[int, int]:
    hc = c.high - c.close
    co = c.close - c.open
    lc = c.low - c.close
    ho = c.high - c.open
    r1 = int(hc > co)
    r2 = int(ho < hc)
    r3 = int(lc > co)
    r4 = int(hc < co)
    r5 = int(ho > hc)
    r6 = int(lc < co)
    return r1 + r2 + r3, r4 + r5 + r6


def excel_signal(c: Candle) -> int:
    """Original Excel rule: S=3 BUY, S=0 SELL, S=1/2 HOLD."""
    up, down = scores(c)
    if up == 3:
        return 1
    if up == 0:
        return -1
    return 0


def body_value(c: Candle) -> float:
    return c.close - c.open


def excel_body_direction(current: Candle, previous: Candle | None) -> int:
    """Exact Excel direction: =IF(L3>L2,1,IF(L3<L2,-1,0))."""
    if previous is None:
        return 0
    current_l = body_value(current)
    previous_l = body_value(previous)
    if current_l > previous_l:
        return 1
    if current_l < previous_l:
        return -1
    return 0


def in_bias(body: float, width: float) -> bool:
    return -width <= body <= width


def pnl(side: int, entry: float, price: float) -> float:
    return side * (price - entry) / entry if entry > 0 else 0.0


def run(candles: list[Candle], width: float) -> dict[str, float | int]:
    equity = CAPITAL
    position: Position | None = None
    previous_prediction = 0
    trades = wins = 0
    peak = CAPITAL
    max_dd = 0.0

    for i, candle in enumerate(candles):
        current_signal = excel_signal(candle)
        body = body_value(candle)
        previous = candles[i - 1] if i > 0 else None
        actual_direction = excel_body_direction(candle, previous)
        previous_correct = previous_prediction != 0 and previous_prediction == actual_direction

        if position is None:
            if current_signal != 0 and not in_bias(body, width) and previous_correct:
                position = Position(current_signal, candle.close, equity)
        elif in_bias(body, width):
            gross = position.capital * pnl(position.side, position.entry, candle.close)
            equity += gross
            trades += 1
            wins += int(gross > 0)
            position = None

        mtm = equity
        if position is not None:
            mtm += position.capital * pnl(position.side, position.entry, candle.close)
        peak = max(peak, mtm)
        max_dd = max(max_dd, (peak - mtm) / peak if peak else 0.0)
        previous_prediction = current_signal

    if position is not None and candles:
        gross = position.capital * pnl(position.side, position.entry, candles[-1].close)
        equity += gross
        trades += 1
        wins += int(gross > 0)

    return {
        "return": (equity / CAPITAL - 1.0) * 100.0,
        "trades": trades,
        "wins": wins,
        "win_rate": wins / trades * 100.0 if trades else 0.0,
        "dd": max_dd * 100.0,
    }


def compound(values: list[float]) -> float:
    x = 1.0
    for value in values:
        x *= 1.0 + value / 100.0
    return (x - 1.0) * 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact Excel candle signal backtest; no commission.")
    ap.add_argument("--input", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--width", type=float, default=1.0)
    args = ap.parse_args()

    path = Path(args.input)
    print(
        f"EXCEL_SIGNAL_BACKTEST | tf=1h | fee=0 | width={args.width:g} | "
        "signal=S3_BUY/S0_SELL/S1,S2_HOLD | direction=L_current_vs_L_previous"
    )
    for symbol in SYMBOLS:
        monthly: list[float] = []
        trades = wins = 0
        dd = 0.0
        print(f"\n{symbol}")
        for month in MONTHS:
            r = run(load(path, month, symbol), args.width)
            monthly.append(float(r["return"]))
            trades += int(r["trades"])
            wins += int(r["wins"])
            dd = max(dd, float(r["dd"]))
            print(
                f"{month} return={float(r['return']):+.2f}% trades={int(r['trades'])} "
                f"win={float(r['win_rate']):.1f}% DD={float(r['dd']):.2f}%"
            )
        print(
            f"4M compound={compound(monthly):+.2f}% avg={sum(monthly)/len(monthly):+.2f}% "
            f"worst={min(monthly):+.2f}% trades={trades} win={wins/trades*100:.1f}% DD={dd:.2f}%"
            if trades else
            f"4M compound=+0.00% avg=+0.00% worst=+0.00% trades=0 win=0.0% DD={dd:.2f}%"
        )


if __name__ == "__main__":
    main()
