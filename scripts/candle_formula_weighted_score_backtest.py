from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from fast_pattern_trader.models import Candle

MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
CAPITAL = 1000.0

# Empirical ranking from the standalone no-fee test:
# R4 > R3 > R2 > R5 > R1 > R6
WEIGHT_SCHEMES = {
    "equal": {"R1": 1, "R2": 1, "R3": 1, "R4": 1, "R5": 1, "R6": 1},
    "rank_6_to_1": {"R1": 2, "R2": 4, "R3": 5, "R4": 6, "R5": 3, "R6": 1},
}

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
                rows.append(Candle(int(float(row["timestamp"])), float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(rows, key=lambda c: c.timestamp)

def active_rules(c: Candle) -> dict[str, bool]:
    hc = c.high - c.close
    co = c.close - c.open
    lc = c.low - c.close
    ho = c.high - c.open
    return {
        "R1": hc > co,
        "R2": ho < hc,
        "R3": lc > co,
        "R4": hc < co,
        "R5": ho > hc,
        "R6": lc < co,
    }

def weighted_signal(c: Candle, weights: dict[str, int]) -> int:
    r = active_rules(c)
    up = sum(weights[k] for k in ("R1", "R2", "R3") if r[k])
    down = sum(weights[k] for k in ("R4", "R5", "R6") if r[k])
    if up > down:
        return 1
    if down > up:
        return -1
    return 0

def body_value(c: Candle) -> float:
    return c.close - c.open

def body_direction(current: Candle, previous: Candle | None) -> int:
    """Exact Excel direction formula: =IF(L3>L2,1,IF(L3<L2,-1,0))."""
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

def run(candles: list[Candle], weights: dict[str, int], width: float) -> dict[str, float | int]:
    equity = CAPITAL
    position: Position | None = None
    previous_prediction = 0
    trades = wins = 0
    peak = CAPITAL
    max_dd = 0.0

    for i, candle in enumerate(candles):
        current_signal = weighted_signal(candle, weights)
        body = body_value(candle)
        previous = candles[i - 1] if i > 0 else None
        current_direction = body_direction(candle, previous)
        previous_correct = previous_prediction != 0 and previous_prediction == current_direction

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
    for v in values:
        x *= 1.0 + v / 100.0
    return (x - 1.0) * 100.0

def main() -> None:
    ap = argparse.ArgumentParser(description="No-fee weighted R1..R6 score backtest")
    ap.add_argument("--input", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--width", type=float, default=1.0)
    args = ap.parse_args()

    path = Path(args.input)
    print(f"WEIGHTED_SCORE_BACKTEST | tf=1h | fee=0 | width={args.width:g} | excel-direction=L3_vs_L2 | previous-prediction-confirmed-entry | same-bias-exit")
    for symbol in SYMBOLS:
        print(f"\n{symbol}")
        candidates = []
        for name, weights in WEIGHT_SCHEMES.items():
            monthly: list[float] = []
            trades = wins = 0
            dd = 0.0
            for month in MONTHS:
                r = run(load(path, month, symbol), weights, args.width)
                monthly.append(float(r["return"]))
                trades += int(r["trades"])
                wins += int(r["wins"])
                dd = max(dd, float(r["dd"]))
            comp = compound(monthly)
            avg = sum(monthly) / len(monthly)
            worst = min(monthly)
            wr = wins / trades * 100.0 if trades else 0.0
            candidates.append((comp, name))
            print(f"{name:12s} weights={weights} 4M={comp:+.2f}% avg={avg:+.2f}% worst={worst:+.2f}% trades={trades} win={wr:.1f}% DD={dd:.2f}%")
        best = max(candidates)
        print(f"BEST {best[1]} | 4M={best[0]:+.2f}%")

if __name__ == "__main__":
    main()
