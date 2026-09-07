from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal


def load_month(path: Path, month: str) -> list[Candle]:
    candles: list[Candle] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("month") != month:
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
    return sorted(candles, key=lambda x: x.timestamp)


def main() -> None:
    ap = argparse.ArgumentParser(description="Test the new 3x3 candle formula on one month of 1h data.")
    ap.add_argument("--input", default="reports/1h/BTCUSDT_1h.csv")
    ap.add_argument("--month", default="2026-07")
    args = ap.parse_args()

    candles = load_month(Path(args.input), args.month)
    if len(candles) < 2:
        raise SystemExit(f"No usable 1h candles found for {args.month}")

    buy = sell = hold = 0
    buy_wins = sell_wins = losses = 0
    ranges = 0

    for i, candle in enumerate(candles[:-1]):
        d = decide(candle)
        ranges += int(d.is_range)
        if d.signal == Signal.BUY:
            buy += 1
            if candles[i + 1].close > candle.close:
                buy_wins += 1
            else:
                losses += 1
        elif d.signal == Signal.SELL:
            sell += 1
            if candles[i + 1].close < candle.close:
                sell_wins += 1
            else:
                losses += 1
        else:
            hold += 1

    signals = buy + sell
    wins = buy_wins + sell_wins
    accuracy = wins / signals * 100.0 if signals else 0.0
    buy_acc = buy_wins / buy * 100.0 if buy else 0.0
    sell_acc = sell_wins / sell * 100.0 if sell else 0.0
    range_pct = ranges / (len(candles) - 1) * 100.0

    print(f"1h {args.month} | candles={len(candles)}")
    print(f"BUY={buy} win={buy_wins} acc={buy_acc:.2f}%")
    print(f"SELL={sell} win={sell_wins} acc={sell_acc:.2f}%")
    print(f"HOLD={hold} RANGE={ranges} ({range_pct:.2f}%)")
    print(f"SIGNALS={signals} WINS={wins} LOSSES={losses} ACCURACY={accuracy:.2f}%")


if __name__ == "__main__":
    main()
