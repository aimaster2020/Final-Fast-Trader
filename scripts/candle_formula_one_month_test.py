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


def signal_name(signal: Signal) -> str:
    if signal == Signal.BUY:
        return "BUY"
    if signal == Signal.SELL:
        return "SELL"
    return "HOLD"


def write_csv(candles: list[Candle], month: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "month", "timestamp", "open", "high", "low", "close", "body_close_minus_open",
        "high_minus_close", "low_minus_close", "high_minus_open",
        "fall_rule_1", "fall_rule_2", "fall_rule_3", "fall_score",
        "rise_rule_1", "rise_rule_2", "rise_rule_3", "rise_score",
        "is_range", "signal", "next_open", "next_close", "actual_body",
        "actual_direction", "correct",
    ]

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, candle in enumerate(candles[:-1]):
            d = decide(candle)
            next_candle = candles[i + 1]
            actual_body = next_candle.close - next_candle.open

            if actual_body > 0:
                actual = "BUY"
            elif actual_body < 0:
                actual = "SELL"
            else:
                actual = "HOLD"

            predicted = signal_name(d.signal)
            correct = "" if predicted == "HOLD" else str(predicted == actual)

            writer.writerow({
                "month": month,
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "body_close_minus_open": candle.close - candle.open,
                "high_minus_close": candle.high - candle.close,
                "low_minus_close": candle.low - candle.close,
                "high_minus_open": candle.high - candle.open,
                "fall_rule_1": int(d.rules.fall_rule_1),
                "fall_rule_2": int(d.rules.fall_rule_2),
                "fall_rule_3": int(d.rules.fall_rule_3),
                "fall_score": d.fall_score,
                "rise_rule_1": int(d.rules.rise_rule_1),
                "rise_rule_2": int(d.rules.rise_rule_2),
                "rise_rule_3": int(d.rules.rise_rule_3),
                "rise_score": d.rise_score,
                "is_range": int(d.is_range),
                "signal": predicted,
                "next_open": next_candle.open,
                "next_close": next_candle.close,
                "actual_body": actual_body,
                "actual_direction": actual,
                "correct": correct,
            })


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Test candle formula direction against the NEXT candle body direction."
    )
    ap.add_argument("--input", default="reports/1h/BTCUSDT_1h.csv")
    ap.add_argument("--month", default="2026-07")
    ap.add_argument("--output", default="reports/candle_formula_1h_2026-07.csv")
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
        next_candle = candles[i + 1]
        actual_body = next_candle.close - next_candle.open

        if d.signal == Signal.BUY:
            buy += 1
            if actual_body > 0:
                buy_wins += 1
            else:
                losses += 1
        elif d.signal == Signal.SELL:
            sell += 1
            if actual_body < 0:
                sell_wins += 1
            else:
                losses += 1
        else:
            hold += 1

    write_csv(candles, args.month, Path(args.output))

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
    print(f"CSV={args.output}")


if __name__ == "__main__":
    main()
