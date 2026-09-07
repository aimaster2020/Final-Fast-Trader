from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import evaluate_rules, is_range
from fast_pattern_trader.models import Candle


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


def actual_direction(candle: Candle, previous_candle: Candle) -> tuple[str, float]:
    """Excel ground truth: IF(current Close > previous Close, UP, DOWN)."""
    change = candle.close - previous_candle.close
    if change > 0:
        return "UP", change
    if change < 0:
        return "DOWN", change
    return "FLAT", change


def weighted_prediction(candle: Candle) -> str:
    """Use the requested threshold of 3 points for the two rule groups."""
    rules = evaluate_rules(candle)
    down = rules.fall_score
    up = rules.rise_score

    if down >= 3.0 and up < 3.0:
        return "DOWN"
    if up >= 3.0 and down < 3.0:
        return "UP"
    return "UNDECIDED"


def write_csv(candles: list[Candle], month: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "month", "timestamp", "open", "high", "low", "close", "previous_close",
        "actual_change", "actual_direction", "body_close_minus_open",
        "high_minus_close", "low_minus_close", "high_minus_open",
        "down_rule_1", "down_rule_2", "down_rule_3", "down_score",
        "up_rule_1", "up_rule_2", "up_rule_3", "up_score",
        "is_range", "predicted_direction", "correct",
    ]

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(1, len(candles)):
            candle = candles[i]
            previous = candles[i - 1]
            rules = evaluate_rules(candle)
            actual, actual_change = actual_direction(candle, previous)
            predicted = weighted_prediction(candle)
            correct = "" if predicted == "UNDECIDED" or actual == "FLAT" else str(predicted == actual)
            writer.writerow({
                "month": month,
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "previous_close": previous.close,
                "actual_change": actual_change,
                "actual_direction": actual,
                "body_close_minus_open": candle.close - candle.open,
                "high_minus_close": candle.high - candle.close,
                "low_minus_close": candle.low - candle.close,
                "high_minus_open": candle.high - candle.open,
                "down_rule_1": int(rules.fall_rule_1),
                "down_rule_2": int(rules.fall_rule_2),
                "down_rule_3": int(rules.fall_rule_3),
                "down_score": rules.fall_score,
                "up_rule_1": int(rules.rise_rule_1),
                "up_rule_2": int(rules.rise_rule_2),
                "up_rule_3": int(rules.rise_rule_3),
                "up_score": rules.rise_score,
                "is_range": int(is_range(candle)),
                "predicted_direction": predicted,
                "correct": correct,
            })


def main() -> None:
    ap = argparse.ArgumentParser(description="Test OHLC formula direction against Close vs previous Close.")
    ap.add_argument("--input", default="reports/1h/BTCUSDT_1h.csv")
    ap.add_argument("--month", default="2026-07")
    ap.add_argument("--output", default="reports/candle_formula_1h_2026-07.csv")
    args = ap.parse_args()

    candles = load_month(Path(args.input), args.month)
    if len(candles) < 2:
        raise SystemExit(f"No usable 1h candles found for {args.month}")

    rule_defs = [
        ("R1_DOWN", "fall_rule_1", "DOWN"),
        ("R2_DOWN", "fall_rule_2", "DOWN"),
        ("R3_DOWN", "fall_rule_3", "DOWN"),
        ("R4_UP", "rise_rule_1", "UP"),
        ("R5_UP", "rise_rule_2", "UP"),
        ("R6_UP", "rise_rule_3", "UP"),
    ]
    stats = {name: [0, 0] for name, _, _ in rule_defs}
    up = down = undecided = ranges = 0
    up_wins = down_wins = comparable = 0
    actual_up = actual_down = actual_flat = 0

    for i in range(1, len(candles)):
        candle = candles[i]
        previous = candles[i - 1]
        rules = evaluate_rules(candle)
        actual, _ = actual_direction(candle, previous)
        predicted = weighted_prediction(candle)
        ranges += int(is_range(candle))

        if actual == "UP":
            actual_up += 1
        elif actual == "DOWN":
            actual_down += 1
        else:
            actual_flat += 1

        if predicted == "UP":
            up += 1
        elif predicted == "DOWN":
            down += 1
        else:
            undecided += 1

        if predicted in {"UP", "DOWN"} and actual in {"UP", "DOWN"}:
            comparable += 1
            if predicted == actual:
                if predicted == "UP":
                    up_wins += 1
                else:
                    down_wins += 1

        for name, attr, expected in rule_defs:
            if getattr(rules, attr):
                stats[name][0] += 1
                stats[name][1] += int(actual == expected)

    write_csv(candles, args.month, Path(args.output))

    wins = up_wins + down_wins
    accuracy = wins / comparable * 100.0 if comparable else 0.0
    up_acc = up_wins / up * 100.0 if up else 0.0
    down_acc = down_wins / down * 100.0 if down else 0.0
    range_pct = ranges / (len(candles) - 1) * 100.0

    print(f"1h {args.month} | candles={len(candles)-1}")
    print(f"ACTUAL UP={actual_up} DOWN={actual_down} FLAT={actual_flat}")
    for name, _, _ in rule_defs:
        correct, activated = stats[name][1], stats[name][0]
        acc = correct / activated * 100.0 if activated else 0.0
        print(f"{name}={correct}/{activated} ({acc:.2f}%)")
    print(f"WEIGHTED UP={up} win={up_wins} acc={up_acc:.2f}% | DOWN={down} win={down_wins} acc={down_acc:.2f}%")
    print(f"UNDECIDED={undecided} RANGE={ranges} ({range_pct:.2f}%)")
    print(f"CHECKED={comparable} WINS={wins} ACCURACY={accuracy:.2f}%")
    print(f"CSV={args.output}")


if __name__ == "__main__":
    main()
