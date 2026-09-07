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


def rule_predictions(candle: Candle) -> tuple[str, ...]:
    rules = evaluate_rules(candle)
    predictions: list[str] = []
    if rules.fall_rule_1:
        predictions.append("FALL")
    if rules.fall_rule_2:
        predictions.append("FALL")
    if rules.fall_rule_3:
        predictions.append("FALL")
    if rules.rise_rule_1:
        predictions.append("RISE")
    if rules.rise_rule_2:
        predictions.append("RISE")
    if rules.rise_rule_3:
        predictions.append("RISE")
    return tuple(predictions)


def weighted_prediction(candle: Candle) -> str:
    rules = evaluate_rules(candle)
    if rules.fall_score > rules.rise_score:
        return "FALL"
    if rules.rise_score > rules.fall_score:
        return "RISE"
    return "UNDECIDED"


def actual_direction(candle: Candle, previous_candle: Candle) -> tuple[str, float]:
    """Excel ground truth: current Close - previous Close."""
    change = candle.close - previous_candle.close
    if change > 0:
        return "RISE", change
    if change < 0:
        return "FALL", change
    return "FLAT", change


def write_csv(candles: list[Candle], month: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "month", "timestamp", "open", "high", "low", "close", "body_close_minus_open",
        "high_minus_close", "low_minus_close", "high_minus_open",
        "fall_rule_1", "fall_rule_2", "fall_rule_3", "fall_score",
        "rise_rule_1", "rise_rule_2", "rise_rule_3", "rise_score",
        "is_range", "predicted_direction", "previous_close", "actual_change",
        "actual_direction", "correct",
    ]

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i in range(1, len(candles)):
            candle = candles[i]
            previous_candle = candles[i - 1]
            rules = evaluate_rules(candle)
            predicted = weighted_prediction(candle)
            actual, actual_change = actual_direction(candle, previous_candle)
            correct = "" if predicted == "UNDECIDED" or actual == "FLAT" else str(predicted == actual)

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
                "fall_rule_1": int(rules.fall_rule_1),
                "fall_rule_2": int(rules.fall_rule_2),
                "fall_rule_3": int(rules.fall_rule_3),
                "fall_score": rules.fall_score,
                "rise_rule_1": int(rules.rise_rule_1),
                "rise_rule_2": int(rules.rise_rule_2),
                "rise_rule_3": int(rules.rise_rule_3),
                "rise_score": rules.rise_score,
                "is_range": int(is_range(candle)),
                "predicted_direction": predicted,
                "previous_close": previous_candle.close,
                "actual_change": actual_change,
                "actual_direction": actual,
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
        ("R1_FALL", "fall_rule_1", "FALL"),
        ("R2_FALL", "fall_rule_2", "FALL"),
        ("R3_FALL", "fall_rule_3", "FALL"),
        ("R4_RISE", "rise_rule_1", "RISE"),
        ("R5_RISE", "rise_rule_2", "RISE"),
        ("R6_RISE", "rise_rule_3", "RISE"),
    ]
    stats = {name: [0, 0] for name, _, _ in rule_defs}
    rise = fall = undecided = ranges = 0
    rise_wins = fall_wins = comparable = 0
    actual_rise = actual_fall = actual_flat = 0

    for i in range(1, len(candles)):
        candle = candles[i]
        previous = candles[i - 1]
        rules = evaluate_rules(candle)
        actual, _ = actual_direction(candle, previous)
        predicted = weighted_prediction(candle)
        ranges += int(is_range(candle))

        if actual == "RISE":
            actual_rise += 1
        elif actual == "FALL":
            actual_fall += 1
        else:
            actual_flat += 1

        if predicted == "RISE":
            rise += 1
        elif predicted == "FALL":
            fall += 1
        else:
            undecided += 1

        if predicted in {"RISE", "FALL"} and actual in {"RISE", "FALL"}:
            comparable += 1
            if predicted == actual:
                if predicted == "RISE":
                    rise_wins += 1
                else:
                    fall_wins += 1

        for name, attr, expected in rule_defs:
            if getattr(rules, attr):
                stats[name][0] += 1
                stats[name][1] += int(actual == expected)

    write_csv(candles, args.month, Path(args.output))

    wins = rise_wins + fall_wins
    accuracy = wins / comparable * 100.0 if comparable else 0.0
    rise_acc = rise_wins / rise * 100.0 if rise else 0.0
    fall_acc = fall_wins / fall * 100.0 if fall else 0.0
    range_pct = ranges / (len(candles) - 1) * 100.0

    print(f"1h {args.month} | candles={len(candles)-1}")
    print(f"ACTUAL RISE={actual_rise} FALL={actual_fall} FLAT={actual_flat}")
    print(f"R1={stats['R1_FALL'][1]}/{stats['R1_FALL'][0]} ({stats['R1_FALL'][1]/stats['R1_FALL'][0]*100:.2f}%) | R2={stats['R2_FALL'][1]}/{stats['R2_FALL'][0]} ({stats['R2_FALL'][1]/stats['R2_FALL'][0]*100:.2f}%) | R3={stats['R3_FALL'][1]}/{stats['R3_FALL'][0]} ({stats['R3_FALL'][1]/stats['R3_FALL'][0]*100:.2f}%)")
    print(f"R4={stats['R4_RISE'][1]}/{stats['R4_RISE'][0]} ({stats['R4_RISE'][1]/stats['R4_RISE'][0]*100:.2f}%) | R5={stats['R5_RISE'][1]}/{stats['R5_RISE'][0]} ({stats['R5_RISE'][1]/stats['R5_RISE'][0]*100:.2f}%) | R6={stats['R6_RISE'][1]}/{stats['R6_RISE'][0]} ({stats['R6_RISE'][1]/stats['R6_RISE'][0]*100:.2f}%)")
    print(f"WEIGHTED RISE={rise} win={rise_wins} acc={rise_acc:.2f}% | FALL={fall} win={fall_wins} acc={fall_acc:.2f}%")
    print(f"UNDECIDED={undecided} RANGE={ranges} ({range_pct:.2f}%)")
    print(f"CHECKED={comparable} WINS={wins} ACCURACY={accuracy:.2f}%")
    print(f"CSV={args.output}")


if __name__ == "__main__":
    main()
