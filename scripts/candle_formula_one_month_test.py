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


def predicted_direction(candle: Candle) -> str:
    """Return direction from the ACTIVE rules only; this is not a trade signal.

    L-C rules are disabled in the strategy. With the two active rules:
      fall_score > rise_score -> FALL
      rise_score > fall_score -> RISE
      otherwise               -> UNDECIDED
    """
    rules = evaluate_rules(candle)
    if rules.fall_score > rules.rise_score:
        return "FALL"
    if rules.rise_score > rules.fall_score:
        return "RISE"
    return "UNDECIDED"


def actual_direction(next_candle: Candle) -> tuple[str, float]:
    """Actual direction is the next candle body: Close(next) - Open(next)."""
    body = next_candle.close - next_candle.open
    if body > 0:
        return "RISE", body
    if body < 0:
        return "FALL", body
    return "FLAT", body


def write_csv(candles: list[Candle], month: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "month", "timestamp", "open", "high", "low", "close", "body_close_minus_open",
        "high_minus_close", "low_minus_close", "high_minus_open",
        "fall_rule_1", "fall_rule_2", "fall_rule_3", "fall_score",
        "rise_rule_1", "rise_rule_2", "rise_rule_3", "rise_score",
        "is_range", "predicted_direction", "next_open", "next_close",
        "actual_body", "actual_direction", "correct",
    ]

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, candle in enumerate(candles[:-1]):
            rules = evaluate_rules(candle)
            predicted = predicted_direction(candle)
            next_candle = candles[i + 1]
            actual, actual_body = actual_direction(next_candle)

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
                "next_open": next_candle.open,
                "next_close": next_candle.close,
                "actual_body": actual_body,
                "actual_direction": actual,
                "correct": correct,
            })


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Test formula-predicted direction against the next candle body direction."
    )
    ap.add_argument("--input", default="reports/1h/BTCUSDT_1h.csv")
    ap.add_argument("--month", default="2026-07")
    ap.add_argument("--output", default="reports/candle_formula_1h_2026-07.csv")
    args = ap.parse_args()

    candles = load_month(Path(args.input), args.month)
    if len(candles) < 2:
        raise SystemExit(f"No usable 1h candles found for {args.month}")

    rise = fall = undecided = 0
    rise_wins = fall_wins = 0
    comparable = 0
    ranges = 0

    for i, candle in enumerate(candles[:-1]):
        predicted = predicted_direction(candle)
        actual, _ = actual_direction(candles[i + 1])
        ranges += int(is_range(candle))

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

    write_csv(candles, args.month, Path(args.output))

    wins = rise_wins + fall_wins
    accuracy = wins / comparable * 100.0 if comparable else 0.0
    rise_acc = rise_wins / rise * 100.0 if rise else 0.0
    fall_acc = fall_wins / fall * 100.0 if fall else 0.0
    range_pct = ranges / (len(candles) - 1) * 100.0

    print(f"1h {args.month} | candles={len(candles)}")
    print(f"RISE={rise} win={rise_wins} acc={rise_acc:.2f}%")
    print(f"FALL={fall} win={fall_wins} acc={fall_acc:.2f}%")
    print(f"UNDECIDED={undecided} RANGE={ranges} ({range_pct:.2f}%)")
    print(f"CHECKED={comparable} WINS={wins} ACCURACY={accuracy:.2f}%")
    print(f"CSV={args.output}")


if __name__ == "__main__":
    main()
