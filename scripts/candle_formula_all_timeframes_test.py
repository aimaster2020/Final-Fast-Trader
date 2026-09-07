from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import evaluate_rules, is_range
from fast_pattern_trader.models import Candle


RULE_DEFS = [
    ("R1_UP", "up_rule_1", "UP"),
    ("R2_UP", "up_rule_2", "UP"),
    ("R3_UP", "up_rule_3", "UP"),
    ("R4_DOWN", "down_rule_1", "DOWN"),
    ("R5_DOWN", "down_rule_2", "DOWN"),
    ("R6_DOWN", "down_rule_3", "DOWN"),
]


def load_month(path: Path, month: str, symbol: str) -> list[Candle]:
    candles: list[Candle] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("month") != month:
                continue
            if symbol and row.get("symbol") != symbol:
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


def actual_direction(candle: Candle, previous: Candle) -> str:
    change = candle.close - previous.close
    if change > 0:
        return "UP"
    if change < 0:
        return "DOWN"
    return "FLAT"


def weighted_prediction(candle: Candle) -> str:
    rules = evaluate_rules(candle)
    if rules.up_score >= 3.0 and rules.down_score < 3.0:
        return "UP"
    if rules.down_score >= 3.0 and rules.up_score < 3.0:
        return "DOWN"
    return "UNDECIDED"


def run_timeframe(path: Path, timeframe: str, month: str, symbol: str, output: Path) -> dict[str, object]:
    candles = load_month(path, month, symbol)
    if len(candles) < 2:
        raise SystemExit(f"No usable {timeframe} candles for {symbol} {month}: {path}")

    stats = {name: [0, 0] for name, _, _ in RULE_DEFS}
    actual_up = actual_down = actual_flat = 0
    predicted_up = predicted_down = undecided = ranges = 0
    up_wins = down_wins = comparable = 0

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "symbol", "timeframe", "month", "timestamp", "open", "high", "low", "close",
        "previous_close", "actual_direction", "up_rule_1", "up_rule_2", "up_rule_3",
        "up_score", "down_rule_1", "down_rule_2", "down_rule_3", "down_score",
        "is_range", "predicted_direction", "correct",
    ]

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i in range(1, len(candles)):
            candle = candles[i]
            previous = candles[i - 1]
            rules = evaluate_rules(candle)
            actual = actual_direction(candle, previous)
            predicted = weighted_prediction(candle)
            range_flag = int(is_range(candle))
            ranges += range_flag

            if actual == "UP":
                actual_up += 1
            elif actual == "DOWN":
                actual_down += 1
            else:
                actual_flat += 1

            if predicted == "UP":
                predicted_up += 1
            elif predicted == "DOWN":
                predicted_down += 1
            else:
                undecided += 1

            if predicted in {"UP", "DOWN"} and actual in {"UP", "DOWN"}:
                comparable += 1
                if predicted == actual:
                    if predicted == "UP":
                        up_wins += 1
                    else:
                        down_wins += 1

            for name, attr, expected in RULE_DEFS:
                if getattr(rules, attr):
                    stats[name][0] += 1
                    stats[name][1] += int(actual == expected)

            correct = "" if predicted == "UNDECIDED" or actual == "FLAT" else str(predicted == actual)
            writer.writerow({
                "symbol": symbol,
                "timeframe": timeframe,
                "month": month,
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "previous_close": previous.close,
                "actual_direction": actual,
                "up_rule_1": int(rules.up_rule_1),
                "up_rule_2": int(rules.up_rule_2),
                "up_rule_3": int(rules.up_rule_3),
                "up_score": rules.up_score,
                "down_rule_1": int(rules.down_rule_1),
                "down_rule_2": int(rules.down_rule_2),
                "down_rule_3": int(rules.down_rule_3),
                "down_score": rules.down_score,
                "is_range": range_flag,
                "predicted_direction": predicted,
                "correct": correct,
            })

    total = len(candles) - 1
    wins = up_wins + down_wins
    accuracy = wins / comparable * 100.0 if comparable else 0.0
    coverage = comparable / total * 100.0 if total else 0.0
    range_pct = ranges / total * 100.0 if total else 0.0
    up_acc = up_wins / predicted_up * 100.0 if predicted_up else 0.0
    down_acc = down_wins / predicted_down * 100.0 if predicted_down else 0.0

    return {
        "timeframe": timeframe,
        "candles": total,
        "actual_up": actual_up,
        "actual_down": actual_down,
        "actual_flat": actual_flat,
        "up": predicted_up,
        "down": predicted_down,
        "undecided": undecided,
        "checked": comparable,
        "wins": wins,
        "accuracy": accuracy,
        "coverage": coverage,
        "range": ranges,
        "range_pct": range_pct,
        "up_acc": up_acc,
        "down_acc": down_acc,
        "stats": stats,
        "output": str(output),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="One-shot validation of candle_formula_weighted_v2 on 5m/15m/1h.")
    ap.add_argument("--month", default="2026-07")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--input-5m", default="reports/prepared_price_action_5m.csv")
    ap.add_argument("--input-15m", default="reports/prepared_price_action_15m.csv")
    ap.add_argument("--input-1h", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--output-dir", default="reports/candle_formula_validation")
    args = ap.parse_args()

    jobs = [
        ("5m", Path(args.input_5m)),
        ("15m", Path(args.input_15m)),
        ("1h", Path(args.input_1h)),
    ]

    results = []
    for timeframe, path in jobs:
        output = Path(args.output_dir) / f"{args.symbol}_{timeframe}_{args.month}.csv"
        result = run_timeframe(path, timeframe, args.month, args.symbol, output)
        results.append(result)

        print(f"{timeframe} {args.month} | candles={result['candles']} | ACTUAL U={result['actual_up']} D={result['actual_down']} F={result['actual_flat']}")
        for name, _, _ in RULE_DEFS:
            correct, activated = result["stats"][name][1], result["stats"][name][0]
            acc = correct / activated * 100.0 if activated else 0.0
            print(f"  {name}={correct}/{activated} ({acc:.2f}%)")
        print(
            f"  WEIGHTED U={result['up']} ({result['up_acc']:.2f}%) | "
            f"D={result['down']} ({result['down_acc']:.2f}%) | "
            f"UNDECIDED={result['undecided']} | RANGE={result['range']} ({result['range_pct']:.2f}%)"
        )
        print(f"  CHECKED={result['checked']} WINS={result['wins']} ACC={result['accuracy']:.2f}% COVERAGE={result['coverage']:.2f}%")
        print(f"  CSV={result['output']}")

    summary = Path(args.output_dir) / f"summary_{args.symbol}_{args.month}.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "symbol", "month", "timeframe", "candles", "actual_up", "actual_down", "actual_flat",
                "predicted_up", "predicted_down", "undecided", "checked", "wins", "accuracy_pct",
                "coverage_pct", "range", "range_pct", "up_accuracy_pct", "down_accuracy_pct",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "symbol": args.symbol,
                "month": args.month,
                "timeframe": r["timeframe"],
                "candles": r["candles"],
                "actual_up": r["actual_up"],
                "actual_down": r["actual_down"],
                "actual_flat": r["actual_flat"],
                "predicted_up": r["up"],
                "predicted_down": r["down"],
                "undecided": r["undecided"],
                "checked": r["checked"],
                "wins": r["wins"],
                "accuracy_pct": f"{r['accuracy']:.6f}",
                "coverage_pct": f"{r['coverage']:.6f}",
                "range": r["range"],
                "range_pct": f"{r['range_pct']:.6f}",
                "up_accuracy_pct": f"{r['up_acc']:.6f}",
                "down_accuracy_pct": f"{r['down_acc']:.6f}",
            })

    print(f"SUMMARY={summary}")


if __name__ == "__main__":
    main()
