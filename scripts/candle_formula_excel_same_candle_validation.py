from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import evaluate_rules
from fast_pattern_trader.models import Candle


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
                        float(row["high"])),
                        float(row["low"])),
                        float(row["close"])),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(candles, key=lambda x: x.timestamp)


def excel_rule_values(candle: Candle) -> tuple[int, int, int, int, int, int]:
    o, h, l, c = candle.open, candle.high, candle.low, candle.close
    hc = h - c
    co = c - o
    ho = h - o
    lc = l - c
    return (
        int(hc > co),
        int(hc > ho),
        int(lc > co),
        int(hc < co),
        int(hc < ho),
        int(lc < co),
    )


def actual_direction(candle: Candle, previous: Candle) -> int:
    if candle.close > previous.close:
        return 1
    if candle.close < previous.close:
        return -1
    return 0


def run_timeframe(path: Path, timeframe: str, month: str, symbol: str) -> dict[str, float | int]:
    candles = load_month(path, month, symbol)
    if len(candles) < 2:
        raise SystemExit(f"No usable {timeframe} candles for {symbol} {month}: {path}")

    up_total = up_correct = 0
    down_total = down_correct = 0
    hold_total = 0
    rule_mismatches = 0
    rows = len(candles) - 1

    for i in range(1, len(candles)):
        candle = candles[i]
        previous = candles[i - 1]
        r1, r2, r3, r4, r5, r6 = excel_rule_values(candle)
        expected_signal = 1 if (r1 + r2 + r3) == 3 else -1 if (r1 + r2 + r3) == 0 else 0
        actual_rules = evaluate_rules(candle)
        actual_tuple = tuple(
            int(getattr(actual_rules, attr))
            for attr in (
                "up_rule_1", "up_rule_2", "up_rule_3",
                "down_rule_1", "down_rule_2", "down_rule_3",
            )
        )
        if actual_tuple != (r1, r2, r3, r4, r5, r6):
            rule_mismatches += 1

        actual_dir = actual_direction(candle, previous)
        if expected_signal == 1:
            up_total += 1
            up_correct += int(actual_dir == 1)
        elif expected_signal == -1:
            down_total += 1
            down_correct += int(actual_dir == -1)
        else:
            hold_total += 1

    up_accuracy = up_correct / up_total * 100.0 if up_total else 0.0
    down_accuracy = down_correct / down_total * 100.0 if down_total else 0.0
    total_signals = up_total + down_total
    total_correct = up_correct + down_correct
    overall_accuracy = total_correct / total_signals * 100.0 if total_signals else 0.0

    return {
        "rows": rows,
        "up_total": up_total,
        "up_correct": up_correct,
        "up_accuracy": up_accuracy,
        "down_total": down_total,
        "down_correct": down_correct,
        "down_accuracy": down_accuracy,
        "total_signals": total_signals,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "hold_total": hold_total,
        "rule_mismatches": rule_mismatches,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact Excel same-candle validation: S=3 vs I=UP and S=0 vs I=DOWN.")
    ap.add_argument("--month", default="2026-07")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--input-5m", default="reports/prepared_price_action_5m.csv")
    ap.add_argument("--input-15m", default="reports/prepared_price_action_15m.csv")
    ap.add_argument("--input-1h", default="reports/prepared_price_action_1h.csv")
    args = ap.parse_args()

    jobs = (
        ("5m", Path(args.input_5m)),
        ("15m", Path(args.input_15m)),
        ("1h", Path(args.input_1h)),
    )

    for timeframe, path in jobs:
        r = run_timeframe(path, timeframe, args.month, args.symbol)
        print(
            f"{timeframe} | rows={r['rows']} | "
            f"UP={r['up_total']}/{r['up_correct']} ({r['up_accuracy']:.2f}%) | "
            f"DOWN={r['down_total']}/{r['down_correct']} ({r['down_accuracy']:.2f}%) | "
            f"TOTAL={r['total_signals']}/{r['total_correct']} ({r['overall_accuracy']:.2f}%) | "
            f"HOLD={r['hold_total']} | RULE_MISMATCH={r['rule_mismatches']}"
        )


if __name__ == "__main__":
    main()
