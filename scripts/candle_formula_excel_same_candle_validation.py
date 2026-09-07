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
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(candles, key=lambda x: x.timestamp)


def excel_rule_values(candle: Candle) -> tuple[int, int, int, int, int, int]:
    """Reproduce Excel P:R and the complementary DOWN comparisons exactly."""
    o, h, l, c = candle.open, candle.high, candle.low, candle.close
    hc = h - c
    co = c - o
    ho = h - o
    lc = l - c
    return (
        int(hc > co),  # P / R1
        int(hc > ho),  # Q / R2
        int(lc > co),  # R / R3
        int(hc < co),  # R4
        int(hc < ho),  # R5
        int(lc < co),  # R6
    )


def candle_body(candle: Candle) -> float:
    """Excel column J = Close - Open (C-O)."""
    return candle.close - candle.open


def excel_i_direction(current: Candle, following: Candle) -> int:
    """Exact Excel I formula on row N: compare J(N+1) with J(N)."""
    current_j = candle_body(current)
    next_j = candle_body(following)
    if next_j > current_j:
        return 1
    if next_j < current_j:
        return -1
    return 0


def run_timeframe(path: Path, timeframe: str, month: str, symbol: str) -> dict[str, float | int]:
    candles = load_month(path, month, symbol)
    if len(candles) < 2:
        raise SystemExit(f"No usable {timeframe} candles for {symbol} {month}: {path}")

    up_total = up_correct = 0
    down_total = down_correct = 0
    hold_total = 0
    actual_flat = 0
    rule_mismatches = 0

    # Streak statistics over signal frames only (UP/DOWN predictions).
    current_correct_streak = 0
    max_correct_streak = 0
    correct_streak_sequences = 0
    current_wrong_streak = 0
    max_wrong_streak = 0
    wrong_streak_sequences = 0

    rows = len(candles) - 1

    # Excel's I2 = compare J3 to J2. Therefore S(N) is evaluated on
    # candle N and correctness is determined by J(N+1) vs J(N).
    for i in range(rows):
        candle = candles[i]
        following = candles[i + 1]
        r1, r2, r3, r4, r5, r6 = excel_rule_values(candle)
        s = r1 + r2 + r3
        expected_signal = 1 if s == 3 else -1 if s == 0 else 0

        actual_rules = evaluate_rules(candle)
        actual_tuple = tuple(
            int(getattr(actual_rules, attr))
            for attr in (
                "up_rule_1",
                "up_rule_2",
                "up_rule_3",
                "down_rule_1",
                "down_rule_2",
                "down_rule_3",
            )
        )
        if actual_tuple != (r1, r2, r3, r4, r5, r6):
            rule_mismatches += 1

        actual_dir = excel_i_direction(candle, following)
        if actual_dir == 0:
            actual_flat += 1

        if expected_signal == 1:
            up_total += 1
        elif expected_signal == -1:
            down_total += 1
        else:
            hold_total += 1
            continue

        correct = actual_dir == expected_signal
        if expected_signal == 1:
            up_correct += int(correct)
        else:
            down_correct += int(correct)

        if correct:
            if current_correct_streak == 0:
                correct_streak_sequences += 1
            current_correct_streak += 1
            max_correct_streak = max(max_correct_streak, current_correct_streak)
            current_wrong_streak = 0
        else:
            if current_wrong_streak == 0:
                wrong_streak_sequences += 1
            current_wrong_streak += 1
            max_wrong_streak = max(max_wrong_streak, current_wrong_streak)
            current_correct_streak = 0

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
        "actual_flat": actual_flat,
        "max_correct_streak": max_correct_streak,
        "correct_streak_sequences": correct_streak_sequences,
        "max_wrong_streak": max_wrong_streak,
        "wrong_streak_sequences": wrong_streak_sequences,
        "rule_mismatches": rule_mismatches,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Exact Excel validation: S(N)=3/S(N)=0 versus J(N+1)>J(N), where J=C-O."
    )
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
            f"DIR={r['total_signals']}/{r['total_correct']} ({r['overall_accuracy']:.2f}%) | "
            f"100PCT_MAX={r['total_signals']} | "
            f"MAX_STREAK={r['max_correct_streak']} | "
            f"STREAKS={r['correct_streak_sequences']} | "
            f"HOLD={r['hold_total']} FLAT={r['actual_flat']} | RULE_MISMATCH={r['rule_mismatches']}"
        )


if __name__ == "__main__":
    main()
