from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide, evaluate_rules, is_range
from fast_pattern_trader.models import Candle


RULES = (
    ("R1_UP", "up_rule_1"),
    ("R2_UP", "up_rule_2"),
    ("R3_UP", "up_rule_3"),
    ("R4_DOWN", "down_rule_1"),
    ("R5_DOWN", "down_rule_2"),
    ("R6_DOWN", "down_rule_3"),
)


def excel_rules(candle: Candle) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Reproduce the spreadsheet's six comparisons literally."""
    o, h, l, c = candle.open, candle.high, candle.low, candle.close
    hc = h - c
    co = c - o
    ho = h - o
    lc = l - c
    return (
        hc > co,  # R1
        hc > ho,  # R2
        lc > co,  # R3
        hc < co,  # R4
        hc < ho,  # R5
        lc < co,  # R6
    )


def excel_signal(candle: Candle) -> str:
    r1, r2, r3, _r4, _r5, _r6 = excel_rules(candle)
    s = int(r1) + int(r2) + int(r3)
    if s == 3:
        return "UP"
    if s == 0:
        return "DOWN"
    return "HOLD"


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


def audit(path: Path, timeframe: str, month: str, symbol: str) -> dict[str, int | bool]:
    candles = load_month(path, month, symbol)
    mismatches = {name: 0 for name, _ in RULES}
    score_mismatches = 0
    signal_mismatches = 0
    range_mismatches = 0
    weighted_mismatches = 0

    for candle in candles:
        expected = excel_rules(candle)
        actual_rules = evaluate_rules(candle)
        actual = tuple(getattr(actual_rules, attr) for _, attr in RULES)

        for (name, _), exp, got in zip(RULES, expected, actual):
            mismatches[name] += int(exp != got)

        expected_s = sum(map(int, expected[:3]))
        actual_up_score = actual_rules.up_score
        actual_down_score = actual_rules.down_score
        expected_up_score = int(expected[0]) * 2 + int(expected[1]) + int(expected[2])
        expected_down_score = int(expected[3]) * 2 + int(expected[4]) + int(expected[5])
        weighted_mismatches += int(
            actual_up_score != expected_up_score or actual_down_score != expected_down_score
        )

        # Excel direction is driven by S = R1+R2+R3.
        actual_s = sum(map(int, actual[:3]))
        score_mismatches += int(actual_s != expected_s)

        expected_signal = excel_signal(candle)
        python_signal = {
            "BUY": "UP",
            "SELL": "DOWN",
            "HOLD": "HOLD",
        }[decide(candle).signal.value]
        signal_mismatches += int(expected_signal != python_signal)
        range_mismatches += int(is_range(candle) != (-100.0 <= candle.close - candle.open <= 100.0))

    total = len(candles)
    return {
        "rows": total,
        **mismatches,
        "score_mismatches": score_mismatches,
        "weighted_mismatches": weighted_mismatches,
        "signal_mismatches": signal_mismatches,
        "range_mismatches": range_mismatches,
        "pass": total > 0 and all(v == 0 for k, v in locals()["mismatches"].items())
        and score_mismatches == 0
        and weighted_mismatches == 0
        and signal_mismatches == 0
        and range_mismatches == 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact Excel R1-R6 audit for candle_formula_strategy.")
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

    all_pass = True
    for timeframe, path in jobs:
        result = audit(path, timeframe, args.month, args.symbol)
        all_pass = all_pass and bool(result["pass"])
        print(
            f"{timeframe} | rows={result['rows']} | "
            f"R1={result['R1_UP']} R2={result['R2_UP']} R3={result['R3_UP']} "
            f"R4={result['R4_DOWN']} R5={result['R5_DOWN']} R6={result['R6_DOWN']} | "
            f"S={result['score_mismatches']} WEIGHTED={result['weighted_mismatches']} "
            f"SIGNAL={result['signal_mismatches']} RANGE={result['range_mismatches']} | "
            f"{'PASS' if result['pass'] else 'FAIL'}"
        )

    print(f"AUDIT={'PASS' if all_pass else 'FAIL'}")
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
