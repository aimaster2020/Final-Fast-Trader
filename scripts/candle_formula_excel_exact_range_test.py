from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import evaluate_rules
from fast_pattern_trader.models import Candle


DATA_FIRST_EXCEL_ROW = 2


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
    return sorted(candles, key=lambda c: c.timestamp)


def j_value(candle: Candle) -> float:
    """Excel column J = Close - Open."""
    return candle.close - candle.open


def excel_rules(candle: Candle) -> tuple[int, int, int, int, int, int]:
    """Exact Excel P:R and the opposite three comparisons."""
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


def excel_i(current: Candle, following: Candle) -> int:
    """Exact Excel I: compare J(next row) with J(current row)."""
    current_j = j_value(current)
    next_j = j_value(following)
    if next_j > current_j:
        return 1
    if next_j < current_j:
        return -1
    return 0


def run(
    path: Path,
    timeframe: str,
    month: str,
    symbol: str,
    start_excel_row: int,
    end_excel_row: int,
) -> dict[str, float | int]:
    if start_excel_row < DATA_FIRST_EXCEL_ROW:
        raise SystemExit(f"start_excel_row must be >= {DATA_FIRST_EXCEL_ROW}")
    if end_excel_row < start_excel_row:
        raise SystemExit("end_excel_row must be >= start_excel_row")

    candles = load_month(path, month, symbol)
    signals_needed = end_excel_row - start_excel_row + 1
    first_index = start_excel_row - DATA_FIRST_EXCEL_ROW
    last_current_index = first_index + signals_needed - 1
    last_following_index = last_current_index + 1

    if first_index < 0 or last_following_index >= len(candles):
        raise SystemExit(
            f"Excel rows {start_excel_row}:{end_excel_row} need data through Excel row "
            f"{end_excel_row + 1}; loaded {len(candles)} data rows starting at Excel row {DATA_FIRST_EXCEL_ROW}."
        )

    up_total = up_correct = 0
    down_total = down_correct = 0
    hold_total = 0
    flat_total = 0
    rule_mismatches = 0

    for offset in range(signals_needed):
        idx = first_index + offset
        current = candles[idx]
        following = candles[idx + 1]

        r1, r2, r3, r4, r5, r6 = excel_rules(current)
        s = r1 + r2 + r3
        expected_signal = 1 if s == 3 else -1 if s == 0 else 0

        project_rules = evaluate_rules(current)
        project_tuple = tuple(
            int(getattr(project_rules, attr))
            for attr in (
                "up_rule_1", "up_rule_2", "up_rule_3",
                "down_rule_1", "down_rule_2", "down_rule_3",
            )
        )
        if project_tuple != (r1, r2, r3, r4, r5, r6):
            rule_mismatches += 1

        actual = excel_i(current, following)

        if expected_signal == 1:
            up_total += 1
            up_correct += int(actual == 1)
        elif expected_signal == -1:
            down_total += 1
            down_correct += int(actual == -1)
        else:
            hold_total += 1

        flat_total += int(actual == 0)

    total_signals = up_total + down_total
    total_correct = up_correct + down_correct
    up_accuracy = up_correct / up_total * 100.0 if up_total else 0.0
    down_accuracy = down_correct / down_total * 100.0 if down_total else 0.0
    overall_accuracy = total_correct / total_signals * 100.0 if total_signals else 0.0

    return {
        "candles_loaded": len(candles),
        "excel_rows_start": start_excel_row,
        "excel_rows_end": end_excel_row,
        "signals": signals_needed,
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
        "flat_total": flat_total,
        "rule_mismatches": rule_mismatches,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reproduce Excel COUNTIF ranges exactly: S[start:end] and T/U against I(next J vs current J)."
    )
    ap.add_argument("--month", default="2026-05")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--input", default="reports/prepared_price_action_15m.csv")
    ap.add_argument("--start-row", type=int, default=2)
    ap.add_argument("--end-row", type=int, default=2953)
    args = ap.parse_args()

    result = run(
        Path(args.input),
        args.timeframe,
        args.month,
        args.symbol,
        args.start_row,
        args.end_row,
    )

    print(
        f"{args.timeframe} | ExcelRows={result['excel_rows_start']}:{result['excel_rows_end']} "
        f"signals={result['signals']} | "
        f"UP={result['up_total']}/{result['up_correct']} ({result['up_accuracy']:.2f}%) | "
        f"DOWN={result['down_total']}/{result['down_correct']} ({result['down_accuracy']:.2f}%) | "
        f"TOTAL={result['total_signals']}/{result['total_correct']} ({result['overall_accuracy']:.2f}%) | "
        f"HOLD={result['hold_total']} FLAT={result['flat_total']} | "
        f"RULE_MISMATCH={result['rule_mismatches']}"
    )


if __name__ == "__main__":
    main()
