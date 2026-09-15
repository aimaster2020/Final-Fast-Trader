#!/usr/bin/env python3
"""Test the simple Excel previous-candle prediction formula.

Excel formula:
=IF(A12="","",IF(E12-B12>0,AVERAGE(E11),IF(E12-B12<0,AVERAGE(C11),E12)))

Assumed columns:
A = timestamp
B = open
C = high
D = low
E = close
F = volume (optional)

For each candle i (after the first usable candle):
- bullish current candle (close > open): predict = previous close
- bearish current candle (close < open): predict = previous high
- doji: predict = current close

CT evaluates the next candle's actual move from the current close.
PT evaluates the formula's predicted move from the current close.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test simple previous-candle Excel prediction formula.")
    parser.add_argument("--input", required=True, help="Input CSV path. Headered or headerless.")
    parser.add_argument("--output", required=True, help="Output CSV path with per-frame results.")
    parser.add_argument("--ct-up", type=float, required=True, help="Actual upward CT threshold in price units.")
    parser.add_argument("--ct-down", type=float, required=True, help="Actual downward CT threshold in price units.")
    parser.add_argument("--year", type=int, default=None, help="Optional calendar year filter.")
    parser.add_argument("--timestamp-column", default=None, help="Timestamp column name when a header is present.")
    return parser.parse_args()


def is_header(row: list[str]) -> bool:
    if not row:
        return False
    joined = ",".join(x.strip().lower() for x in row)
    return any(x in joined for x in ("timestamp", "open", "high", "low", "close"))


def numeric(value: str) -> float:
    return float(str(value).strip())


def parse_timestamp(value: str) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def timestamp_year(value: str) -> Optional[int]:
    ts = parse_timestamp(value)
    if ts is None:
        return None
    # Handle seconds and milliseconds since Unix epoch.
    if ts > 10_000_000_000:
        ts //= 1000
    import datetime as dt
    try:
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).year
    except (OverflowError, OSError, ValueError):
        return None


def load_rows(path: Path, year: Optional[int], timestamp_column: Optional[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError("Input CSV is empty.")

    if is_header(rows[0]):
        raw_header = [x.strip() for x in rows[0]]
        normalized = {name.lower(): idx for idx, name in enumerate(raw_header)}
        required = {"open", "high", "close"}
        missing = sorted(required - normalized.keys())
        if missing:
            raise ValueError(f"Missing required columns in header: {', '.join(missing)}")
        ts_name = (timestamp_column or "timestamp").lower()
        if ts_name not in normalized:
            raise ValueError(f"Timestamp column '{timestamp_column or 'timestamp'}' not found.")
        oi, hi, ci = normalized["open"], normalized["high"], normalized["close"]
        ti = normalized[ts_name]
        parsed: list[dict[str, str]] = []
        for r in rows[1:]:
            if not r or len(r) <= max(oi, hi, ci, ti):
                continue
            item = {"timestamp": r[ti], "open": r[oi], "high": r[hi], "close": r[ci]}
            if year is None or timestamp_year(r[ti]) == year:
                parsed.append(item)
        return parsed

    parsed = []
    for r in rows:
        if len(r) < 5:
            continue
        item = {"timestamp": r[0], "open": r[1], "high": r[2], "close": r[4]}
        if year is None or timestamp_year(r[0]) == year:
            parsed.append(item)
    return parsed


def direction_from_move(move: float, up: float, down: float) -> int:
    if move > up:
        return 1
    if move < -down:
        return -1
    return 0


def pct(value: float, total: int) -> float:
    return (value / total * 100.0) if total else 0.0


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = load_rows(input_path, args.year, args.timestamp_column)

    if len(rows) < 3:
        raise ValueError("Need at least 3 usable candles.")

    results: list[dict[str, object]] = []
    correct = 0
    evaluated = 0
    nonzero_predictions = 0
    pt_correct = 0
    pt_evaluated = 0

    for i in range(1, len(rows) - 1):
        prev = rows[i - 1]
        cur = rows[i]
        nxt = rows[i + 1]

        o = numeric(cur["open"])
        c = numeric(cur["close"])
        prev_high = numeric(prev["high"])
        prev_close = numeric(prev["close"])
        next_close = numeric(nxt["close"])

        body = c - o
        if body > 0:
            predict = prev_close
            branch = "prev_close"
        elif body < 0:
            predict = prev_high
            branch = "prev_high"
        else:
            predict = c
            branch = "current_close"

        actual_move = next_close - c
        predicted_move = predict - c
        ct = direction_from_move(actual_move, args.ct_up, args.ct_down)
        pt = direction_from_move(predicted_move, args.ct_up, args.ct_down)

        if ct != 0:
            evaluated += 1
            if pt == ct:
                correct += 1
        if pt != 0:
            pt_evaluated += 1
            nonzero_predictions += 1

        if pt != 0 and ct != 0:
            if pt == ct:
                pt_correct += 1

        results.append(
            {
                "timestamp": cur["timestamp"],
                "open": o,
                "high": numeric(cur["high"]),
                "low": numeric(rows[i]["close"]),
                "close": c,
                "prev_high": prev_high,
                "prev_close": prev_close,
                "formula_branch": branch,
                "predict": predict,
                "predicted_move": predicted_move,
                "next_close": next_close,
                "actual_move": actual_move,
                "ct": ct,
                "pt": pt,
                "direction_correct": int(ct != 0 and pt == ct),
                "both_nonzero_correct": int(ct != 0 and pt != 0 and pt == ct),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys()) if results else []
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    exact_direction_accuracy = pct(correct, evaluated)
    both_nonzero_accuracy = pct(pt_correct, sum(1 for r in results if r["ct"] != 0 and r["pt"] != 0))

    ct_up_count = sum(1 for r in results if r["ct"] == 1)
    ct_down_count = sum(1 for r in results if r["ct"] == -1)
    pt_up_count = sum(1 for r in results if r["pt"] == 1)
    pt_down_count = sum(1 for r in results if r["pt"] == -1)

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"rows={len(rows)} frames={total}")
    print(f"ct_up={args.ct_up:g} ct_down={args.ct_down:g}")
    if args.year is not None:
        print(f"year={args.year}")
    print()
    print("FORMULA")
    print("bullish current -> previous close")
    print("bearish current -> previous high")
    print("doji -> current close")
    print()
    print("CT / PT")
    print(f"CT up={ct_up_count} down={ct_down_count} neutral={total - ct_up_count - ct_down_count}")
    print(f"PT up={pt_up_count} down={pt_down_count} neutral={total - pt_up_count - pt_down_count}")
    print()
    print("ACCURACY")
    print(f"direction accuracy (all non-neutral CT)={exact_direction_accuracy:.4f}% ({correct}/{evaluated})")
    both_total = sum(1 for r in results if r["ct"] != 0 and r["pt"] != 0)
    print(f"accuracy when both CT and PT non-neutral={both_nonzero_accuracy:.4f}% ({pt_correct}/{both_total})")
    print(f"non-neutral PT signals={nonzero_predictions}")


if __name__ == "__main__":
    main()
