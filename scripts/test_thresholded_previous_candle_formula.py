#!/usr/bin/env python3
"""Test the Excel formula with configurable F3/F4 thresholds.

Excel formula:
=IF(A12="","",IF(E12-B12>$F$3,AVERAGE(E11),IF(E12-B12<$F$4,AVERAGE(C11),E12)))

Assumed columns:
A = timestamp
B = open
C = high
D = low
E = close

The formula is evaluated exactly as:
- current close - current open > F3 -> previous close
- current close - current open < F4 -> previous high
- otherwise -> current close

CT evaluates the next candle's actual move from current close.
PT evaluates the formula's predicted move from current close.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test thresholded Excel previous-candle formula.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--f3", type=float, required=True, help="Excel F3 threshold for bullish branch.")
    p.add_argument("--f4", type=float, required=True, help="Excel F4 threshold for bearish branch.")
    p.add_argument("--ct-up", type=float, required=True)
    p.add_argument("--ct-down", type=float, required=True)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--timestamp-column", default=None)
    return p.parse_args()


def is_header(row: list[str]) -> bool:
    if not row:
        return False
    joined = ",".join(x.strip().lower() for x in row)
    return any(x in joined for x in ("timestamp", "open", "high", "low", "close"))


def num(v: str) -> float:
    return float(str(v).strip())


def parse_ts(v: str) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def ts_year(v: str) -> Optional[int]:
    ts = parse_ts(v)
    if ts is None:
        return None
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
        header = [x.strip() for x in rows[0]]
        idx = {name.lower(): i for i, name in enumerate(header)}
        for required in ("open", "high", "close"):
            if required not in idx:
                raise ValueError(f"Missing required column: {required}")
        ts_name = (timestamp_column or "timestamp").lower()
        if ts_name not in idx:
            raise ValueError(f"Timestamp column '{timestamp_column or 'timestamp'}' not found.")
        ti, oi, hi, ci = idx[ts_name], idx["open"], idx["high"], idx["close"]
        out = []
        for r in rows[1:]:
            if len(r) <= max(ti, oi, hi, ci):
                continue
            if year is None or ts_year(r[ti]) == year:
                out.append({"timestamp": r[ti], "open": r[oi], "high": r[hi], "close": r[ci]})
        return out

    out = []
    for r in rows:
        if len(r) < 5:
            continue
        if year is None or ts_year(r[0]) == year:
            out.append({"timestamp": r[0], "open": r[1], "high": r[2], "close": r[4]})
    return out


def direction(move: float, up: float, down: float) -> int:
    if move > up:
        return 1
    if move < -down:
        return -1
    return 0


def main() -> None:
    a = parse_args()
    rows = load_rows(Path(a.input), a.year, a.timestamp_column)
    if len(rows) < 3:
        raise ValueError("Need at least 3 usable candles.")

    results = []
    correct = 0
    evaluated = 0
    both_correct = 0
    both_total = 0

    for i in range(1, len(rows) - 1):
        prev = rows[i - 1]
        cur = rows[i]
        nxt = rows[i + 1]
        o = num(cur["open"])
        h = num(cur["high"])
        c = num(cur["close"])
        prev_h = num(prev["high"])
        prev_c = num(prev["close"])
        nc = num(nxt["close"])
        body = c - o

        if body > a.f3:
            predict = prev_c
            branch = "prev_close"
        elif body < a.f4:
            predict = prev_h
            branch = "prev_high"
        else:
            predict = c
            branch = "current_close"

        actual_move = nc - c
        predicted_move = predict - c
        ct = direction(actual_move, a.ct_up, a.ct_down)
        pt = direction(predicted_move, a.ct_up, a.ct_down)

        if ct != 0:
            evaluated += 1
            if pt == ct:
                correct += 1
        if ct != 0 and pt != 0:
            both_total += 1
            if pt == ct:
                both_correct += 1

        results.append({
            "timestamp": cur["timestamp"],
            "open": o,
            "high": h,
            "close": c,
            "prev_high": prev_h,
            "prev_close": prev_c,
            "body": body,
            "f3": a.f3,
            "f4": a.f4,
            "formula_branch": branch,
            "predict": predict,
            "predicted_move": predicted_move,
            "next_close": nc,
            "actual_move": actual_move,
            "ct": ct,
            "pt": pt,
            "direction_correct": int(ct != 0 and pt == ct),
        })

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    ct_up = sum(r["ct"] == 1 for r in results)
    ct_down = sum(r["ct"] == -1 for r in results)
    pt_up = sum(r["pt"] == 1 for r in results)
    pt_down = sum(r["pt"] == -1 for r in results)
    branches = {b: sum(r["formula_branch"] == b for r in results) for b in ("prev_close", "prev_high", "current_close")}

    print(f"input={a.input}")
    print(f"output={a.output}")
    print(f"rows={len(rows)} frames={total}")
    print(f"F3={a.f3:g} F4={a.f4:g}")
    print(f"CT_UP={a.ct_up:g} CT_DOWN={a.ct_down:g}")
    if a.year is not None:
        print(f"year={a.year}")
    print()
    print("FORMULA")
    print(f"body > F3 ({a.f3:g}) -> previous close: {branches['prev_close']}")
    print(f"body < F4 ({a.f4:g}) -> previous high: {branches['prev_high']}")
    print(f"otherwise -> current close: {branches['current_close']}")
    print()
    print("CT / PT")
    print(f"CT up={ct_up} down={ct_down} neutral={total-ct_up-ct_down}")
    print(f"PT up={pt_up} down={pt_down} neutral={total-pt_up-pt_down}")
    print()
    print("ACCURACY")
    print(f"direction accuracy (all non-neutral CT)={correct/evaluated*100:.4f}% ({correct}/{evaluated})" if evaluated else "direction accuracy=0.0000% (0/0)")
    print(f"accuracy when both CT and PT non-neutral={both_correct/both_total*100:.4f}% ({both_correct}/{both_total})" if both_total else "accuracy when both CT and PT non-neutral=0.0000% (0/0)")
    print(f"non-neutral PT signals={pt_up+pt_down}")


if __name__ == "__main__":
    main()
