#!/usr/bin/env python3
"""Test the reversed branch mapping of the simple Excel formula.

Original:
  bullish current -> previous close
  bearish current -> previous high

Reversed:
  bullish current -> previous high
  bearish current -> previous close
  doji -> current close

Excel equivalent:
=IF(A12="","",IF(E12-B12<0,AVERAGE(E11),IF(E12-B12>0,AVERAGE(C11),E12)))

Columns: A=timestamp, B=open, C=high, D=low, E=close.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test reversed simple previous-candle Excel prediction formula.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ct-up", type=float, required=True)
    p.add_argument("--ct-down", type=float, required=True)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--timestamp-column", default=None)
    return p.parse_args()


def has_header(row: list[str]) -> bool:
    s = ",".join(x.strip().lower() for x in row)
    return any(x in s for x in ("timestamp", "open", "high", "low", "close"))


def ts_year(value: str) -> Optional[int]:
    try:
        ts = int(float(value))
        if ts > 10_000_000_000:
            ts //= 1000
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).year
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def load(path: Path, year: Optional[int], timestamp_column: Optional[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError("Input CSV is empty.")

    if has_header(rows[0]):
        header = [x.strip() for x in rows[0]]
        idx = {name.lower(): i for i, name in enumerate(header)}
        for name in ("open", "high", "close"):
            if name not in idx:
                raise ValueError(f"Missing required column: {name}")
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
    rows = load(Path(a.input), a.year, a.timestamp_column)
    if len(rows) < 3:
        raise ValueError("Need at least 3 usable candles.")

    results = []
    exact_correct = 0
    exact_total = 0
    both_correct = 0
    both_total = 0

    for i in range(1, len(rows) - 1):
        prev, cur, nxt = rows[i - 1], rows[i], rows[i + 1]
        o = float(cur["open"])
        c = float(cur["close"])
        ph = float(prev["high"])
        pc = float(prev["close"])
        nc = float(nxt["close"])

        body = c - o
        if body > 0:
            predict, branch = ph, "prev_high"
        elif body < 0:
            predict, branch = pc, "prev_close"
        else:
            predict, branch = c, "current_close"

        actual_move = nc - c
        predicted_move = predict - c
        ct = direction(actual_move, a.ct_up, a.ct_down)
        pt = direction(predicted_move, a.ct_up, a.ct_down)

        if ct != 0:
            exact_total += 1
            if pt == ct:
                exact_correct += 1
        if ct != 0 and pt != 0:
            both_total += 1
            if pt == ct:
                both_correct += 1

        results.append({
            "timestamp": cur["timestamp"],
            "open": o,
            "high": float(cur["high"]),
            "close": c,
            "prev_high": ph,
            "prev_close": pc,
            "formula_branch": branch,
            "predict": predict,
            "predicted_move": predicted_move,
            "next_close": nc,
            "actual_move": actual_move,
            "ct": ct,
            "pt": pt,
            "direction_correct": int(ct != 0 and pt == ct),
        })

    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    total = len(results)
    pt_up = sum(r["pt"] == 1 for r in results)
    pt_down = sum(r["pt"] == -1 for r in results)
    ct_up = sum(r["ct"] == 1 for r in results)
    ct_down = sum(r["ct"] == -1 for r in results)

    print(f"input={a.input}")
    print(f"output={a.output}")
    print(f"rows={len(rows)} frames={total}")
    print(f"ct_up={a.ct_up:g} ct_down={a.ct_down:g}")
    if a.year is not None:
        print(f"year={a.year}")
    print()
    print("REVERSED FORMULA")
    print("bullish current -> previous high")
    print("bearish current -> previous close")
    print("doji -> current close")
    print()
    print("CT / PT")
    print(f"CT up={ct_up} down={ct_down} neutral={total - ct_up - ct_down}")
    print(f"PT up={pt_up} down={pt_down} neutral={total - pt_up - pt_down}")
    print()
    print("ACCURACY")
    acc_all = exact_correct / exact_total * 100 if exact_total else 0.0
    acc_both = both_correct / both_total * 100 if both_total else 0.0
    print(f"direction accuracy (all non-neutral CT)={acc_all:.4f}% ({exact_correct}/{exact_total})")
    print(f"accuracy when both CT and PT non-neutral={acc_both:.4f}% ({both_correct}/{both_total})")
    print(f"non-neutral PT signals={pt_up + pt_down}")


if __name__ == "__main__":
    main()
