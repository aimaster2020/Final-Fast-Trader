#!/usr/bin/env python3
"""Fine sweep for the thresholded previous-candle formula.

Formula:
=IF(A12="","",IF(E12-B12>$F$3,AVERAGE(E11),IF(E12-B12<$F$4,AVERAGE(C11),E12)))

Sweep range:
F3 = 100..170, step 1
F4 = -270..-140, step 1

Accuracy is measured only where both CT and PT are non-neutral.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--ct-up", type=float, default=0.0)
    p.add_argument("--ct-down", type=float, default=0.0)
    return p.parse_args()


def is_header(row: list[str]) -> bool:
    if not row:
        return False
    s = ",".join(x.strip().lower() for x in row)
    return any(k in s for k in ("timestamp", "open", "high", "low", "close"))


def ts_year(v: str) -> Optional[int]:
    try:
        import datetime as dt
        ts = int(float(v))
        if ts > 10_000_000_000:
            ts //= 1000
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).year
    except Exception:
        return None


def load_rows(path: Path, year: Optional[int]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError("Input is empty")

    out: list[dict[str, str]] = []
    if is_header(rows[0]):
        hdr = [x.strip().lower() for x in rows[0]]
        idx = {name: i for i, name in enumerate(hdr)}
        for req in ("timestamp", "open", "high", "close"):
            if req not in idx:
                raise ValueError(f"Missing column: {req}")
        for r in rows[1:]:
            if len(r) <= max(idx["timestamp"], idx["open"], idx["high"], idx["close"]):
                continue
            if year is not None and ts_year(r[idx["timestamp"]]) != year:
                continue
            out.append({"timestamp": r[idx["timestamp"]], "open": r[idx["open"]], "high": r[idx["high"]], "close": r[idx["close"]]})
    else:
        for r in rows:
            if len(r) < 5:
                continue
            if year is not None and ts_year(r[0]) != year:
                continue
            out.append({"timestamp": r[0], "open": r[1], "high": r[2], "close": r[4]})
    return out


def direction(move: float, up: float, down: float) -> int:
    if move > up:
        return 1
    if move < -down:
        return -1
    return 0


def evaluate(rows: list[dict[str, str]], f3: float, f4: float, ct_up: float, ct_down: float) -> dict[str, float | int]:
    correct = 0
    both = 0
    pt_signals = 0
    pt_up = 0
    pt_down = 0

    for i in range(1, len(rows) - 1):
        prev = rows[i - 1]
        cur = rows[i]
        nxt = rows[i + 1]
        o = float(cur["open"])
        c = float(cur["close"])
        ph = float(prev["high"])
        pc = float(prev["close"])
        nc = float(nxt["close"])

        body = c - o
        if body > f3:
            predict = pc
        elif body < f4:
            predict = ph
        else:
            predict = c

        ct = direction(nc - c, ct_up, ct_down)
        pt = direction(predict - c, ct_up, ct_down)
        if pt != 0:
            pt_signals += 1
            if pt == 1:
                pt_up += 1
            else:
                pt_down += 1
        if ct != 0 and pt != 0:
            both += 1
            if ct == pt:
                correct += 1

    return {
        "f3": f3,
        "f4": f4,
        "correct": correct,
        "both_nonzero": both,
        "accuracy": (correct / both * 100.0) if both else 0.0,
        "pt_signals": pt_signals,
        "pt_up": pt_up,
        "pt_down": pt_down,
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input), args.year)
    results = []
    for f3 in range(100, 171):
        for f4 in range(-270, -139):
            results.append(evaluate(rows, f3, f4, args.ct_up, args.ct_down))

    results.sort(key=lambda r: (-float(r["accuracy"]), -int(r["both_nonzero"]), -int(r["pt_signals"])))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"input={args.input}")
    print(f"output={out}")
    print(f"rows={len(rows)} frames={max(0, len(rows)-2)}")
    print("range F3=100..170, F4=-270..-140, step=1")
    print(f"combinations={len(results)} CT_UP={args.ct_up:g} CT_DOWN={args.ct_down:g}")
    print()
    print("TOP 20 BY ACCURACY")
    for n, r in enumerate(results[:20], 1):
        print(f"{n:2d}. F3={int(r['f3'])} F4={int(r['f4'])} accuracy={float(r['accuracy']):.4f}% correct={int(r['correct'])}/{int(r['both_nonzero'])} PT_signals={int(r['pt_signals'])} up={int(r['pt_up'])} down={int(r['pt_down'])}")
    print()
    for minimum in (25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500):
        eligible = [r for r in results if int(r["both_nonzero"]) >= minimum]
        if not eligible:
            continue
        best = max(eligible, key=lambda r: (float(r["accuracy"]), int(r["both_nonzero"]), int(r["pt_signals"])))
        print(f"BEST WITH both_nonzero>={minimum}: F3={int(best['f3'])} F4={int(best['f4'])} accuracy={float(best['accuracy']):.4f}% correct={int(best['correct'])}/{int(best['both_nonzero'])} PT_signals={int(best['pt_signals'])}")


if __name__ == "__main__":
    main()
