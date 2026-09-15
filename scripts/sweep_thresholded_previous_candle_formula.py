#!/usr/bin/env python3
"""Sweep F3/F4 for the Excel thresholded previous-candle formula.

Formula:
=IF(A12="","",IF(E12-B12>$F$3,AVERAGE(E11),IF(E12-B12<$F$4,AVERAGE(C11),E12)))

A=timestamp, B=open, C=high, E=close.
F3 and F4 are swept independently from -1000 to +1000 inclusive,
with a step of 50.

Evaluation uses CT_UP=0 / CT_DOWN=0 by default, matching the initial
formula-direction test. A signal is considered directionally correct
when both CT and PT are non-neutral and PT == CT.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep F3/F4 for thresholded previous-candle formula.")
    p.add_argument("--input", required=True, help="Input CSV, headered or headerless.")
    p.add_argument("--output", required=True, help="Output CSV containing every F3/F4 combination.")
    p.add_argument("--min-value", type=float, default=-1000.0)
    p.add_argument("--max-value", type=float, default=1000.0)
    p.add_argument("--step", type=float, default=50.0)
    p.add_argument("--ct-up", type=float, default=0.0)
    p.add_argument("--ct-down", type=float, default=0.0)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--top", type=int, default=20, help="Number of top rows to print.")
    return p.parse_args()


def is_header(row: list[str]) -> bool:
    joined = ",".join(x.strip().lower() for x in row)
    return any(x in joined for x in ("timestamp", "open", "high", "close"))


def num(value: str) -> float:
    return float(str(value).strip())


def ts_year(value: str) -> Optional[int]:
    try:
        ts = int(float(value))
        if ts > 10_000_000_000:
            ts //= 1000
        import datetime as dt
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).year
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def load_rows(path: Path, year: Optional[int]) -> list[tuple[str, float, float, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError("Input CSV is empty.")

    out: list[tuple[str, float, float, float]] = []
    if is_header(rows[0]):
        h = [x.strip().lower() for x in rows[0]]
        idx = {name: i for i, name in enumerate(h)}
        for name in ("timestamp", "open", "high", "close"):
            if name not in idx:
                raise ValueError(f"Missing required column: {name}")
        for r in rows[1:]:
            if len(r) <= max(idx.values()):
                continue
            if year is not None and ts_year(r[idx["timestamp"]]) != year:
                continue
            out.append((r[idx["timestamp"]], num(r[idx["open"]]), num(r[idx["high"]]), num(r[idx["close"]])))
    else:
        for r in rows:
            if len(r) < 5:
                continue
            if year is not None and ts_year(r[0]) != year:
                continue
            out.append((r[0], num(r[1]), num(r[2]), num(r[4])))
    return out


def direction(move: float, up: float, down: float) -> int:
    if move > up:
        return 1
    if move < -down:
        return -1
    return 0


def values(min_value: float, max_value: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    count = int(round((max_value - min_value) / step))
    return [min_value + i * step for i in range(count + 1)]


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input), args.year)
    if len(rows) < 3:
        raise ValueError("Need at least 3 usable candles.")

    f3s = values(args.min_value, args.max_value, args.step)
    f4s = values(args.min_value, args.max_value, args.step)
    combinations = len(f3s) * len(f4s)

    # Precompute the candle-level quantities needed for all combinations.
    frames: list[tuple[float, float, float, float]] = []
    # body, prev_close, prev_high, actual_next_move
    for i in range(1, len(rows) - 1):
        _, o, h, c = rows[i]
        _, prev_o, prev_h, prev_c = rows[i - 1]
        _, next_o, next_h, next_c = rows[i + 1]
        del h, prev_o, next_o, next_h
        frames.append((c - o, prev_c, prev_h, next_c - c))

    results: list[dict[str, float | int]] = []
    for f3 in f3s:
        for f4 in f4s:
            ct_up = ct_down = ct_neutral = 0
            pt_up = pt_down = pt_neutral = 0
            both = correct = 0

            for body, prev_close, prev_high, actual_move in frames:
                if body > f3:
                    predict = prev_close
                elif body < f4:
                    predict = prev_high
                else:
                    # Formula's third branch is current close; its predicted move is 0.
                    predict = 0.0

                ct = direction(actual_move, args.ct_up, args.ct_down)
                if ct == 1:
                    ct_up += 1
                elif ct == -1:
                    ct_down += 1
                else:
                    ct_neutral += 1

                # For the neutral formula branch, predicted move is exactly zero.
                # Otherwise predict is an absolute price and must be converted using current close.
                predicted_move = 0.0 if body >= f4 and body <= f3 else predict
                # Recover current close from actual_move relation is impossible, so use a second
                # precomputed value below; kept separate for clarity.
                if body > f3:
                    predicted_move = prev_close - (frames[frames.index((body, prev_close, prev_high, actual_move))][3] - actual_move) if False else predicted_move

            # Recompute cleanly with current close retained.
            ct_up = ct_down = ct_neutral = 0
            pt_up = pt_down = pt_neutral = 0
            both = correct = 0
            for i in range(1, len(rows) - 1):
                _, o, h, c = rows[i]
                _, _, prev_h, prev_c = rows[i - 1]
                _, _, _, next_c = rows[i + 1]
                body = c - o
                if body > f3:
                    predict = prev_c
                elif body < f4:
                    predict = prev_h
                else:
                    predict = c
                actual_move = next_c - c
                predicted_move = predict - c
                ct = direction(actual_move, args.ct_up, args.ct_down)
                pt = direction(predicted_move, args.ct_up, args.ct_down)
                if ct == 1:
                    ct_up += 1
                elif ct == -1:
                    ct_down += 1
                else:
                    ct_neutral += 1
                if pt == 1:
                    pt_up += 1
                elif pt == -1:
                    pt_down += 1
                else:
                    pt_neutral += 1
                if ct != 0 and pt != 0:
                    both += 1
                    if ct == pt:
                        correct += 1

            accuracy = (correct / both * 100.0) if both else 0.0
            results.append({
                "f3": f3,
                "f4": f4,
                "both_nonzero": both,
                "correct": correct,
                "accuracy_pct": accuracy,
                "pt_up": pt_up,
                "pt_down": pt_down,
                "pt_signals": pt_up + pt_down,
                "ct_up": ct_up,
                "ct_down": ct_down,
            })

    results.sort(key=lambda r: (r["accuracy_pct"], r["both_nonzero"], r["pt_signals"]), reverse=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0].keys())
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    print(f"input={args.input}")
    print(f"output={out}")
    print(f"rows={len(rows)} frames={len(frames)}")
    print(f"F3/F4 range={args.min_value:g}..{args.max_value:g} step={args.step:g}")
    print(f"combinations={combinations}")
    print(f"CT_UP={args.ct_up:g} CT_DOWN={args.ct_down:g}")
    print()
    print("TOP BY ACCURACY (then sample count)")
    for n, r in enumerate(results[: args.top], 1):
        print(
            f"{n:2d}. F3={r['f3']:g} F4={r['f4']:g} "
            f"accuracy={r['accuracy_pct']:.4f}% "
            f"correct={r['correct']}/{r['both_nonzero']} "
            f"PT_signals={r['pt_signals']} "
            f"up={r['pt_up']} down={r['pt_down']}"
        )

    # Also show best rows after requiring minimum sample sizes.
    for minimum in (25, 50, 100, 250, 500):
        eligible = [r for r in results if r["both_nonzero"] >= minimum]
        if eligible:
            best = eligible[0]
            print()
            print(
                f"BEST WITH both_nonzero>={minimum}: "
                f"F3={best['f3']:g} F4={best['f4']:g} "
                f"accuracy={best['accuracy_pct']:.4f}% "
                f"correct={best['correct']}/{best['both_nonzero']} "
                f"PT_signals={best['pt_signals']}"
            )


if __name__ == "__main__":
    main()
