#!/usr/bin/env python3
"""Walk-forward magnitude baseline for the exact Excel S0-S3 candle formula.

For each symbol/month, rows are ordered chronologically.  For row i, the
predicted magnitude is learned only from rows before i with the same S state:
median(abs(next_close-current_close)).  A global same-symbol fallback is used
when that state has no history yet.

The script evaluates:
- MAE of predicted absolute magnitude
- median absolute error
- correlation between predicted and realized absolute magnitude
- directional signed error using the existing S0-S3 signal

No future observations are used to form a prediction.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT = Path("reports/prepared_price_action_1h.csv")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")


@dataclass
class Row:
    symbol: str
    month: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float


def excel_s(r: Row) -> int:
    j = r.close - r.open
    k = r.high - r.close
    m = r.high - r.open
    l = r.low - r.close
    return int(k > j) + int(k > m) + int(l > j)


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            try:
                rows.append(
                    Row(
                        symbol=str(rec["symbol"]),
                        month=str(rec["month"]),
                        timestamp=int(float(rec["timestamp"])),
                        open=float(rec["open"]),
                        high=float(rec["high"]),
                        low=float(rec["low"]),
                        close=float(rec["close"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    rows.sort(key=lambda x: (x.symbol, x.month, x.timestamp))
    return rows


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def evaluate_group(rows: list[Row]) -> dict[int, dict[str, float]]:
    """Evaluate one symbol/month with strict past-only state medians."""
    if len(rows) < 2:
        return {}

    state_history: dict[int, list[float]] = defaultdict(list)
    global_history: list[float] = []
    errors_by_state: dict[int, list[float]] = defaultdict(list)
    pred_by_state: dict[int, list[float]] = defaultdict(list)
    actual_by_state: dict[int, list[float]] = defaultdict(list)

    # The target for row i is only known after row i+1 exists.
    for i in range(len(rows) - 1):
        cur = rows[i]
        nxt = rows[i + 1]
        state = excel_s(cur)
        move = nxt.close - cur.close
        actual_abs = abs(move)

        pred = median_or_none(state_history[state])
        if pred is None:
            pred = median_or_none(global_history)

        # Skip the cold-start rows before any historical target exists.
        if pred is not None:
            errors_by_state[state].append(abs(pred - actual_abs))
            pred_by_state[state].append(pred)
            actual_by_state[state].append(actual_abs)

        # Target becomes available only after evaluation, so it is now safe to
        # add it to history for future rows.
        state_history[state].append(actual_abs)
        global_history.append(actual_abs)

    out: dict[int, dict[str, float]] = {}
    for state in range(4):
        errs = errors_by_state[state]
        preds = pred_by_state[state]
        actuals = actual_by_state[state]
        if not errs:
            continue
        out[state] = {
            "n": float(len(errs)),
            "mae": mean(errs),
            "medae": statistics.median(errs),
            "corr": pearson(preds, actuals),
            "mean_pred": mean(preds),
            "mean_actual": mean(actuals),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = ap.parse_args()

    rows = load_rows(args.input)
    grouped: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for r in rows:
        if r.symbol in SYMBOLS and r.month in MONTHS:
            grouped[(r.symbol, r.month)].append(r)

    print(
        "EXCEL_MAGNITUDE_WF | tf=1h | fee=0 | exact S0-S3 formula | "
        "target=abs(next_close-current_close) | past_only | state_median"
    )
    print("format: N / MAE / MedAE / Corr / PredMean / ActualMean")

    all_metrics: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)

    for symbol in SYMBOLS:
        print(f"\n{symbol}")
        for month in MONTHS:
            metrics = evaluate_group(grouped.get((symbol, month), []))
            all_metrics[symbol].update({(month, s): m for s, m in metrics.items()})
            parts = []
            for state in range(4):
                m = metrics.get(state)
                if not m:
                    parts.append(f"S{state}=0/0/0/0/0/0")
                else:
                    parts.append(
                        f"S{state}={int(m['n'])}/{m['mae']:.2f}/{m['medae']:.2f}/"
                        f"{m['corr']:.3f}/{m['mean_pred']:.2f}/{m['mean_actual']:.2f}"
                    )
            print(f"{month} | " + " ".join(parts))

        # Aggregate raw prediction errors across four months for each state.
        print("4M |", end=" ")
        for state in range(4):
            state_metrics = [
                m
                for (month, s), m in all_metrics[symbol].items()
                if s == state
            ]
            if not state_metrics:
                print(f"S{state}=0/0/0/0/0/0", end=" ")
                continue
            n = sum(int(m["n"]) for m in state_metrics)
            mae = sum(m["mae"] * m["n"] for m in state_metrics) / n
            medae_vals = []
            # Median-of-medians is intentionally not relabeled as an exact
            # pooled median; we expose it simply as the mean monthly MedAE.
            medae = mean(m["medae"] for m in state_metrics)
            corr = mean(m["corr"] for m in state_metrics)
            pred = sum(m["mean_pred"] * m["n"] for m in state_metrics) / n
            actual = sum(m["mean_actual"] * m["n"] for m in state_metrics) / n
            print(
                f"S{state}={n}/{mae:.2f}/{medae:.2f}/{corr:.3f}/{pred:.2f}/{actual:.2f}",
                end=" "
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
