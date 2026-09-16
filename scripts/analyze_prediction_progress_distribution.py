#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze the distribution of directional progress toward the formula prediction.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--f3", type=float, default=800.0)
    p.add_argument("--f4", type=float, default=-150.0)
    p.add_argument("--window", type=int, choices=range(1, 6), default=5)
    p.add_argument("--min-prediction-pct", type=float, default=1.0)
    p.add_argument("--horizons", default="1,2,3,5")
    p.add_argument("--year", type=int, default=None)
    return p.parse_args()


def year_of(v: str) -> Optional[int]:
    try:
        import datetime as dt
        ts = int(float(v))
        if ts > 10_000_000_000:
            ts //= 1000
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).year
    except Exception:
        return None


def load_rows(path: Path, year: Optional[int]):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        raw = list(csv.reader(f))
    if not raw:
        raise ValueError("Input CSV is empty")
    header = [x.strip().lower() for x in raw[0]]
    headered = all(x in header for x in ("timestamp", "open", "high", "low", "close"))
    out = []
    if headered:
        idx = {name: i for i, name in enumerate(header)}
        need = max(idx[k] for k in ("timestamp", "open", "high", "low", "close"))
        for r in raw[1:]:
            if len(r) <= need:
                continue
            ts = r[idx["timestamp"]]
            if year is not None and year_of(ts) != year:
                continue
            out.append((ts, float(r[idx["open"]]), float(r[idx["high"]]), float(r[idx["low"]]), float(r[idx["close"]])))
    else:
        for r in raw:
            if len(r) < 5:
                continue
            ts = r[0]
            if year is not None and year_of(ts) != year:
                continue
            out.append((ts, float(r[1]), float(r[2]), float(r[3]), float(r[4])))
    if not out:
        raise ValueError("No usable OHLC rows found")
    return out


def predicted_price(rows, i: int, window: int, f3: float, f4: float) -> float:
    _, o, _, _, c = rows[i]
    body = c - o
    if body > f3:
        return sum(rows[j][4] for j in range(i - window, i)) / window
    if body < f4:
        return sum(rows[j][2] for j in range(i - window, i)) / window
    return c


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def run(rows, f3: float, f4: float, window: int, min_prediction_pct: float, horizons: list[int]):
    signals = []
    max_h = max(horizons)
    for i in range(window, len(rows) - max_h):
        ts, o, h, l, c = rows[i]
        pred = predicted_price(rows, i, window, f3, f4)
        move = pred - c
        move_pct = abs(move) / c * 100.0 if c else 0.0
        if move == 0 or move_pct < min_prediction_pct:
            continue
        side = 1 if move > 0 else -1
        total_distance = abs(move)
        row = {
            "timestamp": ts,
            "entry_close": c,
            "predicted_price": pred,
            "predicted_move_pct": move_pct,
            "side": "LONG" if side == 1 else "SHORT",
        }
        for h in horizons:
            future = rows[i + 1:i + h + 1]
            if side == 1:
                best_fav = max(r[2] - c for r in future)
                worst_adv = min(r[3] - c for r in future)
            else:
                best_fav = max(c - r[3] for r in future)
                worst_adv = max(r[2] - c for r in future)
            progress = best_fav / total_distance * 100.0 if total_distance else 0.0
            row[f"progress_h{h}"] = progress
            row[f"adverse_h{h}"] = worst_adv / c * 100.0
            row[f"reach25_h{h}"] = int(progress >= 25.0)
            row[f"reach50_h{h}"] = int(progress >= 50.0)
            row[f"reach75_h{h}"] = int(progress >= 75.0)
            row[f"reach100_h{h}"] = int(progress >= 100.0)
        signals.append(row)
    return signals


def main() -> None:
    a = parse_args()
    horizons = sorted(set(int(x.strip()) for x in a.horizons.split(",") if x.strip()))
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("--horizons must contain positive integers")
    rows = load_rows(Path(a.input), a.year)
    signals = run(rows, a.f3, a.f4, a.window, a.min_prediction_pct, horizons)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp", "entry_close", "predicted_price", "predicted_move_pct", "side"]
    for h in horizons:
        fields.extend([f"progress_h{h}", f"adverse_h{h}", f"reach25_h{h}", f"reach50_h{h}", f"reach75_h{h}", f"reach100_h{h}"])
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(signals)

    print(f"input={a.input}")
    print(f"output={out}")
    print(f"rows={len(rows)} signals={len(signals)}")
    print(f"F3={a.f3:g} F4={a.f4:g} WINDOW={a.window}")
    print(f"min_prediction_pct={a.min_prediction_pct:g}%")
    print(f"horizons={','.join(map(str, horizons))}")
    print()
    print("RESULT")
    for h in horizons:
        progress = [float(r[f"progress_h{h}"]) for r in signals]
        adverse = [float(r[f"adverse_h{h}"]) for r in signals]
        print(
            f"horizon={h} "
            f"progress_p25={percentile(progress,0.25):.2f}% "
            f"progress_p50={percentile(progress,0.50):.2f}% "
            f"progress_p75={percentile(progress,0.75):.2f}% "
            f"progress_p90={percentile(progress,0.90):.2f}% "
            f"reach25={sum(int(r[f'reach25_h{h}']) for r in signals)/len(signals)*100:.2f}% "
            f"reach50={sum(int(r[f'reach50_h{h}']) for r in signals)/len(signals)*100:.2f}% "
            f"reach75={sum(int(r[f'reach75_h{h}']) for r in signals)/len(signals)*100:.2f}% "
            f"reach100={sum(int(r[f'reach100_h{h}']) for r in signals)/len(signals)*100:.2f}% "
            f"adverse_p50={percentile(adverse,0.50):.2f}% "
            f"adverse_p75={percentile(adverse,0.75):.2f}%"
        )


if __name__ == "__main__":
    main()
