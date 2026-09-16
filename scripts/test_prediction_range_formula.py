#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test MAX(previous closes) prediction over future candle horizons.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--horizons", default="1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--min-prediction-pct", type=float, default=0.0)
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


def predicted_price(rows, i: int, window: int) -> float:
    # Exact requested formula:
    # =MAX(E7:E11)
    # E7:E11 = previous 5 closes.
    return max(rows[j][4] for j in range(i - window, i))


def main() -> None:
    a = parse_args()
    horizons = sorted(set(int(x.strip()) for x in a.horizons.split(",") if x.strip()))
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("--horizons must contain positive integers")
    if a.window < 1:
        raise ValueError("--window must be >= 1")

    rows = load_rows(Path(a.input), a.year)
    max_h = max(horizons)
    records = []

    for i in range(a.window, len(rows) - max_h):
        ts, _, _, _, current_close = rows[i]
        pred = predicted_price(rows, i, a.window)
        pred_move = pred - current_close
        pred_move_pct = abs(pred_move) / current_close * 100.0 if current_close else 0.0
        if pred_move_pct < a.min_prediction_pct:
            continue

        row = {
            "timestamp": ts,
            "current_close": current_close,
            "predicted_price": pred,
            "predicted_move": pred_move,
            "predicted_move_pct": pred_move_pct,
        }

        for h in horizons:
            future_close = rows[i + h][4]
            actual_move = future_close - current_close
            direction_correct = int(actual_move >= 0)
            target_reached = int(future_close >= pred)
            prediction_error_pct = (future_close - pred) / current_close * 100.0 if current_close else 0.0
            row[f"future_close_h{h}"] = future_close
            row[f"actual_move_pct_h{h}"] = actual_move / current_close * 100.0 if current_close else 0.0
            row[f"direction_correct_h{h}"] = direction_correct
            row[f"target_reached_h{h}"] = target_reached
            row[f"prediction_error_pct_h{h}"] = prediction_error_pct

        records.append(row)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = ["timestamp", "current_close", "predicted_price", "predicted_move", "predicted_move_pct"]
    for h in horizons:
        fields.extend([
            f"future_close_h{h}",
            f"actual_move_pct_h{h}",
            f"direction_correct_h{h}",
            f"target_reached_h{h}",
            f"prediction_error_pct_h{h}",
        ])

    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    print(f"input={a.input}")
    print(f"output={out}")
    print(f"rows={len(rows)} signals={len(records)}")
    print(f"window={a.window}")
    print("formula=MAX(previous_closes)")
    print(f"min_prediction_pct={a.min_prediction_pct:g}%")
    print(f"horizons={','.join(map(str, horizons))}")
    print("commission=0")
    print()
    print("RESULT")

    total = len(records)
    for h in horizons:
        if not total:
            print(f"horizon={h} correct=0/0 accuracy=0.0000% target_reached=0/0 target_hit=0.0000% mae_pct=0.0000% mean_error_pct=0.0000%")
            continue

        correct = sum(int(r[f"direction_correct_h{h}"]) for r in records)
        reached = sum(int(r[f"target_reached_h{h}"]) for r in records)
        errors = [abs(float(r[f"prediction_error_pct_h{h}"])) for r in records]
        signed_errors = [float(r[f"prediction_error_pct_h{h}"]) for r in records]
        mae = sum(errors) / total
        mean_error = sum(signed_errors) / total
        print(
            f"horizon={h} correct={correct}/{total} accuracy={correct/total*100:.4f}% "
            f"target_reached={reached}/{total} target_hit={reached/total*100:.4f}% "
            f"mae_pct={mae:.4f}% mean_error_pct={mean_error:.4f}%"
        )


if __name__ == "__main__":
    main()
