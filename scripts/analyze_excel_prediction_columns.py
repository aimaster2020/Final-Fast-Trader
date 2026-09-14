from __future__ import annotations

import argparse
import csv
from pathlib import Path

RAW_FIELDS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]


def num(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sign(value: float, threshold: float) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def value_range(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if end < start:
        raise ValueError("end must be >= start")
    values: list[float] = []
    x = start
    epsilon = step * 1e-9
    while x <= end + epsilon:
        values.append(round(x, 10))
        x += step
    return values


def looks_like_header(row: list[str]) -> bool:
    normalized = [str(x).strip().lower() for x in row]
    return {"timestamp", "open", "high", "low", "close"}.issubset(normalized)


def load_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return []
        if looks_like_header(first):
            fields = [str(x).strip() for x in first]
            return [
                {k: raw[idx] if idx < len(raw) else "" for idx, k in enumerate(fields)}
                for raw in reader
            ]
        if len(first) < 5:
            raise ValueError("Expected headerless order: Timestamp,Open,High,Low,Close[,Volume].")
        rows = [first] + list(reader)
        return [
            {k: raw[idx] if idx < len(raw) else "" for idx, k in enumerate(RAW_FIELDS)}
            for raw in rows
        ]


def build_observations(rows, h_threshold: float):
    observations = []
    for i in range(5, len(rows) - 1):
        current, nxt = rows[i], rows[i + 1]
        o, high, low, c = (num(current.get(k)) for k in ("Open", "High", "Low", "Close"))
        next_c = num(nxt.get("Close"))
        if None in (o, high, low, c, next_c):
            continue

        window = rows[i - 5:i + 1]
        closes = [num(x.get("Close")) for x in window]
        highs = [num(x.get("High")) for x in window]
        if any(v is None for v in closes + highs):
            continue

        body = c - o
        predict = sum(closes) / 6.0 if body > 0 else (sum(highs) / 6.0 if body < 0 else c)
        pt = sign(predict - c, h_threshold)

        q = c - o
        r = high - c
        s = low - c
        t = high - o
        x = int(r > q)
        y = int(s > q)
        z = int(r > t)
        j = int(x + y + z == 3)
        l = -1 if x + y + z == 0 else 0

        actual_move = next_c - c
        actual_dir = 1 if actual_move > 0 else (-1 if actual_move < 0 else 0)
        move_pct = actual_move / c * 100.0 if c else 0.0
        observations.append({
            "pt": pt,
            "J": j,
            "L": l,
            "actual_dir": actual_dir,
            "move_pct": move_pct,
        })
    return observations


def metric(selected, expected_dir: int, commission_rate: float):
    correct = sum(r["actual_dir"] == expected_dir for r in selected)
    directional_moves = [r["move_pct"] * expected_dir for r in selected]
    net = [x - commission_rate * 100.0 for x in directional_moves]
    return {
        "signals": len(selected),
        "correct": correct,
        "accuracy_pct": correct / len(selected) * 100.0 if selected else 0.0,
        "avg_directional_move_pct": sum(directional_moves) / len(directional_moves) if directional_moves else 0.0,
        "avg_net_pct": sum(net) / len(net) if net else 0.0,
    }


def analyze(rows, h_threshold: float, commission_rate: float):
    observations = build_observations(rows, h_threshold)
    total = len(observations)
    return total, [
        ("H LONG", metric([r for r in observations if r["pt"] == 1], 1, commission_rate)),
        ("H SHORT", metric([r for r in observations if r["pt"] == -1], -1, commission_rate)),
        ("J LONG", metric([r for r in observations if r["J"] == 1], 1, commission_rate)),
        ("L SHORT", metric([r for r in observations if r["L"] == -1], -1, commission_rate)),
    ]


def sweep_pair(observations, pair: str, weight_start: float, weight_end: float, weight_step: float,
               threshold_start: float, threshold_end: float, threshold_step: float,
               commission_rate: float, top: int):
    weights = value_range(weight_start, weight_end, weight_step)
    thresholds = value_range(threshold_start, threshold_end, threshold_step)
    results = []

    expected_dir = 1 if pair == "LONG H+J" else -1
    other_key = "J" if pair == "LONG H+J" else "L"

    for other_weight in weights:
        for threshold in thresholds:
            selected = []
            for r in observations:
                h_vote = 1 if r["pt"] == expected_dir else 0
                other_vote = 1 if (r[other_key] == (1 if expected_dir == 1 else -1)) else 0
                score = h_vote + other_weight * other_vote
                if score >= threshold:
                    selected.append(r)

            m = metric(selected, expected_dir, commission_rate)
            results.append({
                "other_weight": other_weight,
                "threshold": threshold,
                **m,
            })

    results.sort(key=lambda r: (
        r["accuracy_pct"],
        r["avg_directional_move_pct"],
        r["signals"],
    ), reverse=True)
    return results[:top]


def main():
    p = argparse.ArgumentParser(description="Independent H/J/L accuracy and separate weight sweeps")
    p.add_argument("--input", required=True)
    p.add_argument("--h", type=float, default=500.0)
    p.add_argument("--commission", type=float, default=0.0,
                   help="Round-trip commission as decimal; default 0")
    p.add_argument("--sweep-weights", action="store_true")
    p.add_argument("--weight-start", type=float, default=0.0)
    p.add_argument("--weight-end", type=float, default=2.0)
    p.add_argument("--weight-step", type=float, default=0.25)
    p.add_argument("--threshold-start", type=float, default=0.25)
    p.add_argument("--threshold-end", type=float, default=2.0)
    p.add_argument("--threshold-step", type=float, default=0.25)
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()

    if min(args.h, args.commission, args.weight_start, args.threshold_start) < 0:
        raise ValueError("numeric parameters must be >= 0")
    if args.top <= 0:
        raise ValueError("--top must be > 0")

    rows = load_rows(Path(args.input))
    observations = build_observations(rows, args.h)
    total = len(observations)

    if args.sweep_weights:
        print("=" * 115)
        print("SEPARATE WEIGHT SWEEP: LONG H+J / SHORT H+L")
        print("=" * 115)
        print(f"frames={total}")
        print(f"H_THRESHOLD={args.h:g}")
        print(f"COMMISSION_ROUND_TRIP={args.commission * 100:.4f}%")
        print(f"other_weight={args.weight_start:g}..{args.weight_end:g} step={args.weight_step:g}")
        print(f"threshold={args.threshold_start:g}..{args.threshold_end:g} step={args.threshold_step:g}")
        print()

        for pair in ("LONG H+J", "SHORT H+L"):
            results = sweep_pair(
                observations,
                pair,
                args.weight_start,
                args.weight_end,
                args.weight_step,
                args.threshold_start,
                args.threshold_end,
                args.threshold_step,
                args.commission,
                args.top,
            )
            print(pair)
            print("rank  other_w  threshold  signals  accuracy%  avg_move%  avg_net%")
            print("-" * 75)
            for rank, r in enumerate(results, 1):
                print(
                    f"{rank:>4}  {r['other_weight']:>8.2f}  {r['threshold']:>9.2f}  "
                    f"{r['signals']:>7}  {r['accuracy_pct']:>9.3f}  "
                    f"{r['avg_directional_move_pct']:>9.5f}  {r['avg_net_pct']:>9.5f}"
                )
            print()
        return

    total, results = analyze(rows, args.h, args.commission)
    print("=" * 105)
    print("INDEPENDENT PREDICTION ACCURACY: H / J / L")
    print("=" * 105)
    print(f"frames={total}")
    print(f"H_THRESHOLD={args.h:g}")
    print(f"COMMISSION_ROUND_TRIP={args.commission * 100:.4f}%")
    print()
    print("prediction  signals  correct  accuracy%  avg_directional%  avg_net%")
    print("-" * 80)
    for name, r in results:
        print(
            f"{name:<10}  {r['signals']:>7}  {r['correct']:>7}  {r['accuracy_pct']:>9.3f}  "
            f"{r['avg_directional_move_pct']:>17.5f}  {r['avg_net_pct']:>9.5f}"
        )


if __name__ == "__main__":
    main()
