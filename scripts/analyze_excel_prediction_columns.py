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


def analyze(rows, h_threshold: float, commission_rate: float):
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

        # Exact supplied Excel columns:
        # X=--(R>Q), Y=--(S>Q), Z=--(R>T)
        # J=IF(SUM(X:Z)=3,1,0)
        # L=IF(SUM(X:Z)=0,-1,0)
        x = int(r > q)
        y = int(s > q)
        z = int(r > t)
        j = int(x + y + z == 3)
        l = -1 if x + y + z == 0 else 0

        actual_move = next_c - c
        actual_dir = 1 if actual_move > 0 else (-1 if actual_move < 0 else 0)
        move_pct = actual_move / c * 100.0 if c else 0.0
        observations.append({"pt": pt, "J": j, "L": l, "actual_dir": actual_dir, "move_pct": move_pct})

    total = len(observations)

    def metric(name: str, selector, expected_dir: int):
        selected = [r for r in observations if selector(r)]
        correct = [r for r in selected if r["actual_dir"] == expected_dir]
        directional_move = [r["move_pct"] * expected_dir for r in selected]
        net = [x - commission_rate * 100.0 for x in directional_move]
        return {
            "name": name,
            "signals": len(selected),
            "coverage_pct": len(selected) / total * 100.0 if total else 0.0,
            "correct": len(correct),
            "accuracy_pct": len(correct) / len(selected) * 100.0 if selected else 0.0,
            "avg_move_pct": sum(r["move_pct"] for r in selected) / len(selected) if selected else 0.0,
            "avg_directional_move_pct": sum(directional_move) / len(directional_move) if directional_move else 0.0,
            "avg_net_pct": sum(net) / len(net) if net else 0.0,
            "net_positive": sum(x > 0 for x in net),
            "net_negative": sum(x < 0 for x in net),
        }

    return total, [
        metric("H LONG", lambda r: r["pt"] == 1, 1),
        metric("H SHORT", lambda r: r["pt"] == -1, -1),
        metric("J LONG", lambda r: r["J"] == 1, 1),
        metric("L SHORT", lambda r: r["L"] == -1, -1),
    ]


def main():
    p = argparse.ArgumentParser(description="Independent accuracy test for Excel columns H, J and L")
    p.add_argument("--input", required=True)
    p.add_argument("--h", type=float, default=500.0)
    p.add_argument("--commission", type=float, default=0.0026,
                   help="Round-trip commission as decimal; default 0.0026 = 0.26%%")
    args = p.parse_args()

    if args.h < 0 or args.commission < 0:
        raise ValueError("--h and --commission must be >= 0")

    total, results = analyze(load_rows(Path(args.input)), args.h, args.commission)
    print("=" * 105)
    print("INDEPENDENT PREDICTION ACCURACY: H / J / L")
    print("=" * 105)
    print(f"frames={total}")
    print(f"H_THRESHOLD={args.h:g}")
    print(f"COMMISSION_ROUND_TRIP={args.commission * 100:.4f}%")
    print()
    print("prediction  signals  coverage%  correct  accuracy%  avg_move%  avg_directional%  avg_net%  net+  net-")
    print("-" * 105)
    for r in results:
        print(
            f"{r['name']:<10}  {r['signals']:>7}  {r['coverage_pct']:>9.3f}  "
            f"{r['correct']:>7}  {r['accuracy_pct']:>9.3f}  {r['avg_move_pct']:>9.5f}  "
            f"{r['avg_directional_move_pct']:>17.5f}  {r['avg_net_pct']:>9.5f}  "
            f"{r['net_positive']:>4}  {r['net_negative']:>4}"
        )


if __name__ == "__main__":
    main()
