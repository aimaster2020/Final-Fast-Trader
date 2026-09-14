from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


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


def load_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [{k: raw.get(k, "") for k in (reader.fieldnames or [])} for raw in reader]


def evaluate(rows, g_threshold: float, h_threshold: float, magnitude_threshold: float, am_mode: str):
    out = []
    for i in range(5, len(rows) - 1):
        current, nxt = rows[i], rows[i + 1]
        o, h, l, c, next_c = (num(current.get(k)) for k in ("Open", "High", "Low", "Close")) + (num(nxt.get("Close")),)
        if None in (o, h, l, c, next_c):
            continue

        window = rows[i - 5:i + 1]
        closes = [num(x.get("Close")) for x in window]
        highs = [num(x.get("High")) for x in window]
        body = c - o
        predict = sum(closes) / 6 if body > 0 else (sum(highs) / 6 if body < 0 else c)
        ct, pt = sign(next_c - c, g_threshold), sign(predict - c, h_threshold)
        positive_match = int(pt == ct and pt == 1)
        negative_match = int(pt == ct and pt == -1)

        q, r, s, t = c - o, h - c, l - c, h - o
        x, y, z = int(r > q), int(s > q), int(r > t)
        ab, ac, ad = int(r < q), int(s < q), int(r < t)
        if am_mode == "literal":
            bullish = int(x + y + z == 2 and (c - o) > 50)
            bearish = int((ab + ac + ad == 2 and (l - o)) < 50)
        else:
            bullish = int(x + y + z == 2 and (h - o) > 50)
            bearish = int(ab + ac + ad == 2 and (l - o) < 50)

        an = 1 if bullish else (-1 if bearish else 0)
        actual_direction = 1 if next_c > c else (-1 if next_c < c else 0)
        direction_correct = int(an != 0 and actual_direction != 0 and an == actual_direction)
        predicted_close = h - o + c if an == 1 else (l - o + c if an == -1 else o)
        predicted_move = predicted_close - c
        actual_move = next_c - c
        predicted_move_pct = abs(predicted_move) / abs(c) * 100 if c else 0.0
        actual_move_pct = abs(actual_move) / abs(c) * 100 if c else 0.0
        magnitude_correct = int((predicted_move_pct >= magnitude_threshold) == (actual_move_pct >= magnitude_threshold))

        out.append({"Timestamp": current.get("Timestamp", ""), "Open": o, "High": h, "Low": l, "Close": c,
                    "predict": predict, "ct": ct, "pt": pt, "positive_match": positive_match,
                    "negative_match": negative_match, "AM": an, "AN": an, "actual_direction": actual_direction,
                    "direction_correct": direction_correct, "predicted_close": predicted_close,
                    "predicted_move": predicted_move, "predicted_move_pct": predicted_move_pct,
                    "actual_move": actual_move, "actual_move_pct": actual_move_pct,
                    "magnitude_correct": magnitude_correct})

    signals = [r for r in out if r["AN"] != 0]
    correct = sum(int(r["direction_correct"]) for r in signals)
    stats = {"frames": len(out), "signals": len(signals),
             "signal_pct": len(signals) / len(out) * 100 if out else 0.0,
             "direction_correct": correct,
             "direction_accuracy_pct": correct / len(signals) * 100 if signals else 0.0,
             "positive_matches": sum(int(r["positive_match"]) for r in out),
             "negative_matches": sum(int(r["negative_match"]) for r in out),
             "magnitude_correct": sum(int(r["magnitude_correct"]) for r in out)}
    return out, stats


def write_csv(path: Path, rows: Iterable[dict]):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--g", type=float, default=0.0)
    p.add_argument("--h", type=float, default=0.0)
    p.add_argument("--magnitude-threshold", type=float, default=0.0)
    p.add_argument("--am-mode", choices=("literal", "corrected"), default="literal")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result, stats = evaluate(load_rows(Path(args.input)), args.g, args.h, args.magnitude_threshold, args.am_mode)
    write_csv(Path(args.output), result)
    print(f"G={args.g:g} H={args.h:g} MAGNITUDE_THRESHOLD={args.magnitude_threshold:g} AM_MODE={args.am_mode}")
    for key, value in stats.items():
        print(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}")


if __name__ == "__main__":
    main()
