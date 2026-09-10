from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median

DEFAULT_LOOKBACK = 50
MIN_HISTORY = 20


def load(path: Path) -> list[tuple[float, float, float, float]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def ad_score(row):
    o, h, l, c = row
    body = c - o
    hc = h - c
    ho = h - o
    lc = l - c
    return int(hc > body) + int(hc > ho) + int(lc > body)


def features(row):
    o, h, l, c = row
    size = abs(c - o)
    rng = h - l
    if rng <= 1e-12:
        return (1.0, 0.0, 0.0)
    upper = h - max(o, c)
    lower = min(o, c) - l
    return (1.0, upper / rng, lower / rng)


def target(cur, nxt):
    rng = cur[1] - cur[2]
    if rng <= 1e-12:
        return None
    return abs(nxt[3] - nxt[0]) / rng


def solve(samples):
    if len(samples) < MIN_HISTORY:
        return None
    # 3 independent parameters: intercept, upper/range, lower/range.
    n = len(samples)
    s = [0.0] * 6
    t = [0.0] * 3
    for x, y in samples:
        for i in range(3):
            t[i] += x[i] * y
            for j in range(i, 3):
                s[i + j * (j + 1) // 2] += x[i] * x[j]
    # Explicit symmetric matrix.
    a = [
        [sum(x[0] * x[0] for x, _ in samples), sum(x[0] * x[1] for x, _ in samples), sum(x[0] * x[2] for x, _ in samples), t[0]],
        [sum(x[0] * x[1] for x, _ in samples), sum(x[1] * x[1] for x, _ in samples), sum(x[1] * x[2] for x, _ in samples), t[1]],
        [sum(x[0] * x[2] for x, _ in samples), sum(x[1] * x[2] for x, _ in samples), sum(x[2] * x[2] for x, _ in samples), t[2]],
    ]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-10:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        div = a[col][col]
        for j in range(col, 4):
            a[col][j] /= div
        for r in range(3):
            if r == col:
                continue
            factor = a[r][col]
            for j in range(col, 4):
                a[r][j] -= factor * a[col][j]
    coef = tuple(a[i][3] for i in range(3))
    return coef if all(math.isfinite(v) for v in coef) else None


def evaluate(rows, score, lookback):
    errors = []
    rel_errors = []
    correct = 0
    total = 0
    for i in range(1, len(rows) - 1):
        if ad_score(rows[i]) != score:
            continue
        hist = []
        for j in range(max(0, i - lookback), i):
            y = target(rows[j], rows[j + 1])
            if y is not None:
                hist.append((features(rows[j]), y))
        coef = solve(hist)
        if coef is None:
            continue
        ratio = sum(features(rows[i])[k] * coef[k] for k in range(3))
        if not math.isfinite(ratio):
            continue
        pred = max(0.0, ratio) * max(rows[i][1] - rows[i][2], 0.0)
        actual = abs(rows[i + 1][3] - rows[i + 1][0])
        errors.append(abs(pred - actual))
        if actual > 1e-12:
            rel_errors.append(abs(pred - actual) / actual)
        correct += int((actual > abs(rows[i][3] - rows[i][0])) == (pred > abs(rows[i][3] - rows[i][0])))
        total += 1
    if not errors:
        return 0, 0.0, 0.0, 0.0
    return len(errors), sum(errors) / len(errors), median(rel_errors) if rel_errors else 0.0, correct / total * 100 if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    args = ap.parse_args()
    root = Path(args.input_dir)
    print(f"LOOKBACK={max(5, args.lookback)} MIN_HISTORY={MIN_HISTORY}")
    print("MODEL=rolling OLS; features=intercept, upper/range, lower/range; target=next_body_size/current_range")
    for symbol in [s.strip() for s in args.symbols.split(',') if s.strip()]:
        rows = load(root / f"{symbol}_1h.csv")
        print(f"{symbol} rows={len(rows)}")
        for score in (3, 0):
            n, mae, med, acc = evaluate(rows, score, max(5, args.lookback))
            print(f"AD={score} TESTS={n} MAE={mae:.6f} MED_REL_ERR={med*100:.2f}% GROW_SHRINK_ACC={acc:.2f}%")


if __name__ == "__main__":
    main()
