from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median

DEFAULT_LOOKBACK = 50
MIN_HISTORY = 20
EPS = 1e-12


def load(path: Path) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def ad_score(row: tuple[float, float, float, float]) -> int:
    o, h, l, c = row
    body = c - o
    hc = h - c
    ho = h - o
    lc = l - c
    return int(hc > body) + int(ho < hc) + int(lc > body)


def features(row: tuple[float, float, float, float]) -> tuple[float, float, float]:
    o, h, l, c = row
    rng = h - l
    if rng <= EPS:
        return (1.0, 0.0, 0.0)
    size = abs(c - o)
    upper = h - max(o, c)
    # lower/range is redundant because body/range + upper/range + lower/range = 1.
    return (1.0, size / rng, upper / rng)


def target(cur: tuple[float, float, float, float], nxt: tuple[float, float, float, float]) -> float | None:
    rng = cur[1] - cur[2]
    if rng <= EPS:
        return None
    return abs(nxt[3] - nxt[0]) / rng


def solve_ols(samples: list[tuple[tuple[float, float, float], float]]) -> tuple[float, float, float] | None:
    if len(samples) < MIN_HISTORY:
        return None
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for x, y in samples:
        for a in range(3):
            xty[a] += x[a] * y
            for b in range(3):
                xtx[a][b] += x[a] * x[b]

    a = [xtx[i] + [xty[i]] for i in range(3)]
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


def predict_size(cur: tuple[float, float, float, float], coef: tuple[float, float, float]) -> float:
    rng = cur[1] - cur[2]
    x = features(cur)
    ratio = sum(x[i] * coef[i] for i in range(3))
    if not math.isfinite(ratio):
        ratio = abs(cur[3] - cur[0]) / rng if rng > EPS else 0.0
    return max(0.0, ratio) * max(rng, 0.0)


def evaluate(rows: list[tuple[float, float, float, float]], score: int, lookback: int) -> tuple[int, float, float, float]:
    errors: list[float] = []
    rel_errors: list[float] = []
    correct = 0
    total = 0

    for i in range(1, len(rows) - 1):
        cur = rows[i]
        if ad_score(cur) != score:
            continue

        samples: list[tuple[tuple[float, float, float], float]] = []
        start = max(0, i - lookback)
        for j in range(start, i):
            y = target(rows[j], rows[j + 1])
            if y is not None:
                samples.append((features(rows[j]), y))

        coef = solve_ols(samples)
        if coef is None:
            continue

        pred = predict_size(cur, coef)
        actual = abs(rows[i + 1][3] - rows[i + 1][0])
        errors.append(abs(pred - actual))
        if actual > EPS:
            rel_errors.append(abs(pred - actual) / actual)

        current_size = abs(cur[3] - cur[0])
        correct += int((actual > current_size) == (pred > current_size))
        total += 1

    if not errors:
        return 0, 0.0, 0.0, 0.0
    med_rel = median(rel_errors) if rel_errors else 0.0
    return len(errors), sum(errors) / len(errors), med_rel, correct / total * 100 if total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Rolling local body-size model without collinear features; coefficients use past candles only.")
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    args = ap.parse_args()

    lookback = max(5, args.lookback)
    root = Path(args.input_dir)
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]

    print(f"LOOKBACK={lookback} MIN_HISTORY={MIN_HISTORY}")
    print("MODEL=rolling OLS; target=next_body_size/current_range; features=current body/range + upper_wick/range; history=prior candles only")
    for symbol in symbols:
        rows = load(root / f"{symbol}_1h.csv")
        print(f"{symbol} rows={len(rows)}")
        for score in (3, 0):
            n, mae, med_rel, acc = evaluate(rows, score, lookback)
            print(f"AD={score} TESTS={n} MAE={mae:.6f} MED_REL_ERR={med_rel*100:.2f}% GROW_SHRINK_ACC={acc:.2f}%")


if __name__ == "__main__":
    main()
