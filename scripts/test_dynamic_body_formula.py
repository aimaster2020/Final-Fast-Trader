from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def load(path: Path) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((
                    float(r["open"]),
                    float(r["high"]),
                    float(r["low"]),
                    float(r["close"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def safe_ratio(a: float, b: float) -> float | None:
    if abs(b) <= 1e-12:
        return None
    return a / b


def ad_score(o: float, h: float, l: float, c: float) -> int:
    body = c - o
    hc = h - c
    ho = h - o
    lc = l - c
    return int(hc > body) + int(ho < hc) + int(lc > body)


def features(row: tuple[float, float, float, float]) -> dict[str, float]:
    o, h, l, c = row
    body = c - o
    size = abs(body)
    upper = h - max(o, c)
    lower = min(o, c) - l
    rng = h - l
    return {
        "body": body,
        "size": size,
        "upper": upper,
        "lower": lower,
        "range": rng,
        "wicks": upper + lower,
        "body_to_range": safe_ratio(size, rng) or 0.0,
        "range_to_body": safe_ratio(rng, size) or 0.0,
    }


def predict(name: str, cur: dict[str, float], prev: dict[str, float]) -> float | None:
    cs = cur["size"]
    ps = prev["size"]
    cr = cur["range"]
    pr = prev["range"]
    cw = cur["wicks"]
    pw = prev["wicks"]

    if name == "PERSIST_BODY":
        return cs
    if name == "RANGE_RATIO":
        r = safe_ratio(cr, pr)
        return cs * r if r is not None and math.isfinite(r) else None
    if name == "BODY_RATIO":
        r = safe_ratio(cs, ps)
        return cs * r if r is not None and math.isfinite(r) else None
    if name == "COMPRESSION_RATIO":
        a = safe_ratio(cr, cs)
        b = safe_ratio(pr, ps)
        if a is None or b is None or abs(b) <= 1e-12:
            return None
        r = a / b
        return cs * r if math.isfinite(r) else None
    if name == "WICK_RATIO":
        r = safe_ratio(cw, pw)
        return cs * r if r is not None and math.isfinite(r) else None
    if name == "LOWER_RATIO":
        r = safe_ratio(cur["lower"], prev["lower"])
        return cs * r if r is not None and math.isfinite(r) else None
    if name == "UPPER_RATIO":
        r = safe_ratio(cur["upper"], prev["upper"])
        return cs * r if r is not None and math.isfinite(r) else None
    return None


def evaluate(rows: list[tuple[float, float, float, float]], formula: str, score: int) -> tuple[int, float, float, float]:
    errors: list[float] = []
    pct_errors: list[float] = []
    grow_correct = 0
    grow_total = 0

    for i in range(1, len(rows) - 1):
        prev = features(rows[i - 1])
        cur = features(rows[i])
        nxt = features(rows[i + 1])

        if ad_score(*rows[i]) != score:
            continue

        pred = predict(formula, cur, prev)
        if pred is None or not math.isfinite(pred):
            continue

        actual = nxt["size"]
        errors.append(abs(pred - actual))
        if actual > 1e-12:
            pct_errors.append(abs(pred - actual) / actual)

        actual_grow = actual > cur["size"] + 1e-12
        pred_grow = pred > cur["size"]
        grow_correct += int(actual_grow == pred_grow)
        grow_total += 1

    if not errors:
        return 0, 0.0, 0.0, 0.0
    return len(errors), sum(errors) / len(errors), sorted(pct_errors)[len(pct_errors) // 2], grow_correct / grow_total * 100 if grow_total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Test dynamic, no-fixed-coefficient body-size formulas using current and previous candle only.")
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    args = ap.parse_args()

    formulas = [
        "PERSIST_BODY",
        "RANGE_RATIO",
        "BODY_RATIO",
        "COMPRESSION_RATIO",
        "WICK_RATIO",
        "LOWER_RATIO",
        "UPPER_RATIO",
    ]

    root = Path(args.input_dir)
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    for symbol in symbols:
        path = root / f"{symbol}_1h.csv"
        rows = load(path)
        print(f"{symbol} rows={len(rows)}")
        print("FORMULA                 AD   N      MAE         MED_REL_ERR   GROW_ACC")
        for formula in formulas:
            for score in (3, 0):
                n, mae, med_rel, acc = evaluate(rows, formula, score)
                print(f"{formula:<23} {score} {n:5d} {mae:11.6f} {med_rel*100:13.2f}% {acc:9.2f}%")


if __name__ == "__main__":
    main()
