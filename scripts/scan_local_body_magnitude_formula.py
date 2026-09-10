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
                rows.append((float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def ad(o: float, h: float, l: float, c: float) -> int:
    body = c - o
    hc = h - c
    ho = h - o
    lc = l - c
    return int(hc > body) + int(ho < hc) + int(lc > body)


def safe_div(a: float, b: float) -> float | None:
    if abs(b) < 1e-12:
        return None
    return a / b


def candidates(o: float, h: float, l: float, c: float) -> dict[str, float]:
    body = abs(c - o)
    rng = h - l
    upper = h - max(o, c)
    lower = min(o, c) - l
    hc = h - c
    lc_abs = abs(l - c)
    lo_abs = abs(l - o)
    ho_abs = abs(h - o)

    out: dict[str, float] = {}
    if body > 1e-12:
        out["body"] = body
        out["upper/body"] = upper / body
        out["lower/body"] = lower / body
        out["range/body"] = rng / body
    if rng > 1e-12:
        out["body/range"] = body / rng
        out["upper/range"] = upper / rng
        out["lower/range"] = lower / rng
    if abs(hc) > 1e-12:
        out["abs(LC)/HC"] = lc_abs / abs(hc)
        out["abs(LO)/HC"] = lo_abs / abs(hc)
        out["HO/HC"] = abs(h - o) / abs(hc)
    if abs(lc_abs) > 1e-12:
        out["abs(LO)/abs(LC)"] = lo_abs / lc_abs
    if abs(ho_abs) > 1e-12:
        out["abs(LO)/HO"] = lo_abs / ho_abs
    return out


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def run(rows: list[tuple[float, float, float, float]]) -> None:
    by_key: dict[tuple[int, int], tuple[list[float], list[float]]] = {}
    # sign: +1 current bullish, -1 current bearish; score is the Excel AD.
    for i in range(len(rows) - 1):
        o, h, l, c = rows[i]
        no, nh, nl, nc = rows[i + 1]
        body_signed = c - o
        body_size = abs(body_signed)
        next_size = abs(nc - no)
        if body_size <= 1e-12:
            continue
        ratio = next_size / body_size
        score = ad(o, h, l, c)
        sign = 1 if body_signed > 0 else -1 if body_signed < 0 else 0
        if sign == 0:
            continue
        key = (score, sign)
        if key not in by_key:
            by_key[key] = ([], [])
        xs, ys = by_key[key]
        # Use each raw local feature as predictor of next/current body-size ratio.
        # Store all feature values later by rebuilding from the rows for simplicity.
        xs.append(ratio)
        ys.append(0.0)

    print("TARGET = abs(next_body) / abs(current_body)")
    print("LOCAL FEATURES = current candle only; no fixed asset coefficient")
    for score in (3, 0):
        for sign, sign_name in ((1, "UP"), (-1, "DOWN")):
            ratios: list[float] = []
            features: dict[str, list[float]] = {}
            for i in range(len(rows) - 1):
                o, h, l, c = rows[i]
                no, nh, nl, nc = rows[i + 1]
                body_signed = c - o
                body_size = abs(body_signed)
                if body_size <= 1e-12:
                    continue
                if ad(o, h, l, c) != score:
                    continue
                actual_sign = 1 if body_signed > 0 else -1 if body_signed < 0 else 0
                if actual_sign != sign:
                    continue
                ratio = abs(nc - no) / body_size
                ratios.append(ratio)
                for name, value in candidates(o, h, l, c).items():
                    features.setdefault(name, []).append(value)

            print(f"AD={score} BODY={sign_name} N={len(ratios)} median_ratio={sorted(ratios)[len(ratios)//2]:.4f}" if ratios else f"AD={score} BODY={sign_name} N=0")
            scored: list[tuple[float, str]] = []
            for name, vals in features.items():
                if len(vals) == len(ratios):
                    scored.append((abs(corr(vals, ratios)), name))
            scored.sort(reverse=True)
            for cval, name in scored[:8]:
                print(f"  corr={cval:+.4f} feature={name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Find current-candle geometric variables related to next/current body-size ratio.")
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    args = ap.parse_args()
    root = Path(args.input_dir)
    for symbol in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        path = root / f"{symbol}_1h.csv"
        rows = load(path)
        print(f"\n{symbol} rows={len(rows)}")
        run(rows)


if __name__ == "__main__":
    main()
