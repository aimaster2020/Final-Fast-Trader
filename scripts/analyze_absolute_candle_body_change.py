from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    x = sorted(values)
    pos = (len(x) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(x) - 1)
    frac = pos - lo
    return x[lo] * (1.0 - frac) + x[hi] * frac


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


def summarize(values: list[float]) -> tuple[float, float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        mean(values),
        median(values),
        quantile(values, 0.25),
        quantile(values, 0.75),
        quantile(values, 0.90),
    )


def analyze(rows: list[tuple[float, float, float, float]], score: int) -> None:
    signed_delta: list[float] = []
    abs_delta: list[float] = []
    pct_size_change: list[float] = []
    pct_range_change: list[float] = []

    grew = shrank = unchanged = 0

    for i in range(len(rows) - 1):
        o, h, l, c = rows[i]
        no, nh, nl, nc = rows[i + 1]

        body = c - o
        hc = h - c
        ho = h - o
        lc = l - c
        ad = int(hc > body) + int(ho < hc) + int(lc > body)
        if ad != score:
            continue

        current_size = abs(body)
        next_size = abs(nc - no)
        delta = next_size - current_size
        signed_delta.append((nc - no) - body)
        abs_delta.append(delta)

        if delta > 0:
            grew += 1
        elif delta < 0:
            shrank += 1
        else:
            unchanged += 1

        if current_size > 1e-12:
            pct_size_change.append(delta / current_size)

        current_range = h - l
        if current_range > 1e-12:
            pct_range_change.append(delta / current_range)

    mean_d, med_d, q25_d, q75_d, q90_d = summarize(abs_delta)
    mean_pct, med_pct, q25_pct, q75_pct, q90_pct = summarize(pct_size_change)
    mean_rng, med_rng, q25_rng, q75_rng, q90_rng = summarize(pct_range_change)

    n = len(abs_delta)
    print(
        f"AD={score} N={n} "
        f"GROW={grew} ({grew/n*100:.2f}%) "
        f"SHRINK={shrank} ({shrank/n*100:.2f}%) "
        f"SAME={unchanged} ({unchanged/n*100:.2f}%)"
        if n else f"AD={score} N=0"
    )
    print(
        f"  |Body| delta: mean={mean_d:+.6f} median={med_d:+.6f} "
        f"Q25={q25_d:+.6f} Q75={q75_d:+.6f} Q90={q90_d:+.6f}"
    )
    print(
        f"  % size change: mean={mean_pct*100:+.2f}% median={med_pct*100:+.2f}% "
        f"Q25={q25_pct*100:+.2f}% Q75={q75_pct*100:+.2f}% Q90={q90_pct*100:+.2f}%"
    )
    print(
        f"  delta/current-range: mean={mean_rng*100:+.2f}% median={med_rng*100:+.2f}% "
        f"Q25={q25_rng*100:+.2f}% Q75={q75_rng*100:+.2f}% Q90={q90_rng*100:+.2f}%"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure actual absolute candle-body growth/shrink under Excel AD scores.")
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    args = ap.parse_args()

    root = Path(args.input_dir)
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]

    for symbol in symbols:
        path = root / f"{symbol}_1h.csv"
        rows = load(path)
        print(f"{symbol} rows={len(rows)}")
        analyze(rows, 3)
        analyze(rows, 0)


if __name__ == "__main__":
    main()
