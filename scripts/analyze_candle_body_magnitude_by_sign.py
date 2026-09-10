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
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def run_group(rows, ad_score: int, body_sign: int) -> None:
    delta = []
    pct_change = []
    ratios = []
    for i in range(len(rows) - 1):
        o, h, l, c = rows[i]
        _, _, _, nc = rows[i + 1]
        body = c - o
        if body_sign == 1 and body <= 0:
            continue
        if body_sign == -1 and body >= 0:
            continue
        hc = h - c
        ho = h - o
        lc = l - c
        ad = int(hc > body) + int(ho < hc) + int(lc > body)
        if ad != ad_score:
            continue
        current_size = abs(body)
        next_size = abs(nc - rows[i + 1][0])
        d = next_size - current_size
        delta.append(d)
        if current_size > 1e-12:
            pct_change.append(d / current_size * 100.0)
            ratios.append(next_size / current_size)

    n = len(delta)
    if not n:
        print(f"AD={ad_score} BODY={'UP' if body_sign == 1 else 'DOWN'} N=0")
        return
    grew = sum(1 for x in delta if x > 0)
    shrank = sum(1 for x in delta if x < 0)
    print(
        f"AD={ad_score} BODY={'UP' if body_sign == 1 else 'DOWN'} N={n} "
        f"GROW={grew} ({grew/n*100:.2f}%) SHRINK={shrank} ({shrank/n*100:.2f}%)"
    )
    print(
        f"  delta mean={mean(delta):+.6f} median={median(delta):+.6f} "
        f"Q25={quantile(delta,.25):+.6f} Q75={quantile(delta,.75):+.6f}"
    )
    print(
        f"  size_change mean={mean(pct_change):+.2f}% median={median(pct_change):+.2f}% "
        f"Q25={quantile(pct_change,.25):+.2f}% Q75={quantile(pct_change,.75):+.2f}%"
    )
    print(
        f"  next/current size ratio median={median(ratios):.3f} "
        f"Q25={quantile(ratios,.25):.3f} Q75={quantile(ratios,.75):.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Split Excel AD body-size behavior by current candle body sign.")
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    args = ap.parse_args()
    root = Path(args.input_dir)
    for symbol in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        path = root / f"{symbol}_1h.csv"
        rows = load(path)
        print(f"\n{symbol} rows={len(rows)}")
        for ad in (3, 0):
            run_group(rows, ad, 1)
            run_group(rows, ad, -1)


if __name__ == "__main__":
    main()
