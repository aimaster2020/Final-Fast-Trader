from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median, pstdev


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    x = sorted(values)
    pos = (len(x) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(x) - 1)
    frac = pos - lo
    return x[lo] * (1.0 - frac) + x[hi] * frac


def pct(v: float) -> str:
    return f"{v:.2f}%"


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


def analyze(rows: list[tuple[float, float, float, float]], score: int) -> dict[str, float | int]:
    delta: list[float] = []
    rel_body: list[float] = []
    rel_range: list[float] = []

    for i in range(len(rows) - 1):
        o, h, l, c = rows[i]
        no, nh, nl, nc = rows[i + 1]

        body = c - o
        hc = h - c
        lc = l - c
        ho = h - o
        ad = int(hc > body) + int(ho < hc) + int(lc > body)
        if ad != score:
            continue

        next_body = nc - no
        d = next_body - body
        delta.append(d)

        if abs(body) > 1e-12:
            rel_body.append(d / abs(body))

        rng = h - l
        if rng > 1e-12:
            rel_range.append(d / rng)

    abs_delta = [abs(x) for x in delta]

    def summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "q10": 0.0, "q25": 0.0, "q75": 0.0, "q90": 0.0}
        return {
            "mean": mean(values),
            "median": median(values),
            "std": pstdev(values),
            "q10": quantile(values, 0.10),
            "q25": quantile(values, 0.25),
            "q75": quantile(values, 0.75),
            "q90": quantile(values, 0.90),
        }

    s_delta = summary(delta)
    s_abs = summary(abs_delta)
    s_rel_body = summary(rel_body)
    s_rel_range = summary(rel_range)

    return {
        "n": len(delta),
        "mean_delta": s_delta["mean"],
        "median_delta": s_delta["median"],
        "std_delta": s_delta["std"],
        "q25_delta": s_delta["q25"],
        "q75_delta": s_delta["q75"],
        "mean_abs_delta": s_abs["mean"],
        "median_abs_delta": s_abs["median"],
        "q50_rel_body": s_rel_body["median"],
        "q25_rel_body": s_rel_body["q25"],
        "q75_rel_body": s_rel_body["q75"],
        "q50_rel_range": s_rel_range["median"],
        "q25_rel_range": s_rel_range["q25"],
        "q75_rel_range": s_rel_range["q75"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze next-candle body magnitude conditioned on Excel AD score.")
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    args = ap.parse_args()

    root = Path(args.input_dir)
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]

    for symbol in symbols:
        path = root / f"{symbol}_1h.csv"
        rows = load(path)
        print(f"{symbol} rows={len(rows)}")
        for score in (3, 0):
            r = analyze(rows, score)
            print(
                f"  AD={score} N={int(r['n'])} "
                f"dMean={r['mean_delta']:+.6f} dMed={r['median_delta']:+.6f} "
                f"|d|Mean={r['mean_abs_delta']:.6f} |d|Med={r['median_abs_delta']:.6f} "
                f"dQ25={r['q25_delta']:+.6f} dQ75={r['q75_delta']:+.6f}"
            )
            print(
                f"    d/|body| median={pct(r['q50_rel_body'] * 100)} "
                f"Q25={pct(r['q25_rel_body'] * 100)} Q75={pct(r['q75_rel_body'] * 100)}"
            )
            print(
                f"    d/range median={pct(r['q50_rel_range'] * 100)} "
                f"Q25={pct(r['q25_rel_range'] * 100)} Q75={pct(r['q75_rel_range'] * 100)}"
            )


if __name__ == "__main__":
    main()
