from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Pattern:
    name: str
    direction: int  # +1 bullish, -1 bearish
    min_bars: int = 1


def candle_parts(r: dict) -> tuple[float, float, float, float, float]:
    o = float(r["open"]); h = float(r["high"]); l = float(r["low"]); c = float(r["close"])
    rng = h - l
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    return rng, body, upper, lower, c - o


def pattern_matches(rows: list[dict], i: int, name: str) -> bool:
    rng, body, upper, lower, delta = candle_parts(rows[i])
    if rng <= 0:
        return False
    body_ratio = body / rng
    upper_ratio = upper / rng
    lower_ratio = lower / rng
    bull = delta > 0
    bear = delta < 0

    if name == "HAMMER":
        return body_ratio <= 0.35 and lower >= 2.0 * body and upper <= 0.35 * rng
    if name == "INVERTED_HAMMER":
        return body_ratio <= 0.35 and upper >= 2.0 * body and lower <= 0.35 * rng
    if name == "SHOOTING_STAR":
        return body_ratio <= 0.35 and upper >= 2.0 * body and lower <= 0.35 * rng and bear
    if name == "HANGING_MAN":
        return body_ratio <= 0.35 and lower >= 2.0 * body and upper <= 0.35 * rng and bear
    if name == "BULLISH_PIN":
        return lower >= 2.5 * max(body, rng * 0.05) and upper <= 0.30 * rng and c >= rows[i]["low"] if False else lower_ratio >= 0.60 and upper_ratio <= 0.20 and bull
    if name == "BEARISH_PIN":
        return upper_ratio >= 0.60 and lower_ratio <= 0.20 and bear
    if name == "DOJI":
        return body_ratio <= 0.10
    if name == "BULL_MARUBOZU":
        return bull and body_ratio >= 0.80 and upper_ratio <= 0.10 and lower_ratio <= 0.10
    if name == "BEAR_MARUBOZU":
        return bear and body_ratio >= 0.80 and upper_ratio <= 0.10 and lower_ratio <= 0.10

    if i < 1:
        return False
    po = float(rows[i-1]["open"]); ph = float(rows[i-1]["high"]); pl = float(rows[i-1]["low"]); pc = float(rows[i-1]["close"])
    if name == "BULLISH_ENGULFING":
        return pc < po and c > o and o <= pc and c >= po
    if name == "BEARISH_ENGULFING":
        return pc > po and c < o and o >= pc and c <= po
    if name == "INSIDE_BAR":
        return h <= ph and l >= pl
    if name == "OUTSIDE_BAR_BULL":
        return h > ph and l < pl and bull
    if name == "OUTSIDE_BAR_BEAR":
        return h > ph and l < pl and bear
    if name == "TWEEZER_BOTTOM":
        return abs(l - pl) / max(rng, 1e-12) <= 0.05 and bull and pc < po
    if name == "TWEEZER_TOP":
        return abs(h - ph) / max(rng, 1e-12) <= 0.05 and bear and pc > po
    return False

PATTERNS = [
    Pattern("HAMMER", +1),
    Pattern("INVERTED_HAMMER", +1),
    Pattern("SHOOTING_STAR", -1),
    Pattern("HANGING_MAN", -1),
    Pattern("BULLISH_PIN", +1),
    Pattern("BEARISH_PIN", -1),
    Pattern("BULLISH_ENGULFING", +1, 2),
    Pattern("BEARISH_ENGULFING", -1, 2),
    Pattern("DOJI", +1),
    Pattern("BULL_MARUBOZU", +1),
    Pattern("BEAR_MARUBOZU", -1),
    Pattern("INSIDE_BAR", +1),
    Pattern("OUTSIDE_BAR_BULL", +1),
    Pattern("OUTSIDE_BAR_BEAR", -1),
    Pattern("TWEEZER_BOTTOM", +1, 2),
    Pattern("TWEEZER_TOP", -1, 2),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="OHLC candle pattern accuracy on future horizons 1..5.")
    ap.add_argument("--prepared-file", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--months", default="2026-05,2026-06,2026-07,2026-08")
    ap.add_argument("--horizons", default="1,2,3,4,5")
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--output", default="reports/candle_pattern_accuracy_1h.csv")
    args = ap.parse_args()

    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    months = set(x.strip() for x in args.months.split(",") if x.strip())
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    grouped: dict[str, list[dict]] = defaultdict(list)
    path = Path(args.prepared_file)
    if not path.is_absolute(): path = ROOT / path
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["symbol"].upper() in symbols and r["month"] in months:
                grouped[r["symbol"].upper()].append(r)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["timestamp"]))

    results = []
    print("CANDLE_PATTERN_ACCURACY 1H | future horizons = 1..5 candles")
    print("PATTERN                 H   N    ACC     AVG_RET")
    print("-----------------------------------------------------")

    for p in PATTERNS:
        for h in horizons:
            samples = []
            for symbol in symbols:
                rows = grouped[symbol]
                for i in range(p.min_bars - 1, len(rows) - h):
                    if not pattern_matches(rows, i, p.name):
                        continue
                    entry = float(rows[i]["close"])
                    future = float(rows[i+h]["close"])
                    if entry <= 0:
                        continue
                    ret = future / entry - 1.0
                    correct = ret > 0 if p.direction > 0 else ret < 0
                    samples.append((ret, correct))
            n = len(samples)
            acc = sum(x[1] for x in samples) / n * 100.0 if n else 0.0
            avg = sum(x[0] for x in samples) / n * 100.0 if n else 0.0
            if n >= args.min_samples:
                print(f"{p.name:22} H={h} N={n:4d} ACC={acc:6.1f}% AVG={avg:+7.3f}%")
            results.append((p.name, p.direction, h, n, acc, avg))

    ranked = [r for r in results if r[3] >= args.min_samples]
    ranked.sort(key=lambda r: (r[4], r[3]), reverse=True)
    print("\nTOP ACCURACY")
    for name, direction, h, n, acc, avg in ranked[:15]:
        side = "LONG" if direction > 0 else "SHORT"
        print(f"{name:22} H={h} {side:5} N={n:4d} ACC={acc:6.1f}% AVG={avg:+7.3f}%")

    out = Path(args.output)
    if not out.is_absolute(): out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pattern","direction","horizon","samples","accuracy_pct","avg_forward_return_pct"])
        w.writerows(results)
    print(f"SAVED {out.relative_to(ROOT)} rows={len(results)}")

if __name__ == "__main__":
    main()
