from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_pattern_trader.models import Candle
from scripts.prepared_market_loader import load_prepared


@dataclass(frozen=True)
class PatternMatch:
    name: str
    direction: int


def body(c: Candle) -> float:
    return abs(c.close - c.open)


def candle_range(c: Candle) -> float:
    return max(c.high - c.low, 1e-12)


def upper_wick(c: Candle) -> float:
    return c.high - max(c.open, c.close)


def lower_wick(c: Candle) -> float:
    return min(c.open, c.close) - c.low


def is_bull(c: Candle) -> bool:
    return c.close > c.open


def is_bear(c: Candle) -> bool:
    return c.close < c.open


def near(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance


def pattern_matches(rows: list[Candle], i: int, doji_body_pct: float = 0.10, equal_tol_pct: float = 0.20) -> list[PatternMatch]:
    c = rows[i]
    r = candle_range(c)
    b = body(c)
    uw = upper_wick(c)
    lw = lower_wick(c)
    out: list[PatternMatch] = []

    small_body = b <= r * doji_body_pct
    # Single-candle reversal structures.
    if lw >= 2.0 * max(b, r * 0.01) and uw <= max(b * 0.75, r * 0.05) and c.close >= c.open:
        out.append(PatternMatch("HAMMER", 1))
    if uw >= 2.0 * max(b, r * 0.01) and lw <= max(b * 0.75, r * 0.05) and c.close >= c.open:
        out.append(PatternMatch("INVERTED_HAMMER", 1))
    if uw >= 2.0 * max(b, r * 0.01) and lw <= max(b * 0.75, r * 0.05) and c.close <= c.open:
        out.append(PatternMatch("SHOOTING_STAR", -1))
    if lw >= 2.0 * max(b, r * 0.01) and uw <= max(b * 0.75, r * 0.05) and c.close <= c.open:
        out.append(PatternMatch("HANGING_MAN", -1))

    # Pin bars, deliberately stricter than the basic hammer family.
    if lw >= 3.0 * max(b, r * 0.01) and lw / r >= 0.60:
        out.append(PatternMatch("BULLISH_PIN_BAR", 1))
    if uw >= 3.0 * max(b, r * 0.01) and uw / r >= 0.60:
        out.append(PatternMatch("BEARISH_PIN_BAR", -1))

    if small_body:
        out.append(PatternMatch("DOJI", 0))

    # Marubozu: very small opposite wick(s).
    if is_bull(c) and (c.open - c.low) <= r * 0.08 and (c.high - c.close) <= r * 0.08:
        out.append(PatternMatch("BULLISH_MARUBOZU", 1))
    if is_bear(c) and (c.high - c.open) <= r * 0.08 and (c.close - c.low) <= r * 0.08:
        out.append(PatternMatch("BEARISH_MARUBOZU", -1))

    if i >= 1:
        p = rows[i - 1]
        # Engulfing uses real bodies only.
        if is_bear(p) and is_bull(c) and c.open <= p.close and c.close >= p.open:
            out.append(PatternMatch("BULLISH_ENGULFING", 1))
        if is_bull(p) and is_bear(c) and c.open >= p.close and c.close <= p.open:
            out.append(PatternMatch("BEARISH_ENGULFING", -1))

        # Inside / outside bars.
        if c.high <= p.high and c.low >= p.low:
            out.append(PatternMatch("INSIDE_BAR", 0))
        if c.high >= p.high and c.low <= p.low:
            if c.close > p.close:
                out.append(PatternMatch("BULLISH_OUTSIDE_BAR", 1))
            elif c.close < p.close:
                out.append(PatternMatch("BEARISH_OUTSIDE_BAR", -1))

        tol = max(abs(p.high), abs(p.low), abs(c.high), abs(c.low), 1.0) * equal_tol_pct / 100.0
        if near(c.low, p.low, tol) and is_bull(c):
            out.append(PatternMatch("TWEEZER_BOTTOM", 1))
        if near(c.high, p.high, tol) and is_bear(c):
            out.append(PatternMatch("TWEEZER_TOP", -1))

    return out


def evaluate(rows: list[Candle], horizons: list[int]) -> list[dict]:
    records: list[dict] = []
    max_h = max(horizons)
    for i in range(1, len(rows) - max_h):
        matches = pattern_matches(rows, i)
        if not matches:
            continue
        current = rows[i].close
        for match in matches:
            # Neutral patterns are measured separately but do not count as correct/incorrect.
            for h in horizons:
                future = rows[i + h].close
                delta = (future / current - 1.0) * 100.0
                realized = 1 if delta > 0 else -1 if delta < 0 else 0
                correct = match.direction != 0 and realized == match.direction
                records.append({
                    "pattern": match.name,
                    "prediction": match.direction,
                    "horizon": h,
                    "realized": realized,
                    "correct": int(correct),
                    "forward_return_pct": delta * (1 if match.direction == 1 else -1 if match.direction == -1 else 0),
                })
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate numeric OHLC candle patterns on prepared 1H data for forward horizons.")
    ap.add_argument("--prepared-file", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--months", default="2026-05,2026-06,2026-07,2026-08")
    ap.add_argument("--horizons", default="1,2,3,4,5")
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--output", default="reports/candle_pattern_accuracy.csv")
    args = ap.parse_args()

    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    months = [x.strip() for x in args.months.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    if any(h <= 0 for h in horizons):
        ap.error("all horizons must be positive")

    prepared = load_prepared(Path(args.prepared_file))
    rows_by_symbol: dict[str, list[Candle]] = defaultdict(list)
    for symbol in symbols:
        rows: list[Candle] = []
        for month in months:
            rows.extend(prepared.get((symbol, month), []))
        rows.sort(key=lambda c: c.timestamp)
        rows_by_symbol[symbol] = rows

    all_records: list[dict] = []
    for symbol in symbols:
        records = evaluate(rows_by_symbol[symbol], horizons)
        for r in records:
            r["symbol"] = symbol
        all_records.extend(records)

    # Aggregate across all four symbols.
    agg: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in all_records:
        agg[(r["pattern"], r["horizon"])].append(r)

    print(f"CANDLE_PATTERN_ACCURACY source={args.prepared_file} TF=1H symbols={len(symbols)}")
    print("PATTERN               H   N    ACC      AVG_RETURN")
    print("------------------------------------------------------")
    output_rows = []
    for (pattern, horizon), items in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        directional = [x for x in items if x["prediction"] != 0]
        n = len(directional)
        if n < args.min_samples:
            continue
        acc = sum(x["correct"] for x in directional) / n * 100.0
        avg_ret = sum(x["forward_return_pct"] for x in directional) / n
        print(f"{pattern:22} H={horizon} N={n:4d} ACC={acc:6.1f}% AVG={avg_ret:+8.4f}%")
        output_rows.append({
            "pattern": pattern,
            "horizon": horizon,
            "samples": n,
            "accuracy_pct": acc,
            "avg_forward_return_pct": avg_ret,
        })

    ranked = sorted(output_rows, key=lambda x: (x["accuracy_pct"], x["samples"], x["avg_forward_return_pct"]), reverse=True)
    print("TOP ACCURACY")
    for r in ranked[:20]:
        print(f"{r['pattern']:22} H={r['horizon']} N={r['samples']:4d} ACC={r['accuracy_pct']:6.1f}% AVG={r['avg_forward_return_pct']:+8.4f}%")

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pattern", "horizon", "samples", "accuracy_pct", "avg_forward_return_pct"])
        w.writeheader()
        w.writerows(output_rows)
    print(f"SAVED {out.relative_to(ROOT)} rows={len(output_rows)}")


if __name__ == "__main__":
    main()
