from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.july_r_composite_walkforward import (
    RULES,
    TESTS,
    components,
    discover_symbols,
    find_month_file,
    load_binance,
    resample,
)


def rule_signs(candle):
    vals = components(candle)
    out = {}
    for rule in RULES:
        v = vals[rule]
        if abs(v) >= TESTS[rule]:
            out[rule] = 1 if v > 0 else -1 if v < 0 else 0
        else:
            out[rule] = 0
    return out


def make_candidates():
    """Return every rule in normal and inverted orientation plus all pair variants."""
    candidates = {}
    for r in RULES:
        candidates[r] = ((r, 1),)
        candidates[f"{r}_INV"] = ((r, -1),)

    oriented = [(r, d) for r in RULES for d in (1, -1)]
    for (a, da), (b, db) in combinations(oriented, 2):
        # Same base rule in both orientations is intentionally excluded.
        if a == b:
            continue
        name = f"{a}{'' if da == 1 else '_INV'}+{b}{'' if db == 1 else '_INV'}"
        candidates[name] = ((a, da), (b, db))
    return candidates


def evaluate(candles, candidates):
    stats = {
        name: {"signals": 0, "correct": 0, "longs": 0, "shorts": 0}
        for name in candidates
    }
    for i, candle in enumerate(candles[:-1]):
        signs = rule_signs(candle)
        nxt = candles[i + 1].close
        realized = 1 if nxt > candle.close else -1 if nxt < candle.close else 0
        if not realized:
            continue

        for name, rules in candidates.items():
            oriented = [signs[r] * direction for r, direction in rules]
            if len(oriented) == 1:
                sig = oriented[0]
            else:
                sig = oriented[0] if oriented[0] and all(x == oriented[0] for x in oriented) else 0
            if not sig:
                continue
            s = stats[name]
            s["signals"] += 1
            s["correct"] += int(sig == realized)
            s["longs"] += int(sig == 1)
            s["shorts"] += int(sig == -1)
    return stats


def aggregate_eval(datasets, candidates):
    total = {
        name: {"signals": 0, "correct": 0, "longs": 0, "shorts": 0}
        for name in candidates
    }
    for candles in datasets:
        part = evaluate(candles, candidates)
        for name, s in part.items():
            for k in total[name]:
                total[name][k] += s[k]
    return total


def accuracy(s):
    return s["correct"] / s["signals"] * 100.0 if s["signals"] else 0.0


def parse_tfs(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return sorted(dict.fromkeys(values))


def main():
    ap = argparse.ArgumentParser(
        description="R accuracy matrix: every R, inverse R, every pair orientation, and higher timeframes."
    )
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--timeframes", default="5,15,30,60,120,240")
    ap.add_argument("--output", default="reports/july_r_accuracy_matrix.csv")
    ap.add_argument("--top-pairs", type=int, default=10)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    tfs = parse_tfs(args.timeframes)
    if args.symbols.upper() == "ALL":
        symbols = sorted(discover_symbols(input_dir, args.test_month))
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]

    candidates = make_candidates()
    rows: list[dict] = []
    by_tf: dict[int, dict] = {}

    for tf in tfs:
        datasets = []
        for symbol in symbols:
            path = find_month_file(input_dir, symbol, args.test_month)
            if not path:
                continue
            raw = load_binance(path)
            candles = resample(raw, tf)
            if len(candles) >= 2:
                datasets.append((symbol, candles))

        if not datasets:
            continue

        stats = aggregate_eval([x[1] for x in datasets], candidates)
        by_tf[tf] = stats
        for name, s in stats.items():
            kind = "RULE" if "+" not in name else "PAIR"
            rows.append({
                "month": args.test_month,
                "timeframe_min": tf,
                "candidate": name,
                "kind": kind,
                "signals": s["signals"],
                "accuracy_pct": accuracy(s),
                "long_pct": s["longs"] / s["signals"] * 100.0 if s["signals"] else 0.0,
                "short_pct": s["shorts"] / s["signals"] * 100.0 if s["signals"] else 0.0,
                "correct": s["correct"],
            })

    if not rows:
        raise SystemExit(f"No test data found for {args.test_month}")

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Console is deliberately compact: all single rules are shown; only top pairs per TF.
    print(f"R_ACCURACY month={args.test_month} assets={len(symbols)} tfs={','.join(map(str, tfs))}")
    print("RULE      " + " ".join(f"{tf:>6}m" for tf in by_tf))
    for rule in RULES:
        normal = " ".join(
            f"{accuracy(by_tf[tf][rule]):6.2f}" if tf in by_tf else "   -- "
            for tf in by_tf
        )
        inv = " ".join(
            f"{accuracy(by_tf[tf][rule + '_INV']):6.2f}" if tf in by_tf else "   -- "
            for tf in by_tf
        )
        print(f"{rule:<4} N {normal}")
        print(f"{rule:<4} I {inv}")

    for tf in by_tf:
        pairs = [
            (name, s) for name, s in by_tf[tf].items()
            if "+" in name and s["signals"] > 0
        ]
        pairs.sort(key=lambda x: (accuracy(x[1]), x[1]["signals"]), reverse=True)
        print(f"PAIR_TOP tf={tf}m " + " | ".join(
            f"{name}:{accuracy(s):.2f}%/{s['signals']}" for name, s in pairs[: args.top_pairs]
        ))

    print(f"SAVED {p} rows={len(rows)} candidates={len(candidates)}")


if __name__ == "__main__":
    main()
