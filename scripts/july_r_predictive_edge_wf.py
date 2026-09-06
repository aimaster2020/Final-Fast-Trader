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


def evaluate(candles, candidates):
    stats = {name: {"signals": 0, "correct": 0, "longs": 0, "shorts": 0} for name in candidates}
    for i, candle in enumerate(candles[:-1]):
        signs = rule_signs(candle)
        nxt = candles[i + 1].close
        realized = 1 if nxt > candle.close else -1 if nxt < candle.close else 0
        if not realized:
            continue
        for name, rules in candidates.items():
            if len(rules) == 1:
                sig = signs[rules[0]]
            else:
                active = [signs[r] for r in rules]
                sig = active[0] if active[0] and all(x == active[0] for x in active) else 0
            if not sig:
                continue
            s = stats[name]
            s["signals"] += 1
            s["correct"] += int(sig == realized)
            s["longs"] += int(sig == 1)
            s["shorts"] += int(sig == -1)
    return stats


def aggregate_eval(datasets, candidates):
    total = {name: {"signals": 0, "correct": 0, "longs": 0, "shorts": 0} for name in candidates}
    for candles in datasets:
        part = evaluate(candles, candidates)
        for name, s in part.items():
            for k in total[name]:
                total[name][k] += s[k]
    return total


def metric(s):
    n = s["signals"]
    return s["correct"] / n * 100.0 if n else 0.0


def make_candidates():
    candidates = {r: (r,) for r in RULES}
    for a, b in combinations(RULES, 2):
        candidates[f"{a}+{b}"] = (a, b)
    return candidates


def main():
    ap = argparse.ArgumentParser(description="Walk-forward predictive-edge test for individual R rules and pairwise consensus.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--train-month", default="2026-06")
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--timeframes", default="5,15,30,60")
    ap.add_argument("--min-train-signals", type=int, default=100)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--output", default="reports/july_r_predictive_edge_wf.csv")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    tfs = [int(x) for x in args.timeframes.split(",") if x.strip()]
    if args.symbols.upper() == "ALL":
        symbols = sorted(set(discover_symbols(input_dir, args.train_month)) | set(discover_symbols(input_dir, args.test_month)))
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]

    train_sets = []
    test_sets = []
    for symbol in symbols:
        train_path = find_month_file(input_dir, symbol, args.train_month)
        test_path = find_month_file(input_dir, symbol, args.test_month)
        if train_path:
            raw = load_binance(train_path)
            for tf in tfs:
                candles = resample(raw, tf)
                if len(candles) >= 2:
                    train_sets.append((symbol, tf, candles))
        if test_path:
            raw = load_binance(test_path)
            for tf in tfs:
                candles = resample(raw, tf)
                if len(candles) >= 2:
                    test_sets.append((symbol, tf, candles))

    if not train_sets or not test_sets:
        raise SystemExit("Training or test data not found")

    candidates = make_candidates()
    train = aggregate_eval([x[2] for x in train_sets], candidates)
    eligible = [
        (name, s) for name, s in train.items() if s["signals"] >= args.min_train_signals
    ]
    ranked = sorted(eligible, key=lambda x: (metric(x[1]), x[1]["signals"]), reverse=True)
    selected = dict(ranked[: args.top_n])

    test = aggregate_eval([x[2] for x in test_sets], selected)
    rows = []
    for name, tr in ranked:
        if name not in selected:
            continue
        te = test[name]
        rows.append({
            "train_month": args.train_month,
            "test_month": args.test_month,
            "candidate": name,
            "kind": "RULE" if len(name.split("+")) == 1 else "PAIR_CONSENSUS",
            "train_signals": tr["signals"],
            "train_accuracy_pct": metric(tr),
            "train_long_pct": tr["longs"] / tr["signals"] * 100.0 if tr["signals"] else 0.0,
            "test_signals": te["signals"],
            "test_accuracy_pct": metric(te),
            "test_long_pct": te["longs"] / te["signals"] * 100.0 if te["signals"] else 0.0,
            "test_correct": te["correct"],
        })

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["candidate"]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"EDGE_WF train={args.train_month} test={args.test_month} assets={len(symbols)} tfs={','.join(map(str,tfs))}")
    print(f"CANDIDATES rules=16 pairs=120 min_train_signals={args.min_train_signals} top={len(selected)}")
    print("CANDIDATE        TRAIN_ACC TRAIN_N  TEST_ACC TEST_N  TEST_LONG%")
    for r in rows:
        print(f"{r['candidate']:<15} {float(r['train_accuracy_pct']):9.2f} {int(r['train_signals']):7d} {float(r['test_accuracy_pct']):8.2f} {int(r['test_signals']):7d} {float(r['test_long_pct']):10.1f}")
    print(f"SAVED {p} rows={len(rows)}")


if __name__ == "__main__":
    main()
