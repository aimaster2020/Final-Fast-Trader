from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.july_r_composite_walkforward import (
    RULES,
    build_weights,
    components,
    discover_symbols,
    find_month_file,
    load_binance,
    resample,
    summarize_rule,
    TESTS,
)


def composite_filtered(candle, weights, gate: float, min_confirm: int):
    vals = components(candle)
    signed = []
    total = 0.0
    used = 0.0
    for rule in RULES:
        value = vals[rule]
        if abs(value) < TESTS[rule]:
            continue
        sign = 1 if value > 0 else -1 if value < 0 else 0
        if sign:
            signed.append(sign)
            total += weights[rule] * sign
            used += weights[rule]
    if len(signed) < min_confirm or used <= 0:
        return 0, 0.0, len(signed)
    score = total / used
    signal = 1 if score >= gate else -1 if score <= -gate else 0
    return signal, score, len(signed)


def evaluate(candles, weights, gate: float, min_confirm: int):
    accepted = correct = longs = shorts = 0
    active_sum = 0
    total = max(len(candles) - 1, 0)

    for i, c in enumerate(candles[:-1]):
        sig, _score, active = composite_filtered(c, weights, gate, min_confirm)
        active_sum += active
        if not sig:
            continue

        accepted += 1
        if sig == 1:
            longs += 1
        else:
            shorts += 1

        nxt = candles[i + 1].close
        realized = 1 if nxt > c.close else -1 if nxt < c.close else 0
        if realized:
            correct += int(sig == realized)

    days = len({c.timestamp // 86400 for c in candles[:-1]}) if total else 0
    return {
        "candles": len(candles),
        "days": days,
        "signals": accepted,
        "signals_per_day": accepted / days if days else 0.0,
        "accuracy_pct": correct / accepted * 100.0 if accepted else 0.0,
        "long_pct": longs / accepted * 100.0 if accepted else 0.0,
        "avg_active_rules": active_sum / total if total else 0.0,
        "correct_signals": correct,
    }


def main():
    ap = argparse.ArgumentParser(description="Walk-forward precision sweep for R1-R16.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--train-month", default="2026-06")
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--timeframes", default="5,15,30,60")
    ap.add_argument("--gates", default="0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.90,0.95")
    ap.add_argument("--min-confirms", default="2,3,4,5,6,7,8")
    ap.add_argument("--min-signals", type=int, default=30)
    ap.add_argument("--output", default="reports/july_r_precision_sweep.csv")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    tfs = [int(x) for x in args.timeframes.split(",") if x.strip()]
    gates = [float(x) for x in args.gates.split(",") if x.strip()]
    confirms = [int(x) for x in args.min_confirms.split(",") if x.strip()]

    if args.symbols.upper() == "ALL":
        symbols = sorted(set(discover_symbols(input_dir, args.train_month)) | set(discover_symbols(input_dir, args.test_month)))
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]

    train_stats = {}
    for symbol in symbols:
        path = find_month_file(input_dir, symbol, args.train_month)
        if not path:
            continue
        raw = load_binance(path)
        for tf in tfs:
            candles = resample(raw, tf)
            for rule in RULES:
                if candles:
                    train_stats[(symbol, tf, rule)] = summarize_rule(candles, rule)
    if not train_stats:
        raise SystemExit(f"No training data found for {args.train_month}")

    weights, _ = build_weights(train_stats, tfs)
    test_sets = []
    for symbol in symbols:
        path = find_month_file(input_dir, symbol, args.test_month)
        if not path:
            continue
        raw = load_binance(path)
        for tf in tfs:
            candles = resample(raw, tf)
            if len(candles) >= 2:
                test_sets.append((symbol, tf, candles))
    if not test_sets:
        raise SystemExit(f"No test data found for {args.test_month}")

    rows = []
    for gate in gates:
        for min_confirm in confirms:
            per = []
            for symbol, tf, candles in test_sets:
                r = evaluate(candles, weights, gate, min_confirm)
                per.append(r)
                rows.append({
                    "train_month": args.train_month, "test_month": args.test_month,
                    "symbol": symbol, "timeframe": tf, "gate": gate,
                    "min_confirm": min_confirm, "min_signals": args.min_signals,
                    **r,
                })

            signals = sum(x["signals"] for x in per)
            correct = sum(x["correct_signals"] for x in per)
            days = sum(x["days"] for x in per)
            aggregate = {
                "train_month": args.train_month, "test_month": args.test_month,
                "symbol": "ALL", "timeframe": 0, "gate": gate,
                "min_confirm": min_confirm, "min_signals": args.min_signals,
                "candles": sum(x["candles"] for x in per), "days": days,
                "signals": signals, "signals_per_day": signals / days if days else 0.0,
                "accuracy_pct": correct / signals * 100.0 if signals else 0.0,
                "long_pct": sum(x["long_pct"] * x["signals"] for x in per) / signals if signals else 0.0,
                "avg_active_rules": sum(x["avg_active_rules"] for x in per) / len(per) if per else 0.0,
                "correct_signals": correct,
            }
            aggregate["eligible"] = int(signals >= args.min_signals)
            rows.append(aggregate)

    # Ranking is now explicitly quality-first. Trade frequency is reported, not optimized.
    agg = [r for r in rows if r["symbol"] == "ALL" and r["eligible"]]
    ranked = sorted(
        agg,
        key=lambda r: (float(r["accuracy_pct"]), int(r["signals"])),
        reverse=True,
    )

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "train_month", "test_month", "symbol", "timeframe", "gate", "min_confirm",
        "min_signals", "candles", "days", "signals", "signals_per_day",
        "accuracy_pct", "long_pct", "avg_active_rules", "correct_signals", "eligible",
    ]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"PRECISION_WF train={args.train_month} test={args.test_month} assets={len(symbols)} tfs={','.join(map(str,tfs))}")
    print("GATE CONF  SIGNALS/DAY ACC%  LONG% SIGNALS ELIG")
    for r in ranked[:15]:
        print(f"{float(r['gate']):.2f}  {int(r['min_confirm']):>4}  {float(r['signals_per_day']):8.2f} {float(r['accuracy_pct']):5.1f} {float(r['long_pct']):6.1f} {int(r['signals']):7d} {int(r['eligible'])}")
    print(f"SAVED {p} rows={len(rows)}")


if __name__ == "__main__":
    main()
