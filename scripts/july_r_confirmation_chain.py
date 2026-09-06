from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_pattern_trader.models import Candle
from scripts.july_r_composite_walkforward import (
    RULES,
    TESTS,
    components,
    discover_symbols,
    find_month_file,
    load_binance,
    resample,
)


def rule_signs(candle: Candle) -> dict[str, int]:
    vals = components(candle)
    out: dict[str, int] = {}
    for rule in RULES:
        v = vals[rule]
        if abs(v) >= TESTS[rule]:
            out[rule] = 1 if v > 0 else -1 if v < 0 else 0
        else:
            out[rule] = 0
    return out


def direction_name(d: int) -> str:
    return "LONG" if d == 1 else "SHORT" if d == -1 else "HOLD"


def next_direction(candles: list[Candle], i: int) -> int:
    if i + 1 >= len(candles):
        return 0
    if candles[i + 1].close > candles[i].close:
        return 1
    if candles[i + 1].close < candles[i].close:
        return -1
    return 0


def cumulative_signal(signals: list[int], trigger: int) -> int:
    """Unweighted majority of active R signals, with trigger as tie-breaker.

    No coefficient/weight is used. Inactive rules are ignored. If active
    rules split evenly, the original trigger direction is retained rather
    than manufacturing a HOLD, which keeps the sequential chain usable.
    """
    active = [s for s in signals if s]
    if not active:
        return 0
    score = sum(active)
    if score > 0:
        return 1
    if score < 0:
        return -1
    return trigger


def evaluate_dataset(candles: list[Candle]) -> tuple[list[dict], dict]:
    """R1 trigger followed by independent sequential confirmations.

    R1 at t opens a virtual prediction in its own direction. R2 is checked
    one bar later, R3 two bars later, ... R16 fifteen bars later.

    The old implementation treated a disagreement as a permanent rejection,
    which made conditional accuracy collapse to zero. This version keeps all
    stages and separately reports:
      1) individual rule accuracy;
      2) agreement with the original R1 direction;
      3) cumulative unweighted confirmation accuracy using R1..Rn.

    The cumulative signal is a majority vote of active rules seen so far, with
    the R1 direction as a tie-breaker. This is descriptive attribution only;
    it does not introduce optimized weights or a hard trade-frequency cap.
    """
    signs = [rule_signs(c) for c in candles]
    rows: list[dict] = []
    stage_stats = {
        step: {
            "opportunities": 0,
            "active": 0,
            "confirm": 0,
            "correct": 0,
            "cum_signals": 0,
            "cum_correct": 0,
            "cum_agree": 0,
        }
        for step in range(1, len(RULES) + 1)
    }
    trigger_entries = 0

    for entry_i in range(len(candles) - len(RULES)):
        trigger = signs[entry_i]["R1"]
        if not trigger:
            continue

        trigger_entries += 1
        seen_signals: list[int] = []

        for step, rule in enumerate(RULES, start=1):
            i = entry_i + step - 1
            if i >= len(candles) - 1:
                break

            s = signs[i][rule]
            realized = next_direction(candles, i)
            if not realized:
                continue

            st = stage_stats[step]
            st["opportunities"] += 1

            if s:
                st["active"] += 1
                st["confirm"] += int(s == trigger)
                st["correct"] += int(s == realized)
                seen_signals.append(s)

            cum = cumulative_signal(seen_signals, trigger)
            if cum:
                st["cum_signals"] += 1
                st["cum_correct"] += int(cum == realized)
                st["cum_agree"] += int(cum == trigger)

            rows.append({
                "entry_index": entry_i,
                "stage": step,
                "rule": rule,
                "stage_index": i,
                "trigger_direction": direction_name(trigger),
                "rule_direction": direction_name(s),
                "realized_next": direction_name(realized),
                "confirmed": int(bool(s) and s == trigger),
                "correct": int(bool(s) and s == realized),
                "cumulative_direction": direction_name(cum),
                "cumulative_correct": int(bool(cum) and cum == realized),
                "cumulative_agree": int(bool(cum) and cum == trigger),
            })

    return rows, {"trigger_entries": trigger_entries, "stage": stage_stats}


def aggregate(datasets: list[list[Candle]]) -> tuple[list[dict], dict]:
    all_rows: list[dict] = []
    total = {
        "trigger_entries": 0,
        "stage": {
            step: {
                "opportunities": 0,
                "active": 0,
                "confirm": 0,
                "correct": 0,
                "cum_signals": 0,
                "cum_correct": 0,
                "cum_agree": 0,
            }
            for step in range(1, 17)
        },
    }

    for candles in datasets:
        rows, stats = evaluate_dataset(candles)
        all_rows.extend(rows)
        total["trigger_entries"] += stats["trigger_entries"]
        for step in total["stage"]:
            for key in total["stage"][step]:
                total["stage"][step][key] += stats["stage"][step][key]

    return all_rows, total


def pct(n: int, d: int) -> float:
    return n / d * 100.0 if d else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="July R1-triggered sequential confirmation attribution test."
    )
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--timeframes", default="15,30,60,120,240")
    ap.add_argument("--output", default="reports/july_r_confirmation_chain.csv")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    tfs = [int(x.strip()) for x in args.timeframes.split(",") if x.strip()]
    if args.symbols.upper() == "ALL":
        symbols = sorted(discover_symbols(input_dir, args.test_month))
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]

    rows_out: list[dict] = []
    print(f"R_CONFIRM month={args.test_month} assets={len(symbols)} tfs={','.join(map(str,tfs))}")

    for tf in tfs:
        datasets: list[list[Candle]] = []
        for symbol in symbols:
            path = find_month_file(input_dir, symbol, args.test_month)
            if not path:
                continue
            candles = resample(load_binance(path), tf)
            if len(candles) > 17:
                datasets.append(candles)
        if not datasets:
            continue

        rows, stats = aggregate(datasets)
        for r in rows:
            r["timeframe_min"] = tf
        rows_out.extend(rows)

        print(f"\nTF={tf}m TRIGGER_ENTRIES={stats['trigger_entries']}")
        print("STEP RULE OPPORT ACTIVE CONFIRM% ACC% CUM_SIG CUM_ACC% CUM_AGREE%")
        for step in range(1, 17):
            s = stats["stage"][step]
            print(
                f"S{step:<2} {RULES[step-1]:<3} {s['opportunities']:<7} "
                f"{s['active']:<6} {pct(s['confirm'],s['active']):6.2f} "
                f"{pct(s['correct'],s['active']):6.2f} "
                f"{s['cum_signals']:<7} {pct(s['cum_correct'],s['cum_signals']):7.2f} "
                f"{pct(s['cum_agree'],s['cum_signals']):9.2f}"
            )

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "timeframe_min",
            "entry_index",
            "stage",
            "rule",
            "stage_index",
            "trigger_direction",
            "rule_direction",
            "realized_next",
            "confirmed",
            "correct",
            "cumulative_direction",
            "cumulative_correct",
            "cumulative_agree",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    print(f"SAVED {p} rows={len(rows_out)}")


if __name__ == "__main__":
    main()
