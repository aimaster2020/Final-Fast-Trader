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


def evaluate_dataset(candles: list[Candle]) -> tuple[list[dict], dict]:
    """Sequential confirmation: R1 at t, R2 at t+1, ... Rn at t+n-1.

    A chain is valid only when every rule in the prefix emits the same
    direction. Entry is taken on the candle where the last rule confirms.
    For every completed chain, all 16 R signals on the entry candle are also
    scored against the next candle direction.
    """
    signs = [rule_signs(c) for c in candles]
    rows: list[dict] = []
    step_stats = {
        step: {"entries": 0, "correct": 0}
        for step in range(1, len(RULES) + 1)
    }
    rule_stats = {
        rule: {"active": 0, "correct": 0, "long": 0, "short": 0}
        for rule in RULES
    }

    # Step k means R1..Rk confirmed sequentially on consecutive candles.
    for start in range(len(candles) - len(RULES)):
        direction = 0
        valid = True
        completed_steps: list[int] = []
        for j, rule in enumerate(RULES):
            s = signs[start + j][rule]
            if not s:
                valid = False
                break
            if direction == 0:
                direction = s
            elif s != direction:
                valid = False
                break
            completed_steps.append(j + 1)

        # Record each prefix independently. A failed later filter does not
        # erase the accuracy of earlier confirmations.
        for step in completed_steps:
            entry_i = start + step - 1
            if entry_i >= len(candles) - 1:
                continue
            realized = 1 if candles[entry_i + 1].close > candles[entry_i].close else -1 if candles[entry_i + 1].close < candles[entry_i].close else 0
            if not realized:
                continue
            s = direction
            step_stats[step]["entries"] += 1
            step_stats[step]["correct"] += int(s == realized)

            # Snapshot of every R at the actual entry candle for attribution.
            entry_signs = signs[entry_i]
            for rule in RULES:
                rs = entry_signs[rule]
                if not rs:
                    continue
                rule_stats[rule]["active"] += 1
                rule_stats[rule]["correct"] += int(rs == realized)
                rule_stats[rule]["long"] += int(rs == 1)
                rule_stats[rule]["short"] += int(rs == -1)

            rows.append({
                "entry_index": entry_i,
                "chain_step": step,
                "direction": direction_name(direction),
                "realized": direction_name(realized),
                "correct": int(direction == realized),
            })

    return rows, {"step": step_stats, "rule": rule_stats}


def aggregate(datasets: list[list[Candle]]) -> tuple[list[dict], dict]:
    all_rows: list[dict] = []
    total_step = {step: {"entries": 0, "correct": 0} for step in range(1, 17)}
    total_rule = {rule: {"active": 0, "correct": 0, "long": 0, "short": 0} for rule in RULES}
    for candles in datasets:
        rows, stats = evaluate_dataset(candles)
        all_rows.extend(rows)
        for step in total_step:
            for k in total_step[step]:
                total_step[step][k] += stats["step"][step][k]
        for rule in RULES:
            for k in total_rule[rule]:
                total_rule[rule][k] += stats["rule"][rule][k]
    return all_rows, {"step": total_step, "rule": total_rule}


def pct(correct: int, n: int) -> float:
    return correct / n * 100.0 if n else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Sequential R1->R16 confirmation-chain accuracy test.")
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
    print(f"R_CHAIN month={args.test_month} assets={len(symbols)} tfs={','.join(map(str,tfs))}")

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

        print(f"\nTF={tf}m STEP_CONFIRMATION")
        for step in range(1, 17):
            s = stats["step"][step]
            print(f"S{step:<2} R1..R{step:<2} {pct(s['correct'],s['entries']):6.2f}% / {s['entries']}")

        print(f"TF={tf}m ENTRY_SNAPSHOT_R_ACCURACY")
        parts = []
        for rule in RULES:
            s = stats["rule"][rule]
            parts.append(f"{rule}:{pct(s['correct'],s['active']):.2f}%/{s['active']}")
        print(" | ".join(parts))

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        fields = ["timeframe_min", "entry_index", "chain_step", "direction", "realized", "correct"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(f"SAVED {p} rows={len(rows_out)}")


if __name__ == "__main__":
    main()
