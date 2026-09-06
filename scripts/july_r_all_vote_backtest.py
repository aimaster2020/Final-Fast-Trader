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

# Native rule direction is the economic direction implied by the existing
# rule characterization. Rules whose native meaning is SHORT are inverted so
# every active R contributes to one common LONG(+1) / SHORT(-1) vote space.
# This mapping is intentionally explicit for auditability and reproducibility.
NATIVE_TO_LONG = {
    "R1": -1,
    "R2": 1,
    "R3": -1,
    "R4": 1,
    "R5": 1,
    "R6": 1,
    "R7": -1,
    "R8": 1,
    "R9": -1,
    "R10": 1,
    "R11": -1,
    "R12": 1,
    "R13": -1,
    "R14": 1,
    "R15": -1,
    "R16": 1,
}


def raw_sign(c: Candle, rule: str) -> int:
    value = components(c)[rule]
    threshold = TESTS[rule]
    return 1 if value >= threshold else -1 if value <= -threshold else 0


def economic_sign(c: Candle, rule: str) -> int:
    return raw_sign(c, rule) * NATIVE_TO_LONG[rule]


def realized(candles: list[Candle], i: int) -> int:
    if i + 1 >= len(candles):
        return 0
    if candles[i + 1].close > candles[i].close:
        return 1
    if candles[i + 1].close < candles[i].close:
        return -1
    return 0


def vote(signs: list[int]) -> int:
    active = [s for s in signs if s]
    if not active:
        return 0
    total = sum(active)
    if total > 0:
        return 1
    if total < 0:
        return -1
    return 0


def name(d: int) -> str:
    return "LONG" if d == 1 else "SHORT" if d == -1 else "HOLD"


def evaluate_dataset(candles: list[Candle]) -> tuple[list[dict], dict]:
    rule_stats = {
        r: {"active": 0, "correct": 0, "long": 0, "short": 0}
        for r in RULES
    }
    decision_stats = {
        "LONG": {"n": 0, "correct": 0},
        "SHORT": {"n": 0, "correct": 0},
        "HOLD": {"n": 0, "correct": 0},
        "ALL": {"n": 0, "correct": 0},
    }
    rows: list[dict] = []

    for i in range(len(candles) - 1):
        y = realized(candles, i)
        if not y:
            continue

        per_rule: dict[str, int] = {}
        active_votes: list[int] = []
        for r in RULES:
            s = economic_sign(candles[i], r)
            per_rule[r] = s
            if s:
                rule_stats[r]["active"] += 1
                rule_stats[r]["correct"] += int(s == y)
                rule_stats[r]["long"] += int(s == 1)
                rule_stats[r]["short"] += int(s == -1)
                active_votes.append(s)

        decision = vote(active_votes)
        dname = name(decision)
        if decision:
            decision_stats[dname]["n"] += 1
            decision_stats[dname]["correct"] += int(decision == y)
            decision_stats["ALL"]["n"] += 1
            decision_stats["ALL"]["correct"] += int(decision == y)
        else:
            decision_stats["HOLD"]["n"] += 1

        rows.append({
            "bar_index": i,
            "realized_next": name(y),
            "decision": dname,
            "vote_sum": sum(active_votes),
            "active_count": len(active_votes),
            **{f"{r}_decision": name(per_rule[r]) for r in RULES},
            **{f"{r}_correct": int(bool(per_rule[r]) and per_rule[r] == y) for r in RULES},
        })

    return rows, {"rules": rule_stats, "decisions": decision_stats}


def pct(a: int, n: int) -> float:
    return 100.0 * a / n if n else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="July independent all-R vote backtest. Every active R contributes an equal vote after native SHORT rules are inverted into economic LONG/SHORT space."
    )
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--timeframes", default="30,60,120,240")
    ap.add_argument("--output", default="reports/july_r_all_vote_backtest.csv")
    args = ap.parse_args()

    root = Path(args.input_dir)
    tfs = [int(x.strip()) for x in args.timeframes.split(",") if x.strip()]
    if args.symbols.upper() == "ALL":
        symbols = sorted(discover_symbols(root, args.test_month))
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]

    all_rows: list[dict] = []
    print(f"R_ALL_VOTE month={args.test_month} assets={len(symbols)} tfs={','.join(map(str,tfs))}")
    print("RULE_ORIENTATION " + " ".join(f"{r}={'INV' if NATIVE_TO_LONG[r] == -1 else 'NATIVE'}" for r in RULES))

    for tf in tfs:
        datasets: list[tuple[str, list[Candle]]] = []
        for symbol in symbols:
            p = find_month_file(root, symbol, args.test_month)
            if not p:
                continue
            cs = resample(load_binance(p), tf)
            if len(cs) > 20:
                datasets.append((symbol, cs))

        if not datasets:
            continue

        total_rule = {r: {"active": 0, "correct": 0, "long": 0, "short": 0} for r in RULES}
        total_dec = {k: {"n": 0, "correct": 0} for k in ("LONG", "SHORT", "HOLD", "ALL")}

        for symbol, cs in datasets:
            rows, stats = evaluate_dataset(cs)
            for row in rows:
                row["symbol"] = symbol
                row["timeframe_min"] = tf
            all_rows.extend(rows)
            for r in RULES:
                for k in total_rule[r]:
                    total_rule[r][k] += stats["rules"][r][k]
            for k in total_dec:
                for field in total_dec[k]:
                    total_dec[k][field] += stats["decisions"][k][field]

        print(f"\n===== TF={tf}m =====")
        print("RULE ACC% ACTIVE CORRECT LONG SHORT")
        ranked = []
        for r in RULES:
            s = total_rule[r]
            ranked.append((r, pct(s["correct"], s["active"]), s["active"], s["correct"], s["long"], s["short"]))
        ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
        for r, a, n, c, lo, sh in ranked:
            print(f"{r:<4} {a:6.2f} {n:7} {c:7} {lo:4} {sh:5}")

        print("DECISION ACC% N CORRECT")
        for k in ("LONG", "SHORT", "HOLD", "ALL"):
            s = total_dec[k]
            print(f"{k:<6} {pct(s['correct'], s['n']):6.2f} {s['n']:7} {s['correct']:7}")

        active_total = sum(total_rule[r]["active"] for r in RULES)
        correct_total = sum(total_rule[r]["correct"] for r in RULES)
        d = total_dec["ALL"]
        print(f"SUMMARY ALL_RULE_ACTIVE={active_total} ALL_RULE_ACC={pct(correct_total,active_total):.2f}% "
              f"VOTE_ACC={pct(d['correct'],d['n']):.2f}% VOTE_TRADES={d['n']} "
              f"LONG={total_dec['LONG']['n']} SHORT={total_dec['SHORT']['n']} HOLD={total_dec['HOLD']['n']}")

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "symbol", "timeframe_min", "bar_index", "realized_next", "decision", "vote_sum", "active_count",
        *[f"{r}_decision" for r in RULES], *[f"{r}_correct" for r in RULES]
    ]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"SAVED {p} rows={len(all_rows)}")


if __name__ == "__main__":
    main()
