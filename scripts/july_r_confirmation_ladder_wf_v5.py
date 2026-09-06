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


def raw_sig(c: Candle, rule: str) -> int:
    v = components(c)[rule]
    th = TESTS[rule]
    return 1 if v >= th else -1 if v <= -th else 0


def variant_sig(c: Candle, rule: str, variant: str) -> int:
    s = raw_sig(c, rule)
    return s if variant == "N" else -s


def nxt(cs: list[Candle], i: int) -> int:
    if i + 1 >= len(cs):
        return 0
    return 1 if cs[i + 1].close > cs[i].close else -1 if cs[i + 1].close < cs[i].close else 0


def pct(c: int, n: int) -> float:
    return 100.0 * c / n if n else 0.0


def load_sets(root: Path, symbols: list[str], month: str, tf: int) -> list[list[Candle]]:
    out: list[list[Candle]] = []
    for symbol in symbols:
        p = find_month_file(root, symbol, month)
        if not p:
            continue
        cs = resample(load_binance(p), tf)
        if len(cs) > 32:
            out.append(cs)
    return out


def choose_variant(train: list[list[Candle]], rule: str, min_n: int) -> tuple[str, float, int, float, int]:
    stats = {"N": [0, 0], "I": [0, 0]}
    for cs in train:
        for i in range(len(cs) - 1):
            y = nxt(cs, i)
            if not y:
                continue
            for variant in ("N", "I"):
                s = variant_sig(cs[i], rule, variant)
                if not s:
                    continue
                stats[variant][1] += 1
                stats[variant][0] += int(s == y)
    n_acc = pct(stats["N"][0], stats["N"][1])
    i_acc = pct(stats["I"][0], stats["I"][1])
    if stats["N"][1] < min_n and stats["I"][1] < min_n:
        return ("N" if n_acc >= i_acc else "I", n_acc, stats["N"][1], i_acc, stats["I"][1])
    if stats["N"][1] < min_n:
        return "I", n_acc, stats["N"][1], i_acc, stats["I"][1]
    if stats["I"][1] < min_n:
        return "N", n_acc, stats["N"][1], i_acc, stats["I"][1]
    return ("N" if n_acc >= i_acc else "I", n_acc, stats["N"][1], i_acc, stats["I"][1])


def pass_chain(
    cs: list[Candle],
    entry_i: int,
    trigger_rule: str,
    trigger_variant: str,
    selected: list[dict],
) -> tuple[bool, int]:
    trigger = variant_sig(cs[entry_i], trigger_rule, trigger_variant)
    if not trigger:
        return False, 0
    for k, x in enumerate(selected, 1):
        j = entry_i + k
        if j >= len(cs):
            return False, trigger
        s = variant_sig(cs[j], x["rule"], x["variant"])
        if not s:
            return False, trigger
        expected = trigger if x["mode"] == "SAME" else -trigger
        if s != expected:
            return False, trigger
    return True, trigger


def candidate_stats(
    train: list[list[Candle]],
    trigger_rule: str,
    trigger_variant: str,
    selected: list[dict],
    rule: str,
    variant: str,
) -> dict[str, dict[str, int | float]]:
    out = {"SAME": {"correct": 0, "n": 0}, "OPPOSITE": {"correct": 0, "n": 0}}
    depth = len(selected)
    for cs in train:
        for i in range(len(cs) - depth - 2):
            ok, trigger = pass_chain(cs, i, trigger_rule, trigger_variant, selected)
            if not ok:
                continue
            j = i + depth + 1
            s = variant_sig(cs[j], rule, variant)
            y = nxt(cs, j)
            if not s or not y:
                continue
            mode = "SAME" if s == trigger else "OPPOSITE"
            out[mode]["n"] += 1
            out[mode]["correct"] += int(s == y)
    for mode in out:
        out[mode]["acc"] = pct(int(out[mode]["correct"]), int(out[mode]["n"]))
    return out


def choose_next(
    train: list[list[Candle]],
    trigger_rule: str,
    trigger_variant: str,
    selected: list[dict],
    min_n: int,
) -> list[dict]:
    ranked: list[dict] = []
    for rule in RULES:
        if rule == trigger_rule or any(x["rule"] == rule for x in selected):
            continue
        for variant in ("N", "I"):
            stats = candidate_stats(train, trigger_rule, trigger_variant, selected, rule, variant)
            for mode in ("SAME", "OPPOSITE"):
                n = int(stats[mode]["n"])
                if n < min_n:
                    continue
                ranked.append({
                    "rule": rule,
                    "variant": variant,
                    "mode": mode,
                    "acc": float(stats[mode]["acc"]),
                    "n": n,
                    "same_acc": float(stats["SAME"]["acc"]),
                    "same_n": int(stats["SAME"]["n"]),
                    "opp_acc": float(stats["OPPOSITE"]["acc"]),
                    "opp_n": int(stats["OPPOSITE"]["n"]),
                })
    ranked.sort(key=lambda x: (x["acc"], x["n"]), reverse=True)
    return ranked


def evaluate_level(
    test: list[list[Candle]],
    trigger_rule: str,
    trigger_variant: str,
    selected: list[dict],
) -> tuple[int, int, int, int, int]:
    depth = len(selected)
    opportunities = signals = correct = base_correct = base_n = 0
    for cs in test:
        for i in range(len(cs) - depth - 2):
            trigger = variant_sig(cs[i], trigger_rule, trigger_variant)
            if not trigger:
                continue
            opportunities += 1
            final_i = i + depth + 1
            realized = nxt(cs, final_i)
            if not realized:
                continue
            base_n += 1
            base_correct += int(trigger == realized)
            ok, chain_trigger = pass_chain(cs, i, trigger_rule, trigger_variant, selected)
            if not ok:
                continue
            signals += 1
            correct += int(chain_trigger == realized)
    return opportunities, signals, correct, base_n, base_correct


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward R confirmation ladder with explicit Native/Inverse variants")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--train-month", default="2026-06")
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--timeframes", default="30,60,120,240")
    ap.add_argument("--trigger", default="R1")
    ap.add_argument("--trigger-variant", choices=("AUTO", "N", "I"), default="AUTO")
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--output", default="reports/july_r_confirmation_ladder_wf_v5.csv")
    args = ap.parse_args()

    root = Path(args.input_dir)
    tfs = [int(x.strip()) for x in args.timeframes.split(",") if x.strip()]
    if args.symbols.upper() == "ALL":
        symbols = sorted(set(discover_symbols(root, args.train_month)) | set(discover_symbols(root, args.test_month)))
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]

    rows: list[dict] = []
    print(f"R_LADDER_WF_V5 train={args.train_month} test={args.test_month} assets={len(symbols)} tfs={','.join(map(str,tfs))} min_n={args.min_samples}")
    print("Each R is tested explicitly as NATIVE and INVERSE. Candidate relation to the trigger is SAME or OPPOSITE.")

    for tf in tfs:
        train = load_sets(root, symbols, args.train_month, tf)
        test = load_sets(root, symbols, args.test_month, tf)
        if not train or not test:
            continue

        if args.trigger_variant == "AUTO":
            trig_v, tn, tnn, ti, tin = choose_variant(train, args.trigger, args.min_samples)
        else:
            trig_v = args.trigger_variant
            _, tn, tnn, ti, tin = choose_variant(train, args.trigger, args.min_samples)

        selected: list[dict] = []
        print(f"\nTF={tf}m TRIGGER={args.trigger}:{trig_v} TRAIN_TRIGGER N={tn:.2f}%/n{tnn} I={ti:.2f}%/n{tin}")
        print("LEVEL SET TEST_ACC/BASE TEST_N BASE_N OPPORTUNITIES")

        for level in range(args.levels + 1):
            opportunities, signals, correct, base_n, base_correct = evaluate_level(test, args.trigger, trig_v, selected)
            label = f"{args.trigger}:{trig_v}" if not selected else f"{args.trigger}:{trig_v}+" + "+".join(
                f"{x['rule']}:{x['variant']}:{x['mode'][0]}" for x in selected
            )
            test_acc = pct(correct, signals)
            base_acc = pct(base_correct, base_n)
            train_desc = ";".join(f"{x['rule']}:{x['variant']}:{x['mode']}:{x['acc']:.2f}%/n{x['n']}" for x in selected)
            print(f"L{level} {label:<48} TEST {test_acc:6.2f}%/{signals:<5} BASE {base_acc:6.2f}%/{base_n:<5} {opportunities}")
            rows.append({
                "timeframe_min": tf,
                "level": level,
                "trigger_rule": args.trigger,
                "trigger_variant": trig_v,
                "set": label,
                "train_selected": train_desc,
                "test_accuracy_pct": round(test_acc, 6),
                "test_signals": signals,
                "test_correct": correct,
                "horizon_baseline_accuracy_pct": round(base_acc, 6),
                "horizon_baseline_n": base_n,
                "horizon_baseline_correct": base_correct,
                "test_opportunities": opportunities,
            })

            if level == args.levels:
                break
            ranked = choose_next(train, args.trigger, trig_v, selected, args.min_samples)
            print("CANDIDATES " + " ".join(
                f"{x['rule']}:{x['variant']}:{x['mode'][0]}:{x['acc']:.2f}%/n{x['n']}" for x in ranked[:10]
            ))
            if not ranked:
                break
            selected.append(ranked[0])

        print(f"TRIGGER_VARIANT {args.trigger}={trig_v}")

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys()) if rows else ["timeframe_min"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"SAVED {p} rows={len(rows)}")


if __name__ == "__main__":
    main()
