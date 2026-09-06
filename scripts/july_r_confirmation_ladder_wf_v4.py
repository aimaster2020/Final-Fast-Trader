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


def sig(c: Candle, rule: str) -> int:
    v = components(c)[rule]
    th = TESTS[rule]
    return 1 if v >= th else -1 if v <= -th else 0


def nxt(cs: list[Candle], i: int) -> int:
    if i + 1 >= len(cs):
        return 0
    return 1 if cs[i + 1].close > cs[i].close else -1 if cs[i + 1].close < cs[i].close else 0


def pct(c: int, n: int) -> float:
    return 100.0 * c / n if n else 0.0


def load_sets(root: Path, symbols: list[str], month: str, tf: int) -> list[list[Candle]]:
    out = []
    for symbol in symbols:
        p = find_month_file(root, symbol, month)
        if not p:
            continue
        cs = resample(load_binance(p), tf)
        if len(cs) > 32:
            out.append(cs)
    return out


def orientation(train: list[list[Candle]], rule: str) -> int:
    normal = inverse = 0
    for cs in train:
        for i in range(len(cs) - 1):
            s, y = sig(cs[i], rule), nxt(cs, i)
            if not s or not y:
                continue
            normal += int(s == y)
            inverse += int(-s == y)
    return 1 if normal >= inverse else -1


def pass_chain(
    cs: list[Candle],
    entry_i: int,
    trigger_rule: str,
    selected: list[dict],
    ori: dict[str, int],
) -> tuple[bool, int]:
    t = sig(cs[entry_i], trigger_rule)
    if not t:
        return False, 0
    trigger = t * ori[trigger_rule]
    for k, x in enumerate(selected, 1):
        j = entry_i + k
        if j >= len(cs):
            return False, trigger
        s = sig(cs[j], x["rule"])
        if not s:
            return False, trigger
        economic = s * ori[x["rule"]]
        expected = trigger if x["mode"] == "SAME" else -trigger
        if economic != expected:
            return False, trigger
    return True, trigger


def candidate_stats(
    train: list[list[Candle]],
    trigger_rule: str,
    selected: list[dict],
    rule: str,
    ori: dict[str, int],
) -> dict[str, dict[str, float | int]]:
    """Find whether the next R is useful as SAME or OPPOSITE confirmation.

    The candidate is evaluated on its own future candle, so a signal at
    entry+depth+1 predicts the move from that candle to the next candle.
    """
    out = {
        "SAME": {"correct": 0, "n": 0},
        "OPPOSITE": {"correct": 0, "n": 0},
    }
    depth = len(selected)
    for cs in train:
        for i in range(len(cs) - depth - 2):
            ok, trigger = pass_chain(cs, i, trigger_rule, selected, ori)
            if not ok:
                continue
            j = i + depth + 1
            s = sig(cs[j], rule)
            y = nxt(cs, j)
            if not s or not y:
                continue
            economic = s * ori[rule]
            mode = "SAME" if economic == trigger else "OPPOSITE"
            out[mode]["n"] += 1
            out[mode]["correct"] += int(economic == y)
    for mode in out:
        out[mode]["acc"] = pct(int(out[mode]["correct"]), int(out[mode]["n"]))
    return out


def choose_next(
    train: list[list[Candle]],
    trigger_rule: str,
    selected: list[dict],
    ori: dict[str, int],
    min_n: int,
) -> list[dict]:
    ranked: list[dict] = []
    for rule in RULES:
        if rule == trigger_rule or any(x["rule"] == rule for x in selected):
            continue
        stats = candidate_stats(train, trigger_rule, selected, rule, ori)
        for mode in ("SAME", "OPPOSITE"):
            n = int(stats[mode]["n"])
            if n < min_n:
                continue
            ranked.append({
                "rule": rule,
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
    selected: list[dict],
    ori: dict[str, int],
) -> tuple[int, int, int, int, int]:
    """Return opportunities, chain signals/correct, and horizon-matched trigger baseline."""
    depth = len(selected)
    opportunities = signals = correct = base_correct = base_n = 0
    for cs in test:
        # +2 is needed because the final prediction candle must have a next candle.
        for i in range(len(cs) - depth - 2):
            t = sig(cs[i], trigger_rule)
            if not t:
                continue
            opportunities += 1
            trigger = t * ori[trigger_rule]
            final_i = i + depth + 1
            realized = nxt(cs, final_i)
            if not realized:
                continue

            # Horizon-matched baseline: keep the original R1 direction until
            # the same outcome horizon as the confirmation chain.
            base_n += 1
            base_correct += int(trigger == realized)

            ok, chain_trigger = pass_chain(cs, i, trigger_rule, selected, ori)
            if not ok:
                continue
            signals += 1
            correct += int(chain_trigger == realized)
    return opportunities, signals, correct, base_n, base_correct


def main() -> None:
    ap = argparse.ArgumentParser(description="Causal sequential R confirmation ladder with horizon-matched baselines")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--train-month", default="2026-06")
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--timeframes", default="30,60,120,240")
    ap.add_argument("--trigger", default="R1")
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--output", default="reports/july_r_confirmation_ladder_wf_v4.csv")
    args = ap.parse_args()

    root = Path(args.input_dir)
    tfs = [int(x.strip()) for x in args.timeframes.split(",") if x.strip()]
    if args.symbols.upper() == "ALL":
        symbols = sorted(set(discover_symbols(root, args.train_month)) | set(discover_symbols(root, args.test_month)))
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]

    rows: list[dict] = []
    print(
        f"R_LADDER_WF_V4 train={args.train_month} test={args.test_month} "
        f"assets={len(symbols)} tfs={','.join(map(str,tfs))} min_n={args.min_samples}"
    )

    for tf in tfs:
        train = load_sets(root, symbols, args.train_month, tf)
        test = load_sets(root, symbols, args.test_month, tf)
        if not train or not test:
            continue

        ori = {r: orientation(train, r) for r in RULES}
        selected: list[dict] = []

        print(f"\nTF={tf}m")
        print("LEVEL RULE MODE TRAIN_ACC TRAIN_N TEST_ACC TEST_N BASE_H_ACC BASE_H_N OPPORTUNITIES")

        for level in range(args.levels + 1):
            opportunities, signals, correct, base_n, base_correct = evaluate_level(
                test, args.trigger, selected, ori
            )
            label = args.trigger if not selected else args.trigger + "+" + "+".join(
                x["rule"] + ":" + x["mode"][0] for x in selected
            )
            base_acc = pct(base_correct, base_n)
            test_acc = pct(correct, signals)
            print(
                f"L{level} {label:<30} {test_acc:6.2f}% {signals:<7} "
                f"{base_acc:7.2f}% {base_n:<6} {opportunities}"
            )
            rows.append({
                "timeframe_min": tf,
                "level": level,
                "set": label,
                "train_selected": ";".join(f"{x['rule']}:{x['mode']}:{x['acc']:.4f}:{x['n']}" for x in selected),
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

            ranked = choose_next(train, args.trigger, selected, ori, args.min_samples)
            print(
                "CANDIDATES " + " ".join(
                    f"{x['rule']}:{x['mode'][0]}:{x['acc']:.2f}%/n{x['n']}" for x in ranked[:10]
                )
            )
            if not ranked:
                break
            selected.append(ranked[0])

        print("ORIENTATION " + " ".join(f"{r}={'N' if ori[r] == 1 else 'I'}" for r in RULES))

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
