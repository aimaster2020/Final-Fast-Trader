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
    out: list[list[Candle]] = []
    for symbol in symbols:
        p = find_month_file(root, symbol, month)
        if not p:
            continue
        cs = resample(load_binance(p), tf)
        if len(cs) > 35:
            out.append(cs)
    return out


def train_orientation(train: list[list[Candle]], rule: str, min_n: int) -> int:
    normal = inverse = total = 0
    for cs in train:
        for i in range(len(cs) - 1):
            s = sig(cs[i], rule)
            y = nxt(cs, i)
            if not s or not y:
                continue
            total += 1
            normal += int(s == y)
            inverse += int(-s == y)
    if total < min_n:
        return 1
    return 1 if normal >= inverse else -1


def conditional_pair_stats(
    train: list[list[Candle]],
    trigger: str,
    candidate: str,
    trigger_ori: int,
    candidate_ori: int,
    min_n: int,
) -> list[dict]:
    result: list[dict] = []
    # The trigger fires at t. Candidate confirmation is evaluated at t+1.
    # Outcome is the next candle after the confirmation: t+2.
    for trigger_mode in ("SAME", "OPPOSITE"):
        for candidate_mode in ("NATIVE", "INVERSE"):
            n = correct = 0
            for cs in train:
                for i in range(len(cs) - 2):
                    t = sig(cs[i], trigger)
                    c = sig(cs[i + 1], candidate)
                    y = nxt(cs, i + 1)
                    if not t or not c or not y:
                        continue
                    te = t * trigger_ori
                    ce = c * candidate_ori
                    expected = te if trigger_mode == "SAME" else -te
                    if (candidate_mode == "INVERSE"):
                        ce = -ce
                    # candidate_mode is deliberately explicit even though ori is frozen;
                    # this lets every native/inverse relation be tested.
                    if ce != expected:
                        continue
                    n += 1
                    correct += int(expected == y)
            if n >= min_n:
                result.append({
                    "candidate": candidate,
                    "trigger_mode": trigger_mode,
                    "orientation": candidate_mode,
                    "acc": pct(correct, n),
                    "n": n,
                    "correct": correct,
                })
    return result


def choose_next(
    train: list[list[Candle]],
    trigger: str,
    selected: list[dict],
    orientations: dict[str, int],
    min_n: int,
    top: int,
) -> list[dict]:
    ranked: list[dict] = []
    for rule in RULES:
        if rule == trigger or any(x["rule"] == rule for x in selected):
            continue
        # Sequential selection: candidate is evaluated only on rows passing the
        # already-selected chain.
        depth = len(selected)
        modes = ("SAME", "OPPOSITE")
        for relation in modes:
            for orient_name, orient_mult in (("N", 1), ("I", -1)):
                n = correct = 0
                for cs in train:
                    for i in range(len(cs) - depth - 2):
                        t = sig(cs[i], trigger)
                        if not t:
                            continue
                        te = t * orientations[trigger]
                        ok = True
                        for k, x in enumerate(selected, 1):
                            s = sig(cs[i + k], x["rule"])
                            if not s:
                                ok = False
                                break
                            ce = s * orientations[x["rule"]]
                            if x["orientation"] == "I":
                                ce = -ce
                            expected = te if x["relation"] == "SAME" else -te
                            if ce != expected:
                                ok = False
                                break
                        if not ok:
                            continue
                        c = sig(cs[i + depth + 1], rule)
                        y = nxt(cs, i + depth + 1)
                        if not c or not y:
                            continue
                        ce = c * orientations[rule] * orient_mult
                        expected = te if relation == "SAME" else -te
                        n += 1
                        correct += int(ce == y)
                if n >= min_n:
                    ranked.append({
                        "rule": rule,
                        "relation": relation,
                        "orientation": orient_name,
                        "train_acc": pct(correct, n),
                        "train_n": n,
                    })
    ranked.sort(key=lambda x: (x["train_acc"], x["train_n"]), reverse=True)
    return ranked[:top]


def eval_chain(
    test: list[list[Candle]], trigger: str, selected: list[dict], orientations: dict[str, int]
) -> tuple[int, int, int, int, int]:
    depth = len(selected)
    opportunities = signals = correct = long_count = short_count = 0
    for cs in test:
        for i in range(len(cs) - depth - 2):
            t = sig(cs[i], trigger)
            if not t:
                continue
            opportunities += 1
            te = t * orientations[trigger]
            ok = True
            for k, x in enumerate(selected, 1):
                s = sig(cs[i + k], x["rule"])
                if not s:
                    ok = False
                    break
                ce = s * orientations[x["rule"]]
                if x["orientation"] == "I":
                    ce = -ce
                expected = te if x["relation"] == "SAME" else -te
                if ce != expected:
                    ok = False
                    break
            if not ok:
                continue
            y = nxt(cs, i + depth + 1)
            if not y:
                continue
            signals += 1
            correct += int(te == y)
            long_count += int(te == 1)
            short_count += int(te == -1)
    return opportunities, signals, correct, long_count, short_count


def trigger_baseline(test: list[list[Candle]], trigger: str, orientations: dict[str, int], depth: int) -> tuple[int, int, int, int]:
    opportunities = signals = correct = 0
    for cs in test:
        for i in range(len(cs) - depth - 2):
            t = sig(cs[i], trigger)
            if not t:
                continue
            y = nxt(cs, i + depth + 1)
            if not y:
                continue
            opportunities += 1
            pred = t * orientations[trigger]
            signals += 1
            correct += int(pred == y)
    return opportunities, signals, correct, depth


def main() -> None:
    ap = argparse.ArgumentParser(description="All-R-trigger sequential confirmation walk-forward test")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--train-month", default="2026-06")
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--timeframes", default="30,60,120,240")
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--output", default="reports/july_r_all_trigger_confirmation_wf_v5.csv")
    args = ap.parse_args()

    root = Path(args.input_dir)
    tfs = [int(x.strip()) for x in args.timeframes.split(",") if x.strip()]
    if args.symbols.upper() == "ALL":
        symbols = sorted(set(discover_symbols(root, args.train_month)) | set(discover_symbols(root, args.test_month)))
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]

    rows: list[dict] = []
    print(f"R_ALL_WF_V5 train={args.train_month} test={args.test_month} assets={len(symbols)} tfs={','.join(map(str,tfs))} min_n={args.min_n}")

    for tf in tfs:
        train = load_sets(root, symbols, args.train_month, tf)
        test = load_sets(root, symbols, args.test_month, tf)
        if not train or not test:
            continue
        print(f"\n===== TF={tf}m =====")
        trigger_rank: list[tuple[str, float, int]] = []
        orientations: dict[str, int] = {}
        for r in RULES:
            orientations[r] = train_orientation(train, r, args.min_n)
            # Same one-step accuracy using frozen orientation, for trigger ranking.
            c = n = 0
            for cs in train:
                for i in range(len(cs) - 1):
                    s = sig(cs[i], r); y = nxt(cs, i)
                    if not s or not y:
                        continue
                    n += 1; c += int(s * orientations[r] == y)
            if n >= args.min_n:
                trigger_rank.append((r, pct(c, n), n))
        trigger_rank.sort(key=lambda x: (x[1], x[2]), reverse=True)
        print("TRIGGER_RANK " + " ".join(f"{r}:{a:.2f}%/n{n}" for r,a,n in trigger_rank))
        print("TRIGGER LEVEL SET ACC SIGNALS CORRECT LONG SHORT BASE_H_ACC BASE_H_N")

        for trigger, train_acc, train_n in trigger_rank:
            selected: list[dict] = []
            for level in range(args.levels + 1):
                opp, sigs, corr, lng, sht = eval_chain(test, trigger, selected, orientations)
                bop, bsn, bcr, _ = trigger_baseline(test, trigger, orientations, level)
                label = trigger if not selected else trigger + "+" + "+".join(
                    x["rule"] + ":" + x["relation"][0] + x["orientation"] for x in selected
                )
                print(f"{trigger} L{level} {label:<35} {pct(corr,sigs):6.2f}% {sigs:<6} {corr:<6} {lng:<5} {sht:<5} {pct(bcr,bsn):6.2f}% {bsn}")
                rows.append({
                    "timeframe_min": tf,
                    "trigger": trigger,
                    "level": level,
                    "set": label,
                    "test_accuracy_pct": round(pct(corr, sigs), 6),
                    "test_signals": sigs,
                    "test_correct": corr,
                    "test_long": lng,
                    "test_short": sht,
                    "test_opportunities": opp,
                    "base_h_accuracy_pct": round(pct(bcr, bsn), 6),
                    "base_h_n": bsn,
                    "train_trigger_accuracy_pct": round(train_acc, 6),
                    "train_trigger_n": train_n,
                    "selected": ";".join(f"{x['rule']}:{x['relation']}:{x['orientation']}:{x['train_acc']:.4f}:{x['train_n']}" for x in selected),
                })
                if level == args.levels:
                    break
                ranked = choose_next(train, trigger, selected, orientations, args.min_n, args.top)
                print("  CAND " + " ".join(f"{x['rule']}:{x['relation'][0]}{x['orientation']}:{x['train_acc']:.2f}%/n{x['train_n']}" for x in ranked))
                if not ranked:
                    break
                selected.append(ranked[0])

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys()) if rows else ["timeframe_min"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"SAVED {p} rows={len(rows)}")


if __name__ == "__main__":
    main()
