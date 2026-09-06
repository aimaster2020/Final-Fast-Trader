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


def raw_sign(candle: Candle, rule: str) -> int:
    v = components(candle)[rule]
    threshold = TESTS[rule]
    return 1 if v >= threshold else -1 if v <= -threshold else 0


def next_direction(candles: list[Candle], i: int) -> int:
    if i + 1 >= len(candles):
        return 0
    if candles[i + 1].close > candles[i].close:
        return 1
    if candles[i + 1].close < candles[i].close:
        return -1
    return 0


def sign_accuracy(pairs: list[tuple[int, int]]) -> tuple[float, int, int]:
    pairs = [(a, y) for a, y in pairs if a and y]
    n = len(pairs)
    correct = sum(int(a == y) for a, y in pairs)
    return (100.0 * correct / n if n else 0.0, n, correct)


def collect_pair_stats(
    candles: list[Candle],
    trigger_rule: str,
    candidate: str,
    orientations: dict[str, int] | None = None,
) -> dict[str, tuple[float, int, int]]:
    same: list[tuple[int, int]] = []
    opposite: list[tuple[int, int]] = []
    cand_all: list[tuple[int, int]] = []
    trigger_all: list[tuple[int, int]] = []

    for i in range(len(candles) - 1):
        trigger = raw_sign(candles[i], trigger_rule)
        cand = raw_sign(candles[i], candidate)
        y = next_direction(candles, i)
        if not trigger or not cand or not y:
            continue
        if orientations:
            trigger *= orientations[trigger_rule]
            cand *= orientations[candidate]
        trigger_all.append((trigger, y))
        cand_all.append((cand, y))
        (same if cand == trigger else opposite).append((cand, y))

    return {
        "trigger": sign_accuracy(trigger_all),
        "candidate": sign_accuracy(cand_all),
        "same": sign_accuracy(same),
        "opposite": sign_accuracy(opposite),
    }


def choose_orientations(train_sets: list[list[Candle]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rule in RULES:
        normal: list[tuple[int, int]] = []
        inverse: list[tuple[int, int]] = []
        for candles in train_sets:
            for i in range(len(candles) - 1):
                s = raw_sign(candles[i], rule)
                y = next_direction(candles, i)
                if not s or not y:
                    continue
                normal.append((s, y))
                inverse.append((-s, y))
        na, nn, _ = sign_accuracy(normal)
        ia, _, _ = sign_accuracy(inverse)
        out[rule] = 1 if na >= ia else -1
        print(
            f"ORIENT {rule}={'N' if out[rule] == 1 else 'I'} "
            f"normal={na:.2f}% inverse={ia:.2f}% n={nn}"
        )
    return out


def rank_candidates(
    train_sets: list[list[Candle]],
    trigger: str,
    orientations: dict[str, int],
    min_pair_samples: int,
) -> list[dict]:
    ranked: list[dict] = []
    for candidate in RULES:
        if candidate == trigger:
            continue
        same_all: list[tuple[int, int]] = []
        opposite_all: list[tuple[int, int]] = []
        trigger_all: list[tuple[int, int]] = []
        candidate_all: list[tuple[int, int]] = []
        for candles in train_sets:
            stats = collect_pair_stats(candles, trigger, candidate, orientations)
            for key, target in (
                ("same", same_all),
                ("opposite", opposite_all),
                ("trigger", trigger_all),
                ("candidate", candidate_all),
            ):
                # collect_pair_stats returns only aggregate values, so rebuild
                # the pair observations here to preserve exact sample counts.
                for i in range(len(candles) - 1):
                    t = raw_sign(candles[i], trigger) * orientations[trigger]
                    c = raw_sign(candles[i], candidate) * orientations[candidate]
                    y = next_direction(candles, i)
                    if not t or not c or not y:
                        continue
                    if key == "same" and c == t:
                        target.append((c, y))
                    elif key == "opposite" and c != t:
                        target.append((c, y))
                    elif key == "trigger":
                        target.append((t, y))
                    elif key == "candidate":
                        target.append((c, y))
        same_acc, same_n, _ = sign_accuracy(same_all)
        opp_acc, opp_n, _ = sign_accuracy(opposite_all)
        trig_acc, trig_n, _ = sign_accuracy(trigger_all)
        cand_acc, cand_n, _ = sign_accuracy(candidate_all)
        best_n = max(same_n, opp_n)
        if best_n < min_pair_samples:
            continue
        # Rank by best conditional predictive accuracy, then sample count.
        best_acc = max(same_acc, opp_acc)
        best_mode = "SAME" if same_acc >= opp_acc else "OPPOSITE"
        ranked.append(
            {
                "rule": candidate,
                "best_acc": best_acc,
                "best_mode": best_mode,
                "same_acc": same_acc,
                "same_n": same_n,
                "opposite_acc": opp_acc,
                "opposite_n": opp_n,
                "trigger_acc": trig_acc,
                "trigger_n": trig_n,
                "candidate_acc": cand_acc,
                "candidate_n": cand_n,
            }
        )
    return sorted(ranked, key=lambda x: (x["best_acc"], max(x["same_n"], x["opposite_n"])), reverse=True)


def conditional_sequence(
    candles: list[Candle],
    trigger: str,
    selected: list[dict],
    orientations: dict[str, int],
) -> tuple[int, int, int]:
    """Strict sequential confirmation, but each selected R can be SAME or OPPOSITE.

    The mode for every confirmation is frozen from TRAIN. A test entry survives
    only when the candidate at its assigned future bar is active and matches the
    frozen relationship with the original trigger. Outcome is the candle after
    the final confirmation.
    """
    if not selected:
        selected = []
    depth = len(selected)
    opportunities = accepted = correct = 0

    for entry_i in range(len(candles) - depth - 1):
        trigger_native = raw_sign(candles[entry_i], trigger)
        if not trigger_native:
            continue
        trig = trigger_native * orientations[trigger]
        opportunities += 1
        ok = True
        for k, item in enumerate(selected, start=1):
            s = raw_sign(candles[entry_i + k], item["rule"])
            if not s:
                ok = False
                break
            econ = s * orientations[item["rule"]]
            if item["mode"] == "SAME":
                if econ != trig:
                    ok = False
                    break
            else:
                if econ != -trig:
                    ok = False
                    break
        if not ok:
            continue
        final_i = entry_i + depth
        y = next_direction(candles, final_i)
        if not y:
            continue
        accepted += 1
        correct += int(y == trig)

    return opportunities, accepted, correct


def load_sets(input_dir: Path, symbols: list[str], month: str, tf: int) -> list[list[Candle]]:
    out: list[list[Candle]] = []
    for symbol in symbols:
        path = find_month_file(input_dir, symbol, month)
        if not path:
            continue
        candles = resample(load_binance(path), tf)
        if len(candles) > 20:
            out.append(candles)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward R1 pairwise SAME/OPPOSITE confirmation test")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--train-month", default="2026-06")
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--timeframes", default="30,60,120,240")
    ap.add_argument("--trigger", default="R1")
    ap.add_argument("--min-pair-samples", type=int, default=100)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--output", default="reports/july_r_pairwise_confirmation_wf.csv")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    tfs = [int(x.strip()) for x in args.timeframes.split(",") if x.strip()]
    if args.symbols.upper() == "ALL":
        symbols = sorted(set(discover_symbols(input_dir, args.train_month)) | set(discover_symbols(input_dir, args.test_month)))
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]

    output_rows: list[dict] = []
    print(
        f"R_PAIR_WF train={args.train_month} test={args.test_month} assets={len(symbols)} "
        f"tfs={','.join(map(str,tfs))} trigger={args.trigger} min_pair_n={args.min_pair_samples}"
    )

    for tf in tfs:
        train_sets = load_sets(input_dir, symbols, args.train_month, tf)
        test_sets = load_sets(input_dir, symbols, args.test_month, tf)
        if not train_sets or not test_sets:
            continue

        orientations = choose_orientations(train_sets)
        ranked = rank_candidates(train_sets, args.trigger, orientations, args.min_pair_samples)
        print(f"\nTF={tf}m")
        print("PAIR_RANK RULE BEST_MODE BEST_ACC SAME_ACC/N OPP_ACC/N")
        for item in ranked[:10]:
            print(
                f"{item['rule']:<7} {item['best_mode']:<10} {item['best_acc']:6.2f}% "
                f"{item['same_acc']:6.2f}/{item['same_n']:<4} "
                f"{item['opposite_acc']:6.2f}/{item['opposite_n']:<4}"
            )

        selected: list[dict] = []
        print("LEVEL SET ACC% SIGNALS CORRECT OPPORTUNITIES")
        for level in range(0, min(args.levels, len(ranked)) + 1):
            if level == 0:
                confs: list[dict] = []
            else:
                # Greedy conditional selection is applied on TRAIN by the pairwise
                # score only; TEST remains untouched until final evaluation.
                item = ranked[level - 1]
                mode = item["best_mode"]
                selected.append({"rule": item["rule"], "mode": mode})
                confs = selected.copy()

            total_opp = total_sig = total_correct = 0
            for candles in test_sets:
                opp, sig, cor = conditional_sequence(candles, args.trigger, confs, orientations)
                total_opp += opp
                total_sig += sig
                total_correct += cor
            accuracy = 100.0 * total_correct / total_sig if total_sig else 0.0
            label = args.trigger if not confs else args.trigger + "+" + "+".join(
                f"{x['rule']}:{x['mode'][0]}" for x in confs
            )
            print(f"L{level} {label:<28} {accuracy:6.2f}% {total_sig:<7} {total_correct:<7} {total_opp}")
            output_rows.append(
                {
                    "timeframe_min": tf,
                    "level": level,
                    "set": label,
                    "test_opportunities": total_opp,
                    "test_signals": total_sig,
                    "test_correct": total_correct,
                    "test_accuracy_pct": round(accuracy, 6),
                    "orientations": ";".join(f"{r}:{'N' if orientations[r] == 1 else 'I'}" for r in RULES),
                    "train_pair_rank": ";".join(
                        f"{x['rule']}:{x['best_mode']}:{x['best_acc']:.4f}:{x['same_n']}:{x['opposite_n']}" for x in ranked[:10]
                    ),
                }
            )

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        fields = list(output_rows[0].keys()) if output_rows else [
            "timeframe_min", "level", "set", "test_opportunities", "test_signals", "test_correct",
            "test_accuracy_pct", "orientations", "train_pair_rank"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(output_rows)
    print(f"SAVED {p} rows={len(output_rows)}")


if __name__ == "__main__":
    main()
