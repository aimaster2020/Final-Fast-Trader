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


def raw_sign(c: Candle, rule: str) -> int:
    value = components(c)[rule]
    threshold = TESTS[rule]
    if value >= threshold:
        return 1
    if value <= -threshold:
        return -1
    return 0


def next_direction(candles: list[Candle], i: int) -> int:
    if i + 1 >= len(candles):
        return 0
    if candles[i + 1].close > candles[i].close:
        return 1
    if candles[i + 1].close < candles[i].close:
        return -1
    return 0


def pct(correct: int, n: int) -> float:
    return 100.0 * correct / n if n else 0.0


def load_sets(root: Path, symbols: list[str], month: str, tf: int) -> list[list[Candle]]:
    out: list[list[Candle]] = []
    for symbol in symbols:
        path = find_month_file(root, symbol, month)
        if not path:
            continue
        candles = resample(load_binance(path), tf)
        if len(candles) > 20:
            out.append(candles)
    return out


def choose_orientation(train: list[list[Candle]], rule: str) -> tuple[int, float, float, int]:
    normal_correct = inverse_correct = active = 0
    for candles in train:
        for i in range(len(candles) - 1):
            s = raw_sign(candles[i], rule)
            y = next_direction(candles, i)
            if not s or not y:
                continue
            active += 1
            normal_correct += int(s == y)
            inverse_correct += int(-s == y)
    normal_acc = pct(normal_correct, active)
    inverse_acc = pct(inverse_correct, active)
    orientation = 1 if normal_acc >= inverse_acc else -1
    return orientation, normal_acc, inverse_acc, active


def selected_sign(candles: list[Candle], i: int, rule: str, orientation: int) -> int:
    return raw_sign(candles[i], rule) * orientation


def evaluate_rules(test: list[list[Candle]], orientations: dict[str, int]) -> dict[str, dict[str, int]]:
    """Evaluate the already-frozen orientation for every R on the test set."""
    missing = [r for r in RULES if r not in orientations]
    if missing:
        raise ValueError(f"Missing frozen orientations: {', '.join(missing)}")

    stats = {
        r: {
            "native_active": 0,
            "native_correct": 0,
            "inverse_correct": 0,
            "selected_active": 0,
            "selected_correct": 0,
            "selected_long": 0,
            "selected_short": 0,
        }
        for r in RULES
    }

    for candles in test:
        for i in range(len(candles) - 1):
            y = next_direction(candles, i)
            if not y:
                continue
            for r in RULES:
                s = raw_sign(candles[i], r)
                if not s:
                    continue
                st = stats[r]
                st["native_active"] += 1
                st["native_correct"] += int(s == y)
                st["inverse_correct"] += int(-s == y)

                selected = s * orientations[r]
                st["selected_active"] += 1
                st["selected_correct"] += int(selected == y)
                st["selected_long"] += int(selected == 1)
                st["selected_short"] += int(selected == -1)

    return stats


def vote_dataset(
    candles: list[Candle],
    orientations: dict[str, int],
    min_active: int,
) -> tuple[list[dict], dict[str, int]]:
    missing = [r for r in RULES if r not in orientations]
    if missing:
        raise ValueError(f"Missing frozen orientations: {', '.join(missing)}")

    rows: list[dict] = []
    decision = {"LONG": [0, 0], "SHORT": [0, 0], "HOLD": [0, 0], "ALL": [0, 0]}

    for i in range(len(candles) - 1):
        y = next_direction(candles, i)
        if not y:
            continue

        per_rule: dict[str, int] = {}
        long_score = short_score = 0
        active = 0

        for r in RULES:
            s = selected_sign(candles, i, r, orientations[r])
            per_rule[r] = s
            if not s:
                continue
            active += 1
            if s == 1:
                long_score += 1
            else:
                short_score += 1

        if active < min_active:
            final = 0
        elif long_score > short_score:
            final = 1
        elif short_score > long_score:
            final = -1
        else:
            final = 0

        name = "LONG" if final == 1 else "SHORT" if final == -1 else "HOLD"

        if final:
            decision[name][0] += 1
            decision[name][1] += int(final == y)
            decision["ALL"][0] += 1
            decision["ALL"][1] += int(final == y)
        else:
            decision["HOLD"][0] += 1

        rows.append(
            {
                "bar_index": i,
                "realized_next": "LONG" if y == 1 else "SHORT",
                "long_score": long_score,
                "short_score": short_score,
                "active_rules": active,
                "decision": name,
                **{f"{r}_score": s for r, s in per_rule.items()},
                **{
                    f"{r}_decision": "LONG" if s == 1 else "SHORT" if s == -1 else "HOLD"
                    for r, s in per_rule.items()
                },
            }
        )

    return rows, {
        "long_n": decision["LONG"][0],
        "long_correct": decision["LONG"][1],
        "short_n": decision["SHORT"][0],
        "short_correct": decision["SHORT"][1],
        "hold_n": decision["HOLD"][0],
        "all_n": decision["ALL"][0],
        "all_correct": decision["ALL"][1],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Walk-forward dual-variant R voting: every rule has Native and Inverse "
            "candidates; one orientation is frozen from training, all selected rule "
            "votes are scored equally on test."
        )
    )
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--train-month", default="2026-06")
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--timeframes", default="30,60,120,240")
    ap.add_argument("--min-active", type=int, default=1)
    ap.add_argument("--output", default="reports/july_r_dual_variant_vote_wf_v2.csv")
    ap.add_argument("--rule-output", default="reports/july_r_dual_variant_rule_stats_v2.csv")
    args = ap.parse_args()

    root = Path(args.input_dir)
    tfs = [int(x.strip()) for x in args.timeframes.split(",") if x.strip()]
    if args.symbols.upper() == "ALL":
        symbols = sorted(
            set(discover_symbols(root, args.train_month))
            | set(discover_symbols(root, args.test_month))
        )
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]

    all_bar_rows: list[dict] = []
    rule_rows: list[dict] = []

    print(
        f"R_DUAL_VARIANT_VOTE_WF train={args.train_month} test={args.test_month} "
        f"assets={len(symbols)} tfs={','.join(map(str, tfs))}"
    )
    print(
        "Each R has NATIVE and INVERSE. Training freezes the better orientation; "
        "Test scores only that selected variant equally in the vote."
    )

    for tf in tfs:
        train = load_sets(root, symbols, args.train_month, tf)
        test = load_sets(root, symbols, args.test_month, tf)
        if not train or not test:
            continue

        orientations: dict[str, int] = {}
        print(f"\n===== TF={tf}m =====")
        print("RULE NATIVE_ACC INV_ACC SELECT N TRAIN_SELECTED_ACC")

        # Freeze ALL 16 orientations first. Never evaluate a partial map.
        train_meta: dict[str, tuple[float, float, int]] = {}
        for r in RULES:
            o, nacc, iacc, n = choose_orientation(train, r)
            orientations[r] = o
            train_meta[r] = (nacc, iacc, n)

        # Test every R only after all 16 orientations are available.
        stats = evaluate_rules(test, orientations)
        print("RULE NATIVE_ACC INV_ACC SELECT N TRAIN_SELECTED_ACC TEST_ACTIVE TEST_ACC LONG SHORT")
        for r in RULES:
            nacc, iacc, n = train_meta[r]
            o = orientations[r]
            selected_name = "N" if o == 1 else "I"
            st = stats[r]
            selected_acc = nacc if o == 1 else iacc
            print(
                f"{r:<4} {nacc:7.2f} {iacc:7.2f} {selected_name:^6} {n:5} "
                f"{selected_acc:16.2f} {st['selected_active']:10} "
                f"{pct(st['selected_correct'], st['selected_active']):7.2f} "
                f"{st['selected_long']:4} {st['selected_short']:5}"
            )
            rule_rows.append(
                {
                    "timeframe_min": tf,
                    "rule": r,
                    "train_native_acc_pct": round(nacc, 6),
                    "train_inverse_acc_pct": round(iacc, 6),
                    "selected_orientation": selected_name,
                    "train_active": n,
                    "test_active": st["selected_active"],
                    "test_selected_acc_pct": round(
                        pct(st["selected_correct"], st["selected_active"]), 6
                    ),
                    "test_selected_correct": st["selected_correct"],
                    "test_selected_wrong": st["selected_active"] - st["selected_correct"],
                    "test_native_acc_pct": round(
                        pct(st["native_correct"], st["native_active"]), 6
                    ),
                    "test_inverse_acc_pct": round(
                        pct(st["inverse_correct"], st["native_active"]), 6
                    ),
                    "test_long": st["selected_long"],
                    "test_short": st["selected_short"],
                }
            )

        print(
            "ORIENTATION "
            + " ".join(f"{r}={'N' if orientations[r] == 1 else 'I'}" for r in RULES)
        )

        agg = {k: [0, 0] for k in ("LONG", "SHORT", "HOLD", "ALL")}
        for symbol in symbols:
            path = find_month_file(root, symbol, args.test_month)
            if not path:
                continue
            candles = resample(load_binance(path), tf)
            if len(candles) <= 20:
                continue
            rows, st = vote_dataset(candles, orientations, args.min_active)
            for row in rows:
                row["symbol"] = symbol
                row["timeframe_min"] = tf
            all_bar_rows.extend(rows)
            agg["LONG"][0] += st["long_n"]
            agg["LONG"][1] += st["long_correct"]
            agg["SHORT"][0] += st["short_n"]
            agg["SHORT"][1] += st["short_correct"]
            agg["HOLD"][0] += st["hold_n"]
            agg["ALL"][0] += st["all_n"]
            agg["ALL"][1] += st["all_correct"]

        print("VOTE LONG_N LONG_ACC SHORT_N SHORT_ACC HOLD ALL_N ALL_ACC")
        print(
            f"VOTE {agg['LONG'][0]:6} "
            f"{pct(agg['LONG'][1], agg['LONG'][0]):8.2f} "
            f"{agg['SHORT'][0]:7} "
            f"{pct(agg['SHORT'][1], agg['SHORT'][0]):9.2f} "
            f"{agg['HOLD'][0]:5} {agg['ALL'][0]:5} "
            f"{pct(agg['ALL'][1], agg['ALL'][0]):7.2f}"
        )

    rp = Path(args.rule_output)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with rp.open("w", newline="", encoding="utf-8") as f:
        fields = list(rule_rows[0].keys()) if rule_rows else ["timeframe_min", "rule"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rule_rows)

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        fields = list(all_bar_rows[0].keys()) if all_bar_rows else ["symbol", "timeframe_min", "bar_index", "decision"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_bar_rows)

    print(f"SAVED {p} rows={len(all_bar_rows)}")
    print(f"SAVED {rp} rows={len(rule_rows)}")


if __name__ == "__main__":
    main()
