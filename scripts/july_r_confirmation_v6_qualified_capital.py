from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_pattern_trader.models import Candle
from scripts.july_r_confirmation_all_triggers_wf_v6 import (
    RULES,
    choose_next,
    choose_variant,
    chain_ok,
    evaluate_level,
    load_sets,
)
from scripts.july_r_composite_walkforward import (
    discover_symbols,
    find_month_file,
    load_binance,
    resample,
)


def pct(correct: int, n: int) -> float:
    return 100.0 * correct / n if n else 0.0


def signal_indices(
    cs: list[Candle], trigger_rule: str, trigger_variant: str, selected: list[dict]
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    depth = len(selected)
    for i in range(len(cs) - depth - 2):
        ok, direction = chain_ok(cs, i, trigger_rule, trigger_variant, selected)
        if ok and direction and i + 1 < len(cs):
            out.append((i, direction))
    return out


def run_capital(
    series: dict[str, list[Candle]],
    trigger_rule: str,
    trigger_variant: str,
    selected: list[dict],
    initial: float,
    fee_per_side_pct: float,
) -> dict:
    capital = initial
    peak = initial
    max_dd = 0.0
    trades = wins = losses = 0
    gross_sum = 0.0
    fees_paid = 0.0

    for _, cs in series.items():
        for i, direction in signal_indices(cs, trigger_rule, trigger_variant, selected):
            entry = cs[i].close
            exit_price = cs[i + 1].close
            if entry <= 0 or exit_price <= 0:
                continue
            move = (exit_price / entry - 1.0) * direction
            gross_pnl = capital * move
            fee = capital * (fee_per_side_pct / 100.0) * 2.0
            capital = max(0.0, capital + gross_pnl - fee)
            trades += 1
            wins += int(gross_pnl > 0)
            losses += int(gross_pnl <= 0)
            gross_sum += move
            fees_paid += fee
            peak = max(peak, capital)
            if peak > 0:
                max_dd = max(max_dd, (peak - capital) / peak)
            if capital <= 0:
                break
        if capital <= 0:
            break

    return {
        "initial_capital": initial,
        "final_capital": capital,
        "return_pct": 100.0 * (capital / initial - 1.0) if initial else 0.0,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": pct(wins, trades),
        "max_dd_pct": 100.0 * max_dd,
        "gross_sum_pct": 100.0 * gross_sum,
        "fees_paid": fees_paid,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-run V6 discovery, retain every July result with TEST_ACC >= threshold, and calculate compounded capital at 0% and 1.3% per side."
    )
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--train-month", default="2026-06")
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--timeframes", default="30,60,120,240")
    ap.add_argument("--threshold", type=float, default=55.0)
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--output", default="reports/july_r_confirmation_v6_qualified_capital.csv")
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

    rows: list[dict] = []
    print(
        f"V6_QUALIFIED_CAPITAL train={args.train_month} test={args.test_month} "
        f"assets={len(symbols)} tfs={','.join(map(str, tfs))} threshold={args.threshold:.1f}% "
        f"initial={args.initial_capital:.2f}"
    )
    print("Qualification uses V6 July TEST_ACC; every qualifying row is retained, including low-N results.")
    print("Capital compounds at 100% position size; results are shown with 0% and 1.3% per-side fees.")
    print("This is retrospective July capital on V6-discovered strategies, not a leakage-free capital walk-forward.")

    for tf in tfs:
        train = load_sets(root, symbols, args.train_month, tf)
        test = load_sets(root, symbols, args.test_month, tf)
        if not train or not test:
            continue

        series: dict[str, list[Candle]] = {}
        for symbol in symbols:
            p = find_month_file(root, symbol, args.test_month)
            if not p:
                continue
            cs = resample(load_binance(p), tf)
            if len(cs) > 32:
                series[symbol] = cs

        qualified: list[dict] = []
        for trig in RULES:
            tv, tn, nn, ti, ni = choose_variant(train, trig, args.min_samples)
            selected: list[dict] = []

            for level in range(args.levels + 1):
                opp, sn, correct, bn, bc = evaluate_level(test, trig, tv, selected)
                test_acc = pct(correct, sn)
                base_acc = pct(bc, bn)

                if test_acc >= args.threshold:
                    label = (
                        f"{trig}:{tv}"
                        if not selected
                        else f"{trig}:{tv}+" + "+".join(
                            f"{x['rule']}:{x['variant']}:{x['mode'][0]}" for x in selected
                        )
                    )
                    qualified.append(
                        {
                            "tf": tf,
                            "trigger_rule": trig,
                            "trigger_variant": tv,
                            "level": level,
                            "set": label,
                            "test_accuracy": test_acc,
                            "test_signals": sn,
                            "test_correct": correct,
                            "test_wrong": sn - correct,
                            "baseline_accuracy": base_acc,
                            "baseline_n": bn,
                            "lift_pp": test_acc - base_acc,
                            "opportunities": opp,
                            "train_trigger_native_acc": tn,
                            "train_trigger_native_n": nn,
                            "train_trigger_inverse_acc": ti,
                            "train_trigger_inverse_n": ni,
                            "selected": [dict(x) for x in selected],
                        }
                    )

                if level == args.levels:
                    break
                ranked = choose_next(train, trig, tv, selected, args.min_samples)
                if not ranked:
                    break
                selected.append(ranked[0])

        qualified.sort(
            key=lambda x: (x["test_accuracy"], x["test_signals"], x["lift_pp"]),
            reverse=True,
        )

        print(f"\n===== TF={tf}m QUALIFIED >= {args.threshold:.1f}% =====")
        if not qualified:
            print("NONE")
            continue

        print("RANK SET TEST_ACC N CORRECT WRONG BASE LIFT CAPITAL_0 RET_0 CAPITAL_FEE1.3 RET_FEE1.3 FEES")
        for rank, q in enumerate(qualified, 1):
            no_fee = run_capital(
                series,
                q["trigger_rule"],
                q["trigger_variant"],
                q["selected"],
                args.initial_capital,
                0.0,
            )
            fee = run_capital(
                series,
                q["trigger_rule"],
                q["trigger_variant"],
                q["selected"],
                args.initial_capital,
                1.3,
            )
            print(
                f"{rank:4} {q['set']:<38} {q['test_accuracy']:7.2f}% {q['test_signals']:4} "
                f"{q['test_correct']:4} {q['test_wrong']:4} {q['baseline_accuracy']:7.2f}% {q['lift_pp']:+6.2f} "
                f"{no_fee['final_capital']:10.2f} {no_fee['return_pct']:+7.2f}% "
                f"{fee['final_capital']:12.2f} {fee['return_pct']:+9.2f}% {fee['fees_paid']:10.2f}"
            )

            rows.append(
                {
                    "timeframe_min": tf,
                    "rank_within_tf": rank,
                    "trigger_rule": q["trigger_rule"],
                    "trigger_variant": q["trigger_variant"],
                    "level": q["level"],
                    "set": q["set"],
                    "test_accuracy_pct": round(q["test_accuracy"], 6),
                    "test_signals": q["test_signals"],
                    "test_correct": q["test_correct"],
                    "test_wrong": q["test_wrong"],
                    "baseline_accuracy_pct": round(q["baseline_accuracy"], 6),
                    "lift_pp": round(q["lift_pp"], 6),
                    "initial_capital": args.initial_capital,
                    "final_no_fee": round(no_fee["final_capital"], 10),
                    "return_no_fee_pct": round(no_fee["return_pct"], 8),
                    "trades_no_fee": no_fee["trades"],
                    "wins_no_fee": no_fee["wins"],
                    "losses_no_fee": no_fee["losses"],
                    "win_rate_no_fee_pct": round(no_fee["win_rate_pct"], 6),
                    "max_dd_no_fee_pct": round(no_fee["max_dd_pct"], 6),
                    "final_fee_1_3_per_side": round(fee["final_capital"], 10),
                    "return_fee_1_3_per_side_pct": round(fee["return_pct"], 8),
                    "trades_fee_1_3_per_side": fee["trades"],
                    "wins_fee_1_3_per_side": fee["wins"],
                    "losses_fee_1_3_per_side": fee["losses"],
                    "win_rate_fee_1_3_per_side_pct": round(fee["win_rate_pct"], 6),
                    "max_dd_fee_1_3_per_side_pct": round(fee["max_dd_pct"], 6),
                    "fees_paid": round(fee["fees_paid"], 10),
                    "train_trigger_native_acc_pct": round(q["train_trigger_native_acc"], 6),
                    "train_trigger_native_n": q["train_trigger_native_n"],
                    "train_trigger_inverse_acc_pct": round(q["train_trigger_inverse_acc"], 6),
                    "train_trigger_inverse_n": q["train_trigger_inverse_n"],
                }
            )

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys()) if rows else ["timeframe_min"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSAVED {p} rows={len(rows)}")


if __name__ == "__main__":
    main()
