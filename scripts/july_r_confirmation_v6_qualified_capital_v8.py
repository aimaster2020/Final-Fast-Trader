from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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


def pct(c: int, n: int) -> float:
    return 100.0 * c / n if n else 0.0


@dataclass
class CapitalResult:
    initial: float
    final: float
    trades: int
    wins: int
    losses: int
    fees_paid: float
    max_dd_pct: float


def signal_indices(cs: list[Candle], trigger_rule: str, trigger_variant: str, selected: list[dict]) -> list[tuple[int, int]]:
    """Return tradable signals at the CLOSE of the last confirmation candle.

    L0: signal candle i => entry at close(i), exit at close(i+1).
    L1: trigger i, confirmation i+1 => entry at close(i+1), exit at close(i+2).
    L2: trigger i, confirmations i+1/i+2 => entry at close(i+2), exit at close(i+3).
    """
    out: list[tuple[int, int]] = []
    depth = len(selected)
    for i in range(len(cs) - depth - 2):
        ok, direction = chain_ok(cs, i, trigger_rule, trigger_variant, selected)
        if ok and direction:
            entry_i = i + depth
            if entry_i + 1 < len(cs):
                out.append((entry_i, direction))
    return out


def run_capital(
    series: dict[str, list[Candle]],
    trigger_rule: str,
    trigger_variant: str,
    selected: list[dict],
    initial: float,
    fee_per_side_pct: float,
) -> CapitalResult:
    capital = initial
    peak = initial
    max_dd = 0.0
    trades = wins = losses = 0
    fees_paid = 0.0
    # Process each symbol independently but use the same initial capital allocation.
    # This preserves the old test's per-symbol equal-allocation interpretation.
    per_symbol = initial / max(1, len(series))
    ending = 0.0
    for _, cs in series.items():
        local = per_symbol
        local_peak = local
        for entry_i, direction in signal_indices(cs, trigger_rule, trigger_variant, selected):
            entry = cs[entry_i].close
            exit_price = cs[entry_i + 1].close
            if entry <= 0 or exit_price <= 0:
                continue
            move = (exit_price / entry - 1.0) * direction
            gross = local * move
            fee = local * (2.0 * fee_per_side_pct / 100.0)
            local = max(0.0, local + gross - fee)
            fees_paid += fee
            trades += 1
            wins += int(gross > 0)
            losses += int(gross <= 0)
            local_peak = max(local_peak, local)
            if local_peak > 0:
                max_dd = max(max_dd, 100.0 * (local_peak - local) / local_peak)
            if local <= 0:
                break
        ending += local
    capital = ending
    return CapitalResult(initial, capital, trades, wins, losses, fees_paid, max_dd)


def choose_best_v6(train, threshold: float, min_samples: int, levels: int):
    best = None
    for trig in RULES:
        tv, tn, nn, ti, ni = choose_variant(train, trig, min_samples)
        selected: list[dict] = []
        for level in range(levels + 1):
            # Re-use V6 candidate-selection logic: score the current chain on TRAIN only.
            dummy_test = train
            _, signals, correct, _, _ = evaluate_level(dummy_test, trig, tv, selected)
            acc = pct(correct, signals)
            if signals >= min_samples and acc >= threshold:
                cand = (acc, signals, (trig, tv), [dict(x) for x in selected])
                if best is None or (acc, signals) > (best[0], best[1]):
                    best = cand
            if level == levels:
                break
            ranked = choose_next(train, trig, tv, selected, min_samples)
            ranked = [x for x in ranked if x["acc"] >= threshold]
            if not ranked:
                break
            selected.append(ranked[0])
    return None if best is None else best[2], best[3]


def main() -> None:
    ap = argparse.ArgumentParser(description="Corrected V6 capital calculation: enter only after the final confirmation candle closes.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--train-month", default="2026-06")
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--timeframes", default="30,60,120,240")
    ap.add_argument("--threshold", type=float, default=55.0)
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--output", default="reports/july_r_confirmation_v6_qualified_capital_v8.csv")
    args = ap.parse_args()
    root = Path(args.input_dir)
    tfs = [int(x.strip()) for x in args.timeframes.split(",") if x.strip()]
    symbols = (sorted(set(discover_symbols(root, args.train_month)) | set(discover_symbols(root, args.test_month)))
               if args.symbols.upper() == "ALL" else [x.strip().upper() for x in args.symbols.split(",") if x.strip()])
    rows=[]
    print(f"V6_QUALIFIED_CAPITAL_V8 train={args.train_month} test={args.test_month} threshold={args.threshold:.1f}% initial={args.initial_capital:.2f}")
    print("ENTRY = close of the final confirmation candle; EXIT = next candle close. Both 0% and 1.3%/side are reported.")
    for tf in tfs:
        train = load_sets(root, symbols, args.train_month, tf)
        test = load_sets(root, symbols, args.test_month, tf)
        if not train or not test:
            continue
        qualified=[]
        for trig in RULES:
            tv, tn, nn, ti, ni = choose_variant(train, trig, args.min_samples)
            selected=[]
            for level in range(args.levels+1):
                opp,sn,co,bn,bc = evaluate_level(test,trig,tv,selected)
                acc=pct(co,sn)
                if acc>=args.threshold:
                    label=f"{trig}:{tv}" if not selected else f"{trig}:{tv}+"+"+".join(f"{x['rule']}:{x['variant']}:{x['mode'][0]}" for x in selected)
                    qualified.append((acc,sn,label,trig,tv,[dict(x) for x in selected],bc,bn))
                if level==args.levels: break
                ranked=choose_next(train,trig,tv,selected,args.min_samples)
                if not ranked: break
                selected.append(ranked[0])
        qualified.sort(reverse=True,key=lambda x:(x[0],x[1]))
        print(f"\nTF={tf}m QUALIFIED >= {args.threshold:.1f}%")
        print("RANK SET ACC N BASE LIFT FINAL_0 RET_0 FINAL_1.3 RET_1.3 TRADES")
        for rank,q in enumerate(qualified,1):
            acc,n,label,trig,tv,sel,bc,bn=q
            base=pct(bc,bn); lift=acc-base
            series={}
            for s in symbols:
                p=find_month_file(root,s,args.test_month)
                if not p: continue
                cs=resample(load_binance(p),tf)
                if len(cs)>32: series[s]=cs
            a=run_capital(series,trig,tv,sel,args.initial_capital,0.0)
            b=run_capital(series,trig,tv,sel,args.initial_capital,1.3)
            r0=100*(a.final/args.initial_capital-1); r1=100*(b.final/args.initial_capital-1)
            print(f"{rank:4} {label:<38} {acc:6.2f}% {n:4} {base:6.2f}% {lift:+6.2f} {a.final:8.2f} {r0:+7.2f}% {b.final:8.2f} {r1:+8.2f}% {a.trades}")
            rows.append({"tf":tf,"rank":rank,"set":label,"accuracy_pct":acc,"signals":n,"base_accuracy_pct":base,"lift_pp":lift,"final_no_fee":a.final,"return_no_fee_pct":r0,"final_fee_1_3_side":b.final,"return_fee_1_3_side_pct":r1,"trades":a.trades,"wins":a.wins,"losses":a.losses,"fees":b.fees_paid})
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()) if rows else ["tf"]); w.writeheader(); w.writerows(rows)
    print(f"SAVED {p} rows={len(rows)}")

if __name__=="__main__": main()
