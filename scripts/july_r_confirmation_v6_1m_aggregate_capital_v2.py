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
    RULES, choose_next, choose_variant, chain_ok, load_sets,
)
from scripts.july_r_composite_walkforward import (
    TESTS, components, discover_symbols, find_month_file, load_binance,
)


def pct(c: int, n: int) -> float:
    return 100.0 * c / n if n else 0.0


def raw_sig(c: Candle, rule: str) -> int:
    v = components(c)[rule]
    th = TESTS[rule]
    return 1 if v >= th else -1 if v <= -th else 0


def variant_sig(c: Candle, rule: str, variant: str) -> int:
    s = raw_sig(c, rule)
    return s if variant == "N" else -s


def next_direction(cs: list[Candle], i: int) -> int:
    if i + 1 >= len(cs):
        return 0
    return 1 if cs[i + 1].close > cs[i].close else -1 if cs[i + 1].close < cs[i].close else 0


def load_1m(root: Path, symbols: list[str], month: str) -> dict[str, list[Candle]]:
    out = {}
    for s in symbols:
        p = find_month_file(root, s, month)
        if not p:
            continue
        cs = load_binance(p)
        if len(cs) > 32:
            out[s] = cs
    return out


def train_acc(train, trig, tv, selected):
    depth = len(selected)
    correct = n = 0
    for cs in train:
        for i in range(len(cs) - depth - 2):
            ok, d = chain_ok(cs, i, trig, tv, selected)
            if not ok:
                continue
            y = next_direction(cs, i + depth + 1)
            if not y:
                continue
            n += 1
            correct += int(d == y)
    return pct(correct, n), n


def select_v6_candidates(train, threshold: float, min_samples: int, levels: int):
    candidates = []
    for trig in RULES:
        tv, tn, nn, ti, ni = choose_variant(train, trig, min_samples)
        selected = []
        for level in range(levels + 1):
            a, n = train_acc(train, trig, tv, selected)
            if n >= min_samples and a >= threshold:
                label = trig + ":" + tv
                if selected:
                    label += "+" + "+".join(f"{x['rule']}:{x['variant']}:{x['mode'][0]}" for x in selected)
                candidates.append({"trigger": trig, "variant": tv, "selected": [dict(x) for x in selected], "train_acc": a, "train_n": n, "label": label, "level": level})
            if level == levels:
                break
            ranked = choose_next(train, trig, tv, selected, min_samples)
            ranked = [x for x in ranked if x["acc"] >= threshold]
            if not ranked:
                break
            selected.append(ranked[0])
    # Preserve every qualifying candidate, but rank strongest first.
    candidates.sort(key=lambda x: (x["train_acc"], x["train_n"]), reverse=True)
    return candidates


def one_min_predictions(cs, cand):
    out = {}
    trig = cand["trigger"]
    tv = cand["variant"]
    selected = cand["selected"]
    depth = len(selected)
    for i in range(len(cs) - depth - 2):
        ok, d = chain_ok(cs, i, trig, tv, selected)
        if ok and d:
            signal_bar = i + depth
            out[signal_bar] = d
    return out


def block_vote(preds: dict[int, int], start: int, end: int) -> tuple[int, int, int]:
    long_n = short_n = hold_n = 0
    for i in range(start, end):
        d = preds.get(i, 0)
        if d > 0: long_n += 1
        elif d < 0: short_n += 1
        else: hold_n += 1
    return long_n, short_n, hold_n


def aggregate_blocks(preds: dict[int, int], block_size: int, strict_majority: float, n_bars: int):
    out = []
    for end in range(block_size, n_bars + 1, block_size):
        start = end - block_size
        l, s, h = block_vote(preds, start, end)
        total = end - start
        if l > s and 100.0 * l / total > strict_majority:
            d = 1
        elif s > l and 100.0 * s / total > strict_majority:
            d = -1
        else:
            d = 0
        out.append((end, d, l, s, h))
    return out


def run_capital(series, candidate, tf: int, initial: float, fee_side_pct: float, vote_majority: float):
    per_symbol = initial / max(1, len(series))
    total = 0.0; trades = wins = losses = 0; fees = 0.0; peak_global = initial; max_dd = 0.0
    for _, cs in series.items():
        local = per_symbol
        preds = one_min_predictions(cs, candidate)
        blocks = aggregate_blocks(preds, tf, vote_majority, len(cs))
        peak = local
        for end, direction, _, _, _ in blocks:
            if not direction:
                continue
            # The vote is known at close(end-1). That close is the entry.
            entry_i = end - 1
            exit_i = entry_i + tf
            if exit_i >= len(cs):
                break
            entry = cs[entry_i].close; exit_price = cs[exit_i].close
            if entry <= 0 or exit_price <= 0:
                continue
            move = (exit_price / entry - 1.0) * direction
            gross = local * move
            fee = local * (2.0 * fee_side_pct / 100.0)
            local = max(0.0, local + gross - fee)
            fees += fee; trades += 1; wins += int(gross > 0); losses += int(gross <= 0)
            peak = max(peak, local); max_dd = max(max_dd, 100.0 * (peak-local)/peak if peak else 0.0)
            if local <= 0: break
        total += local
    return {"final": total, "return_pct": 100.0*(total/initial-1.0), "trades": trades, "wins": wins, "losses": losses, "win_rate_pct": pct(wins,trades), "fees": fees, "max_dd_pct": max_dd}


def main():
    ap=argparse.ArgumentParser(description="Generate predictions only on 1m candles, aggregate them into higher-TF majority decisions, then calculate capital for every V6-qualified candidate.")
    ap.add_argument("--input-dir",required=True); ap.add_argument("--train-month",default="2026-06"); ap.add_argument("--test-month",default="2026-07")
    ap.add_argument("--symbols",default="ALL"); ap.add_argument("--timeframes",default="5,15,30,60,120,240"); ap.add_argument("--vote-majority",type=float,default=50.0)
    ap.add_argument("--threshold",type=float,default=55.0); ap.add_argument("--min-samples",type=int,default=100); ap.add_argument("--levels",type=int,default=2); ap.add_argument("--initial-capital",type=float,default=1000.0)
    ap.add_argument("--output",default="reports/july_r_confirmation_v6_1m_aggregate_capital_v2.csv")
    args=ap.parse_args(); root=Path(args.input_dir); tfs=[int(x) for x in args.timeframes.split(',') if x.strip()]
    symbols=sorted(set(discover_symbols(root,args.train_month))|set(discover_symbols(root,args.test_month))) if args.symbols.upper()=="ALL" else [x.strip().upper() for x in args.symbols.split(',') if x.strip()]
    train=load_1m(root,symbols,args.train_month); test=load_1m(root,symbols,args.test_month)
    train_sets=list(train.values()); candidates=select_v6_candidates(train_sets,args.threshold,args.min_samples,args.levels)
    print(f"V6_1M_AGGREGATE_CAPITAL_V2 train={args.train_month} test={args.test_month} assets={len(test)} candidates={len(candidates)} majority>{args.vote_majority:.1f}%")
    print("Every candidate with TRAIN accuracy >= threshold is retained. Predictions are generated on 1m; higher-TF trade uses the majority of the preceding 1m predictions.")
    rows=[]
    for tf in tfs:
        print(f"\n===== TF={tf}m =====")
        for rank,c in enumerate(candidates,1):
            series={s:test[s] for s in symbols if s in test}
            no=run_capital(series,c,tf,args.initial_capital,0.0,args.vote_majority)
            fe=run_capital(series,c,tf,args.initial_capital,1.3,args.vote_majority)
            print(f"{rank:3} {c['label']:<45} TRAIN={c['train_acc']:.2f}%/n{c['train_n']} F0={no['final']:.2f} ({no['return_pct']:+.2f}%) T={no['trades']} WIN={no['win_rate_pct']:.2f}% F1.3={fe['final']:.2f} ({fe['return_pct']:+.2f}%) FEES={fe['fees']:.2f}")
            rows.append({"tf":tf,"rank":rank,"candidate":c["label"],"train_acc_pct":c["train_acc"],"train_n":c["train_n"],"final_no_fee":no["final"],"return_no_fee_pct":no["return_pct"],"trades_no_fee":no["trades"],"win_rate_no_fee_pct":no["win_rate_pct"],"max_dd_no_fee_pct":no["max_dd_pct"],"final_fee_1_3_side":fe["final"],"return_fee_1_3_side_pct":fe["return_pct"],"trades_fee_1_3_side":fe["trades"],"win_rate_fee_1_3_side_pct":fe["win_rate_pct"],"max_dd_fee_1_3_side_pct":fe["max_dd_pct"],"fees_paid":fe["fees"]})
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()) if rows else ['tf']); w.writeheader(); w.writerows(rows)
    print(f"SAVED {out} rows={len(rows)}")

if __name__=='__main__': main()
