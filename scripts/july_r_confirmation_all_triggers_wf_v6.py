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
    RULES, TESTS, components, discover_symbols, find_month_file, load_binance, resample,
)


def raw_sig(c: Candle, rule: str) -> int:
    v = components(c)[rule]; th = TESTS[rule]
    return 1 if v >= th else -1 if v <= -th else 0


def variant_sig(c: Candle, rule: str, variant: str) -> int:
    s = raw_sig(c, rule)
    return s if variant == "N" else -s


def nxt(cs: list[Candle], i: int) -> int:
    if i + 1 >= len(cs): return 0
    return 1 if cs[i+1].close > cs[i].close else -1 if cs[i+1].close < cs[i].close else 0


def pct(c: int, n: int) -> float:
    return 100.0*c/n if n else 0.0


def load_sets(root: Path, symbols: list[str], month: str, tf: int) -> list[list[Candle]]:
    out=[]
    for symbol in symbols:
        p=find_month_file(root,symbol,month)
        if not p: continue
        cs=resample(load_binance(p),tf)
        if len(cs)>32: out.append(cs)
    return out


def choose_variant(train: list[list[Candle]], rule: str, min_n: int) -> tuple[str,float,int,float,int]:
    a={"N":[0,0],"I":[0,0]}
    for cs in train:
        for i in range(len(cs)-1):
            y=nxt(cs,i)
            if not y: continue
            for v in ("N","I"):
                s=variant_sig(cs[i],rule,v)
                if not s: continue
                a[v][1]+=1; a[v][0]+=int(s==y)
    na,ia=pct(a["N"][0],a["N"][1]),pct(a["I"][0],a["I"][1])
    if a["N"][1]>=min_n and a["I"][1]>=min_n: v="N" if na>=ia else "I"
    elif a["N"][1]>=min_n: v="N"
    elif a["I"][1]>=min_n: v="I"
    else: v="N" if na>=ia else "I"
    return v,na,a["N"][1],ia,a["I"][1]


def chain_ok(cs, i, trigger_rule, trigger_variant, selected):
    trigger=variant_sig(cs[i],trigger_rule,trigger_variant)
    if not trigger: return False,0
    for k,x in enumerate(selected,1):
        j=i+k
        if j>=len(cs): return False,trigger
        s=variant_sig(cs[j],x["rule"],x["variant"])
        if not s: return False,trigger
        expected=trigger if x["mode"]=="SAME" else -trigger
        if s!=expected: return False,trigger
    return True,trigger


def candidate_stats(train, trigger_rule, trigger_variant, selected, rule, variant):
    out={"SAME":{"n":0,"correct":0},"OPPOSITE":{"n":0,"correct":0}}
    depth=len(selected)
    for cs in train:
        for i in range(len(cs)-depth-2):
            ok,t=chain_ok(cs,i,trigger_rule,trigger_variant,selected)
            if not ok: continue
            j=i+depth+1; s=variant_sig(cs[j],rule,variant); y=nxt(cs,j)
            if not s or not y: continue
            mode="SAME" if s==t else "OPPOSITE"
            out[mode]["n"]+=1; out[mode]["correct"]+=int(s==y)
    for m in out: out[m]["acc"]=pct(out[m]["correct"],out[m]["n"])
    return out


def choose_next(train, trigger_rule, trigger_variant, selected, min_n):
    ranked=[]
    for rule in RULES:
        if rule==trigger_rule or any(x["rule"]==rule for x in selected): continue
        for variant in ("N","I"):
            st=candidate_stats(train,trigger_rule,trigger_variant,selected,rule,variant)
            for mode in ("SAME","OPPOSITE"):
                n=st[mode]["n"]
                if n<min_n: continue
                ranked.append({"rule":rule,"variant":variant,"mode":mode,"acc":st[mode]["acc"],"n":n})
    ranked.sort(key=lambda x:(x["acc"],x["n"]),reverse=True)
    return ranked


def evaluate_level(test, trigger_rule, trigger_variant, selected):
    depth=len(selected); opp=sig_n=correct=base_n=base_correct=0
    for cs in test:
        for i in range(len(cs)-depth-2):
            trigger=variant_sig(cs[i],trigger_rule,trigger_variant)
            if not trigger: continue
            opp+=1; final_i=i+depth+1; y=nxt(cs,final_i)
            if not y: continue
            base_n+=1; base_correct+=int(trigger==y)
            ok,t=chain_ok(cs,i,trigger_rule,trigger_variant,selected)
            if ok: sig_n+=1; correct+=int(t==y)
    return opp,sig_n,correct,base_n,base_correct


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",required=True); ap.add_argument("--train-month",default="2026-06"); ap.add_argument("--test-month",default="2026-07")
    ap.add_argument("--symbols",default="ALL"); ap.add_argument("--timeframes",default="30,60,120,240"); ap.add_argument("--min-samples",type=int,default=100); ap.add_argument("--levels",type=int,default=2)
    ap.add_argument("--output",default="reports/july_r_confirmation_all_triggers_wf_v6.csv")
    args=ap.parse_args(); root=Path(args.input_dir); tfs=[int(x) for x in args.timeframes.split(',') if x.strip()]
    symbols=sorted(set(discover_symbols(root,args.train_month))|set(discover_symbols(root,args.test_month))) if args.symbols.upper()=="ALL" else [x.strip().upper() for x in args.symbols.split(',') if x.strip()]
    rows=[]
    print(f"R_LADDER_ALL_TRIGGERS_V6 train={args.train_month} test={args.test_month} assets={len(symbols)} tfs={','.join(map(str,tfs))}")
    for tf in tfs:
        train=load_sets(root,symbols,args.train_month,tf); test=load_sets(root,symbols,args.test_month,tf)
        if not train or not test: continue
        print(f"\n===== TF={tf}m =====")
        for trig in RULES:
            tv,tn,nn,ti,ni=choose_variant(train,trig,args.min_samples)
            selected=[]
            print(f"TRIGGER {trig}:{tv} TRAIN_N={tn:.2f}/n{nn} TRAIN_I={ti:.2f}/n{ni}")
            for level in range(args.levels+1):
                opp,sn,co,bn,bc=evaluate_level(test,trig,tv,selected)
                label=f"{trig}:{tv}" if not selected else f"{trig}:{tv}+"+"+".join(f"{x['rule']}:{x['variant']}:{x['mode'][0]}" for x in selected)
                acc=pct(co,sn); base=pct(bc,bn); lift=acc-base
                print(f"L{level} {label:<42} TEST={acc:6.2f}%/{sn:<4} BASE={base:6.2f}%/{bn:<4} LIFT={lift:+6.2f} OPP={opp}")
                rows.append({"timeframe_min":tf,"trigger_rule":trig,"trigger_variant":tv,"level":level,"set":label,"test_accuracy_pct":round(acc,6),"test_signals":sn,"test_correct":co,"horizon_baseline_accuracy_pct":round(base,6),"horizon_baseline_n":bn,"horizon_baseline_correct":bc,"lift_pp":round(lift,6),"opportunities":opp,"train_trigger_native_acc_pct":round(tn,6),"train_trigger_native_n":nn,"train_trigger_inverse_acc_pct":round(ti,6),"train_trigger_inverse_n":ni})
                if level==args.levels: break
                ranked=choose_next(train,trig,tv,selected,args.min_samples)
                print("  CAND " + " ".join(f"{x['rule']}:{x['variant']}:{x['mode'][0]}={x['acc']:.2f}%/n{x['n']}" for x in ranked[:5]))
                if not ranked: break
                selected.append(ranked[0])
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        fields=list(rows[0].keys()) if rows else ['timeframe_min']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"SAVED {p} rows={len(rows)}")

if __name__=='__main__': main()
