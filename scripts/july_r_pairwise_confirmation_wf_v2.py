from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_pattern_trader.models import Candle
from scripts.july_r_composite_walkforward import RULES, TESTS, components, discover_symbols, find_month_file, load_binance, resample


def sig(c: Candle, r: str) -> int:
    v = components(c)[r]
    th = TESTS[r]
    return 1 if v >= th else -1 if v <= -th else 0


def nxt(cs: list[Candle], i: int) -> int:
    if i + 1 >= len(cs): return 0
    return 1 if cs[i + 1].close > cs[i].close else -1 if cs[i + 1].close < cs[i].close else 0


def pct(c: int, n: int) -> float:
    return 100.0 * c / n if n else 0.0


def orient(train: list[list[Candle]], rule: str) -> tuple[int, float, float, int]:
    n = inv = total = 0
    for cs in train:
        for i in range(len(cs) - 1):
            s, y = sig(cs[i], rule), nxt(cs, i)
            if not s or not y: continue
            total += 1; n += s == y; inv += -s == y
    a, b = pct(n,total), pct(inv,total)
    return (1 if a >= b else -1), a, b, total


def pair(train: list[list[Candle]], tr: str, cand: str, stage: int, ori: dict[str,int], min_n: int) -> dict:
    buckets = {k:[0,0] for k in ('same_tr','same_c','opp_tr','opp_c')}
    for cs in train:
        for i in range(len(cs) - stage - 1):
            t, c = sig(cs[i], tr), sig(cs[i+stage], cand)
            y = nxt(cs, i+stage)
            if not t or not c or not y: continue
            t *= ori[tr]; c *= ori[cand]
            p = 'same' if c == t else 'opp'
            buckets[p+'_tr'][0] += t == y; buckets[p+'_tr'][1] += 1
            buckets[p+'_c'][0] += c == y; buckets[p+'_c'][1] += 1
    ts, tc = buckets['same_tr']; tn = tc
    os, oc = buckets['opp_tr']; on = oc
    sc, sn = buckets['same_c']; rc, rn = buckets['opp_c']
    choices=[]
    if tn >= min_n: choices.append((pct(sc,sn),'SAME',tn))
    if on >= min_n: choices.append((pct(rc,rn),'OPPOSITE',on))
    if not choices: return {}
    best = max(choices,key=lambda x:(x[0],x[2]))
    return {'rule':cand,'mode':best[1],'score':best[0],'same_trigger':pct(ts,tn),'same_candidate':pct(sc,sn),'same_n':tn,'opp_trigger':pct(os,on),'opp_candidate':pct(rc,rn),'opp_n':on}


def eval_chain(test: list[list[Candle]], trigger: str, selected: list[dict], ori: dict[str,int]) -> tuple[int,int,int]:
    d=len(selected); opp=sigc=correct=0
    for cs in test:
        for i in range(len(cs)-d-1):
            t=sig(cs[i],trigger)
            if not t: continue
            t*=ori[trigger]; opp+=1; ok=True; pred=t
            for k,x in enumerate(selected,1):
                s=sig(cs[i+k],x['rule'])
                if not s: ok=False; break
                s*=ori[x['rule']]
                expected=t if x['mode']=='SAME' else -t
                if s!=expected: ok=False; break
                pred=expected
            if not ok: continue
            y=nxt(cs,i+d)
            if not y: continue
            sigc+=1; correct += pred==y
    return opp,sigc,correct


def load_sets(root:Path, symbols:list[str], month:str, tf:int)->list[list[Candle]]:
    out=[]
    for s in symbols:
        p=find_month_file(root,s,month)
        if p:
            cs=resample(load_binance(p),tf)
            if len(cs)>30: out.append(cs)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',required=True); ap.add_argument('--train-month',default='2026-06'); ap.add_argument('--test-month',default='2026-07')
    ap.add_argument('--symbols',default='ALL'); ap.add_argument('--timeframes',default='30,60,120,240'); ap.add_argument('--trigger',default='R1'); ap.add_argument('--min-pair-samples',type=int,default=100); ap.add_argument('--levels',type=int,default=3)
    ap.add_argument('--output',default='reports/july_r_pairwise_confirmation_wf_v2.csv'); args=ap.parse_args()
    root=Path(args.input_dir); tfs=[int(x) for x in args.timeframes.split(',') if x.strip()]
    symbols=sorted(set(discover_symbols(root,args.train_month))|set(discover_symbols(root,args.test_month))) if args.symbols.upper()=='ALL' else [x.strip().upper() for x in args.symbols.split(',')]
    rows=[]
    print(f'R_PAIR_WF_V2 train={args.train_month} test={args.test_month} assets={len(symbols)} tfs={",".join(map(str,tfs))} trigger={args.trigger} min_pair_n={args.min_pair_samples}')
    for tf in tfs:
        train=load_sets(root,symbols,args.train_month,tf); test=load_sets(root,symbols,args.test_month,tf)
        if not train or not test: continue
        ori={}
        for r in RULES:
            o,a,b,n=orient(train,r); ori[r]=o; print(f'ORIENT {r}={"N" if o==1 else "I"} train={max(a,b):.2f}% n={n}')
        ranked=[]
        for r in RULES:
            if r==args.trigger: continue
            # First confirmation is always evaluated one candle after R1.
            x=pair(train,args.trigger,r,1,ori,args.min_pair_samples)
            if x: ranked.append(x)
        ranked.sort(key=lambda x:(x['score'],max(x['same_n'],x['opp_n'])),reverse=True)
        print(f'\nTF={tf}m')
        print('PAIR_RANK RULE MODE SCORE SAME_TRIG/N SAME_R/N OPP_TRIG/N OPP_R/N')
        for x in ranked[:12]:
            print(f"{x['rule']:<7} {x['mode']:<9} {x['score']:6.2f}% {x['same_trigger']:6.2f}/{x['same_n']:<4} {x['same_candidate']:6.2f}/{x['same_n']:<4} {x['opp_trigger']:6.2f}/{x['opp_n']:<4} {x['opp_candidate']:6.2f}/{x['opp_n']:<4}")
        selected=ranked[:args.levels]
        print('LEVEL SET ACC% SIGNALS CORRECT OPPORTUNITIES')
        for level in range(args.levels+1):
            confs=selected[:level]; O=S=C=0
            for cs in test:
                o,s,c=eval_chain([cs],args.trigger,confs,ori); O+=o; S+=s; C+=c
            label=args.trigger if level==0 else args.trigger+'+' '+'.join(x['rule']+':'+x['mode'][0] for x in confs)
            print(f'L{level} {label:<28} {pct(C,S):6.2f}% {S:<7} {C:<7} {O}')
            rows.append({'timeframe_min':tf,'level':level,'set':label,'test_opportunities':O,'test_signals':S,'test_correct':C,'test_accuracy_pct':round(pct(C,S),6),'train_pair_rank':';'.join(f"{x['rule']}:{x['mode']}:{x['score']:.4f}:{x['same_n']}:{x['opp_n']}" for x in ranked[:12])})
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ['timeframe_min']); w.writeheader(); w.writerows(rows)
    print(f'SAVED {p} rows={len(rows)}')

if __name__=='__main__': main()
