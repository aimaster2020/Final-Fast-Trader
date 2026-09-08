from __future__ import annotations

import argparse, csv, statistics
from collections import defaultdict
from pathlib import Path

SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
MONTHS=("2026-05","2026-06","2026-07","2026-08")
FEE_RT=0.0026
BINS=6


def load(path:Path,symbol:str):
    out=[]
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol')!=symbol: continue
            try:
                out.append((r['month'],int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close'])))
            except (KeyError,TypeError,ValueError): pass
    return sorted(out,key=lambda x:x[1])


def s(row):
    _,_,o,h,l,c=row
    j=c-o; k=h-c; m=h-o
    return int(k>j)+int(k>m)+int(l>j)


def qbin(x, edges):
    for i,e in enumerate(edges):
        if x <= e: return i
    return len(edges)


def pct(side, a, b):
    return side*(b-a)/a*100.0 if a>0 else 0.0


def run(rows):
    # Past-only empirical distribution keyed by exact Excel S and body-size bin.
    hist=defaultdict(list)
    probs=defaultdict(lambda:[0,0])
    all_body=[]
    for i in range(len(rows)-1):
        o,h,l,c=rows[i][2:]
        body=abs(c-o); px=c
        if px>0: all_body.append(body/px)
    global_edges=[statistics.quantiles(all_body,n=BINS,method='inclusive')[i] for i in range(BINS-1)] if len(all_body)>=BINS else []

    rows_by_month={m:[r for r in rows if r[0]==m] for m in MONTHS}
    results=[]
    for month in MONTHS:
        mr=rows_by_month[month]
        # Build month training context from all earlier rows only.
        for i in range(len(mr)-1):
            cur=mr[i]; fut=mr[i+1]
            body=abs(cur[5]-cur[2]); px=cur[5]
            if px<=0: continue
            size=body/px
            b=qbin(size,global_edges) if global_edges else 0
            st=s(cur); key=(st,b)
            vals=hist[key]
            pred_p=(sum(v>0 for v in vals)/len(vals)) if vals else 0.5
            # Direction maximizing historical next-price probability.
            side=1 if pred_p>=0.5 else -1
            gross=pct(side,cur[5],fut[5])
            move=abs((fut[5]-cur[5])/cur[5])*100.0
            # Store actual outcome in current-state bucket for future use.
            delta=(fut[5]-fut[2])-(cur[5]-cur[2])
            actual_price=1 if fut[5]>cur[5] else -1 if fut[5]<cur[5] else 0
            hist[key].append(actual_price)
            # Track signed outcomes only for confidence levels.
            results.append((month,st,b,pred_p,side,gross,move))

    print('EXPECTANCY_BY_REGIME_WF | tf=1h | fee=0.13%/side | one-bar | past_only')
    print('Key=(Excel S, current-body-size quantile). Direction uses past-only empirical next-price sign.')
    print('For each symbol, shows confidence buckets; avg_gross must exceed +0.2600% to cover round-trip fee.')
    for sym in SYMBOLS:
        rr=run_for_symbol(rows_by_month=None)


def run_for_symbol(rows):
    # Separate implementation so training never crosses symbol boundaries.
    all_size=[]
    for r in rows[:-1]:
        if r[5]>0: all_size.append(abs(r[5]-r[2])/r[5])
    edges=statistics.quantiles(all_size,n=BINS,method='inclusive')[:-1] if len(all_size)>=BINS else []
    hist=defaultdict(list)
    accum=defaultdict(list)
    monthly=defaultdict(list)
    for month in MONTHS:
        mr=[r for r in rows if r[0]==month]
        for i in range(len(mr)-1):
            cur=mr[i]; fut=mr[i+1]
            if cur[5]<=0: continue
            size=abs(cur[5]-cur[2])/cur[5]
            b=qbin(size,edges)
            st=s(cur); key=(st,b)
            vals=hist[key]
            if len(vals)>=8:
                p=sum(vals)/len(vals)
                side=1 if p>=0.5 else -1
                gross=pct(side,cur[5],fut[5]); net=gross-FEE_RT*100
                monthly[(month,st,b)].append((p,gross,net))
                accum[(st,b)].append((p,gross,net))
            actual=1 if fut[5]>cur[5] else 0 if fut[5]==cur[5] else -1
            hist[key].append(actual)
    return accum,monthly


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); a=ap.parse_args()
    print('EXPECTANCY_BY_REGIME_WF | tf=1h | fee=0.13%/side | one-bar | past_only')
    print('Key=(S,size_quantile); prediction is empirical next-price sign. Min history per regime=8.')
    print('Reported bucket avg_gross/avg_net across the full May-Aug walk-forward. Break-even gross=+0.2600%.')
    for sym in SYMBOLS:
        rows=load(a.input,sym); accum,monthly=run_for_symbol(rows)
        print(f'\n{sym}')
        best=[]
        for key,vals in accum.items():
            if len(vals)<20: continue
            p=sum(x[0] for x in vals)/len(vals); ag=sum(x[1] for x in vals)/len(vals); an=sum(x[2] for x in vals)/len(vals)
            n=len(vals); best.append((an,key,p,ag,n))
        best.sort(reverse=True)
        for an,key,p,ag,n in best[:12]:
            st,b=key
            print(f' S{st}/Q{b} n={n:4} pred_p={p:.3f} avg_gross={ag:+.4f}% avg_net={an:+.4f}%')
        print(' TOP_NET_BUCKETS=',len(best))
        for threshold in (0.55,0.60,0.65,0.70,0.75):
            vals=[]
            for k,vs in accum.items():
                for p,g,n in vs:
                    if p>=threshold: vals.append((g,n))
            if vals:
                print(f' p>={threshold:.2f} n={len(vals):4} avg_gross={sum(x[0] for x in vals)/len(vals):+.4f}% avg_net={sum(x[1] for x in vals)/len(vals):+.4f}%')
            else:
                print(f' p>={threshold:.2f} n=   0 avg_gross=+0.0000% avg_net=+0.0000%')

if __name__=='__main__': main()
