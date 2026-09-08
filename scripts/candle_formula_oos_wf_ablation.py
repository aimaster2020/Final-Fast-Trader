from __future__ import annotations
import argparse, csv, statistics
from collections import defaultdict
from pathlib import Path

SYMS=('BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT')
TRAIN_MONTHS=('2026-05','2026-06','2026-07')
TEST_MONTH='2026-08'
CAPITAL=1000.0
MIN_BODY=0.0001
MODES=('FIXED2','STATE','S12_S3','S13_S2')

def load_rows(p: Path, sym: str):
    out=[]
    with p.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol')!=sym: continue
            try: out.append((r['month'],int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close'])))
            except (KeyError,TypeError,ValueError): continue
    return sorted(out,key=lambda x:x[1])

def S(r):
    _,_,o,h,l,c=r; j=c-o; k=h-c; m=h-o
    return int(k>j)+int(k>m)+int(l>j)

def side(s): return 1 if s in (2,3) else -1

def bucket(mode,s):
    if mode=='STATE': return s
    if mode=='S12_S3': return 12 if s in (1,2) else (3 if s==3 else 0)
    if mode=='S13_S2': return 13 if s in (1,3) else (2 if s==2 else 0)
    return 0

def run(rows, mode):
    hist=defaultdict(list); glob=[]; pos=None; ei=-1; eq=CAPITAL; peak=CAPITAL; dd=0.0; trades=wins=0
    # Train chronologically, then test. History accumulated from train and then updated only inside test after each target is known.
    for r in rows:
        month=r[0]
        if month not in TRAIN_MONTHS and month!=TEST_MONTH: continue
    train=[r for r in rows if r[0] in TRAIN_MONTHS]
    test=[r for r in rows if r[0]==TEST_MONTH]
    for i,r in enumerate(train[:-1]):
        b=abs(r[5]-r[2])
        if r[5]!=0 and b/abs(r[5])>=MIN_BODY:
            q=abs(train[i+1][5]-r[5])/b
            hist[bucket(mode,S(r))].append(q); glob.append(q)
    for i,r in enumerate(test[:-1]):
        s=S(r); b=abs(r[5]-r[2]); gr=statistics.median(glob) if glob else 2.0
        ratio=statistics.median(hist[bucket(mode,s)]) if hist[bucket(mode,s)] else gr
        if pos is None:
            sd=side(s); eff=max(b,abs(r[5])*MIN_BODY); tp=eff*(2.0 if mode=='FIXED2' else ratio); sl=0.5*tp
            pos=(sd,r[5],tp,sl); ei=i
        if pos is not None and i>ei:
            sd,en,tp,sl=pos; px=r[5]; hit=None
            if sd==1:
                if px>=en+tp: hit=tp/en*100
                elif px<=en-sl: hit=-sl/en*100
            else:
                if px<=en-tp: hit=tp/en*100
                elif px>=en+sl: hit=-sl/en*100
            if hit is not None:
                eq += CAPITAL*hit/100; trades+=1; wins+=int(hit>0); pos=None; ei=-1; peak=max(peak,eq); dd=max(dd,(peak-eq)/peak if peak else 0)
        if r[5]!=0 and b>0:
            q=abs(test[i+1][5]-r[5])/b; hist[bucket(mode,s)].append(q); glob.append(q)
    if pos is not None:
        sd,en,tp,sl=pos; px=test[-1][5]; ret=sd*(px-en)/en*100; eq+=CAPITAL*ret/100; trades+=1; wins+=int(ret>0); peak=max(peak,eq); dd=max(dd,(peak-eq)/peak if peak else 0)
    return (eq-CAPITAL)/CAPITAL*100, trades, 100*wins/trades if trades else 0, 100*dd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); a=ap.parse_args()
    print('EXCEL_OOS_WF_ABLATION | tf=1h | train=May-Jul | test=Aug | fixed trade capital=$1000 | close_only | past_only | SL=0.5xTP')
    print('Each mode builds history only from May-Jul; August is one held-out test with rolling past-only updates inside August.')
    for sym in SYMS:
        rows=load_rows(a.input,sym); print(sym)
        for mode in MODES:
            x=run(rows,mode); print(f'  {mode:7s} Aug={x[0]:+.2f}% trades={x[1]} win={x[2]:.1f}% DD={x[3]:.2f}%')

if __name__=='__main__': main()
