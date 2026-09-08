from __future__ import annotations

import argparse, csv, statistics
from collections import defaultdict
from pathlib import Path

SYMS=('BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT')
TRAIN_MONTHS=('2026-05','2026-06','2026-07')
TEST_MONTH='2026-08'
CAPITAL=1000.0
MIN_BODY=0.0001
FEE_SIDE=0.0013
THRESHOLDS=(0.0026,0.0030,0.0050,0.0075,0.0100)  # expected TP distance as fraction of entry


def load_rows(p: Path, sym: str):
    out=[]
    with p.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol')!=sym: continue
            try: out.append((r['month'],int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close'])))
            except (KeyError,TypeError,ValueError): continue
    return sorted(out,key=lambda x:x[1])


def S(r):
    _,_,o,h,l,c=r
    j=c-o; k=h-c; m=h-o
    return int(k>j)+int(k>m)+int(l>j)


def side(s): return 1 if s in (2,3) else -1

def run(rows, threshold):
    hist=defaultdict(list); glob=[]
    train=[r for r in rows if r[0] in TRAIN_MONTHS]
    test=[r for r in rows if r[0]==TEST_MONTH]
    for i,r in enumerate(train[:-1]):
        s=S(r); b=abs(r[5]-r[2]); valid=r[5]!=0 and b/abs(r[5])>=MIN_BODY
        if valid:
            x=abs(train[i+1][5]-r[5])/b
            if x==x: hist[s].append(x); glob.append(x)
    eq=CAPITAL; peak=CAPITAL; max_dd=0.0; trades=wins=0; pos=None; ei=-1
    filtered=0
    for i,r in enumerate(test[:-1]):
        s=S(r); b=abs(r[5]-r[2]); valid=r[5]!=0 and b/abs(r[5])>=MIN_BODY
        gr=statistics.median(glob) if glob else 2.0
        ratio=statistics.median(hist[s]) if hist[s] else gr
        eff=max(b,abs(r[5])*MIN_BODY)
        tp=eff*ratio
        tp_pct=tp/r[5] if r[5] else 0.0
        if pos is None:
            if threshold>0 and tp_pct < threshold:
                filtered += 1
            else:
                pos=(side(s),r[5],tp,0.5*tp); ei=i
        if pos is not None and i>ei:
            sd,en,tpd,sld=pos; px=r[5]; gross_pct=None
            if sd==1:
                if px>=en+tpd: gross_pct=tpd/en*100
                elif px<=en-sld: gross_pct=-sld/en*100
            else:
                if px<=en-tpd: gross_pct=tpd/en*100
                elif px>=en+sld: gross_pct=-sld/en*100
            if gross_pct is not None:
                gross=gross_pct/100; exit_value=CAPITAL*(1+gross)
                fees=CAPITAL*FEE_SIDE+max(exit_value,0.0)*FEE_SIDE
                net_pct=(CAPITAL*gross-fees)/CAPITAL*100
                eq += CAPITAL*net_pct/100; trades += 1; wins += int(net_pct>0)
                pos=None; ei=-1; peak=max(peak,eq); max_dd=max(max_dd,(peak-eq)/peak if peak else 0.0)
        if valid:
            x=abs(test[i+1][5]-r[5])/b
            if x==x: hist[s].append(x); glob.append(x)
    if pos is not None:
        sd,en,tpd,sld=pos; gross=sd*(test[-1][5]-en)/en if en else 0.0
        exit_value=CAPITAL*(1+gross); fees=CAPITAL*FEE_SIDE+max(exit_value,0.0)*FEE_SIDE
        net_pct=(CAPITAL*gross-fees)/CAPITAL*100
        eq += CAPITAL*net_pct/100; trades += 1; wins += int(net_pct>0)
        peak=max(peak,eq); max_dd=max(max_dd,(peak-eq)/peak if peak else 0.0)
    return (eq-CAPITAL)/CAPITAL*100,trades,wins,max_dd*100,filtered


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); a=ap.parse_args()
    print('EXCEL_OOS_FEE_FILTER | tf=1h | train=May-Jul | test=Aug | STATE | fee_side=0.13% | round_trip≈0.26% | fixed trade capital=$1000 | past_only | SL=0.5xTP')
    print('Filter is based only on predicted TP distance / entry price; thresholds are fixed sensitivity levels, not fitted to August.')
    for sym in SYMS:
        rows=load_rows(a.input,sym); print(sym)
        for t in THRESHOLDS:
            r=run(rows,t)
            print(f'  min_tp={100*t:.2f}% Aug_net={r[0]:+.2f}% trades={r[1]} win={100*r[2]/r[1] if r[1] else 0:.1f}% DD={r[3]:.2f}% filtered_entries={r[4]}')

if __name__=='__main__': main()
