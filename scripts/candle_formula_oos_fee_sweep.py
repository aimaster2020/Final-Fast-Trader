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
FEE_SIDE=0.0013  # user's actual fee per side = 0.13%


def load_rows(p: Path, sym: str):
    out=[]
    with p.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol')!=sym: continue
            try:
                out.append((r['month'], int(float(r['timestamp'])), float(r['open']), float(r['high']), float(r['low']), float(r['close'])))
            except (KeyError,TypeError,ValueError):
                continue
    return sorted(out,key=lambda x:x[1])


def S(r):
    _,_,o,h,l,c=r
    j=c-o; k=h-c; m=h-o
    return int(k>j)+int(k>m)+int(l>j)


def side(s): return 1 if s in (2,3) else -1

def bucket(mode,s):
    if mode=='STATE': return s
    if mode=='S12_S3': return 12 if s in (1,2) else (3 if s==3 else 0)
    if mode=='S13_S2': return 13 if s in (1,3) else (2 if s==2 else 0)
    return 0


def run(rows, mode):
    hist=defaultdict(list); glob=[]
    eq=CAPITAL; peak=CAPITAL; max_dd=0.0
    pos=None; ei=-1; trades=wins=0
    all_rows=[r for r in rows if r[0] in TRAIN_MONTHS or r[0]==TEST_MONTH]
    train=[r for r in all_rows if r[0] in TRAIN_MONTHS]
    test=[r for r in all_rows if r[0]==TEST_MONTH]

    for i,r in enumerate(train[:-1]):
        s=S(r); b=abs(r[5]-r[2]); valid=r[5]!=0 and b/abs(r[5])>=MIN_BODY
        if valid:
            x=abs(train[i+1][5]-r[5])/b
            if x==x:
                key=s if mode=='STATE' else bucket(mode,s)
                hist[key].append(x); glob.append(x)

    for i,r in enumerate(test[:-1]):
        s=S(r); b=abs(r[5]-r[2]); valid=r[5]!=0 and b/abs(r[5])>=MIN_BODY
        gr=statistics.median(glob) if glob else 2.0
        key=bucket(mode,s)
        ratio=2.0 if mode=='FIXED2' else (statistics.median(hist[key]) if hist[key] else gr)
        if pos is None:
            sd=side(s); eff=max(b,abs(r[5])*MIN_BODY)
            tp=eff*ratio; sl=0.5*tp
            pos=(sd,r[5],tp,sl); ei=i
        if pos is not None and i>ei:
            sd,en,tpd,sld=pos; px=r[5]; gross_pct=None
            if sd==1:
                if px>=en+tpd: gross_pct=tpd/en*100
                elif px<=en-sld: gross_pct=-sld/en*100
            else:
                if px<=en-tpd: gross_pct=tpd/en*100
                elif px>=en+sld: gross_pct=-sld/en*100
            if gross_pct is not None:
                gross=gross_pct/100
                exit_value=CAPITAL*(1+gross)
                fees=CAPITAL*FEE_SIDE + max(exit_value,0.0)*FEE_SIDE
                net_pct=(CAPITAL*gross-fees)/CAPITAL*100
                eq += CAPITAL*net_pct/100
                trades += 1; wins += int(net_pct>0); pos=None; ei=-1
                peak=max(peak,eq); max_dd=max(max_dd,(peak-eq)/peak if peak else 0.0)
        if valid:
            x=abs(test[i+1][5]-r[5])/b
            if x==x:
                key=s if mode=='STATE' else bucket(mode,s)
                hist[key].append(x); glob.append(x)

    if pos is not None:
        sd,en,tpd,sld=pos; px=test[-1][5]; gross=sd*(px-en)/en if en>0 else 0.0
        exit_value=CAPITAL*(1+gross)
        fees=CAPITAL*FEE_SIDE + max(exit_value,0.0)*FEE_SIDE
        net_pct=(CAPITAL*gross-fees)/CAPITAL*100
        eq += CAPITAL*net_pct/100
        trades += 1; wins += int(net_pct>0); pos=None
        peak=max(peak,eq); max_dd=max(max_dd,(peak-eq)/peak if peak else 0.0)

    return (eq-CAPITAL)/CAPITAL*100, trades, wins, max_dd*100


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv'))
    a=ap.parse_args()
    print('EXCEL_OOS_FEE | tf=1h | train=May-Jul | test=Aug | fixed trade capital=$1000 | fee_side=0.13% | round_trip≈0.26% | past_only | SL=0.5xTP')
    for sym in SYMS:
        rows=load_rows(a.input,sym)
        print(sym)
        for mode in MODES:
            r=run(rows,mode)
            print(f'  {mode:7s} Aug_net={r[0]:+.2f}% trades={r[1]} win={100*r[2]/r[1] if r[1] else 0:.1f}% DD={r[3]:.2f}%')

if __name__=='__main__':
    main()
