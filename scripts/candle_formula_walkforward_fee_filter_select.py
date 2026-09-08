from __future__ import annotations

import argparse, csv, statistics
from collections import defaultdict
from pathlib import Path

SYMS=('BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT')
MONTHS=('2026-05','2026-06','2026-07','2026-08')
CAPITAL=1000.0
FEE_SIDE=0.0013
MIN_BODY=0.0001
THRESHOLDS=(0.0030,0.0050,0.0075,0.0100)
MODES=('FIXED2','STATE')


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

def bucket(mode,s): return s

def net_from_gross(gross, fee):
    exit_value=CAPITAL*(1+gross)
    fees=CAPITAL*fee + max(exit_value,0.0)*fee
    return (CAPITAL*gross-fees)/CAPITAL*100


def run_month(hist, glob, rows, mode, threshold, allow_update=True):
    eq=CAPITAL; pos=None; ei=-1; trades=wins=0
    peak=CAPITAL; max_dd=0.0
    for i,r in enumerate(rows[:-1]):
        s=S(r); b=abs(r[5]-r[2]); valid=r[5]!=0 and b/abs(r[5])>=MIN_BODY
        gr=statistics.median(glob) if glob else 2.0
        key=bucket(mode,s)
        ratio=2.0 if mode=='FIXED2' else (statistics.median(hist[key]) if hist[key] else gr)
        eff=max(b,abs(r[5])*MIN_BODY)
        pred_tp_pct=(eff*ratio/abs(r[5])) if r[5] else 0.0
        if pos is None and pred_tp_pct >= threshold:
            sd=side(s); tp=eff*ratio; sl=0.5*tp; pos=(sd,r[5],tp,sl); ei=i
        if pos is not None and i>ei:
            sd,en,tpd,sld=pos; px=r[5]; gross=None
            if sd==1:
                if px>=en+tpd: gross=tpd/en
                elif px<=en-sld: gross=-sld/en
            else:
                if px<=en-tpd: gross=tpd/en
                elif px>=en+sld: gross=-sld/en
            if gross is not None:
                net=net_from_gross(gross,FEE_SIDE)
                eq += CAPITAL*net/100; trades+=1; wins+=int(net>0); pos=None; ei=-1
                peak=max(peak,eq); max_dd=max(max_dd,(peak-eq)/peak if peak else 0.0)
        if allow_update and valid:
            x=abs(rows[i+1][5]-r[5])/b
            if x==x:
                hist[key].append(x); glob.append(x)
    if pos is not None:
        sd,en,tpd,sld=pos; px=rows[-1][5]
        gross=sd*(px-en)/en if en else 0.0
        net=net_from_gross(gross,FEE_SIDE)
        eq += CAPITAL*net/100; trades+=1; wins+=int(net>0); pos=None
        peak=max(peak,eq); max_dd=max(max_dd,(peak-eq)/peak if peak else 0.0)
    return (eq-CAPITAL)/CAPITAL*100,trades,wins,max_dd*100


def select_threshold(train_rows, mode):
    scores=[]
    for th in THRESHOLDS:
        hist=defaultdict(list); glob=[]
        total=0.0
        for month in MONTHS:
            rs=[r for r in train_rows if r[0]==month]
            if not rs: continue
            r=run_month(hist,glob,rs,mode,th,True)
            total += r[0]
        scores.append((total,th))
    return max(scores)[1]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); a=ap.parse_args()
    print('EXCEL_WF_FEE_FILTER_SELECT | tf=1h | fee_side=0.13% | round_trip≈0.26% | fixed trade capital=$1000 | SL=0.5xTP')
    print('Selection is walk-forward: threshold for month t is chosen only from earlier months; no August fitting.')
    for sym in SYMS:
        rows=load_rows(a.input,sym)
        print(sym)
        for mode in MODES:
            print(f'  {mode}')
            hist=defaultdict(list); glob=[]
            for mi,month in enumerate(MONTHS):
                rs=[r for r in rows if r[0]==month]
                if not rs: continue
                if mi==0:
                    th=0.0030
                    train_label='coldstart'
                else:
                    prior=[r for r in rows if MONTHS.index(r[0])<mi]
                    th=select_threshold(prior,mode)
                    train_label='prior_months'
                test_r=run_month(hist,glob,rs,mode,th,True)
                print(f'    {month} threshold={th*100:.2f}% source={train_label} net={test_r[0]:+.2f}% trades={test_r[1]} win={100*test_r[2]/test_r[1] if test_r[1] else 0:.1f}% DD={test_r[3]:.2f}%')
                # hist/glob already updated by test_r, so carry history forward.
        print()

if __name__=='__main__': main()
