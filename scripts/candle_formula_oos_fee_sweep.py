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
FEE_LEVELS=(0.0,0.0005,0.0010)  # fee per side as fraction of notional


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


def trade_return_pct(sd,en,px,fee):
    if en<=0: return 0.0
    gross=sd*(px-en)/en
    # Fixed $1000 trade capital. Apply fee on both entry and exit notionals.
    # Entry notional is CAPITAL; exit notional is CAPITAL*(1+gross).
    exit_value=CAPITAL*(1+gross)
    fees=CAPITAL*fee + max(exit_value,0.0)*fee
    net_value=CAPITAL+CAPITAL*gross-fees
    return (net_value-CAPITAL)/CAPITAL*100


def run(rows, mode, fee):
    hist=defaultdict(list); glob=[]
    eq=CAPITAL; peak=CAPITAL; max_dd=0.0
    pos=None; ei=-1; trades=wins=0
    test_start=0
    all_rows=[r for r in rows if r[0] in TRAIN_MONTHS or r[0]==TEST_MONTH]
    train=[r for r in all_rows if r[0] in TRAIN_MONTHS]
    test=[r for r in all_rows if r[0]==TEST_MONTH]

    # Build history from May-Jul only.
    for rs in (train,):
        for i,r in enumerate(rs[:-1]):
            s=S(r); b=abs(r[5]-r[2]); valid=r[5]!=0 and b/abs(r[5])>=MIN_BODY
            if valid:
                x=abs(rs[i+1][5]-r[5])/b
                if x==x:
                    key=s if mode=='STATE' else bucket(mode,s)
                    hist[key].append(x); glob.append(x)

    # Start August with frozen May-Jul history, then update rolling past-only within August.
    month_start=eq
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
                fees=CAPITAL*fee + max(exit_value,0.0)*fee
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
        fees=CAPITAL*fee + max(exit_value,0.0)*fee
        net_pct=(CAPITAL*gross-fees)/CAPITAL*100
        eq += CAPITAL*net_pct/100
        trades += 1; wins += int(net_pct>0); pos=None
        peak=max(peak,eq); max_dd=max(max_dd,(peak-eq)/peak if peak else 0.0)

    return (eq-CAPITAL)/CAPITAL*100, trades, wins, max_dd*100


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv'))
    a=ap.parse_args()
    print('EXCEL_OOS_FEE_SWEEP | tf=1h | train=May-Jul | test=Aug | fixed trade capital=$1000 | past_only | SL=0.5xTP')
    print('Fee is charged per side; total round-trip is approximately 2x fee for unchanged notional.')
    for sym in SYMS:
        rows=load_rows(a.input,sym)
        print(sym)
        for fee in FEE_LEVELS:
            label=f'fee_side={fee*100:.2f}%'
            vals=[]
            for mode in MODES:
                r=run(rows,mode,fee); vals.append((mode,r))
            line=' '.join(f'{m}={r[0]:+.2f}%/{r[1]}t/{(100*r[2]/r[1] if r[1] else 0):.1f}%w/DD{r[3]:.2f}%' for m,r in vals)
            print(f'  {label} | {line}')

if __name__=='__main__':
    main()
