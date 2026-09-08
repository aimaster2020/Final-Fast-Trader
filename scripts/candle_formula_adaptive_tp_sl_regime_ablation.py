from __future__ import annotations

import argparse, csv, statistics
from collections import defaultdict
from pathlib import Path

SYMS=('BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT')
MONTHS=('2026-05','2026-06','2026-07','2026-08')
CAPITAL=1000.0
MIN_BODY=0.0001


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

def pnl_pct(sd,en,px): return sd*(px-en)/en*100 if en>0 else 0.0

def bucket(mode,s):
    if mode=='STATE': return s
    if mode=='S12_S3': return 12 if s in (1,2) else (3 if s==3 else 0)
    if mode=='S13_S2': return 13 if s in (1,3) else (2 if s==2 else 0)
    return 0

def run(rows, mode):
    hist=defaultdict(list); glob=[]; eq=CAPITAL; peak=CAPITAL; dd=0.0
    pos=None; ei=-1; trades=wins=0
    monthly={m:0.0 for m in MONTHS}
    for month in MONTHS:
        rs=[r for r in rows if r[0]==month]
        if not rs: continue
        month_start=eq
        for i,r in enumerate(rs[:-1]):
            s=S(r); b=abs(r[5]-r[2]); valid=r[5]!=0 and b/abs(r[5])>=MIN_BODY
            gr=statistics.median(glob) if glob else 2.0
            key=bucket(mode,s)
            ratio=statistics.median(hist[key]) if hist[key] else gr
            if pos is None:
                sd=side(s); eff=max(b,abs(r[5])*MIN_BODY); tp=eff*ratio; sl=tp*0.5
                pos=(sd,r[5],tp,sl); ei=i
            if pos is not None and i>ei:
                sd,en,tpd,sld=pos; px=r[5]; hit=None
                if sd==1:
                    if px>=en+tpd: hit=tpd/en*100
                    elif px<=en-sld: hit=-sld/en*100
                else:
                    if px<=en-tpd: hit=tpd/en*100
                    elif px>=en+sld: hit=-sld/en*100
                if hit is not None:
                    eq += CAPITAL*hit/100; trades+=1; wins+=int(hit>0); pos=None; ei=-1
                    peak=max(peak,eq); dd=max(dd,(peak-eq)/peak if peak else 0.0)
            if valid:
                ratio_actual=abs(rs[i+1][5]-r[5])/b
                if ratio_actual==ratio_actual:
                    hist[s].append(ratio_actual) if mode=='STATE' else hist[bucket(mode,s)].append(ratio_actual)
                    glob.append(ratio_actual)
        if pos is not None:
            sd,en,tpd,sld=pos; px=rs[-1][5]; ret=pnl_pct(sd,en,px)
            eq += CAPITAL*ret/100; trades+=1; wins+=int(ret>0); pos=None; ei=-1
            peak=max(peak,eq); dd=max(dd,(peak-eq)/peak if peak else 0.0)
        monthly[month]=(eq-month_start)/CAPITAL*100
    return sum(monthly.values()), trades, wins, dd, monthly


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); a=ap.parse_args()
    print('EXCEL_REGIME_ABLATION | tf=1h | continuous May-Aug | fixed trade capital=$1000 | close_only | past_only | SL=0.5xTP')
    print('STATE=4 buckets | S1S2=pooled S1/S2 + separate S3 | S1S3=pooled S1/S3 + separate S2')
    for sym in SYMS:
        rows=load_rows(a.input,sym)
        out={m:run(rows,m) for m in ('STATE','S12_S3','S13_S2')}
        print(sym)
        for mode in ('STATE','S12_S3','S13_S2'):
            r=out[mode]
            print(f'  {mode:7s} sum={r[0]:+.2f}% trades={r[1]} win={100*r[2]/r[1] if r[1] else 0:.1f}% DD={r[3]*100:.2f}%')
        print(f'  delta_S12S3_vs_STATE={out["S12_S3"][0]-out["STATE"][0]:+.2f}pp')

if __name__=='__main__': main()
