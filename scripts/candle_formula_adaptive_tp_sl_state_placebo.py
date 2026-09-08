from __future__ import annotations

import argparse, csv, itertools, statistics
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


def run(rows, mapping):
    # History is always assigned to the REAL observed S bucket.
    # mapping only changes which bucket is consulted for the current S.
    # Identity mapping (0,1,2,3) is the real STATE model.
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
            key=mapping[s]
            ratio=statistics.median(hist[key]) if hist[key] else gr
            if pos is None:
                sd=side(s); eff=max(b,abs(r[5])*MIN_BODY); tp=eff*ratio; sl=tp*0.5
                pos=(sd,r[5],tp,sl); ei=i
            if pos is not None and i>ei:
                sd,en,tpd,sld=pos; px=r[5]; hit=None
                if sd==1:
                    if px>=en+tpd: hit=(en+tpd-en)/en*100
                    elif px<=en-sld: hit=(en-sld-en)/en*100
                else:
                    if px<=en-tpd: hit=(en-(en-tpd))/en*100
                    elif px>=en+sld: hit=(en-(en+sld))/en*100
                if hit is not None:
                    eq += CAPITAL*hit/100; trades+=1; wins+=int(hit>0); pos=None; ei=-1
                    peak=max(peak,eq); dd=max(dd,(peak-eq)/peak if peak else 0.0)
            if valid:
                ratio_actual=abs(rs[i+1][5]-r[5])/b
                if ratio_actual==ratio_actual:
                    # Keep target observations in their REAL S bucket.
                    # The permutation changes lookup only, preventing cancellation.
                    hist[s].append(ratio_actual)
                    glob.append(ratio_actual)
        if pos is not None:
            sd,en,tpd,sld=pos; px=rs[-1][5]; ret=pnl_pct(sd,en,px)
            eq += CAPITAL*ret/100; trades+=1; wins+=int(ret>0); pos=None; ei=-1
            peak=max(peak,eq); dd=max(dd,(peak-eq)/peak if peak else 0.0)
        monthly[month]=(eq-month_start)/CAPITAL*100
    return sum(monthly.values()), trades, wins, dd, monthly


def trace_entries(rows, mapping, limit=20):
    hist=defaultdict(list); glob=[]; pos=False; traces=[]
    for month in MONTHS:
        rs=[r for r in rows if r[0]==month]
        for i,r in enumerate(rs[:-1]):
            s=S(r); b=abs(r[5]-r[2]); valid=r[5]!=0 and b/abs(r[5])>=MIN_BODY
            gr=statistics.median(glob) if glob else 2.0
            key=mapping[s]
            ratio=statistics.median(hist[key]) if hist[key] else gr
            if not pos:
                traces.append((month,i,s,key,ratio))
                pos=True
            # Mirror the backtest's close-only position lifetime: the next candle
            # gets a chance to close the position. This trace only tracks entry ratios.
            if pos and i > 0:
                # We only need entry-ratio diagnostics, so approximate a one-candle hold
                # to expose lookup differences without changing the actual backtest.
                pos=False
            if valid:
                ratio_actual=abs(rs[i+1][5]-r[5])/b
                if ratio_actual==ratio_actual:
                    hist[s].append(ratio_actual); glob.append(ratio_actual)
            if len(traces)>=limit: return traces
    return traces


def diagnostic(rows):
    ident=(0,1,2,3); probe=(1,0,2,3)
    a=trace_entries(rows,ident,20); b=trace_entries(rows,probe,20)
    diffs=[]
    for x,y in zip(a,b):
        if abs(x[4]-y[4])>1e-12: diffs.append((x,y))
    meds={s:statistics.median([abs(rs[i+1][5]-r[5])/abs(r[5]-r[2]) for i,r in enumerate(rows[:-1]) if S(r)==s and abs(r[5]-r[2])>0 and r[5]!=0]) for s in range(4)}
    print(f'  DIAG overall_ratio_medians={meds}')
    print(f'  DIAG first20_entry_ratios_identity={[round(x[4],4) for x in a]}')
    print(f'  DIAG first20_entry_ratios_probe={ [round(x[4],4) for x in b] }')
    print(f'  DIAG differing_first20={len(diffs)}/20')
    if diffs:
        print(f'  DIAG first_difference identity={diffs[0][0]} probe={diffs[0][1]}')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); ap.add_argument('--diagnostic',action='store_true'); a=ap.parse_args()
    print('EXCEL_STATE_PLACEBO | tf=1h | exact S0-S3 | fixed trade capital=$1000 | close_only | past_only')
    print('Actual STATE vs 24 lookup permutations; history remains bucketed by real S; SL=0.5xTP')
    for sym in SYMS:
        rows=load_rows(a.input,sym)
        if a.diagnostic: diagnostic(rows)
        maps=list(itertools.permutations(range(4))); res=[]; actual=(0,0,0,0,{})
        for mp in maps:
            r=run(rows,mp); res.append((mp,r))
            if mp==(0,1,2,3): actual=r
        sums=[x[1][0] for x in res]
        mean=sum(sums)/len(sums); med=statistics.median(sums)
        better=sum(x>actual[0]+1e-12 for x in sums); worse=sum(x<actual[0]-1e-12 for x in sums)
        print(f'{sym} STATE_sum={actual[0]:+.2f}% placebo_mean={mean:+.2f}% placebo_median={med:+.2f}% better_than_STATE={better}/23 worse={worse}/23 range=[{min(sums):+.2f},{max(sums):+.2f}]')
        print(f'  STATE trades={actual[1]} win={100*actual[2]/actual[1] if actual[1] else 0:.1f}% DD={actual[3]:.2f}%')
        best=max(res,key=lambda x:x[1][0]); worst=min(res,key=lambda x:x[1][0])
        print(f'  best_perm={best[0]} sum={best[1][0]:+.2f}% | worst_perm={worst[0]} sum={worst[1][0]:+.2f}%')

if __name__=='__main__': main()
