from __future__ import annotations

import argparse, csv
from pathlib import Path

SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
MONTHS=("2026-05","2026-06","2026-07","2026-08")
FEE_SIDE=0.0013


def load(path: Path, symbol: str):
    out=[]
    with path.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol') != symbol:
                continue
            try:
                out.append((r['month'], int(float(r['timestamp'])), float(r['open']), float(r['high']), float(r['low']), float(r['close'])))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(out, key=lambda x:x[1])


def s(row):
    _,_,o,h,l,c=row
    j=c-o; k=h-c; m=h-o
    return int(k>j)+int(k>m)+int(l>j)


def run(rows, mode, lookback):
    n=correct=0; gross_sum=net_sum=0.0; gross_wins=net_wins=0
    for i in range(lookback, len(rows)-1):
        cur=rows[i]; nxt=rows[i+1]
        st=s(cur)
        highs=[r[3] for r in rows[i-lookback:i+1]]
        lows=[r[4] for r in rows[i-lookback:i+1]]
        prev_high=max(highs[:-1]); prev_low=min(lows[:-1])
        side=0
        # Entry is at the current close only when current candle is a confirmed
        # breakout of the preceding range. S is used solely as a filter.
        up_break=cur[5] > prev_high
        down_break=cur[5] < prev_low
        if mode=='ALL':
            if up_break: side=1
            elif down_break: side=-1
        elif mode=='S_FILTER':
            if up_break and st==3: side=1
            elif down_break and st==1: side=-1
        elif mode=='S_FILTER_REV':
            if up_break and st==1: side=1
            elif down_break and st==3: side=-1
        if side==0:
            continue
        actual=1 if nxt[5]>cur[5] else -1 if nxt[5]<cur[5] else 0
        gross=side*(nxt[5]-cur[5])/cur[5]*100
        net=gross-2*FEE_SIDE*100
        n+=1; correct+=int(actual==side); gross_sum+=gross; net_sum+=net
        gross_wins+=int(gross>0); net_wins+=int(net>0)
    return n,correct,gross_sum/n if n else 0.0,net_sum/n if n else 0.0,gross_wins,net_wins,gross_sum,net_sum


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); a=ap.parse_args()
    print('BREAKOUT_S_FILTER_WF | tf=1h | fee=0.13%/side | one-bar close-to-close | causal')
    print('Baseline = range breakout. S is only a filter; no direct S->BUY/SELL mapping.')
    print('Lookback windows: 3, 6, 12, 24 hours. Filter: S3 for upside breakout, S1 for downside breakout; reverse is control.')
    for sym in SYMBOLS:
        rows=load(a.input,sym); print(f'\n{sym}')
        for lb in (3,6,12,24):
            for mode in ('ALL','S_FILTER','S_FILTER_REV'):
                n,c,ag,an,gw,nw,gs,ns=run(rows,mode,lb)
                print(f'LB={lb:2} {mode:13} n={n:4} acc={100*c/n if n else 0:5.1f}% avg_gross={ag:+.4f}% avg_net={an:+.4f}% gross_win={100*gw/n if n else 0:4.1f}% net_win={100*nw/n if n else 0:4.1f}%')

if __name__=='__main__': main()
