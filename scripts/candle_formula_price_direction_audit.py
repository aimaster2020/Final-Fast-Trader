from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

SYMS = ('BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT')
MONTHS = ('2026-05','2026-06','2026-07','2026-08')
FEE_SIDE = 0.0013
ROUND_TRIP_FEE = 2.0 * FEE_SIDE
MIN_BODY = 0.0001


def load_rows(p: Path, sym: str):
    out=[]
    with p.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol') != sym:
                continue
            try:
                out.append((r['month'], int(float(r['timestamp'])), float(r['open']), float(r['high']), float(r['low']), float(r['close'])))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(out, key=lambda x:x[1])


def S(r):
    _,_,o,h,l,c = r
    j = c-o
    k = h-c
    m = h-o
    return int(k>j) + int(k>m) + int(l>j)


def side(s):
    return 1 if s in (2,3) else -1


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('reports/prepared_price_action_1h.csv'))
    a=ap.parse_args()
    print('EXCEL_PRICE_DIRECTION_AUDIT | tf=1h | exact S0-S3 | next_close direction | fee_side=0.13% | past_only')
    print('Trade-side test uses BUY for S2/S3 and SELL for S0/S1; target is next_close-current_close (the quantity traded).')
    for sym in SYMS:
        rows=load_rows(a.input,sym)
        stats=defaultdict(lambda: {'n':0,'correct':0,'gross_sum':0.0,'net_sum':0.0,'gt_fee':0})
        allst=stats['all']
        for i,r in enumerate(rows[:-1]):
            s=S(r)
            sd=side(s)
            c=r[5]; nc=rows[i+1][5]
            if c <= 0:
                continue
            delta=(nc-c)/c
            correct = (delta>0 and sd==1) or (delta<0 and sd==-1)
            gross=sd*delta*100.0
            net=gross - ROUND_TRIP_FEE*100.0
            keys=('all',s)
            for key in keys:
                z=stats[key]
                z['n']+=1
                z['correct']+=int(correct)
                z['gross_sum']+=gross
                z['net_sum']+=net
                z['gt_fee']+=int(gross > ROUND_TRIP_FEE*100.0)
        print(sym)
        z=stats['all']
        print(f"  all n={z['n']} price_direction_acc={100*z['correct']/z['n'] if z['n'] else 0:.1f}% gross_mean={z['gross_sum']/z['n'] if z['n'] else 0:+.4f}% net_mean={z['net_sum']/z['n'] if z['n'] else 0:+.4f}% net_win={100*z['gt_fee']/z['n'] if z['n'] else 0:.1f}%")
        for s in range(4):
            z=stats[s]
            print(f"  S{s} n={z['n']} acc={100*z['correct']/z['n'] if z['n'] else 0:.1f}% gross_mean={z['gross_sum']/z['n'] if z['n'] else 0:+.4f}% net_mean={z['net_sum']/z['n'] if z['n'] else 0:+.4f}% net_win={100*z['gt_fee']/z['n'] if z['n'] else 0:.1f}%")

if __name__=='__main__':
    main()
