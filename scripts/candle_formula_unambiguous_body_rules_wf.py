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


def evaluate(rows, horizon, mode):
    total=correct=gross_sum=net_sum=0.0
    gross_wins=net_wins=0
    by_month={m:[0,0,0.0,0.0] for m in MONTHS}
    for i in range(len(rows)-horizon):
        cur=rows[i]; fut=rows[i+horizon]
        st=s(cur); j=cur[5]-cur[2]
        side=0
        if st==1:
            if mode=='UNAMBIG' and j<0: side=-1
            elif mode=='REV' and j>0: side=-1
        elif st==3:
            if mode=='UNAMBIG' and j>0: side=1
            elif mode=='REV' and j<0: side=1
        if side==0:
            continue
        total+=1
        actual=1 if fut[5]>cur[5] else -1 if fut[5]<cur[5] else 0
        correct += int(actual==side)
        gross=side*(fut[5]-cur[5])/cur[5]*100
        net=gross-2*FEE_SIDE*100
        gross_sum+=gross; net_sum+=net; gross_wins+=int(gross>0); net_wins+=int(net>0)
        z=by_month[cur[0]]; z[0]+=1; z[1]+=int(actual==side); z[2]+=gross; z[3]+=net
    return total,correct,gross_sum,net_sum,gross_wins,net_wins,by_month


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); a=ap.parse_args()
    print('UNAMBIG_BODY_RULES_WF | tf=1h | fee=0.13%/side | close-to-close | causal')
    print('UNAMBIG: S1+bearish body => SHORT; S3+bullish body => LONG. REV is opposite conditional mapping.')
    for sym in SYMBOLS:
        rows=load(a.input,sym); print(f'\n{sym}')
        for h in (1,3,6,12,24):
            for mode in ('UNAMBIG','REV'):
                n,c,gs,ns,gw,nw,months=evaluate(rows,h,mode)
                print(f'H={h:2} {mode:7} n={int(n):4} acc={100*c/n if n else 0:5.1f}% avg_gross={gs/n if n else 0:+.4f}% avg_net={ns/n if n else 0:+.4f}% gross_win={100*gw/n if n else 0:4.1f}% net_win={100*nw/n if n else 0:4.1f}%')
                print('  '+' '.join(f'{m}:{z[3]:+.2f}%({int(z[0])})' for m,z in months.items()))

if __name__=='__main__': main()
