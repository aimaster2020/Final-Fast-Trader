from __future__ import annotations

import argparse, csv
from collections import defaultdict
from pathlib import Path

SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
MONTHS=("2026-05","2026-06","2026-07","2026-08")
FEE_SIDE=0.0013


def load(path:Path,symbol:str):
    out=[]
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol')!=symbol: continue
            try:
                out.append((r['month'],int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close'])))
            except (KeyError,TypeError,ValueError): pass
    return sorted(out,key=lambda x:x[1])


def s(row):
    _,_,o,h,l,c=row
    j=c-o; k=h-c; m=h-o
    return int(k>j)+int(k>m)+int(l>j)


def sg(x,eps=1e-12):
    return 1 if x>eps else -1 if x<-eps else 0


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); a=ap.parse_args()
    print('TRANSITION_AUDIT | tf=1h | exact S | no lookahead')
    print('Measures current-body sign, next-body sign, S-conditioned transitions, and price-direction performance.')
    for sym in SYMBOLS:
        rows=load(a.input,sym)
        print(f'\n{sym}')
        for st in (0,1,2,3):
            n=cur_pos=next_pos=next_neg=delta_pos=delta_neg=price_pos=0
            for i in range(len(rows)-1):
                r=rows[i]; nxt=rows[i+1]
                if s(r)!=st: continue
                j=r[5]-r[2]; jn=nxt[5]-nxt[2]; d=jn-j; dp=nxt[5]-r[5]
                n+=1; cur_pos+=int(j>0); next_pos+=int(jn>0); next_neg+=int(jn<0)
                delta_pos+=int(d>0); delta_neg+=int(d<0); price_pos+=int(dp>0)
            print(f' S{st} n={n} current_body_pos={100*cur_pos/n if n else 0:.1f}% next_body_pos={100*next_pos/n if n else 0:.1f}% next_body_neg={100*next_neg/n if n else 0:.1f}% delta_pos={100*delta_pos/n if n else 0:.1f}% delta_neg={100*delta_neg/n if n else 0:.1f}% next_price_up={100*price_pos/n if n else 0:.1f}%')

        # Most direct transition rules: S1 predicts lower J, S3 predicts higher J.
        # Test whether that implies crossing zero in J often enough to give next-body direction.
        counts=defaultdict(lambda:[0,0])
        for i in range(len(rows)-1):
            r=rows[i]; nxt=rows[i+1]; st=s(r); j=sg(r[5]-r[2]); jn=sg(nxt[5]-nxt[2])
            key=(st,j)
            counts[key][0]+=1; counts[key][1]+=int(jn==j)
        print(' CONDITIONAL next-body same-sign accuracy:')
        for key in sorted(counts):
            n,c=counts[key]; print(f'  S{key[0]} current_sign={key[1]:+d} n={n} same_sign={100*c/n if n else 0:.1f}%')

if __name__=='__main__': main()
