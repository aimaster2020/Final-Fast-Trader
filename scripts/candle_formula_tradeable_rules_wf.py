from __future__ import annotations

import argparse, csv
from pathlib import Path

SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
MONTHS=("2026-05","2026-06","2026-07","2026-08")
FEE=0.0013


def load(path:Path,symbol:str):
    out=[]
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol')!=symbol: continue
            try:
                out.append((r['month'],int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close'])))
            except (KeyError,TypeError,ValueError):
                pass
    return sorted(out,key=lambda x:x[1])


def s(row):
    _,_,o,h,l,c=row
    j=c-o; k=h-c; m=h-o
    return int(k>j)+int(k>m)+int(l>j)


def sign(x,eps=1e-12):
    if x>eps: return 1
    if x<-eps: return -1
    return 0


def signal(row, rule):
    st=s(row); jb=sign(row[5]-row[2])
    # S3 = delta-J positive, S1 = delta-J negative.
    # S2 is deliberately excluded from the primary S-only logic.
    if rule=='S13':
        return 1 if st==3 else (-1 if st==1 else 0)
    if rule=='BODY':
        return jb
    if rule=='REV_BODY':
        return -jb
    if rule=='S3_CONT_S1_REV':
        if not jb: return 0
        if st==3: return jb
        if st==1: return -jb
        return 0
    if rule=='S3_REV_S1_CONT':
        if not jb: return 0
        if st==3: return -jb
        if st==1: return jb
        return 0
    if rule=='S3_ONLY_BODY':
        return jb if st==3 else 0
    if rule=='S1_ONLY_REV_BODY':
        return -jb if st==1 else 0
    raise ValueError(rule)


def run(rows,rule):
    net=0.0; wins=trades=correct=predn=0; monthly=[]
    for month in MONTHS:
        mr=[r for r in rows if r[0]==month]
        mnet=0.0; mtr=mwin=mcor=mp=0
        for i in range(len(mr)-1):
            cur,nxt=mr[i],mr[i+1]
            sd=signal(cur,rule)
            if not sd: continue
            predn+=1; mp+=1
            price_move=sign(nxt[5]-cur[5])
            correct += int(sd==price_move); mcor += int(sd==price_move)
            gross=sd*(nxt[5]-cur[5])/cur[5]
            r=gross-2*FEE
            net += r*100; mnet += r*100
            trades+=1; mtr+=1; wins+=int(r>0); mwin+=int(r>0)
        monthly.append((month,mnet,mtr,mwin,mcor,mp))
    return net,trades,wins,correct,predn,monthly


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); a=ap.parse_args()
    rules=('S13','BODY','REV_BODY','S3_CONT_S1_REV','S3_REV_S1_CONT','S3_ONLY_BODY','S1_ONLY_REV_BODY')
    print('TRADEABLE_RULES_WF | tf=1h | fee=0.13%/side | one-bar close-to-close | no lookahead')
    print('S13 = exact Excel body-change direction mapping (S1=-1, S3=+1), tested as PRICE direction only as a diagnostic.')
    print('Other rules test whether S1/S3 should modulate current candle-body direction.')
    for sym in SYMBOLS:
        rows=load(a.input,sym)
        print(f'\n{sym}')
        for rule in rules:
            net,tr,w,cor,pn,mon=run(rows,rule)
            acc=100*cor/pn if pn else 0
            print(f'{rule:18s} price_acc={acc:.1f}% trades={tr} win={100*w/tr if tr else 0:.1f}% net_sum={net:+.2f}%')
            print('  '+' '.join(f'{m}={r:+.2f}%' for m,r,*_ in mon))

if __name__=='__main__': main()
