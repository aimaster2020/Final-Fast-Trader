from __future__ import annotations

import argparse, csv
from pathlib import Path

SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
MONTHS=("2026-05","2026-06","2026-07","2026-08")
FEE_RT=0.0026
HORIZONS=(1,6,12,24)
BODY_THRESHOLDS=(0.0005,0.001,0.002,0.003,0.005,0.01)


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


def side_from_s(st):
    if st==1: return -1
    if st==3: return 1
    return 0


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); a=ap.parse_args()
    print('SELECTIVE_EDGE_WF | tf=1h | fee=0.13%/side | past_only')
    print('Signals only S1/S3. Fixed current-body/price thresholds; no parameter fitting.')
    print('Break-even gross return for a round trip is 0.2600%.')
    for sym in SYMBOLS:
        rows=load(a.input,sym)
        print(f'\n{sym}')
        for h in HORIZONS:
            for th in BODY_THRESHOLDS:
                vals=[]
                bys={1:[],3:[]}
                for i in range(len(rows)-h):
                    cur=rows[i]; fut=rows[i+h]; st=s(cur)
                    sd=side_from_s(st)
                    if sd==0: continue
                    body=abs(cur[5]-cur[2]); px=abs(cur[5])
                    if px<=0 or body/px < th: continue
                    gross=sd*(fut[5]-cur[5])/cur[5]
                    vals.append(gross)
                    bys[st].append(gross)
                n=len(vals)
                if n:
                    avg=sum(vals)/n*100
                    net=avg-FEE_RT*100
                    acc=sum(1 for x in vals if x>0)/n*100
                    s1=(sum(bys[1])/len(bys[1])*100) if bys[1] else 0
                    s3=(sum(bys[3])/len(bys[3])*100) if bys[3] else 0
                    print(f'H={h:2d} body>={th*100:.02f}% n={n:4d} acc={acc:5.1f}% avg_gross={avg:+.4f}% avg_net={net:+.4f}% S1={s1:+.4f}% S3={s3:+.4f}%')

if __name__=='__main__': main()
