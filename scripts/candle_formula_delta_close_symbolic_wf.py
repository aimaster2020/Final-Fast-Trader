from __future__ import annotations
import argparse, csv, math
from pathlib import Path

SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")


def load(path: Path, symbol: str):
    out=[]
    with path.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol') != symbol:
                continue
            try:
                out.append((int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close'])))
            except (KeyError,TypeError,ValueError):
                continue
    return sorted(out,key=lambda x:x[0])


def metrics(rows, fn):
    ys=[]; ps=[]
    for i in range(len(rows)-1):
        _,o,h,l,c=rows[i]
        y=rows[i+1][4]-c
        p=fn(o,h,l,c)
        ys.append(y); ps.append(p)
    n=len(ys)
    if not n: return 0,0,0,0
    mae=sum(abs(p-y) for p,y in zip(ps,ys))/n
    rmse=math.sqrt(sum((p-y)**2 for p,y in zip(ps,ys))/n)
    acc=sum((p>0 and y>0) or (p<0 and y<0) or (p==0 and y==0) for p,y in zip(ps,ys))/n*100
    return n,mae,rmse,acc


def candidates():
    return [
        ("ZERO", lambda o,h,l,c: 0.0),
        ("BODY", lambda o,h,l,c: c-o),
        ("REV_BODY", lambda o,h,l,c: o-c),
        ("UPPER_WICK", lambda o,h,l,c: h-c),
        ("LOWER_WICK_NEG", lambda o,h,l,c: l-c),
        ("LOWER_WICK", lambda o,h,l,c: c-l),
        ("RANGE_HALF", lambda o,h,l,c: (h-l)/2),
        ("RANGE_NEG_HALF", lambda o,h,l,c: -(h-l)/2),
        ("UPPER_MINUS_LOWER", lambda o,h,l,c: (h-c)-(c-l)),
        ("LOWER_MINUS_UPPER", lambda o,h,l,c: (c-l)-(h-c)),
        ("BODY_PLUS_UPPER", lambda o,h,l,c: (c-o)+(h-c)),
        ("BODY_MINUS_UPPER", lambda o,h,l,c: (c-o)-(h-c)),
        ("BODY_PLUS_LOWER", lambda o,h,l,c: (c-o)+(c-l)),
        ("BODY_MINUS_LOWER", lambda o,h,l,c: (c-o)-(c-l)),
        ("MIDPOINT_MINUS_CLOSE", lambda o,h,l,c: (h+l)/2-c),
        ("CLOSE_MINUS_MIDPOINT", lambda o,h,l,c: c-(h+l)/2),
        ("OC_CENTER_DELTA", lambda o,h,l,c: (o+c)/2-c),
        ("HL_CENTER_DELTA", lambda o,h,l,c: (h+l)/2-c),
        ("OPEN_TO_HIGH_HALF", lambda o,h,l,c: (h-o)/2),
        ("OPEN_TO_LOW_HALF", lambda o,h,l,c: (l-o)/2),
        ("HIGH_PLUS_LOW_MINUS_2C", lambda o,h,l,c: h+l-2*c),
    ]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); a=ap.parse_args()
    print('SYMBOLIC_DELTA_CLOSE_WF | tf=1h | fee=0 | no lookahead')
    print('Target = Close_next - Close_current. Candidates are simple algebraic OHLC transforms only.')
    for sym in SYMBOLS:
        rows=load(a.input,sym)
        res=[]
        for name,fn in candidates():
            n,mae,rmse,acc=metrics(rows,fn); res.append((mae,rmse,-acc,name,n))
        res.sort()
        print(sym)
        for mae,rmse,nacc,name,n in res:
            acc=-nacc
            print(f' {name:24s} MAE={mae:.6g} RMSE={rmse:.6g} dir_acc={acc:5.2f}%')

if __name__=='__main__': main()
