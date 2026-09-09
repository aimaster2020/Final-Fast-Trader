from __future__ import annotations

import argparse, csv, itertools, math
from pathlib import Path

SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")


def load(path: Path, symbol: str):
    out=[]
    with path.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol') != symbol:
                continue
            try:
                out.append((int(float(r['timestamp'])), float(r['open']), float(r['high']), float(r['low']), float(r['close'])))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(out, key=lambda x:x[0])


def sign(x, eps=1e-12):
    if x > eps: return 1
    if x < -eps: return -1
    return 0


def components(r):
    _,o,h,l,c=r
    body=c-o
    upper=h-max(o,c)
    lower=min(o,c)-l
    rng=h-l
    oc=(o+c)/2-c
    hl=(h+l)/2-c
    return (body,upper,lower,rng,oc,hl)

NAMES=("BODY","UPPER","LOWER","RANGE","OC_CENTER","HL_CENTER")
COEFFS=(-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv'))
    ap.add_argument('--top',type=int,default=12)
    args=ap.parse_args()
    print('DELTA_CLOSE_GRID_SEARCH | tf=1h | fee=0 | no lookahead')
    print('Target = next_close-current_close. Fixed coefficient grid; no fitted regression.')
    print('Formula = sum(coefficient_i * current_component_i). Best formulas minimize MAE, then RMSE.')

    # Keep the search compact: BODY, UPPER, LOWER plus RANGE, and require at least 1 nonzero coefficient.
    specs=[]
    for coef in itertools.product(COEFFS, repeat=4):
        if all(x==0 for x in coef):
            continue
        specs.append(coef)

    for sym in SYMBOLS:
        rows=load(args.input,sym)
        scored=[]
        for coef in specs:
            abs_sum=sq_sum=0.0; correct=n=0
            for i in range(len(rows)-1):
                cur=rows[i]; nxt=rows[i+1]
                body,upper,lower,rng,_,_=components(cur)
                pred=coef[0]*body+coef[1]*upper+coef[2]*lower+coef[3]*rng
                actual=nxt[4]-cur[4]
                err=pred-actual
                abs_sum+=abs(err); sq_sum+=err*err
                correct+=int(sign(pred)==sign(actual)); n+=1
            mae=abs_sum/n if n else math.inf
            rmse=math.sqrt(sq_sum/n) if n else math.inf
            acc=100*correct/n if n else 0.0
            scored.append((mae,rmse,-acc,coef,n))
        scored.sort()
        print(sym)
        for mae,rmse,negacc,coef,n in scored[:args.top]:
            terms=[]
            for name,k in zip(NAMES[:4],coef):
                if k==0: continue
                terms.append(f'{k:g}*{name}')
            formula=' + '.join(terms).replace('+ -','- ')
            print(f'  MAE={mae:.6g} RMSE={rmse:.6g} dir_acc={-negacc:.2f}% | {formula}')
        print()

if __name__=='__main__':
    main()
