from __future__ import annotations

import argparse, csv, itertools, math
from collections import defaultdict
from pathlib import Path

SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
COEFFS=(-1.0,-0.5,0.0,0.5,1.0)


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


def components(r):
    _,o,h,l,c=r
    body=c-o
    upper=h-max(o,c)
    lower=min(o,c)-l
    rng=h-l
    return body, upper, lower, rng


def regime(r, kind):
    body, upper, lower, rng = components(r)
    eps=1e-12
    body_sign = 1 if body>eps else (-1 if body<-eps else 0)
    if kind == 'BODY_SIGN':
        return (body_sign,)
    if kind == 'WICK_DOMINANCE':
        return ('U' if upper>lower else ('L' if lower>upper else 'E'),)
    if kind == 'BODY_WICK_SIGN':
        return (body_sign, 'U' if upper>lower else ('L' if lower>upper else 'E'))
    # Three broad body/range bins; computed only from current candle.
    ratio = abs(body)/rng if rng>eps else 0.0
    bucket = 0 if ratio < 0.33 else (1 if ratio < 0.66 else 2)
    return (body_sign, bucket)


def fit_coeff(train_rows, kind, min_count):
    # For each current-candle regime, choose the fixed coefficient triple
    # minimizing training MAE for target next_close-current_close.
    grouped=defaultdict(list)
    for i in range(len(train_rows)-1):
        cur=train_rows[i]; nxt=train_rows[i+1]
        grouped[regime(cur, kind)].append((components(cur), nxt[4]-cur[4]))
    models={}
    for key, data in grouped.items():
        if len(data) < min_count:
            models[key]=(0.0,0.0,0.0,len(data))
            continue
        best=None
        for coef in itertools.product(COEFFS, repeat=3):
            ae=0.0
            for (body,upper,lower,rng), actual in data:
                pred=coef[0]*body + coef[1]*upper + coef[2]*lower
                ae += abs(pred-actual)
            score=(ae/len(data), coef)
            if best is None or score < best:
                best=score
        models[key]=(*best[1],len(data))
    return models


def eval_model(test_rows, kind, models):
    ae=sq=0.0; covered=correct=n=0
    for i in range(len(test_rows)-1):
        cur=test_rows[i]; nxt=test_rows[i+1]
        key=regime(cur, kind)
        body,upper,lower,_=components(cur)
        coef=models.get(key)
        if coef is None or coef[3] < 1:
            pred=0.0
        else:
            pred=coef[0]*body + coef[1]*upper + coef[2]*lower
            covered += 1
        actual=nxt[4]-cur[4]
        err=pred-actual
        ae+=abs(err); sq+=err*err
        correct += int((pred>0 and actual>0) or (pred<0 and actual<0) or (pred==0 and actual==0))
        n += 1
    return ae/n, math.sqrt(sq/n), 100*correct/n, 100*covered/n


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv'))
    ap.add_argument('--train',type=float,default=0.70)
    ap.add_argument('--min-count',type=int,default=40)
    args=ap.parse_args()
    print(f'PIECEWISE_DELTA_SEARCH | tf=1h | fee=0 | train={args.train:.0%} | no lookahead')
    print('Target = next_close-current_close. Each current-candle regime gets its own fixed body/upper/lower coefficients.')
    print('Coefficient grid = (-1,-0.5,0,0.5,1); coefficients fit on train only.')
    kinds=('BODY_SIGN','WICK_DOMINANCE','BODY_WICK_SIGN','BODY_SIGN_BODY_RATIO')
    for sym in SYMBOLS:
        rows=load(args.input,sym)
        cut=max(2,int(len(rows)*args.train))
        train=rows[:cut]; test=rows[cut-1:]
        print(sym)
        print('  ZERO_TEST', end='')
        zero=eval_model(test,'BODY_SIGN',{})
        print(f' MAE={zero[0]:.6g} RMSE={zero[1]:.6g} dir_acc={zero[2]:.2f}%')
        scored=[]
        for kind in kinds:
            models=fit_coeff(train,kind,args.min_count)
            mae,rmse,acc,cov=eval_model(test,kind,models)
            scored.append((mae,rmse,-acc,-cov,kind,models))
        scored.sort()
        for mae,rmse,negacc,negcov,kind,models in scored:
            print(f'  {kind:22s} MAE={mae:.6g} RMSE={rmse:.6g} dir_acc={-negacc:.2f}% coverage={-negcov:.1f}%')
            if kind==scored[0][4]:
                for key,coef in sorted(models.items(), key=lambda kv: str(kv[0])):
                    print(f'    {key}: body={coef[0]:g} upper={coef[1]:g} lower={coef[2]:g} n={coef[3]}')
        print()

if __name__=='__main__':
    main()
