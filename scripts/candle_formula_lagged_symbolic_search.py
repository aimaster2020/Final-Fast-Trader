from __future__ import annotations

import argparse, csv, itertools, math
from pathlib import Path

SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
COEFFS=(-1.0,-0.5,-0.25,0.0,0.25,0.5,1.0)


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


def feats(cur, prev):
    _,o,h,l,c=cur
    _,po,ph,pl,pc=prev
    body=c-o
    pbody=pc-po
    upper=h-max(o,c)
    pupper=ph-max(po,pc)
    lower=min(o,c)-l
    plower=min(po,pc)-pl
    rng=h-l
    prng=ph-pl
    # Dollar-space features, intentionally simple and interpretable.
    return (
        body,
        pbody,
        upper-lower,
        pupper-plower,
        rng-prng,
        c-pc,
    )

NAMES=("BODY0","BODY1","WICKDIFF0","WICKDIFF1","RANGE_DELTA","CLOSE_DELTA0")


def score(rows, coef, start, end):
    abs_sum=sq_sum=0.0
    correct=n=0
    for i in range(max(1,start), min(end,len(rows)-1)):
        x=feats(rows[i], rows[i-1])
        pred=sum(a*b for a,b in zip(coef,x))
        actual=rows[i+1][4]-rows[i][4]
        err=pred-actual
        abs_sum += abs(err)
        sq_sum += err*err
        correct += int(sign(pred)==sign(actual))
        n += 1
    if not n:
        return math.inf, math.inf, 0.0, 0
    return abs_sum/n, math.sqrt(sq_sum/n), 100*correct/n, n


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('reports/prepared_price_action_1h.csv'))
    ap.add_argument('--top', type=int, default=10)
    args=ap.parse_args()
    print('LAGGED_SYMBOLIC_SEARCH | tf=1h | fee=0 | no lookahead | train=70%')
    print('Target = next_close-current_close. Current + previous candle numeric formulas only.')
    print('Coefficients selected on train only; no fitted regression or ML.')

    specs=list(itertools.product(COEFFS, repeat=6))
    specs=[s for s in specs if any(v != 0 for v in s)]

    for sym in SYMBOLS:
        rows=load(args.input,sym)
        cut=max(2,int(len(rows)*0.70))
        scored=[]
        for coef in specs:
            tr=score(rows,coef,1,cut)
            te=score(rows,coef,cut,len(rows))
            if not math.isfinite(te[0]):
                continue
            scored.append((te[0],te[1],-te[2],tr[0],coef,te[3]))
        scored.sort()
        zero=score(rows,(0,0,0,0,0,0),cut,len(rows))
        print(sym)
        print(f'  ZERO_TEST MAE={zero[0]:.6g} RMSE={zero[1]:.6g} dir_acc={zero[2]:.2f}%')
        for mae,rmse,negacc,trmae,coef,n in scored[:args.top]:
            terms=[]
            for name,k in zip(NAMES,coef):
                if k == 0: continue
                terms.append(f'{k:g}*{name}')
            print(f'  {mae:.6g} / {rmse:.6g} / {-negacc:.2f}% | train_mae={trmae:.6g} | ' + ' + '.join(terms).replace('+ -','- '))
        print()

if __name__=='__main__':
    main()
