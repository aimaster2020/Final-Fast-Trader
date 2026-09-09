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
                out.append((int(float(r['timestamp'])), float(r['open']), float(r['high']), float(r['low']), float(r['close'])))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(out, key=lambda x:x[0])


def sign(x, eps=1e-12):
    return 1 if x>eps else (-1 if x<-eps else 0)


def feats(r):
    _,o,h,l,c=r
    rng=h-l
    body=c-o
    upper=h-max(o,c)
    lower=min(o,c)-l
    if rng<=1e-12:
        return body,upper,lower,rng,0,0,0
    return body,upper,lower,rng,body/rng,upper/rng,lower/rng


def bucket(r):
    body,upper,lower,rng,br,ur,lr=feats(r)
    bs=sign(body)
    # Three coarse geometry states; intentionally simple and deterministic.
    if bs>0:
        bstate='B+'
    elif bs<0:
        bstate='B-'
    else:
        bstate='B0'
    wick='U_DOM' if ur>lr+0.05 else ('L_DOM' if lr>ur+0.05 else 'BAL')
    ratio='SMALL' if abs(br)<0.25 else ('MED' if abs(br)<0.60 else 'LARGE')
    return bstate+'|'+wick+'|'+ratio


def candidate_values(r):
    body,upper,lower,rng,br,ur,lr=feats(r)
    d=upper-lower
    q=(upper*upper-lower*lower)/rng if rng>1e-12 else 0.0
    a=body*d/rng if rng>1e-12 else 0.0
    b=body*(upper*upper-lower*lower)/(rng*rng) if rng>1e-12 else 0.0
    s=math.copysign(1.0, body) if abs(body)>1e-12 else 0.0
    return {
        'ZERO':0.0,
        'WICK_DIFF':d,
        'WICK_DIFF_HALF':0.5*d,
        'WICK_SQ_DIFF_OVER_RANGE':q,
        'BODY_WICK_RATIO':a,
        'BODY_WICK_SQ_RATIO':b,
        'SIGNED_WICK_ENERGY':rng*((d/rng)**2)*s if rng>1e-12 else 0.0,
    }


def score(preds, actuals):
    n=len(actuals)
    mae=sum(abs(p-y) for p,y in zip(preds,actuals))/n
    rmse=math.sqrt(sum((p-y)**2 for p,y in zip(preds,actuals))/n)
    return mae,rmse


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv'))
    ap.add_argument('--min-count',type=int,default=30)
    args=ap.parse_args()
    print('REGIME_SYMBOLIC_SEARCH | tf=1h | fee=0 | train=70% | no lookahead')
    print('Selects a symbolic candidate independently inside each train regime; evaluates on unseen test candles.')
    print('Regime = body sign + wick dominance + body/range size. Candidate coefficients are fixed.')
    for sym in SYMBOLS:
        rows=load(args.input,sym)
        split=int(len(rows)*0.70)
        train=rows[:split]; test=rows[split:]
        hist={}
        names=list(candidate_values(rows[0]).keys()) if rows else []
        for i in range(len(train)-1):
            k=bucket(train[i]); y=train[i+1][4]-train[i][4]
            hist.setdefault(k,{n:[] for n in names})
            vals=candidate_values(train[i])
            for n in names: hist[k][n].append(y)
        sums={k:{n:sum(v)/len(v) if len(v)>=args.min_count else 0.0 for n,v in dd.items()} for k,dd in hist.items()}
        errors={n:[] for n in names}; used={n:0 for n in names}
        for i in range(len(test)-1):
            k=bucket(test[i]); y=test[i+1][4]-test[i][4]
            vals=candidate_values(test[i])
            for n in names:
                p=(sums.get(k,{}).get(n,0.0))*0.25 if n!='ZERO' else 0.0
                # Candidate signal is normalized by a train-scale coefficient from the
                # selected regime: mean target / mean absolute candidate magnitude.
                src=sums.get(k,{}).get(n,0.0)
                abs_c=sum(abs(candidate_values(r)[n]) for r in train if bucket(r)==k)
                cnt=sum(1 for r in train if bucket(r)==k)
                mean_abs=abs_c/cnt if cnt else 0.0
                coef=src/mean_abs if mean_abs>1e-12 else 0.0
                p=coef*vals[n]
                errors[n].append((p,y)); used[n]+=int(k in sums and (len(hist[k][n])>=args.min_count))
        print(sym)
        base=errors['ZERO']
        bm,br=score([p for p,y in base],[y for p,y in base])
        print(f'  ZERO_TEST MAE={bm:.6g} RMSE={br:.6g}')
        ranked=[]
        for n,pairs in errors.items():
            if n=='ZERO': continue
            mae,rmse=score([p for p,y in pairs],[y for p,y in pairs])
            ranked.append((mae,rmse,n))
        ranked.sort()
        for mae,rmse,n in ranked[:6]:
            print(f'  {n:24s} MAE={mae:.6g} RMSE={rmse:.6g} coverage={100*used[n]/max(1,len(test)-1):.1f}%')
        print()

if __name__=='__main__':
    main()
