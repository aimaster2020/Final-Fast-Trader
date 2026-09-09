from __future__ import annotations

import argparse, csv, math
from collections import defaultdict
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


def mean(xs):
    return sum(xs)/len(xs) if xs else 0.0


def mae(preds, actuals):
    return sum(abs(p-a) for p,a in zip(preds, actuals))/len(actuals) if actuals else math.inf


def rmse(preds, actuals):
    return math.sqrt(sum((p-a)*(p-a) for p,a in zip(preds,actuals))/len(actuals)) if actuals else math.inf


def sign(x, eps=1e-12):
    if x > eps: return 1
    if x < -eps: return -1
    return 0


def features(row):
    _,o,h,l,c=row
    body=c-o
    upper=h-max(o,c)
    lower=min(o,c)-l
    rng=h-l
    s=int(upper > body)+int(upper > (h-o))+int(lower > body)
    body_sign=sign(body)
    wick_bias=sign(upper-lower)
    # Relative body size bucket, based only on the current candle.
    ratio=abs(body)/rng if rng > 0 else 0.0
    body_bucket=0 if ratio < 0.25 else (1 if ratio < 0.50 else 2)
    return {
        'S': s,
        'BODY_SIGN': body_sign,
        'WICK_BIAS': wick_bias,
        'BODY_BUCKET': body_bucket,
    }


def key_for(f, name):
    if name == 'S': return (f['S'],)
    if name == 'BODY_SIGN': return (f['BODY_SIGN'],)
    if name == 'S_BODY_SIGN': return (f['S'], f['BODY_SIGN'])
    if name == 'S_WICK_BIAS': return (f['S'], f['WICK_BIAS'])
    if name == 'S_BODY_BUCKET': return (f['S'], f['BODY_BUCKET'])
    if name == 'S_BODY_SIGN_BUCKET': return (f['S'], f['BODY_SIGN'], f['BODY_BUCKET'])
    if name == 'S_WICK_BUCKET': return (f['S'], f['WICK_BIAS'], f['BODY_BUCKET'])
    raise ValueError(name)


def evaluate(rows, key_name, split, min_count):
    cut=max(1, int(len(rows)*split))
    train=rows[:cut]
    test=rows[cut:]
    buckets=defaultdict(list)
    for i in range(len(train)-1):
        delta=train[i+1][4]-train[i][4]
        buckets[key_for(features(train[i]), key_name)].append(delta)

    preds=[]; actuals=[]; covered=0
    for i in range(len(test)-1):
        actual=test[i+1][4]-test[i][4]
        key=key_for(features(test[i]), key_name)
        vals=buckets.get(key, [])
        if len(vals) < min_count:
            pred=0.0
        else:
            pred=mean(vals)
            covered+=1
        preds.append(pred); actuals.append(actual)

    return mae(preds,actuals), rmse(preds,actuals), 100*covered/len(actuals) if actuals else 0.0, len(train), len(actuals)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv'))
    ap.add_argument('--split',type=float,default=0.70)
    ap.add_argument('--min-count',type=int,default=20)
    args=ap.parse_args()

    methods=("S","BODY_SIGN","S_BODY_SIGN","S_WICK_BIAS","S_BODY_BUCKET","S_BODY_SIGN_BUCKET","S_WICK_BUCKET")
    print(f'CONDITIONAL_DELTA_AUDIT | tf=1h | fee=0 | train={args.split:.0%} | no lookahead')
    print('Target = next_close-current_close. Prediction = historical mean delta for current-candle numeric bucket.')
    print(f'min_count={args.min_count}; fallback=0 when bucket is too sparse.')

    for sym in SYMBOLS:
        rows=load(args.input,sym)
        # Baseline on the same test segment.
        cut=max(1,int(len(rows)*args.split))
        actual=[rows[i+1][4]-rows[i][4] for i in range(cut,len(rows)-1)]
        zero_mae=sum(abs(a) for a in actual)/len(actual)
        zero_rmse=math.sqrt(sum(a*a for a in actual)/len(actual))
        print(sym)
        print(f'  ZERO_TEST MAE={zero_mae:.6g} RMSE={zero_rmse:.6g}')
        results=[]
        for name in methods:
            m,r,cov,ntr,nte=evaluate(rows,name,args.split,args.min_count)
            results.append((m,r,-cov,name))
        results.sort()
        for m,r,negcov,name in results:
            print(f'  {name:<22} MAE={m:.6g} RMSE={r:.6g} coverage={-negcov:.1f}%')
        print()

if __name__=='__main__':
    main()
