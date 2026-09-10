from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median


def load(path: Path) -> list[tuple[float, float, float, float]]:
    rows = []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r['open']), float(r['high']), float(r['low']), float(r['close'])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def f(row):
    o,h,l,c=row
    body=abs(c-o)
    rng=max(h-l,0.0)
    upper=max(h-max(o,c),0.0)
    lower=max(min(o,c)-l,0.0)
    return body,rng,upper,lower


def ad(row):
    o,h,l,c=row
    body=c-o
    return int((h-c)>body)+int((h-o)<(h-c))+int((l-c)>body)


def ratio(a,b):
    return a/b if abs(b)>1e-12 else None


def preds(cur, prev):
    cb,cr,cu,cl=f(cur)
    pb,pr,pu,pl=f(prev)
    out={}
    # All predictions are scale-free transformations of current/previous candle geometry.
    # No asset-specific or fitted constants.
    out['CUR_RANGE_MINUS_WICKS']=max(cr-cu-cl,0.0)
    out['CUR_RANGE_X_PREV_BODY_OVER_PREV_RANGE'] = cr*ratio(pb,pr) if ratio(pb,pr) is not None else None
    out['CUR_BODY_X_CUR_COMPRESSION_OVER_PREV_COMPRESSION'] = cb*ratio(ratio(cb,cr) or 0.0, ratio(pb,pr) or 1e99) if pr>1e-12 and pb>1e-12 else None
    out['CUR_RANGE_X_PREV_COMPRESSION'] = cr*ratio(pb,pr) if pr>1e-12 else None
    out['CUR_BODY_X_PREV_RANGE_OVER_CUR_RANGE'] = cb*ratio(pr,cr) if cr>1e-12 else None
    out['GEOMEAN_BODY_RANGE'] = math.sqrt(cb*cr) if cb>0 and cr>0 else None
    out['HARMONIC_BODY_RANGE'] = 2*cb*cr/(cb+cr) if cb>0 and cr>0 else None
    out['CUR_BODY_PLUS_PREV_WICK_IMBALANCE'] = cb + abs(cl-pl)
    out['CUR_BODY_PLUS_PREV_WICK_RATIO'] = cb*(1 + abs(cl-pl)/max(pr,1e-12))
    out['CUR_RANGE_X_PREV_WICK_SHARE'] = cr*((pu+pl)/max(pr,1e-12)) if pr>1e-12 else None
    return out


def evaluate(rows, name, score):
    errs=[]; rel=[]; correct=0; total=0
    for i in range(1,len(rows)-1):
        if ad(rows[i])!=score: continue
        p=preds(rows[i],rows[i-1]).get(name)
        if p is None or not math.isfinite(p): continue
        actual=f(rows[i+1])[0]
        cur=f(rows[i])[0]
        errs.append(abs(p-actual))
        if actual>1e-12: rel.append(abs(p-actual)/actual)
        correct += int((p>cur)==(actual>cur))
        total += 1
    return len(errs), sum(errs)/len(errs) if errs else 0.0, median(rel) if rel else 0.0, correct/total*100 if total else 0.0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',default='reports/1h')
    ap.add_argument('--symbols',default='BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT')
    args=ap.parse_args()
    formulas=list(preds((1,1,0,1),(1,1,0,1)).keys())
    root=Path(args.input_dir)
    for symbol in [x.strip() for x in args.symbols.split(',') if x.strip()]:
        rows=load(root/f'{symbol}_1h.csv')
        print(f'{symbol} rows={len(rows)}')
        print('FORMULA                                  AD    N       MAE      MED_REL_ERR  GROW_ACC')
        for name in formulas:
            for score in (3,0):
                n,mae,med,acc=evaluate(rows,name,score)
                print(f'{name:<42} {score} {n:5d} {mae:10.6f} {med*100:12.2f}% {acc:9.2f}%')

if __name__=='__main__':
    main()
