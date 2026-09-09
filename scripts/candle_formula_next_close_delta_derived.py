from __future__ import annotations

import argparse, csv, math
from pathlib import Path

SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")


def load(path:Path,sym:str):
    out=[]
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol')!=sym: continue
            try: out.append((int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close'])))
            except (KeyError,TypeError,ValueError): pass
    return sorted(out)


def sign(x,eps=1e-12): return 1 if x>eps else -1 if x<-eps else 0

# Candidate estimators of next-close delta, using only current OHLC.
def candidates(o,h,l,c):
    body=c-o
    upper=h-c
    lower=c-l
    rng=h-l
    return {
        'ZERO':0.0,
        'BODY':body,
        'REV_BODY':-body,
        'UPPER_WICK':upper,
        'LOWER_WICK':-lower,
        'RANGE_MID':0.5*rng,
        'RANGE_NEG_MID':-0.5*rng,
        'CLOSE_TO_HIGH':h-c,
        'CLOSE_TO_LOW':l-c,
        'HIGH_OPEN_AVG':0.5*((h-o)+(c-o)),
        'LOW_OPEN_AVG':0.5*((l-o)+(c-o)),
        'HL_CENTER_DELTA':0.5*(h+l)-c,
        'OC_CENTER_DELTA':0.5*(o+c)-c,
        'BODY_PLUS_UPPER':body+upper,
        'BODY_MINUS_LOWER':body-lower,
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); a=ap.parse_args()
    print('NEXT_CLOSE_DELTA_DERIVED | tf=1h | fee=0 | no lookahead')
    print('Target = next_close-current_close. All candidates use only current O/H/L/C.')
    for sym in SYMBOLS:
        rows=load(a.input,sym); stats={k:[0,0.0,0.0,0] for k in candidates(*rows[0][1:]).keys()}
        for i in range(len(rows)-1):
            _,o,h,l,c=rows[i]; nxtc=rows[i+1][4]; y=nxtc-c
            for k,p in candidates(o,h,l,c).items():
                z=stats[k]; e=p-y; z[0]+=1; z[1]+=abs(e); z[2]+=e*e; z[3]+=int(sign(p)==sign(y))
        print(sym)
        for k,(n,ae,se,cor) in sorted(stats.items(), key=lambda kv: kv[1][1]/kv[1][0]):
            print(f' {k:20s} MAE={ae/n:.6g} RMSE={math.sqrt(se/n):.6g} dir_acc={100*cor/n:.2f}%')

if __name__=='__main__': main()
