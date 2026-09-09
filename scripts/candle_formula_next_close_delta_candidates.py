from __future__ import annotations

import argparse, csv, math
from pathlib import Path

SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
MONTHS=("2026-05","2026-06","2026-07","2026-08")


def load(path: Path, symbol: str):
    out=[]
    with path.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol') != symbol:
                continue
            try:
                out.append((r['month'], int(float(r['timestamp'])), float(r['open']), float(r['high']), float(r['low']), float(r['close'])))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(out, key=lambda x:x[1])


def sign(x, eps=1e-12):
    if x > eps: return 1
    if x < -eps: return -1
    return 0


def candidates(o,h,l,c):
    # Return predicted next-close DELTA, not predicted absolute price.
    return {
        'ZERO': 0.0,
        'BODY': c-o,
        'REV_BODY': o-c,
        'UPPER_WICK': h-c,
        'LOWER_WICK': c-l,
        'REV_UPPER': c-h,
        'REV_LOWER': l-c,
        'RANGE_HALF': 0.5*(h-l),
        'RANGE_NEG_HALF': -0.5*(h-l),
        'CLOSE_MID': 0.5*(h+l)-c,
        'OPEN_MID': 0.5*(h+l)-o,
        'BODY_PLUS_UPPER': (c-o)+(h-c),
        'BODY_MINUS_UPPER': (c-o)-(h-c),
        'BODY_PLUS_LOWER': (c-o)+(c-l),
        'BODY_MINUS_LOWER': (c-o)-(c-l),
        'UPPER_MINUS_LOWER': (h-c)-(c-l),
        'LOWER_MINUS_UPPER': (c-l)-(h-c),
        'OPEN_TO_HIGH': h-o,
        'OPEN_TO_LOW': l-o,
        'HIGH_PLUS_LOW_MINUS_2C': h+l-2*c,
        'HIGH_PLUS_LOW_MINUS_2O': h+l-2*o,
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); a=ap.parse_args()
    print('NEXT_CLOSE_DELTA_CANDIDATES | tf=1h | fee=0 | no lookahead')
    print('Target = next_close - current_close. Candidates predict DELTA directly from current O/H/L/C.')
    for sym in SYMBOLS:
        rows=load(a.input,sym)
        names=list(candidates(*rows[0][2:]).keys()) if rows else []
        agg={n:[0,0.0,0.0,0] for n in names}  # n, abs error sum, sq error sum, direction correct
        for i in range(len(rows)-1):
            cur=rows[i]; nxt=rows[i+1]
            o,h,l,c=cur[2:]; actual=nxt[5]-c
            for name,pred in candidates(o,h,l,c).items():
                z=agg[name]; z[0]+=1; z[1]+=abs(pred-actual); z[2]+=(pred-actual)**2; z[3]+=int(sign(pred)==sign(actual))
        ranked=sorted(agg.items(), key=lambda kv:(kv[1][1]/kv[1][0], -(kv[1][3]/kv[1][0])))
        print(f'\n{sym}')
        for name,z in ranked[:10]:
            n,ae,se,dc=z
            rmse=math.sqrt(se/n) if n else 0
            acc=100*dc/n if n else 0
            print(f' {name:22} MAE={ae/n:.6g} RMSE={rmse:.6g} dir_acc={acc:5.2f}%')
        print(' Best formula above predicts NEXT_CLOSE - CURRENT_CLOSE directly; add it to current close for absolute prediction.')

if __name__=='__main__':
    main()
