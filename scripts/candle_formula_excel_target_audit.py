from __future__ import annotations

import argparse, csv
from collections import defaultdict
from pathlib import Path

SYMS = ('BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT')
MONTHS = ('2026-05','2026-06','2026-07','2026-08')


def load_rows(p: Path, sym: str):
    out=[]
    with p.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol') != sym:
                continue
            try:
                out.append((r['month'], int(float(r['timestamp'])), float(r['open']), float(r['high']), float(r['low']), float(r['close'])))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(out, key=lambda x: x[1])


def S(r):
    _, _, o, h, l, c = r
    j = c-o
    k = h-c
    m = h-o
    return int(k>j) + int(k>m) + int(l>j)


def sign(x, eps=1e-12):
    if x > eps: return 1
    if x < -eps: return -1
    return 0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('reports/prepared_price_action_1h.csv'))
    a=ap.parse_args()
    print('EXCEL_TARGET_AUDIT | tf=1h | exact Excel target I=sign(J_next-J_current) | past_only')
    print('Prediction from S: S2/S3=+1, S0/S1=-1; target uses candle body J=C-O, not next-close price.')
    for sym in SYMS:
        rows=load_rows(a.input, sym)
        agg=defaultdict(lambda:[0,0,0,0.0,0.0])
        alln=allcorrect=0; sum_target=0.0
        for i,r in enumerate(rows[:-1]):
            s=S(r)
            j=r[5]-r[2]
            jn=rows[i+1][5]-rows[i+1][2]
            pred=1 if s in (2,3) else -1
            target=sign(jn-j)
            z=agg[s]
            z[0]+=1
            z[1]+=int(target==pred)
            z[2]+=int(target!=0)
            z[3]+=jn-j
            z[4]+=abs(jn-j)
            alln+=1; allcorrect+=int(target==pred); sum_target+=jn-j
        print(sym)
        print(f'  ALL n={alln} acc={100*allcorrect/alln if alln else 0:.1f}% nonzero_target={100*sum(v[2] for v in agg.values())/alln if alln else 0:.1f}% mean_delta_J={sum_target/alln if alln else 0:.6g}')
        for s in range(4):
            n,c,nz,sm,sa=agg[s]
            print(f'  S{s} n={n} acc={100*c/n if n else 0:.1f}% nonzero_target={100*nz/n if n else 0:.1f}% mean_delta_J={sm/n if n else 0:.6g} mean_abs_delta_J={sa/n if n else 0:.6g}')

if __name__ == '__main__':
    main()
