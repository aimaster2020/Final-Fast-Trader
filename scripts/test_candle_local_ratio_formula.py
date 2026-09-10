from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load(path: Path) -> list[tuple[float, float, float, float]]:
    rows=[]
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r['open']), float(r['high']), float(r['low']), float(r['close'])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def evaluate(rows):
    methods={
        'BODY*RANGE_RATIO': [],
        'BODY*COMPRESSION_RATIO': [],
        'RANGE*BODY_RATIO': [],
        'WICK_ADJUSTED': [],
    }
    for i in range(1, len(rows)-1):
        o,h,l,c=rows[i]
        po,ph,pl,pc=rows[i-1]
        no,nh,nl,nc=rows[i+1]
        body=abs(c-o); prev_body=abs(pc-po)
        rng=h-l; prev_rng=ph-pl
        if body<=1e-12 or prev_rng<=1e-12 or rng<=1e-12:
            continue
        target=abs(nc-no)
        # Dynamic multipliers derived only from current/previous candles.
        range_ratio=rng/prev_rng
        compression=(body/rng)/(prev_body/prev_rng) if prev_body>1e-12 else None
        body_ratio=body/prev_body if prev_body>1e-12 else None
        upper=h-max(o,c); lower=min(o,c)-l
        prev_upper=ph-max(po,pc); prev_lower=min(po,pc)-pl
        wick_ratio=((upper+lower)/rng)/(((prev_upper+prev_lower)/prev_rng)) if (prev_upper+prev_lower)>1e-12 else None
        preds={
            'BODY*RANGE_RATIO': body*range_ratio,
            'BODY*COMPRESSION_RATIO': body*compression if compression is not None else None,
            'RANGE*BODY_RATIO': rng*body_ratio if body_ratio is not None else None,
            'WICK_ADJUSTED': body*wick_ratio if wick_ratio is not None else None,
        }
        for name,p in preds.items():
            if p is not None and p < float('inf'):
                methods[name].append((p,target))
    return methods


def stats(pairs):
    if not pairs: return (0,0,0,0)
    errs=[abs(p-t) for p,t in pairs]
    rel=[e/t for e,(p,t) in zip(errs,pairs) if t>1e-12]
    acc=sum(1 for p,t in pairs if (p>abs(rows_target_dummy:=0)))
    return len(pairs), sum(errs)/len(errs), sorted(rel)[len(rel)//2] if rel else 0.0, sum(1 for p,t in pairs if (p>0)==(t>0))/len(pairs)*100


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='reports/1h')
    ap.add_argument('--symbols', default='BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT')
    args=ap.parse_args()
    root=Path(args.input_dir)
    for symbol in [s.strip() for s in args.symbols.split(',') if s.strip()]:
        rows=load(root/f'{symbol}_1h.csv')
        print(f'{symbol} rows={len(rows)}')
        methods=evaluate(rows)
        print('FORMULA                         N      MAE        MED_REL_ERR')
        for name,pairs in methods.items():
            errs=[abs(p-t) for p,t in pairs]
            rel=[e/t for e,(p,t) in zip(errs,pairs) if t>1e-12]
            med=sorted(rel)[len(rel)//2] if rel else 0.0
            print(f'{name:<30} {len(pairs):<6} {sum(errs)/len(errs):10.6f} {med*100:14.2f}%')

if __name__=='__main__':
    main()
