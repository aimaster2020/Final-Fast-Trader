#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--f3', type=float, required=True)
    p.add_argument('--f4', type=float, required=True)
    p.add_argument('--year', type=int, default=None)
    return p.parse_args()


def year_of(v: str) -> Optional[int]:
    try:
        import datetime as dt
        ts = int(float(v))
        if ts > 10_000_000_000:
            ts //= 1000
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).year
    except Exception:
        return None


def load(path: Path, year: Optional[int]):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    if rows and any(x.strip().lower() in {'timestamp','open','high','low','close'} for x in rows[0]):
        h = {x.strip().lower(): i for i, x in enumerate(rows[0])}
        out=[]
        for r in rows[1:]:
            if len(r) <= max(h['timestamp'],h['open'],h['high'],h['close']): continue
            if year is not None and year_of(r[h['timestamp']]) != year: continue
            out.append((r[h['timestamp']],float(r[h['open']]),float(r[h['high']]),float(r[h['close']])))
        return out
    out=[]
    for r in rows:
        if len(r)<5: continue
        if year is not None and year_of(r[0]) != year: continue
        out.append((r[0],float(r[1]),float(r[2]),float(r[4])))
    return out


def direction(move: float) -> int:
    return 1 if move > 0 else -1 if move < 0 else 0


def main():
    a=args(); rows=load(Path(a.input),a.year)
    results=[]
    for w in (1,2,3,4,5):
        correct=both=pt_sig=pt_up=pt_down=0
        start=w
        for i in range(start,len(rows)-1):
            _,o,h,c=rows[i]
            _,_,_,nc=rows[i+1]
            body=c-o
            if body>a.f3:
                vals=[rows[j][3] for j in range(i-w,i)]
                pred=sum(vals)/w
                branch='avg_prev_close'
            elif body<a.f4:
                vals=[rows[j][2] for j in range(i-w,i)]
                pred=sum(vals)/w
                branch='avg_prev_high'
            else:
                pred=c
                branch='current_close'
            pt=direction(pred-c); ct=direction(nc-c)
            if pt:
                pt_sig+=1
                pt_up += pt==1
                pt_down += pt==-1
            if pt and ct:
                both+=1; correct += pt==ct
            results.append({'window':w,'timestamp':rows[i][0],'open':o,'high':h,'close':c,'predict':pred,'branch':branch,'next_close':nc,'ct':ct,'pt':pt,'correct':int(pt and ct and pt==ct)})
        n=both
        print(f'WINDOW={w} accuracy={(correct/n*100 if n else 0):.4f}% correct={correct}/{n} PT_signals={pt_sig} up={pt_up} down={pt_down}')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    print(f'input={a.input}')
    print(f'output={out}')
    print(f'rows={len(rows)}')
    print(f'F3={a.f3:g} F4={a.f4:g}')
    print('Formula: body > F3 -> average previous N closes; body < F4 -> average previous N highs; otherwise current close')

if __name__=='__main__': main()
