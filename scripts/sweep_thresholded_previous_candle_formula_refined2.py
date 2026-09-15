#!/usr/bin/env python3
"""Fine sweep for the thresholded previous-candle Excel formula.

Formula:
=IF(A12="","",IF(E12-B12>$F$3,AVERAGE(E11),IF(E12-B12<$F$4,AVERAGE(C11),E12)))

F3 range: 50..160 step 5
F4 range: -260..-40 step 5

CT/PT thresholds are kept separate from F3/F4.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import Optional


def args():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    p.add_argument('--year',type=int,default=None)
    p.add_argument('--ct-up',type=float,default=0); p.add_argument('--ct-down',type=float,default=0)
    return p.parse_args()


def is_header(r):
    s=','.join(x.strip().lower() for x in r)
    return any(x in s for x in ('timestamp','open','high','low','close'))

def year(ts):
    import datetime as dt
    try:
        x=int(float(ts)); x//=1000 if x>10_000_000_000 else 1
        return dt.datetime.fromtimestamp(x,tz=dt.timezone.utc).year
    except: return None

def load(path, yr):
    with path.open(encoding='utf-8-sig',newline='') as f:r=list(csv.reader(f))
    if not r: raise ValueError('empty CSV')
    out=[]
    if is_header(r[0]):
        h={x.strip().lower():i for i,x in enumerate(r[0])}; oi,hi,ci,ti=h['open'],h['high'],h['close'],h.get('timestamp',0)
        for x in r[1:]:
            if len(x)>max(oi,hi,ci,ti) and (yr is None or year(x[ti])==yr): out.append((x[ti],float(x[oi]),float(x[hi]),float(x[ci])))
    else:
        for x in r:
            if len(x)>=5 and (yr is None or year(x[0])==yr): out.append((x[0],float(x[1]),float(x[2]),float(x[4])))
    return out

def d(move,up,down): return 1 if move>up else -1 if move<-down else 0

def main():
    a=args(); rows=load(Path(a.input),a.year)
    if len(rows)<3: raise ValueError('need at least 3 rows')
    results=[]
    for f3 in range(50,161,5):
        for f4 in range(-260,-39,5):
            correct=both=pts=up=down=0
            for i in range(1,len(rows)-1):
                _,o,ph,c=rows[i]; _,_,_,nc=rows[i+1]; _,_,prev_h,prev_c=rows[i-1]
                body=c-o
                pred=prev_c if body>f3 else prev_h if body<f4 else c
                pt=d(pred-c,a.ct_up,a.ct_down); ct=d(nc-c,a.ct_up,a.ct_down)
                pts += pt!=0; up += pt==1; down += pt==-1
                if pt!=0 and ct!=0:
                    both += 1; correct += pt==ct
            acc=100*correct/both if both else 0
            results.append({'f3':f3,'f4':f4,'correct':correct,'both_nonzero':both,'accuracy':acc,'pt_signals':pts,'pt_up':up,'pt_down':down})
    results.sort(key=lambda x:(x['accuracy'],x['both_nonzero']),reverse=True)
    thresholds=[25,50,75,100,125,150,175,200,250,300,400,500]
    print(f'input={a.input}\noutput={a.output}\nrows={len(rows)} frames={len(rows)-2}')
    print(f'range F3=50..160, F4=-260..-40, step=5\ncombinations={len(results)} CT_UP={a.ct_up:g} CT_DOWN={a.ct_down:g}')
    print('\nTOP 20 BY ACCURACY')
    for n,r in enumerate(results[:20],1): print(f"{n:2}. F3={r['f3']} F4={r['f4']} accuracy={r['accuracy']:.4f}% correct={r['correct']}/{r['both_nonzero']} PT_signals={r['pt_signals']} up={r['pt_up']} down={r['pt_down']}")
    for t in thresholds:
        cand=[r for r in results if r['both_nonzero']>=t]
        if cand:
            r=max(cand,key=lambda x:(x['accuracy'],x['both_nonzero']))
            print(f"BEST WITH both_nonzero>={t}: F3={r['f3']} F4={r['f4']} accuracy={r['accuracy']:.4f}% correct={r['correct']}/{r['both_nonzero']} PT_signals={r['pt_signals']}")
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with Path(a.output).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)

if __name__=='__main__': main()
