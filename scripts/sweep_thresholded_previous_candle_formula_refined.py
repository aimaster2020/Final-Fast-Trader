#!/usr/bin/env python3
"""Refined sweep for the thresholded previous-candle Excel formula.

Formula:
=IF(A12="","",IF(E12-B12>$F$3,AVERAGE(E11),IF(E12-B12<$F$4,AVERAGE(C11),E12)))

F3 and F4 are scanned independently. CT/PT thresholds are kept separate.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from datetime import datetime, timezone


def args():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--year',type=int,default=None)
    p.add_argument('--ct-up',type=float,default=0)
    p.add_argument('--ct-down',type=float,default=0)
    p.add_argument('--f3-min',type=float,default=0)
    p.add_argument('--f3-max',type=float,default=250)
    p.add_argument('--f4-min',type=float,default=-400)
    p.add_argument('--f4-max',type=float,default=0)
    p.add_argument('--step',type=float,default=10)
    return p.parse_args()


def year_of(v):
    try:
        ts=int(float(v))
        if ts>10_000_000_000: ts//=1000
        return datetime.fromtimestamp(ts,tz=timezone.utc).year
    except Exception:
        return None


def load(path,year):
    with open(path,'r',encoding='utf-8-sig',newline='') as f: rows=list(csv.reader(f))
    if not rows: raise ValueError('Input empty')
    head=rows[0]
    is_head=any(x.strip().lower() in {'timestamp','open','high','low','close'} for x in head)
    if is_head:
        m={x.strip().lower():i for i,x in enumerate(head)}
        req=['timestamp','open','high','close']
        miss=[x for x in req if x not in m]
        if miss: raise ValueError('Missing columns: '+','.join(miss))
        data=rows[1:]
        ti,oi,hi,ci=[m[x] for x in req]
        out=[]
        for r in data:
            if len(r)<=max(ti,oi,hi,ci): continue
            if year is None or year_of(r[ti])==year: out.append((r[ti],float(r[oi]),float(r[hi]),float(r[ci])))
        return out
    out=[]
    for r in rows:
        if len(r)<5: continue
        if year is None or year_of(r[0])==year: out.append((r[0],float(r[1]),float(r[2]),float(r[4])))
    return out


def direction(x,up,down):
    return 1 if x>up else -1 if x<-down else 0


def frange(a,b,s):
    n=round((b-a)/s)
    return [round(a+i*s,10) for i in range(n+1)]


def main():
    a=args(); rows=load(a.input,a.year)
    if len(rows)<3: raise ValueError('Need at least 3 rows')
    frames=[]
    for i in range(1,len(rows)-1):
        ts,o,h,c=rows[i]; _,_,_,nxtc=rows[i+1]
        _,po,ph,pc=rows[i-1]
        frames.append((ts,o,h,c,ph,pc,nxtc,c-o,nxtc-c))
    f3s=frange(a.f3_min,a.f3_max,a.step); f4s=frange(a.f4_min,a.f4_max,a.step)
    results=[]
    for f3 in f3s:
        for f4 in f4s:
            correct=0; both=0; signals=0; up=down=0
            for ts,o,h,c,ph,pc,nxtc,body,actual in frames:
                if body>f3: pred=pc
                elif body<f4: pred=ph
                else: pred=c
                pt=direction(pred-c,a.ct_up,a.ct_down)
                ct=direction(actual,a.ct_up,a.ct_down)
                if pt!=0: signals+=1; up += pt==1; down += pt==-1
                if ct!=0 and pt!=0:
                    both+=1
                    if ct==pt: correct+=1
            results.append({'f3':f3,'f4':f4,'accuracy':100*correct/both if both else 0.0,'correct':correct,'both_nonzero':both,'pt_signals':signals,'pt_up':up,'pt_down':down})
    results.sort(key=lambda r:(r['accuracy'],r['both_nonzero'],r['pt_signals']),reverse=True)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    print(f'input={a.input}'); print(f'output={out}'); print(f'rows={len(rows)} frames={len(frames)}')
    print(f'range F3={a.f3_min:g}..{a.f3_max:g}, F4={a.f4_min:g}..{a.f4_max:g}, step={a.step:g}')
    print(f'combinations={len(results)} CT_UP={a.ct_up:g} CT_DOWN={a.ct_down:g}')
    print('\nTOP 20 BY ACCURACY (then sample count)')
    for i,r in enumerate(results[:20],1):
        print(f"{i:2d}. F3={r['f3']:g} F4={r['f4']:g} accuracy={r['accuracy']:.4f}% correct={r['correct']}/{r['both_nonzero']} PT_signals={r['pt_signals']} up={r['pt_up']} down={r['pt_down']}")
    print()
    for minimum in (25,50,100,150,200,250,500):
        eligible=[r for r in results if r['both_nonzero']>=minimum]
        if eligible:
            r=max(eligible,key=lambda x:(x['accuracy'],x['both_nonzero'],x['pt_signals']))
            print(f"BEST WITH both_nonzero>={minimum}: F3={r['f3']:g} F4={r['f4']:g} accuracy={r['accuracy']:.4f}% correct={r['correct']}/{r['both_nonzero']} PT_signals={r['pt_signals']}")

if __name__=='__main__': main()
