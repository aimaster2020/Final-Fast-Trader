#!/usr/bin/env python3
"""Walk-forward test of relative next-candle movement for exact S0-S3 formula.

Target is abs(next_close-current_close) / max(abs(current_close-open), eps).
Prediction uses only past targets for the same S state, with a same-symbol
fallback median during cold start.  No future observations are used.
"""
from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INPUT = Path("reports/prepared_price_action_1h.csv")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
EPS = 1e-12

@dataclass
class Row:
    symbol: str
    month: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float

def excel_s(r: Row) -> int:
    j = r.close-r.open
    k = r.high-r.close
    m = r.high-r.open
    l = r.low-r.close
    return int(k > j) + int(k > m) + int(l > j)

def load_rows(path: Path) -> list[Row]:
    import csv
    out=[]
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        for rec in csv.DictReader(f):
            try:
                out.append(Row(str(rec["symbol"]),str(rec["month"]),int(float(rec["timestamp"])),float(rec["open"]),float(rec["high"]),float(rec["low"]),float(rec["close"])))
            except (KeyError,TypeError,ValueError):
                continue
    out.sort(key=lambda r:(r.symbol,r.month,r.timestamp))
    return out

def mean(xs):
    return sum(xs)/len(xs) if xs else 0.0

def pearson(xs,ys):
    if len(xs)<2: return 0.0
    mx,my=mean(xs),mean(ys)
    num=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx=math.sqrt(sum((x-mx)**2 for x in xs)); dy=math.sqrt(sum((y-my)**2 for y in ys))
    return 0.0 if dx==0.0 or dy==0.0 else num/(dx*dy)

def evaluate(rows):
    hist=defaultdict(list); global_hist=[]
    e=defaultdict(list); p=defaultdict(list); a=defaultdict(list)
    for i in range(len(rows)-1):
        cur,nxt=rows[i],rows[i+1]
        s=excel_s(cur)
        body=abs(cur.close-cur.open)
        actual=abs(nxt.close-cur.close)/max(body,EPS)
        pred=statistics.median(hist[s]) if hist[s] else (statistics.median(global_hist) if global_hist else None)
        if pred is not None:
            e[s].append(abs(pred-actual)); p[s].append(pred); a[s].append(actual)
        hist[s].append(actual); global_hist.append(actual)
    out={}
    for s in range(4):
        if e[s]:
            out[s]={"n":len(e[s]),"mae":mean(e[s]),"medae":statistics.median(e[s]),"corr":pearson(p[s],a[s]),"pred":mean(p[s]),"actual":mean(a[s])}
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,default=DEFAULT_INPUT); args=ap.parse_args()
    rows=load_rows(args.input); groups=defaultdict(list)
    for r in rows:
        if r.symbol in SYMBOLS and r.month in MONTHS: groups[(r.symbol,r.month)].append(r)
    print("EXCEL_REL_MAG_WF | tf=1h | exact S0-S3 | target=abs(next_close-close)/abs(close-open) | past_only | state_median")
    print("format: N / MAE / MedAE / Corr / PredMean / ActualMean")
    for sym in SYMBOLS:
        print(f"\n{sym}"); allm=defaultdict(list)
        for mo in MONTHS:
            m=evaluate(groups[(sym,mo)])
            for s,v in m.items(): allm[s].append(v)
            parts=[]
            for s in range(4):
                v=m.get(s)
                parts.append(f"S{s}=0/0/0/0/0/0" if not v else f"S{s}={v['n']}/{v['mae']:.3f}/{v['medae']:.3f}/{v['corr']:.3f}/{v['pred']:.3f}/{v['actual']:.3f}")
            print(mo+" | "+" ".join(parts))
        parts=[]
        for s in range(4):
            vs=allm[s]
            if not vs: parts.append(f"S{s}=0/0/0/0/0/0"); continue
            n=sum(v['n'] for v in vs)
            mae=sum(v['mae']*v['n'] for v in vs)/n
            medae=mean(v['medae'] for v in vs); corr=mean(v['corr'] for v in vs)
            pred=sum(v['pred']*v['n'] for v in vs)/n; actual=sum(v['actual']*v['n'] for v in vs)/n
            parts.append(f"S{s}={n}/{mae:.3f}/{medae:.3f}/{corr:.3f}/{pred:.3f}/{actual:.3f}")
        print("4M | "+" ".join(parts))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
