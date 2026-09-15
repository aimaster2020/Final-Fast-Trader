#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional


def parse_args():
    p = argparse.ArgumentParser(description="Sweep F3/F4 for previous-window formula using compounded capital.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--commission-per-side", type=float, default=0.0013)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--f3-min", type=float, default=0)
    p.add_argument("--f3-max", type=float, default=250)
    p.add_argument("--f4-min", type=float, default=-400)
    p.add_argument("--f4-max", type=float, default=0)
    p.add_argument("--step", type=float, default=10)
    return p.parse_args()


def year_of(v: str) -> Optional[int]:
    try:
        import datetime as dt
        ts = int(float(v)); ts = ts // 1000 if ts > 10_000_000_000 else ts
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).year
    except Exception:
        return None


def load(path: Path, year: Optional[int]):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header = [x.strip().lower() for x in rows[0]] if rows else []
    headered = any(x in header for x in ("timestamp", "open", "high", "close"))
    out=[]
    if headered:
        idx={name:i for i,name in enumerate(header)}
        for r in rows[1:]:
            if len(r) <= max(idx["timestamp"], idx["open"], idx["high"], idx["close"]): continue
            if year is not None and year_of(r[idx["timestamp"]]) != year: continue
            out.append((r[idx["timestamp"]],float(r[idx["open"]]),float(r[idx["high"]]),float(r[idx["close"]])))
    else:
        for r in rows:
            if len(r)<5: continue
            if year is not None and year_of(r[0]) != year: continue
            out.append((r[0],float(r[1]),float(r[2]),float(r[4])))
    return out


def frange(a,b,s):
    n=int(round((b-a)/s))
    return [a+i*s for i in range(n+1)]


def backtest(rows, initial, f3, f4, window, fee):
    if len(rows) <= window+1: return None
    capital=initial; pos=0; entry=0.0; trades=0; wins=losses=0; fees=0.0
    for i in range(window, len(rows)-1):
        _,o,_,c=rows[i]
        body=c-o
        if body>f3:
            pred=sum(rows[j][3] for j in range(i-window,i))/window
        elif body<f4:
            pred=sum(rows[j][2] for j in range(i-window,i))/window
        else:
            pred=c
        pt=1 if pred>c else -1 if pred<c else 0
        if pos==0 and pt:
            fee_amt=capital*fee; capital-=fee_amt; fees+=fee_amt
            entry=c; pos=pt
        elif pos and pt and pt!=pos:
            gross=((c-entry)/entry) if pos==1 else ((entry-c)/entry)
            pnl=capital*gross; capital+=pnl
            if pnl>0: wins+=1
            else: losses+=1
            trades+=1
            fee_amt=capital*fee; capital-=fee_amt; fees+=fee_amt
            entry=c; pos=pt
    return {"f3":f3,"f4":f4,"trades":trades,"wins":wins,"losses":losses,
            "win_rate":(wins/trades*100 if trades else 0.0),"commission":fees,
            "final_capital":capital,"net_return":(capital/initial-1)*100}


def main():
    a=parse_args(); rows=load(Path(a.input),a.year)
    results=[]
    for f3 in frange(a.f3_min,a.f3_max,a.step):
        for f4 in frange(a.f4_min,a.f4_max,a.step):
            r=backtest(rows,a.initial_capital,f3,f4,a.window,a.commission_per_side)
            if r: results.append(r)
    results.sort(key=lambda r:(-r["net_return"],-r["trades"],-r["win_rate"]))
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    print(f"input={a.input}"); print(f"output={out}"); print(f"rows={len(rows)} window={a.window}")
    print(f"initial_capital={a.initial_capital:.2f} commission_per_side={a.commission_per_side*100:.4f}%")
    print(f"combinations={len(results)}")
    print("TOP 20 BY NET RETURN")
    for i,r in enumerate(results[:20],1):
        print(f"{i:2d}. F3={r['f3']:g} F4={r['f4']:g} net={r['net_return']:.4f}% final={r['final_capital']:.2f} trades={r['trades']} wins={r['wins']} losses={r['losses']} win_rate={r['win_rate']:.2f}% fees={r['commission']:.2f}")

if __name__=='__main__': main()
