from __future__ import annotations
import csv
from pathlib import Path
from fast_pattern_trader.models import Candle

SYMS=('BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT')
MONTHS=('2026-05','2026-06','2026-07','2026-08')
TFS=('5m','15m','1h')
FEE=.0013


def read(p):
    with Path(p).open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def cv(r):
    try:
        return Candle(int(float(r['timestamp'])), float(r['open']), float(r['high']), float(r['low']), float(r['close']))
    except (KeyError, TypeError, ValueError):
        return None


def load(rs, m, s):
    return sorted([c for r in rs if r.get('month') == m and r.get('symbol') == s for c in [cv(r)] if c], key=lambda c: c.timestamp)


def dirs(cs):
    out=[]
    for i in range(len(cs)-1):
        j0=cs[i].close-cs[i].open
        j1=cs[i+1].close-cs[i+1].open
        out.append(1 if j1>j0 else -1 if j1<j0 else 0)
    return out


def run(cs, want):
    if len(cs)<2:
        return 0.0,0.0,0.0,0
    ds=dirs(cs)
    eq=1000.0
    gross_pct=0.0
    fee_pct=0.0
    runs=0
    i=0
    while i<len(ds):
        if ds[i]!=want:
            i+=1; continue
        start=i; i+=1
        while i<len(ds) and ds[i]==want:
            i+=1
        end=i-1
        entry=cs[start].close
        exitp=cs[end+1].close
        gross=want*(exitp-entry)/entry
        gross_pct += gross*100.0
        fee_pct += 2.0*FEE*100.0
        eq *= (1.0 + gross - 2.0*FEE)
        runs += 1
    net_pct=(eq/1000.0-1.0)*100.0
    return gross_pct, fee_pct, net_pct, runs


def main():
    src={tf:read('reports/prepared_price_action_'+tf+'.csv') for tf in TFS}
    print('ORACLE_GROSS_NET | exact Excel J direction | 100% correct | hold same direction | fee=0.13%/side')
    for s in SYMS:
        print('\n'+s)
        for m in MONTHS:
            parts=[]
            for tf in TFS:
                cs=load(src[tf],m,s)
                l=run(cs,1); sh=run(cs,-1)
                parts.append(f'{tf}:L G={l[0]:+.1f}% F={l[1]:.1f}% N={l[2]:+.2f}%/{l[3]}R | S G={sh[0]:+.1f}% F={sh[1]:.1f}% N={sh[2]:+.2f}%/{sh[3]}R')
            print(m+' | '+' | '.join(parts))

if __name__=='__main__':
    main()
