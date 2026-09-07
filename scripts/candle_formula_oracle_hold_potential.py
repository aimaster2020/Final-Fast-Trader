from __future__ import annotations
import csv
from pathlib import Path
from fast_pattern_trader.models import Candle

SYMS=('BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT')
MONTHS=('2026-05','2026-06','2026-07','2026-08')
TFS=('5m','15m','1h')
FEE=.0013
INITIAL=1000.0

def read(p):
    with Path(p).open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def cv(r):
    try:
        return Candle(int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close']))
    except Exception:
        return None

def load(rs,m,s):
    out=[cv(r) for r in rs if r.get('month')==m and r.get('symbol')==s]
    return sorted([c for c in out if c],key=lambda c:c.timestamp)

def direction(prev: Candle, cur: Candle) -> int:
    prev_j=prev.close-prev.open
    cur_j=cur.close-cur.open
    return 1 if cur_j>prev_j else -1 if cur_j<prev_j else 0

def run(cs,want):
    # Oracle: the direction is known perfectly from the next J comparison.
    # Stay in the wanted direction across consecutive correct predictions.
    # Enter on the close of the first candle whose NEXT direction is 'want'.
    # Exit on the close when the next direction becomes opposite (or at month end).
    if len(cs)<2:
        return 0.0,0,0.0
    eq=INITIAL
    entry=None
    trades=0
    gross_total=0.0
    i=1
    while i < len(cs):
        d=direction(cs[i-1],cs[i])
        if entry is None:
            if d==want:
                entry=cs[i].close
                i+=1
                continue
            i+=1
            continue
        if d==-want:
            gross=want*(cs[i].close-entry)/entry
            net=gross-2*FEE
            gross_total += gross
            eq *= (1.0+net)
            trades+=1
            entry=None
        i+=1
    if entry is not None:
        gross=want*(cs[-1].close-entry)/entry
        net=gross-2*FEE
        gross_total += gross
        eq *= (1.0+net)
        trades+=1
    return (eq/INITIAL-1)*100,trades,gross_total*100

def main():
    src={tf:read('reports/prepared_price_action_'+tf+'.csv') for tf in TFS}
    print('ORACLE_HOLD | exact Excel J-direction | perfect prediction | hold until opposite | fee=0.13%/side')
    for s in SYMS:
        print('\n'+s)
        for m in MONTHS:
            out=[]
            for tf in TFS:
                cs=load(src[tf],m,s)
                l=run(cs,1); sh=run(cs,-1)
                out.append(f'{tf}:L={l[0]:+.2f}%/{l[1]}T S={sh[0]:+.2f}%/{sh[1]}T')
            print(m+' | '+' | '.join(out))

if __name__=='__main__':
    main()
