from __future__ import annotations
import csv
from pathlib import Path
from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal
BIAS=(-200,-175); FEE=.0013; MONTHS=('2026-05','2026-06','2026-07','2026-08')
def rows(p):
    with Path(p).open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def cv(r):
    try:return Candle(int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close']))
    except:return None
def load(rs,m):return sorted([c for r in rs if r.get('month')==m and r.get('symbol')=='ETHUSDT' for c in [cv(r)] if c],key=lambda x:x.timestamp)
def sig(c):
    s=decide(c).signal;return 1 if s==Signal.BUY else -1 if s==Signal.SELL else 0
def run(cs):
    eq=1000.;pos=None;tr=win=0
    for c in cs:
        s=sig(c);b=c.close-c.open
        if pos is None:
            if s==1 and not(BIAS[0]<=b<=BIAS[1]):pos=(c.close,eq)
            continue
        e,cap=pos
        if s==-1 and b<=BIAS[1]:
            gross=cap*(c.close-e)/e;net=gross-cap*FEE*2;eq+=net;tr+=1;win+=net>0;pos=None
    if pos:
        e,cap=pos;gross=cap*(cs[-1].close-e)/e;net=gross-cap*FEE*2;eq+=net;tr+=1;win+=net>0
    return (eq/1000-1)*100,tr,win/tr*100 if tr else 0
def main():
    src={tf:rows('reports/prepared_price_action_'+tf+'.csv') for tf in ('5m','15m','1h')}
    print('ETH_LONG_ONLY | bias=(-200,-175) | fee=0.13%/side')
    vals=[]
    for m in MONTHS:
        out=[]
        for tf in ('5m','15m','1h'):
            r=run(load(src[tf],m));vals.append(r[0]);out.append(f'{tf}={r[0]:+.2f}% T={r[1]} W={r[2]:.0f}%')
        print(f'{m} | '+' | '.join(out))
    print(f'AVG={sum(vals)/len(vals):+.2f}% | POS={sum(x>0 for x in vals)}/{len(vals)} | WORST={min(vals):+.2f}%')
if __name__=='__main__':main()
