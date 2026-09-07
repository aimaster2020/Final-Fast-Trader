from __future__ import annotations
import csv
from datetime import datetime, timezone
from pathlib import Path
from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal

BIAS=(-200,-175); FEE=.0013; MONTHS=('2026-05','2026-06','2026-07','2026-08')

def rs(p):
 with Path(p).open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def cv(r):
 try:return Candle(int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close']))
 except:return None
def load(r,m):return sorted([c for x in r if x.get('symbol')=='ETHUSDT' and x.get('month')==m for c in [cv(x)] if c],key=lambda x:x.timestamp)
def sig(c):
 s=decide(c).signal;return 1 if s==Signal.BUY else -1 if s==Signal.SELL else 0
def run(cs):
 eq=1000.;pos=None;longs=shorts=0;lg=0.;sg=0.
 for c in cs:
  s=sig(c);b=c.close-c.open
  if pos is None:
   if s and not(BIAS[0]<=b<=BIAS[1]):pos=(s,c.close,eq)
   continue
  side,e,cap=pos;opp=s and s!=side
  exit_now=(side>0 and opp and b<=BIAS[1]) or (side<0 and opp and b>=BIAS[0])
  if exit_now:
   gross=cap*side*(c.close-e)/e;net=gross-cap*FEE*2;eq+=net
   if side>0:longs+=1;lg+=net
   else:shorts+=1;sg+=net
   pos=None
 if pos:
  side,e,cap=pos;gross=cap*side*(cs[-1].close-e)/e;net=gross-cap*FEE*2;eq+=net
  if side>0:longs+=1;lg+=net
  else:shorts+=1;sg+=net
 return (eq/1000-1)*100,longs,shorts,lg,sg

def main():
 src={tf:rs('reports/prepared_price_action_'+tf+'.csv') for tf in ('5m','15m','1h')}
 print('ETH_SIDE | bias=(-200,-175) | fee=0.13%/side')
 for m in MONTHS:
  print(m,end=' | ')
  for tf in ('5m','15m','1h'):
   r=run(load(src[tf],m));print(f'{tf}={r[0]:+.2f}% L={r[1]}({r[3]:+.1f}) S={r[2]}({r[4]:+.1f})',end=' | ')
  print()
if __name__=='__main__':main()
