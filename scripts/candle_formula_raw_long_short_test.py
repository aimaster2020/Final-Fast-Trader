from __future__ import annotations
import csv
from pathlib import Path
from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal

SYMS=('BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT')
MONTHS=('2026-05','2026-06','2026-07','2026-08')
TFS=('5m','15m','1h')
FEE=.0013

def read(p):
 with Path(p).open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def cv(r):
 try:return Candle(int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close']))
 except:return None
def load(rs,m,s):return sorted([c for r in rs if r.get('month')==m and r.get('symbol')==s for c in [cv(r)] if c],key=lambda c:c.timestamp)
def side(c):
 s=decide(c).signal
 return 1 if s==Signal.BUY else -1 if s==Signal.SELL else 0

def run(cs, wanted):
 eq=1000.; pos=None; trades=wins=0
 for c in cs:
  s=side(c)
  if pos is None:
   if s==wanted: pos=(c.close,eq); continue
  else:
   entry,cap=pos
   if s==-wanted:
    gross=cap*wanted*(c.close-entry)/entry; net=gross-cap*FEE*2; eq+=net; trades+=1; wins+=net>0; pos=None
 if pos:
  entry,cap=pos; gross=cap*wanted*(cs[-1].close-entry)/entry; net=gross-cap*FEE*2; eq+=net; trades+=1; wins+=net>0
 return (eq/1000-1)*100,trades,wins/trades*100 if trades else 0

def main():
 src={tf:read('reports/prepared_price_action_'+tf+'.csv') for tf in TFS}
 print('RAW_100PCT | exact Excel score | fee=0.13%/side | no bias | full-capital')
 for s in SYMS:
  print('\n'+s)
  for m in MONTHS:
   out=[]
   for tf in TFS:
    cs=load(src[tf],m,s); l=run(cs,1); sh=run(cs,-1)
    out.append(f'{tf}:L={l[0]:+.2f}%/{l[1]}T S={sh[0]:+.2f}%/{sh[1]}T')
   print(m+' | '+' | '.join(out))
if __name__=='__main__': main()
