from __future__ import annotations
import csv
from datetime import datetime, timezone
from pathlib import Path
from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal
BIAS=(-200,-175); MONTHS=('2026-05','2026-06','2026-07','2026-08')
def rows(p):
 with Path(p).open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def cv(r):
 try:return Candle(int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close']))
 except:return None
def main():
 for tf in ('5m','15m','1h'):
  rs=rows(f'reports/prepared_price_action_{tf}.csv');print(tf)
  for m in MONTHS:
   cs=sorted([c for r in rs if r.get('symbol')=='ETHUSDT' and r.get('month')==m for c in [cv(r)] if c],key=lambda x:x.timestamp)
   pos=None
   for c in cs:
    s=decide(c).signal; side=1 if s==Signal.BUY else -1 if s==Signal.SELL else 0; b=c.close-c.open
    if pos is None:
     if side==1 and not(BIAS[0]<=b<=BIAS[1]):pos=(c,side)
    else:
     e,side0=pos
     if side== -1 and b<=BIAS[1]:
      print(f' {m} ENTRY={datetime.fromtimestamp(e.timestamp,timezone.utc):%m-%d %H:%M} {e.close:.2f} EXIT={datetime.fromtimestamp(c.timestamp,timezone.utc):%m-%d %H:%M} {c.close:.2f} move={(c.close/e.close-1)*100:+.2f}%');pos=None
   if pos:
    e,_=pos;c=cs[-1];print(f' {m} ENTRY={datetime.fromtimestamp(e.timestamp,timezone.utc):%m-%d %H:%M} {e.close:.2f} EXIT=END {c.close:.2f} move={(c.close/e.close-1)*100:+.2f}%')
if __name__=='__main__':main()
