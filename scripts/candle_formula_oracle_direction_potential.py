from __future__ import annotations
import csv
from pathlib import Path
from fast_pattern_trader.models import Candle

SYMS=('BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT'); MONTHS=('2026-05','2026-06','2026-07','2026-08'); TFS=('5m','15m','1h'); FEE=.0013

def read(p):
 with Path(p).open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def cv(r):
 try:return Candle(int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close']))
 except:return None
def load(rs,m,s):return sorted([c for r in rs if r.get('month')==m and r.get('symbol')==s for c in [cv(r)] if c],key=lambda c:c.timestamp)
def run(cs,want):
 if len(cs)<2:return (0,0)
 eq=1000.; trades=0
 for i in range(len(cs)-1):
  a,b=cs[i],cs[i+1]
  d=1 if b.close>a.close else -1 if b.close<a.close else 0
  if d!=want:continue
  ret=want*(b.close-a.close)/a.close
  eq += eq*(ret-2*FEE)
  trades+=1
 return (eq/1000-1)*100,trades

def main():
 src={tf:read('reports/prepared_price_action_'+tf+'.csv') for tf in TFS}
 print('ORACLE_DIRECTION_POTENTIAL | 100% correct | 1-candle holds | fee=0.13%/side')
 for s in SYMS:
  print('\n'+s)
  for m in MONTHS:
   out=[]
   for tf in TFS:
    cs=load(src[tf],m,s); l=run(cs,1); sh=run(cs,-1)
    out.append(f'{tf}:L={l[0]:+.2f}%/{l[1]}T S={sh[0]:+.2f}%/{sh[1]}T')
   print(m+' | '+' | '.join(out))
if __name__=='__main__':main()
