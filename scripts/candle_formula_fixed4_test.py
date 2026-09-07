from __future__ import annotations
import argparse, csv
from datetime import datetime, timezone
from pathlib import Path
from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal

CASES={
 'ETHUSDT':((-200,-175),(-200,-175)),
 'BTCUSDT':((-200,-125),(-50,-25)),
 'XRPUSDT':((-200,-175),(-200,-175)),
 'SOLUSDT':((25,50),(25,50)),
}
MONTHS=('2026-05','2026-06','2026-07','2026-08')


def rows(path):
 with Path(path).open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def candle(r):
 try:return Candle(int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close']))
 except:return None

def load(rs,month,sym):
 out=[candle(r) for r in rs if r.get('month')==month and r.get('symbol')==sym];return sorted([x for x in out if x],key=lambda x:x.timestamp)

def month(ts):return datetime.fromtimestamp(ts,tz=timezone.utc).strftime('%Y-%m')

def build4h(rs,month_name,sym):
 h=sorted([candle(r) for r in rs if r.get('symbol')==sym and candle(r)],key=lambda x:x.timestamp)
 b={}
 for c in h:
  t=c.timestamp-c.timestamp%14400;b.setdefault(t,[]).append(c)
 out=[]
 for t,g in sorted(b.items()):
  if month(t)!=month_name:continue
  exp=[t+3600*i for i in range(4)]
  if len(g)==4 and [c.timestamp for c in g]==exp:
   out.append(Candle(t,g[0].open,max(c.high for c in g),min(c.low for c in g),g[-1].close))
 return out

def sig(c):
 s=decide(c).signal
 return 1 if s==Signal.BUY else -1 if s==Signal.SELL else 0

def run(cs,lo,hi,fee):
 eq=1000.;pos=None;tr=win=0
 for c in cs:
  s=sig(c);body=c.close-c.open
  if pos is None:
   if s and not(lo<=body<=hi):pos=(s,c.close,eq)
   continue
  side,entry,cap=pos;opp=s and s!=side
  if (side>0 and opp and body<=hi) or (side<0 and opp and body>=lo):
   gross=cap*side*(c.close-entry)/entry;net=gross-cap*fee*2;eq+=net;tr+=1;win+=net>0;pos=None
 if pos:
  side,entry,cap=pos;gross=cap*side*(cs[-1].close-entry)/entry;eq+=gross-cap*fee*2;tr+=1;win+=gross-cap*fee*2>0
 return (eq/1000-1)*100,tr,win/tr*100 if tr else 0

def main():
 p=argparse.ArgumentParser();p.add_argument('--commission',type=float,default=.0013);p.add_argument('--symbols',default=','.join(CASES));p.add_argument('--top',type=int,default=20);a=p.parse_args()
 src={tf:rows('reports/prepared_price_action_'+tf+'.csv') for tf in ('5m','15m','1h')}
 print('FIXED4 | months=May-Aug | fee=0.13%/side')
 for sym in a.symbols.split(','):
  fast,h4bias=CASES.get(sym,(( -200,-175),(-200,-175)))
  vals=[];print('\n'+sym+' | FAST='+str(fast)+' | 4H='+str(h4bias))
  for m in MONTHS:
   data={tf:load(src[tf],m,sym) for tf in ('5m','15m','1h')};data['4h']=build4h(src['1h'],m,sym)
   rs=[run(data[tf],*fast,a.commission) for tf in ('5m','15m','1h')]+[run(data['4h'],*h4bias,a.commission)]
   rets=[x[0] for x in rs];vals+=rets
   print(f'{m} | 5m={rets[0]:+.2f}% 15m={rets[1]:+.2f}% 1h={rets[2]:+.2f}% 4h={rets[3]:+.2f}% | trades={sum(x[1] for x in rs)}')
  avg=sum(vals)/len(vals);positive=sum(x>0 for x in vals);print(f'4M_AVG={avg:+.2f}% | POS={positive}/16 | WORST={min(vals):+.2f}%')
if __name__=='__main__':main()
