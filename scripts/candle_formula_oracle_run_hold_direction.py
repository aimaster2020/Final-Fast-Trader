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
    if len(cs)<2:return (0,0,0)
    eq=1000.;pos=False;entry=0.;trades=0;wins=0
    for i in range(len(cs)-1):
        a,b=cs[i],cs[i+1]
        j1=a.close-a.open; j2=b.close-b.open
        d=1 if j2>j1 else -1 if j2<j1 else 0
        if not pos:
            if d==want: pos=True;entry=a.close
            continue
        if d==-want:
            gross=eq*want*(a.close-entry)/entry
            net=gross-eq*FEE*2
            eq+=net; trades+=1; wins+=net>0; pos=False
            if d==want: pos=True; entry=a.close
    if pos:
        gross=eq*want*(cs[-1].close-entry)/entry; net=gross-eq*FEE*2; eq+=net; trades+=1; wins+=net>0
    return (eq/1000-1)*100,trades,wins

def main():
    src={tf:read('reports/prepared_price_action_'+tf+'.csv') for tf in TFS}
    print('ORACLE_RAW_DIRECTION | exact Excel J next-direction | perfect prediction | hold until opposite | fee=0.13%/side')
    for s in SYMS:
        print('\n'+s)
        for m in MONTHS:
            out=[]
            for tf in TFS:
                cs=load(src[tf],m,s);l=run(cs,1);sh=run(cs,-1)
                out.append(f'{tf}:L={l[0]:+.2f}%/{l[1]}T S={sh[0]:+.2f}%/{sh[1]}T')
            print(m+' | '+' | '.join(out))
if __name__=='__main__':main()
