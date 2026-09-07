from __future__ import annotations
import csv
from pathlib import Path

SYMS=('BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT')
MONTHS=('2026-05','2026-06','2026-07','2026-08')
TFS=('5m','15m','1h')
FEE=.0013

def read(p):
    with Path(p).open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def cv(r):
    try:return (float(r['timestamp']),float(r['open']),float(r['high']),float(r['low']),float(r['close']))
    except:return None
def load(rs,m,s):
    return sorted([c for r in rs if r.get('month')==m and r.get('symbol')==s for c in [cv(r)] if c],key=lambda x:x[0])

def run(cs,wanted):
    eq=1000.0; trades=0
    for i in range(len(cs)-1):
        cur,nxt=cs[i],cs[i+1]
        # Exact Excel target: next candle body (C-O) compared with current candle body.
        cur_body=cur[4]-cur[1]; nxt_body=nxt[4]-nxt[1]
        direction=1 if nxt_body>cur_body else -1 if nxt_body<cur_body else 0
        if direction!=wanted: continue
        if cur[4]<=0 or nxt[4]<=0: continue
        gross_pct=wanted*(nxt[4]-cur[4])/cur[4]
        net_pct=gross_pct-2*FEE
        eq*=1.0+net_pct
        trades+=1
    return (eq/1000.0-1)*100,trades

def main():
    src={tf:read('reports/prepared_price_action_'+tf+'.csv') for tf in TFS}
    print('ORACLE_100PCT | exact Excel next-body direction | fee=0.13%/side | full-capital')
    for s in SYMS:
        print('\n'+s)
        for m in MONTHS:
            out=[]
            for tf in TFS:
                cs=load(src[tf],m,s); l=run(cs,1); sh=run(cs,-1)
                out.append(f'{tf}:L={l[0]:+.2f}%/{l[1]}T S={sh[0]:+.2f}%/{sh[1]}T')
            print(m+' | '+' | '.join(out))
if __name__=='__main__':main()
