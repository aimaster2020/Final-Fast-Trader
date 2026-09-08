from __future__ import annotations

import argparse, csv, statistics
from collections import defaultdict
from pathlib import Path

SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
MONTHS=("2026-05","2026-06","2026-07","2026-08")
CAPITAL=1000.0
MIN_BODY_RATIO=0.0001


def load(path:Path,symbol:str):
    out=[]
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol')!=symbol: continue
            try: out.append((r['month'],int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close'])))
            except (KeyError,TypeError,ValueError): pass
    return sorted(out,key=lambda x:x[1])


def s(row):
    _,_,o,h,l,c=row
    j=c-o; k=h-c; m=h-o
    return int(k>j)+int(k>m)+int(l>j)


def side(st): return 1 if st in (2,3) else -1

def pct(sd,en,px): return sd*(px-en)/en*100 if en>0 else 0.0

def compound(xs):
    v=1.0
    for x in xs: v*=1+x/100
    return (v-1)*100


def run(rows,mode):
    hist=defaultdict(list); glob=[]
    equity=CAPITAL; pos=None; entry_i=-1; peak=CAPITAL; dd=0.0
    monthly=[]; idx=0
    for month in MONTHS:
        mr=[r for r in rows if r[0]==month]
        start_equity=equity; trades=wins=0; month_peak=equity; month_dd=0.0
        for i in range(len(mr)-1):
            cur=mr[i]; nxt=mr[i+1]; st=s(cur); sd=side(st)
            body=abs(cur[5]-cur[2]); valid=cur[5]!=0 and body/abs(cur[5])>=MIN_BODY_RATIO
            if mode=='fixed2': ratio=2.0
            elif mode=='global': ratio=statistics.median(glob) if glob else 2.0
            else: ratio=statistics.median(hist[st]) if hist[st] else (statistics.median(glob) if glob else 2.0)
            if pos is None:
                eff=max(body,abs(cur[5])*MIN_BODY_RATIO); tp=eff*ratio; sl=tp*0.5
                pos=(sd,cur[5],cur[5]+tp if sd==1 else cur[5]-tp,cur[5]-sl if sd==1 else cur[5]+sl)
                entry_i=i
            ps,en,target,stop=pos
            if i>entry_i:
                tpnl=None
                if ps==1:
                    if cur[5]>=target: tpnl=(target-en)/en*100
                    elif cur[5]<=stop: tpnl=(stop-en)/en*100
                else:
                    if cur[5]<=target: tpnl=(en-target)/en*100
                    elif cur[5]>=stop: tpnl=(en-stop)/en*100
                if tpnl is not None:
                    equity += CAPITAL*tpnl/100; trades+=1; wins+=int(tpnl>0); pos=None; entry_i=-1
            if valid:
                mr_ratio=abs(nxt[5]-cur[5])/body
                if mr_ratio==mr_ratio and mr_ratio!=float('inf'):
                    hist[st].append(mr_ratio); glob.append(mr_ratio)
            mtm=equity if pos is None else equity+CAPITAL*pct(pos[0],pos[1],cur[5])/100
            peak=max(peak,mtm); dd=max(dd,(peak-mtm)/peak if peak else 0)
            month_peak=max(month_peak,mtm); month_dd=max(month_dd,(month_peak-mtm)/month_peak if month_peak else 0)
        if pos is not None:
            ret=pct(pos[0],pos[1],mr[-1][5]); equity += CAPITAL*ret/100; trades+=1; wins+=int(ret>0); pos=None; entry_i=-1
        monthly.append((month,(equity-start_equity)/CAPITAL*100,trades,wins,month_dd))
    return monthly,dd*100,equity


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); args=ap.parse_args()
    print('EXCEL_CONTINUOUS_WF | tf=1h | fee=0 | exact S0-S3 | fixed_trade_capital=$1000 | close_only')
    print('history carries continuously May->Jun->Jul->Aug; no monthly reset; modes=FIXED2/GLOBAL/STATE')
    for sym in SYMBOLS:
        rows=load(args.input,sym); results={}
        print(f'\n{sym}')
        for mode in ('fixed2','global','state'):
            monthly,dd,equity=run(rows,mode); results[mode]=monthly
            pnl=sum(x[1] for x in monthly); tr=sum(x[2] for x in monthly); w=sum(x[3] for x in monthly)
            print(f'{mode.upper()} 4M_sum={pnl:+.2f}% compound={compound([x[1] for x in monthly]):+.2f}% trades={tr} win={(100*w/tr if tr else 0):.1f}% DD={dd:.2f}% end=$'+f'{equity:.2f}')
            print('  '+' '.join(f'{m}={r:+.2f}%' for m,r,_,_,_ in monthly))
        print(f'STATE-GLOBAL(sum)={sum(x[1] for x in results["state"])-sum(x[1] for x in results["global"]):+.2f}pp')
        print(f'STATE-FIXED2(sum)={sum(x[1] for x in results["state"])-sum(x[1] for x in results["fixed2"]):+.2f}pp')

if __name__=='__main__': main()
