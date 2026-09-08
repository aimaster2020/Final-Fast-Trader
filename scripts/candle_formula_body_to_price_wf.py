from __future__ import annotations

import argparse, csv, statistics
from collections import defaultdict
from pathlib import Path

SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
MONTHS=("2026-05","2026-06","2026-07","2026-08")
FEE_SIDE=0.0013
CAPITAL=1000.0


def load(path:Path,symbol:str):
    out=[]
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol')!=symbol: continue
            try:
                out.append((r['month'],int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close'])))
            except (KeyError,TypeError,ValueError): pass
    return sorted(out,key=lambda x:x[1])


def s(row):
    _,_,o,h,l,c=row
    j=c-o; k=h-c; m=h-o
    return int(k>j)+int(k>m)+int(l>j)


def sign(x,eps=1e-12):
    if x>eps: return 1
    if x<-eps: return -1
    return 0


def trade_ret(side, entry, exit_):
    if side==0 or entry<=0: return 0.0
    gross=side*(exit_-entry)/entry
    return gross-2*FEE_SIDE


def run(rows, mode):
    # Past-only estimates of delta-J by S. Each observation at i predicts i+1.
    hist=defaultdict(list)
    equity=CAPITAL; trades=wins=0; correct=0; pred_n=0; sum_gross=sum_net=0.0
    monthly=[]
    for month in MONTHS:
        mr=[r for r in rows if r[0]==month]
        start_equity=equity; mtr=0; mw=0; mc=0; mp=0
        for i in range(len(mr)-1):
            cur=mr[i]; nxt=mr[i+1]
            st=s(cur); j=cur[5]-cur[2]
            # Past-only prediction of delta J.
            if mode=='S_DELTA':
                pred_delta=statistics.median(hist[st]) if hist[st] else None
                pred_body=sign(j+(pred_delta or 0.0)) if pred_delta is not None else 0
            elif mode=='S_DELTA_SIGN':
                vals=hist[st]
                pred_delta=statistics.median(vals) if vals else None
                pred_body=sign(pred_delta) if pred_delta is not None else 0
            elif mode=='CURRENT_BODY':
                pred_body=sign(j)
            elif mode=='REVERSE_CURRENT_BODY':
                pred_body=-sign(j)
            else:
                raise ValueError(mode)

            actual_body=sign(nxt[5]-nxt[2])
            px_side=sign(nxt[5]-cur[5])
            if pred_body:
                pred_n+=1; correct+=int(pred_body==actual_body); mc+=int(pred_body==actual_body); mp+=1
                ret=trade_ret(pred_body,cur[5],nxt[5])*100
                sum_gross += pred_body*(nxt[5]-cur[5])/cur[5]*100
                sum_net += ret
                equity += CAPITAL*ret/100
                trades+=1; mtr+=1
                wins+=int(ret>0); mw+=int(ret>0)

            # Update with the now-observed next candle body change; no look-ahead.
            if j==j and nxt[5]-nxt[2]==nxt[5]-nxt[2]:
                hist[st].append((nxt[5]-nxt[2])-j)

        monthly.append((month,(equity-start_equity)/CAPITAL*100,mtr,mw,(100*mc/mp if mp else 0)))
    return monthly,pred_n,correct,trades,wins,sum_gross,sum_net,equity


def compound(monthly):
    v=1.0
    for _,r,*_ in monthly: v*=1+r/100
    return (v-1)*100


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); a=ap.parse_args()
    print('BODY_TO_PRICE_WF | tf=1h | fee=0.13%/side | one-bar close-to-close | past_only')
    print('S_DELTA: median past delta-J by S -> predicted next body sign via J_current + predicted delta')
    print('S_DELTA_SIGN: median past delta-J by S -> predicted delta sign only')
    print('CURRENT_BODY/REVERSE_CURRENT_BODY are causal benchmarks')
    for sym in SYMBOLS:
        rows=load(a.input,sym)
        print(f'\n{sym}')
        for mode in ('S_DELTA','S_DELTA_SIGN','CURRENT_BODY','REVERSE_CURRENT_BODY'):
            monthly,pn,pc,tr,w,sg,sn,end=run(rows,mode)
            acc=100*pc/pn if pn else 0
            print(f'{mode} pred_acc_to_next_body={acc:.1f}% trades={tr} win={100*w/tr if tr else 0:.1f}% net_sum={sn:+.2f}% net_compound={compound(monthly):+.2f}% end=${end:.2f}')
            print('  '+' '.join(f'{m}={r:+.2f}%' for m,r,_,_,_ in monthly))

if __name__=='__main__': main()
