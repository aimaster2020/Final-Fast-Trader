from __future__ import annotations

import argparse, csv, statistics
from pathlib import Path

SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
MONTHS=("2026-05","2026-06","2026-07","2026-08")
FEE_SIDE=0.0013


def load(path: Path, symbol: str):
    out=[]
    with path.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol') != symbol:
                continue
            try:
                out.append((r['month'], int(float(r['timestamp'])), float(r['open']), float(r['high']), float(r['low']), float(r['close'])))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(out, key=lambda x:x[1])


def s(row):
    _,_,o,h,l,c=row
    j=c-o; k=h-c; m=h-o
    return int(k>j)+int(k>m)+int(l>j)


def sign(x, eps=1e-12):
    if x > eps: return 1
    if x < -eps: return -1
    return 0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv'))
    args=ap.parse_args()
    print('EXHAUSTION_REVERSAL_WF | tf=1h | fee=0.13%/side | fixed OHLC only | past_only')
    print('Hypothesis: S1= bullish-body contraction regime, S3= bearish-body contraction regime.')
    print('Trade only large exhaustion candles using past median body; reversal=opposite current body direction.')
    print('Body multiple thresholds are fixed diagnostics, not optimized to Aug.')

    for sym in SYMBOLS:
        rows=load(args.input,sym)
        print(f'\n{sym}')
        for lookback in (12,24,48):
            for mult in (1.0,1.5,2.0):
                for horizon in (1,3,6,12):
                    n=correct=0; gross=net=0.0; wins=0
                    for month in MONTHS:
                        mr=[r for r in rows if r[0]==month]
                        prior=[]
                        for i in range(len(mr)-horizon):
                            cur=mr[i]; fut=mr[i+horizon]
                            st=s(cur)
                            cur_body=cur[5]-cur[2]
                            if st not in (1,3) or cur_body==0:
                                prior.append(abs(cur_body)); prior=prior[-lookback:]
                                continue
                            med=statistics.median(prior) if len(prior)>=5 else None
                            if med is None or abs(cur_body) < mult*med:
                                prior.append(abs(cur_body)); prior=prior[-lookback:]
                                continue
                            # S1 is bullish current body => short; S3 is bearish => long.
                            side=-1 if st==1 else 1
                            ret=side*(fut[5]-cur[5])/cur[5]*100 - 2*FEE_SIDE*100
                            gross += side*(fut[5]-cur[5])/cur[5]*100
                            net += ret
                            wins += int(ret>0); n += 1
                            actual=sign(fut[5]-cur[5]); correct += int(actual==side)
                            prior.append(abs(cur_body)); prior=prior[-lookback:]
                    if n:
                        print(f'LB={lookback:2}x{mult:.1f} H={horizon:2} n={n:4} acc={100*correct/n:5.1f}% avg_gross={gross/n:+.4f}% avg_net={net/n:+.4f}% net_win={100*wins/n:5.1f}%')

if __name__=='__main__':
    main()
