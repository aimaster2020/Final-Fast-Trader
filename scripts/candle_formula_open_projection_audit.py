from __future__ import annotations

import argparse, csv
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


def sign(x, eps=1e-12):
    if x > eps: return 1
    if x < -eps: return -1
    return 0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('reports/prepared_price_action_1h.csv'))
    a=ap.parse_args()
    print('OPEN_PROJECTION_AUDIT | tf=1h | exact rule: prediction = current close + (current open-current close) = current open')
    print('Compare predicted price with NEXT candle close. Also report direction, absolute error, percentage error, and fee-adjusted one-bar trading edge.')
    for sym in SYMBOLS:
        rows=load(a.input, sym)
        n=correct=0; abs_err=0.0; pct_err=0.0; gross_sum=0.0; net_sum=0.0; wins=0
        by_month={m:[0,0,0.0,0.0,0.0] for m in MONTHS}
        for i in range(len(rows)-1):
            cur=rows[i]; nxt=rows[i+1]
            pred=cur[2]
            actual=nxt[5]
            ok=sign(pred-cur[5]) == sign(actual-cur[5])
            err=abs(pred-actual)
            pe=err/actual*100 if actual else 0.0
            gross=sign(pred-cur[5])*(actual-cur[5])/cur[5]*100
            net=gross-2*FEE_SIDE*100
            n += 1; correct += int(ok); abs_err += err; pct_err += pe; gross_sum += gross; net_sum += net; wins += int(net > 0)
            z=by_month[cur[0]]; z[0]+=1; z[1]+=int(ok); z[2]+=err; z[3]+=gross; z[4]+=net
        print(sym)
        print(f'  ALL n={n} dir_acc={100*correct/n if n else 0:.2f}% MAE={abs_err/n if n else 0:.6g} MAPE={pct_err/n if n else 0:.4f}% avg_gross={gross_sum/n if n else 0:+.4f}% avg_net={net_sum/n if n else 0:+.4f}% net_win={100*wins/n if n else 0:.1f}%')
        for m,z in by_month.items():
            print(f'  {m} n={int(z[0])} dir_acc={100*z[1]/z[0] if z[0] else 0:.2f}% MAE={z[2]/z[0] if z[0] else 0:.6g} gross_sum={z[3]:+.2f}% net_sum={z[4]:+.2f}%')

if __name__=='__main__':
    main()
