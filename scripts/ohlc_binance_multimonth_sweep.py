from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path: sys.path.insert(0,str(ROOT/'src'))
from fast_pattern_trader.models import Candle
from ohlc_continuous_account_candidates_365d import evaluate, NOBITEX_TAKER_FEE_PER_SIDE, parse_candidates, resample


def load_binance_vision(path: Path) -> list[Candle]:
    out=[]
    with path.open('r',encoding='utf-8') as f:
        for line in f:
            parts=line.strip().split(',')
            if len(parts)<5: continue
            try: out.append(Candle(int(parts[0])//1_000_000,float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])))
            except (ValueError,TypeError): continue
    return sorted(out,key=lambda x:x.timestamp)


def find_month_file(root: Path,symbol: str,month: str)->Path|None:
    d=root/symbol/'csv'
    candidates=[d/f'{symbol}-1m-{month}.csv',d/f'{symbol}-1m-{month}.csv.zip']
    for p in candidates:
        if p.exists() and p.suffix=='.csv': return p
    matches=list(d.glob(f'*{month}*.csv')) if d.exists() else []
    return matches[0] if matches else None


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',required=True); ap.add_argument('--symbols',default='BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT'); ap.add_argument('--months',default='2026-02,2026-03,2026-04,2026-05,2026-06,2026-07'); ap.add_argument('--candidates',default='5:5,30:4'); ap.add_argument('--initial-capital',type=float,default=1000); ap.add_argument('--trade-allocation',type=float,default=.30); ap.add_argument('--max-positions',type=int,default=2); ap.add_argument('--score-threshold',type=float,default=1.0); ap.add_argument('--output',default='reports/binance_multimonth_sweep.csv'); args=ap.parse_args()
    symbols=[x.strip() for x in args.symbols.split(',') if x.strip()]; months=[x.strip() for x in args.months.split(',') if x.strip()]; candidates=parse_candidates(args.candidates)
    detail=[]; missing=[]
    for symbol in symbols:
        for month in months:
            path=find_month_file(Path(args.input_dir),symbol,month)
            if not path: missing.append(f'{symbol},{month}'); print(f'MISSING {symbol} {month}'); continue
            raw=load_binance_vision(path); print(f'LOADED {symbol} {month} | 1m bars={len(raw)}')
            for tf,n in candidates:
                candles=resample(raw,tf)
                for fee_name,fee in [('no_commission',0.0),('nobitex_taker',NOBITEX_TAKER_FEE_PER_SIDE)]:
                    s,_=evaluate(candles,args.initial_capital,args.trade_allocation,args.max_positions,args.score_threshold,n,fee)
                    detail.append({'symbol':symbol,'month':month,'candidate':f'{tf}m:N{n}','fee_mode':fee_name,**s})
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=detail[0].keys()); w.writeheader(); w.writerows(detail)
    summary_path=p.with_name(p.stem+'_summary.csv')
    groups={}
    for r in detail: groups.setdefault((r['symbol'],r['candidate'],r['fee_mode']),[]).append(r)
    sums=[]
    for (symbol,candidate,fee),items in groups.items():
        compound=1.0
        for r in items: compound*=1+float(r['return_pct'])/100
        sums.append({'symbol':symbol,'candidate':candidate,'fee_mode':fee,'months':len(items),'profitable_months':sum(float(r['return_pct'])>0 for r in items),'compound_return_pct':(compound-1)*100,'avg_month_return_pct':sum(float(r['return_pct']) for r in items)/len(items),'best_month_pct':max(float(r['return_pct']) for r in items),'worst_month_pct':min(float(r['return_pct']) for r in items),'total_trades':sum(int(r['completed_trades']) for r in items),'avg_win_rate_pct':sum(float(r['win_rate_pct']) for r in items)/len(items),'avg_max_drawdown_pct':sum(float(r['max_drawdown_pct']) for r in items)/len(items)})
    with summary_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=sums[0].keys()); w.writeheader(); w.writerows(sorted(sums,key=lambda x:float(x['compound_return_pct']),reverse=True))
    p.with_name(p.stem+'_missing.txt').write_text('\n'.join(missing)+('\n' if missing else ''),encoding='utf-8')
    print(f'SAVED {p}'); print(f'SAVED {summary_path}'); print(f'Missing files: {len(missing)}')

if __name__=='__main__': main()
