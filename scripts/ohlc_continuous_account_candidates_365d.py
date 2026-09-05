from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from fast_pattern_trader.models import Candle, Signal
from fast_pattern_trader.ohlc_rule_strategy import DEFAULT_RULE_WEIGHTS, decide_movement

NOBITEX_TAKER_FEE_PER_SIDE = 0.001

@dataclass
class Position:
    side: int
    entry: float
    allocation: float
    opened_at: int


def parse_candidates(text: str) -> list[tuple[int,int]]:
    out=[]
    for item in text.split(','):
        tf,n=item.strip().split(':')
        out.append((int(tf),int(n)))
    return out


def resample(candles: list[Candle], minutes: int) -> list[Candle]:
    if minutes==1: return candles
    bucket=minutes*60; groups={}
    for c in candles: groups.setdefault((c.timestamp//bucket)*bucket,[]).append(c)
    out=[]
    for key,group in sorted(groups.items()):
        group=sorted(group,key=lambda x:x.timestamp)
        ids=sorted({x.timestamp//60 for x in group})
        if len(ids)!=minutes or ids[-1]-ids[0]+1!=minutes: continue
        out.append(Candle(key,group[0].open,max(x.high for x in group),min(x.low for x in group),group[-1].close))
    return out


def evaluate(candles: list[Candle], initial_capital: float, allocation: float, max_positions: int, score_threshold: float, n: int, fee_per_side: float=0.0):
    capital=initial_capital; peak=capital; max_dd=0.0; positions=[]; trades=[]; opposite_count=0; last_signal=0; fees=0.0
    for c in candles:
        weights=dict(DEFAULT_RULE_WEIGHTS); weights['R16']=0.0
        d=decide_movement(c,score_threshold=score_threshold,rule_weights=weights)
        sig=int(d.signal)
        if sig and sig==last_signal: opposite_count=0
        elif sig and sig==-last_signal: opposite_count+=1
        elif sig: opposite_count=1
        else: opposite_count=0

        if positions and sig and sig==-positions[0].side and opposite_count>=n:
            new=[]
            for p in positions:
                gross=(c.close/p.entry-1) if p.side==1 else (p.entry/c.close-1)
                net=gross-2*fee_per_side
                capital += p.allocation*net
                fees += p.allocation*2*fee_per_side
                trades.append(net)
            positions=new; opposite_count=0

        if sig and not positions and len(positions)<max_positions:
            alloc=capital*allocation
            capital-=alloc*fee_per_side
            fees += alloc*fee_per_side
            positions.append(Position(sig,c.close,alloc,c.timestamp))

        last_signal=sig if sig else last_signal
        equity=capital
        for p in positions:
            equity += p.allocation*((c.close/p.entry-1) if p.side==1 else (p.entry/c.close-1))
        peak=max(peak,equity); max_dd=max(max_dd,(peak-equity)/peak*100 if peak else 0)

    for p in positions:
        gross=(candles[-1].close/p.entry-1) if p.side==1 else (p.entry/candles[-1].close-1)
        net=gross-2*fee_per_side
        capital += p.allocation*net
        fees += p.allocation*2*fee_per_side
        trades.append(net)
    wins=sum(x>0 for x in trades)
    days=max((candles[-1].timestamp-candles[0].timestamp)/86400,1/24)
    summary={
        'initial_capital':initial_capital,'final_capital':capital,'return_pct':(capital/initial_capital-1)*100,
        'completed_trades':len(trades),'win_rate_pct':wins/len(trades)*100 if trades else 0.0,
        'max_drawdown_pct':max_dd,'commission_paid':fees,'calendar_days':days,
    }
    return summary,trades


def load_1m(path: Path)->list[Candle]:
    with path.open('r',newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f); required={'timestamp','open','high','low','close'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames): raise ValueError('CSV must contain timestamp,open,high,low,close')
        rows=[]
        for r in reader:
            try: rows.append(Candle(int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close'])))
            except (ValueError,TypeError): continue
    return sorted(rows,key=lambda x:x.timestamp)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',default='reports/ohlc_candidates_365d.csv'); ap.add_argument('--initial-capital',type=float,default=1000); ap.add_argument('--trade-allocation',type=float,default=.30); ap.add_argument('--max-positions',type=int,default=2); ap.add_argument('--score-threshold',type=float,default=1.0); ap.add_argument('--candidates',default='1:5,5:5,15:5,30:4,60:5,60:6'); args=ap.parse_args()
    raw=load_1m(Path(args.input)); rows=[]
    for tf,n in parse_candidates(args.candidates):
        candles=resample(raw,tf)
        for name,fee in [('no_commission',0.0),('nobitex_taker',NOBITEX_TAKER_FEE_PER_SIDE)]:
            s,_=evaluate(candles,args.initial_capital,args.trade_allocation,args.max_positions,args.score_threshold,n,fee)
            rows.append({'timeframe_min':tf,'N':n,'fee_mode':name,**s})
            print(f"{tf}m:N{n} {name} return={s['return_pct']:.2f}% trades={s['completed_trades']} win={s['win_rate_pct']:.2f}% DD={s['max_drawdown_pct']:.2f}%")
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

if __name__=='__main__': main()
