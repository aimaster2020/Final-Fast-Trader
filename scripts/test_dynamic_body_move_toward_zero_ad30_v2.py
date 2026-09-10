from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median

EPS = 1e-12

@dataclass
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float


def load(path: Path) -> list[Candle]:
    out=[]
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            try:
                out.append(Candle(str(r.get('timestamp','')), float(r['open']), float(r['high']), float(r['low']), float(r['close'])))
            except (KeyError, TypeError, ValueError):
                pass
    return out


def body(c: Candle) -> float:
    return c.close - c.open


def ad(c: Candle) -> int:
    b=body(c)
    hc=c.high-c.close
    ho=c.high-c.open
    lc=c.low-c.close
    return int(hc>b)+int(ho<hc)+int(lc>b)


def move_ratio(cur: float, nxt: float) -> float | None:
    if abs(cur)<=EPS:
        return None
    return math.copysign(1.0,cur)*(cur-nxt)/abs(cur)


def predict(rows: list[Candle], i: int, lookback: int, minh: int) -> tuple[float|None,int,float|None]:
    score=ad(rows[i])
    vals=[]
    start=max(1,i-lookback)
    for j in range(start,i):
        if ad(rows[j])!=score:
            continue
        x=move_ratio(body(rows[j]),body(rows[j+1]))
        if x is not None and math.isfinite(x):
            vals.append(x)
    if len(vals)<minh:
        return None,len(vals),None
    pred=median(vals)
    b=body(rows[i])
    pred_next=b-math.copysign(abs(b)*pred,b)
    return pred,len(vals),pred_next


def trade_return(side: str, entry: float, exit_price: float, fee: float) -> tuple[float,float]:
    gross=(exit_price/entry-1.0) if side=='LONG' else (1.0-exit_price/entry)
    return gross, gross-2.0*fee


def run(rows: list[Candle], symbol: str, score_filter: int|None, threshold: float, fee: float, lookback: int, minh: int, capital0: float):
    capital=capital0
    trades=[]
    candidates=signals=eligible=0
    hist_skips=0
    ratio_actual=[]
    pred_ratios=[]
    for i in range(1,len(rows)-1):
        s=ad(rows[i])
        if s not in (0,3) or (score_filter is not None and s!=score_filter):
            continue
        candidates+=1
        pred,hist_n,pred_next=predict(rows,i,lookback,minh)
        if pred is None:
            hist_skips+=1
            continue
        signals+=1
        actual=body(rows[i+1])
        ar=move_ratio(body(rows[i]),actual)
        if ar is not None:
            ratio_actual.append(ar); pred_ratios.append(pred)
        if pred<threshold:
            continue
        eligible+=1
        side='LONG' if s==3 else 'SHORT'
        entry=rows[i].close
        exit_price=rows[i+1].close
        if entry<=EPS:
            continue
        gross,net=trade_return(side,entry,exit_price,fee)
        before=capital
        after=before*(1.0+net)
        trades.append((i,rows[i].timestamp,rows[i+1].timestamp,side,s,body(rows[i]),pred,pred_next,entry,exit_price,gross,2*fee,net,before,after,ar))
        capital=after
    wins=sum(1 for t in trades if t[12]>0)
    gross_wins=sum(1 for t in trades if t[10]>0)
    gp=sum(t[10] for t in trades if t[10]>0); gl=sum(t[10] for t in trades if t[10]<0)
    np=sum(t[12] for t in trades if t[12]>0); nl=sum(t[12] for t in trades if t[12]<0)
    peak=capital0; dd=0.0; path=capital0
    for t in trades:
        path=t[14]; peak=max(peak,path); dd=max(dd,(peak-path)/peak if peak>EPS else 0.0)
    return {
        'candidates':candidates,'signals':signals,'eligible':eligible,'trades':len(trades),'wins':wins,'gross_wins':gross_wins,
        'wr':wins/len(trades)*100 if trades else 0.0,'gross_wr':gross_wins/len(trades)*100 if trades else 0.0,
        'ret':capital/capital0-1.0,'final':capital,
        'pf':(np/abs(nl) if nl< -EPS else math.inf),'gross_pf':(gp/abs(gl) if gl< -EPS else math.inf),'dd':dd,'hist_skips':hist_skips,
        'pred_med':median(pred_ratios) if pred_ratios else 0.0,'actual_med':median(ratio_actual) if ratio_actual else 0.0,
        'trades_data':trades,'avg_gross':(sum(t[10] for t in trades)/len(trades) if trades else 0.0),
        'avg_net':(sum(t[12] for t in trades)/len(trades) if trades else 0.0),
        'longs':sum(1 for t in trades if t[3]=='LONG'),'shorts':sum(1 for t in trades if t[3]=='SHORT'),
        'positive_signals':sum(1 for t in trades if t[12]>0),'negative_signals':sum(1 for t in trades if t[12]<0),
    }


def write_csv(path: Path, trades) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f)
        w.writerow(['signal_index','entry_timestamp','exit_timestamp','side','ad','current_body','predicted_move_ratio','predicted_next_body','entry_price','exit_price','gross_return','fee_return','net_return','capital_before','capital_after','actual_move_ratio'])
        for t in trades:
            w.writerow(t)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',default='reports/1h'); p.add_argument('--symbols',default='BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT')
    p.add_argument('--lookback',type=int,default=50); p.add_argument('--min-history',type=int,default=15)
    p.add_argument('--capital',type=float,default=1000.0); p.add_argument('--commission',type=float,default=0.0013)
    p.add_argument('--thresholds',default='0.10,0.20,0.30,0.40,0.50,0.60,0.70')
    p.add_argument('--log-dir',default='reports/move_to_zero_trades_ad30_v2')
    a=p.parse_args(); root=Path(a.input_dir); log=Path(a.log_dir); syms=[x.strip() for x in a.symbols.split(',') if x.strip()]; ths=[float(x) for x in a.thresholds.split(',')]
    print(f'MODEL=causal rolling median move-to-zero | LB={a.lookback} MINH={a.min_history} CAPITAL={a.capital:.2f} | fee={a.commission*100:.2f}%/side')
    print('CORRECT ratio=sign(body)*(current_body-next_body)/abs(body); positive=toward-zero')
    print('ENTRY=current close EXIT=next close | AD3=LONG AD0=SHORT | 100% capital | one-bar hold')
    for sym in syms:
        rows=load(root/f'{sym}_1h.csv'); print(f'\n=== {sym} rows={len(rows)} ===')
        for sf,label in [(3,'AD3'),(0,'AD0'),(None,'ALL')]:
            print(f'[{label}]')
            for t in ths:
                r0=run(rows,sym,sf,t,0.0,a.lookback,a.min_history,a.capital)
                r1=run(rows,sym,sf,t,a.commission,a.lookback,a.min_history,a.capital)
                pf0='INF' if math.isinf(r0['pf']) else f"{r0['pf']:.2f}"; pf1='INF' if math.isinf(r1['pf']) else f"{r1['pf']:.2f}"
                print(f"T={t:.2f} C={r0['candidates']} SIG={r0['signals']} TR={r0['trades']} WR0={r0['wr']:.1f}% R0={r0['ret']*100:.2f}% F0={r0['final']:.2f} PF0={pf0} DD0={r0['dd']*100:.2f}% | WRfee={r1['wr']:.1f}% Rfee={r1['ret']*100:.2f}% Ffee={r1['final']:.2f} PFfee={pf1} DDfee={r1['dd']*100:.2f}% | PRED_MED={r0['pred_med']:.3f} ACT_MED={r0['actual_med']:.3f} SKIP_H={r0['hist_skips']} LONG={r0['longs']} SHORT={r0['shorts']}")
                write_csv(log/f'{sym}_{label}_T{t:.2f}_F0.csv',r0['trades_data']); write_csv(log/f'{sym}_{label}_T{t:.2f}_F0_13.csv',r1['trades_data'])

if __name__=='__main__': main()
