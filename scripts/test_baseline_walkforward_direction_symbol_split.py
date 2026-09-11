from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
NOISE = 1.0
FEE = 0.0013
CANDIDATES = [0.0075, 0.0080, 0.0083, 0.0085, 0.0090]
MONTHS = ["2026-05", "2026-06", "2026-07", "2026-08"]

def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "month" in df.columns:
        month = df["month"].astype(str)
    else:
        month = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", utc=True).dt.strftime("%Y-%m")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    body = df.close - df.open
    hc = df.high - df.close
    ho = df.high - df.open
    lc = df.low - df.close
    score = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)
    upper = (df.high - df.open).abs()
    lower = (df.open - df.low).abs()
    formula = np.where(lower > upper, 1, np.where(upper > lower, -1, 0))
    direction = np.where(score <= 1, -1, 1).astype(int)
    direction[(score == 1) & (formula == 1)] = 0
    direction[(score == 2) & (formula == -1)] = 0
    magnitude = np.where(direction == 1, upper, np.where(direction == -1, lower, np.nan))
    out = df.copy()
    out["month"] = month.values
    out["direction"] = direction
    out["magnitude_pct"] = magnitude / out.close
    out["body"] = body
    return out

def backtest(df: pd.DataFrame, start: int, end: int, entry_min: float, initial: float, direction_filter: int | None) -> tuple[float,int,int]:
    factor=(1-FEE)**2; capital=initial; trades=wins=0; i=start; end=min(end,len(df)-1)
    while i<end:
        row=df.iloc[i]; d=int(row.direction); exp=float(row.magnitude_pct); body=abs(float(row.body))
        if direction_filter is not None and d!=direction_filter: i+=1; continue
        if d==0 or not np.isfinite(exp) or exp<entry_min or body<=0: i+=1; continue
        entry=float(row.close); first=0.0; count=0; exit_i=None; j=i+1
        while j<end:
            cur=df.iloc[j]; pdirection=int(cur.direction); pmag=float(cur.magnitude_pct) if np.isfinite(cur.magnitude_pct) else np.nan
            if pdirection==0 or not np.isfinite(pmag): j+=1; continue
            move=abs(float(cur.close)-entry)/max(abs(entry),1e-12); ref=max(move,body/max(abs(entry),1e-12),1e-12)
            if pdirection==d: count=0; first=0.0; j+=1; continue
            count+=1
            if count==1:
                first=pmag
                if pmag<=NOISE*ref: j+=1; continue
                exit_i=j; break
            if count==2:
                if pmag<2*first or pmag<=2*NOISE*ref: j+=1; continue
                exit_i=j; break
            exit_i=j; break
        if exit_i is None: break
        ep=float(df.iloc[exit_i].close); gross=(ep-entry)/entry if d==1 else (entry-ep)/entry
        capital*=max((1+gross)*factor,0.0); trades+=1; wins+=gross>0; i=exit_i+1
    return capital,trades,wins

def train_select(loaded,symbols,train_end,initial,direction_filter):
    best=None
    months=MONTHS[:train_end+1]
    for cand in CANDIDATES:
        total=0.0
        for s in symbols:
            df=loaded[s]; idx=df.index[df.month.isin(months)]
            if len(idx): total+=backtest(df,int(idx.min()),int(idx.max()+1),cand,initial,direction_filter)[0]
        ret=total/(initial*len(symbols))-1
        item=(ret,cand)
        if best is None or item[0]>best[0]: best=item
    return best[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); ap.add_argument('--symbols',default=SYMBOLS); ap.add_argument('--initial-capital',type=float,default=1000.0); args=ap.parse_args()
    symbols=[x.strip() for x in args.symbols.split(',') if x.strip()]; root=Path(args.data_dir); loaded={s:load(root/f'{s}_1h.csv') for s in symbols}
    print('='*100); print('WALK-FORWARD DIRECTION + SYMBOL BREAKDOWN'); print('Same candidates | Noise=1.00 | Fee=0.13%/side | Monthly OOS | Compounded'); print('='*100)
    for direction_filter,label in [(None,'ALL'),(1,'LONG'),(-1,'SHORT')]:
        capital={s:args.initial_capital for s in symbols}; total_trades=total_wins=0
        print(f'\n=== {label} ===')
        for pos in range(1,len(MONTHS)):
            selected=train_select(loaded,symbols,pos-1,args.initial_capital,direction_filter)
            month=MONTHS[pos]; before=sum(capital.values());
            for s in symbols:
                df=loaded[s]; idx=df.index[df.month==month]
                if len(idx):
                    initial_symbol=capital[s]; final,t,w=backtest(df,int(idx.min()),int(idx.max()+1),selected,initial_symbol,direction_filter); capital[s]=final; total_trades+=t; total_wins+=w
            after=sum(capital.values()); print(f'{month}: selected={selected*100:.2f}% pooled={after:.2f} return={after/before-1:+.2%} '+ ' '.join(f'{s}={capital[s]:.2f}' for s in symbols))
        total=sum(capital.values()); print(f'FINAL {label}: ${total:.2f} from ${args.initial_capital*len(symbols):.2f} return={(total/(args.initial_capital*len(symbols))-1)*100:+.2f}% trades={total_trades} win={total_wins/total_trades*100 if total_trades else 0:.1f}%')

if __name__=='__main__': main()
