from __future__ import annotations

import argparse
import csv
from pathlib import Path

MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")

def load(path: Path, month: str, symbol: str):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("month") != month or row.get("symbol") != symbol:
                continue
            try:
                rows.append({k: float(row[k]) for k in ("open","high","low","close")} | {"timestamp": int(float(row["timestamp"]))})
            except (KeyError, TypeError, ValueError):
                pass
    return sorted(rows, key=lambda r: r["timestamp"])

def direction(cur_l, prev_l):
    if prev_l is None: return 0
    return 1 if cur_l > prev_l else -1 if cur_l < prev_l else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--output", default="reports/candle_formula_excel_signal_detail.csv")
    args = ap.parse_args()
    out=[]
    for symbol in SYMBOLS:
      for month in MONTHS:
        data=load(Path(args.input),month,symbol); prev_pred=0; pos=0; entry_price=None; trade_id=0
        for i,r in enumerate(data):
          o,h,l,c=r["open"],r["high"],r["low"],r["close"]; lc=c-o; pl=None if i==0 else data[i-1]["close"]-data[i-1]["open"]
          actual=direction(lc,pl); j=lc; k=h-c; ll=l-c; m=h-o
          R1=int(k>j); R2=int(k>m); R3=int(ll>j); R4=int(k<j); R5=int(k<m); R6=int(ll<j); S=R1+R2+R3
          sig=1 if S==3 else -1 if S==0 else 0; correct=int(prev_pred!=0 and prev_pred==actual); bias=int(-args.width<=lc<=args.width)
          entry=exit_=0; pnl=""
          if pos==0 and sig and not bias and correct:
            pos=sig; entry_price=c; trade_id+=1; entry=1
          elif pos and bias and entry_price is not None:
            pnl=pos*(c-entry_price)/entry_price*100.0; pos=0; entry_price=None; exit_=1
          out.append({"symbol":symbol,"month":month,"row":i+1,"timestamp":r["timestamp"],"open":o,"high":h,"low":l,"close":c,"L_current":lc,"L_previous":"" if pl is None else pl,"actual_direction_code":actual,"R1":R1,"R2":R2,"R3":R3,"R4":R4,"R5":R5,"R6":R6,"S":S,"signal_code":sig,"previous_prediction_code":prev_pred,"previous_prediction_correct":correct,"bias_width":args.width,"in_bias":bias,"position_after":pos,"entry":entry,"exit":exit_,"trade_id":trade_id if(pos or entry or exit_) else "","entry_price":"" if entry_price is None else entry_price,"trade_pnl_pct":pnl})
          prev_pred=sig
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8-sig",newline="") as f:
      w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"DETAIL_CSV={p}"); print(f"ROWS={len(out)}"); print("SIGNAL=S3_BUY/S0_SELL/S1,S2_HOLD"); print("DIRECTION=IF(L_current>L_previous,1,IF(L_current<L_previous,-1,0))"); print("FEE=0"); print(f"WIDTH={args.width:g}")
if __name__=="__main__": main()
