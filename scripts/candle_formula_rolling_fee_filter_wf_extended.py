from __future__ import annotations

import argparse, csv, statistics
from collections import defaultdict
from pathlib import Path

SYMS = ('BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT')
MONTHS = ('2026-05','2026-06','2026-07','2026-08')
CAPITAL = 1000.0
MIN_BODY = 0.0001
FEE_SIDE = 0.0013
# Fixed sensitivity grid. Threshold is selected only from the immediately prior month.
THRESHOLDS = (0.005, 0.0075, 0.010, 0.0125, 0.015, 0.020, 0.025, 0.030)


def load_rows(p: Path, sym: str):
    out=[]
    with p.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol') != sym: continue
            try:
                out.append((r['month'], int(float(r['timestamp'])), float(r['open']), float(r['high']), float(r['low']), float(r['close'])))
            except (KeyError, TypeError, ValueError): continue
    return sorted(out, key=lambda x:x[1])


def S(r):
    _,_,o,h,l,c=r
    j=c-o; k=h-c; m=h-o
    return int(k>j)+int(k>m)+int(l>j)


def side(s): return 1 if s in (2,3) else -1


def fee_net_pct(gross_pct):
    gross=gross_pct/100.0
    exit_value=CAPITAL*(1.0+gross)
    fees=CAPITAL*FEE_SIDE + max(exit_value,0.0)*FEE_SIDE
    return (CAPITAL*gross-fees)/CAPITAL*100.0


def build_history(rows):
    hist=defaultdict(list); glob=[]
    for i,r in enumerate(rows[:-1]):
        b=abs(r[5]-r[2])
        if r[5]==0 or b/abs(r[5])<MIN_BODY: continue
        x=abs(rows[i+1][5]-r[5])/b
        if x==x:
            hist[S(r)].append(x); glob.append(x)
    return hist,glob


def run_month(rows, seed_hist, seed_glob, threshold):
    hist=defaultdict(list,{k:list(v) for k,v in seed_hist.items()})
    glob=list(seed_glob)
    eq=CAPITAL; peak=CAPITAL; max_dd=0.0
    pos=None; ei=-1; trades=wins=0; filtered=0
    for i,r in enumerate(rows[:-1]):
        s=S(r); b=abs(r[5]-r[2]); gr=statistics.median(glob) if glob else 2.0
        bucket=hist[s]; ratio=statistics.median(bucket) if bucket else gr
        eff=max(b,abs(r[5])*MIN_BODY)
        predicted_tp_pct=eff*ratio/r[5]*100.0 if r[5] else 0.0
        if pos is None:
            if predicted_tp_pct < threshold*100.0:
                filtered += 1
            else:
                sd=side(s); tp=eff*ratio; sl=0.5*tp
                pos=(sd,r[5],tp,sl); ei=i
        if pos is not None and i>ei:
            sd,en,tp,sl=pos; px=r[5]; gross_pct=None
            if sd==1:
                if px>=en+tp: gross_pct=tp/en*100.0
                elif px<=en-sl: gross_pct=-sl/en*100.0
            else:
                if px<=en-tp: gross_pct=tp/en*100.0
                elif px>=en+sl: gross_pct=-sl/en*100.0
            if gross_pct is not None:
                net_pct=fee_net_pct(gross_pct)
                eq += CAPITAL*net_pct/100.0; trades+=1; wins+=int(net_pct>0)
                pos=None; ei=-1; peak=max(peak,eq); max_dd=max(max_dd,(peak-eq)/peak if peak else 0.0)
        if r[5]!=0 and b/abs(r[5])>=MIN_BODY if r[5] else False:
            x=abs(rows[i+1][5]-r[5])/b if b else float('nan')
            if x==x: hist[s].append(x); glob.append(x)
    if pos is not None:
        sd,en,tp,sl=pos; px=rows[-1][5]
        gross_pct=sd*(px-en)/en*100.0 if en else 0.0
        net_pct=fee_net_pct(gross_pct)
        eq += CAPITAL*net_pct/100.0; trades+=1; wins+=int(net_pct>0)
        peak=max(peak,eq); max_dd=max(max_dd,(peak-eq)/peak if peak else 0.0)
    return {'net':(eq-CAPITAL)/CAPITAL*100.0,'trades':trades,'win':100.0*wins/trades if trades else 0.0,'dd':100.0*max_dd,'filtered':filtered}


def select(train_rows, seed_rows):
    seed_hist,seed_glob=build_history(seed_rows)
    scores=[]
    for t in THRESHOLDS:
        r=run_month(train_rows,seed_hist,seed_glob,t)
        scores.append((r['net'],t,r))
    return max(scores,key=lambda x:x[0]),scores


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); a=ap.parse_args()
    print('EXCEL_ROLLING_FEE_FILTER_WF_EXTENDED | tf=1h | fee_side=0.13% | round_trip≈0.26% | STATE | fixed trade capital=$1000')
    print('Threshold grid=0.50..3.00%; each OOS month uses only the previous month to select threshold.')
    for sym in SYMS:
        rows=load_rows(a.input,sym)
        mr={m:[r for r in rows if r[0]==m] for m in MONTHS}
        # JUL selection from JUNE, seeded with MAY.
        sel_jul, _ = select(mr['2026-06'], mr['2026-05'])
        t_jul=sel_jul[1]
        h_jul,g_jul=build_history(mr['2026-05']+mr['2026-06'])
        jul=run_month(mr['2026-07'],h_jul,g_jul,t_jul)
        # AUG selection from JUL, seeded with history through JUNE.
        sel_aug,_ = select(mr['2026-07'], mr['2026-05']+mr['2026-06'])
        t_aug=sel_aug[1]
        h_aug,g_aug=build_history(mr['2026-05']+mr['2026-06']+mr['2026-07'])
        aug=run_month(mr['2026-08'],h_aug,g_aug,t_aug)
        base_j=run_month(mr['2026-07'],h_jul,g_jul,0.0)
        base_a=run_month(mr['2026-08'],h_aug,g_aug,0.0)
        print(sym)
        print(f"  JUL selected={t_jul*100:.2f}% via_JUN={sel_jul[2]['net']:+.2f}% | OOS_JUL={jul['net']:+.2f}% trades={jul['trades']} win={jul['win']:.1f}% DD={jul['dd']:.2f}%")
        print(f"  AUG selected={t_aug*100:.2f}% via_JUL={sel_aug[2]['net']:+.2f}% | OOS_AUG={aug['net']:+.2f}% trades={aug['trades']} win={aug['win']:.1f}% DD={aug['dd']:.2f}%")
        print(f"  OOS_2M filter_sum={jul['net']+aug['net']:+.2f}% baseline_sum={base_j['net']+base_a['net']:+.2f}% delta={jul['net']+aug['net']-base_j['net']-base_a['net']:+.2f}pp")

if __name__=='__main__': main()
