from __future__ import annotations
import argparse, csv, statistics
from pathlib import Path

SYMS=('BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT')
MONTHS=('2026-05','2026-06','2026-07','2026-08')
FEE_SIDE=0.0013
CAPITAL=1000.0
MIN_BODY=0.0001


def load_rows(p: Path, sym: str):
    out=[]
    with p.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('symbol')!=sym: continue
            try: out.append((r['month'],int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close'])))
            except (KeyError,TypeError,ValueError): continue
    return sorted(out,key=lambda x:x[1])

def S(r):
    _,_,o,h,l,c=r
    j=c-o; k=h-c; m=h-o
    return int(k>j)+int(k>m)+int(l>j)

def side(s): return 1 if s in (2,3) else -1

def fee_net(gross_pct):
    g=gross_pct/100.0
    exit_value=CAPITAL*(1+g)
    fees=CAPITAL*FEE_SIDE+max(exit_value,0.0)*FEE_SIDE
    return (CAPITAL*g-fees)/CAPITAL*100

def audit(rows):
    gross=[]; net=[]; by={s:[] for s in range(4)}; wins=0; n=0; profitable=0
    for i,r in enumerate(rows[:-1]):
        b=abs(r[5]-r[2]);
        if r[5]==0 or b/abs(r[5])<MIN_BODY: continue
        s=S(r); sd=side(s)
        # One-bar close-to-close directional outcome, matching the direction target.
        move=sd*(rows[i+1][5]-r[5])/r[5]*100
        g=move
        ng=fee_net(g)
        gross.append(g); net.append(ng); by[s].append((g,ng)); n+=1; wins+=int(ng>0); profitable+=int(g>2*FEE_SIDE*100)
    def stats(v):
        if not v: return (0,0,0,0)
        return statistics.mean(v), statistics.median(v), min(v), max(v)
    print(f'  all n={n} gross_mean={stats(gross)[0]:+.3f}% gross_median={stats(gross)[1]:+.3f}% net_mean={stats(net)[0]:+.3f}% net_median={stats(net)[1]:+.3f}% net_win={100*wins/n if n else 0:.1f}%')
    print(f'  gross_move_gt_roundtrip_fee={100*profitable/n if n else 0:.1f}% (threshold≈0.26%)')
    for s in range(4):
        g=[x[0] for x in by[s]]; nn=[x[1] for x in by[s]]
        print(f'  S{s} n={len(g)} gross_mean={stats(g)[0]:+.3f}% gross_median={stats(g)[1]:+.3f}% net_mean={stats(nn)[0]:+.3f}% net_median={stats(nn)[1]:+.3f}%')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); a=ap.parse_args()
    print('EXCEL_DIRECTION_FEE_AUDIT | tf=1h | exact S0-S3 | fee_side=0.13% | one-bar close target | past_only')
    for sym in SYMS:
        rows=load_rows(a.input,sym); print(sym); audit(rows)

if __name__=='__main__': main()
