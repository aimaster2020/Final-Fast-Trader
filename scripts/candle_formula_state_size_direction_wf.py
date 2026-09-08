from __future__ import annotations

import argparse, csv
from collections import defaultdict, deque
from pathlib import Path

SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
MONTHS=("2026-05","2026-06","2026-07","2026-08")
FEE_SIDE=0.0013
MIN_HISTORY=100
BINS=4


def load(path: Path, symbol: str):
    out=[]
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("symbol") != symbol:
                continue
            try:
                out.append((r["month"], int(float(r["timestamp"])), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(out, key=lambda x:x[1])


def s(row):
    _,_,o,h,l,c=row
    j=c-o; k=h-c; m=h-o
    return int(k>j)+int(k>m)+int(l>j)


def sign(x):
    return 1 if x>0 else -1 if x<0 else 0


def qbin(x, cuts):
    for i,c in enumerate(cuts):
        if x<=c:
            return i
    return len(cuts)


def quantile_cuts(vals):
    if not vals:
        return None
    a=sorted(vals)
    cuts=[]
    for q in (0.25,0.50,0.75):
        idx=min(len(a)-1, int(q*(len(a)-1)))
        cuts.append(a[idx])
    return cuts


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("reports/prepared_price_action_1h.csv"))
    args=ap.parse_args()
    print("STATE_SIZE_DIRECTION_WF | tf=1h | fee=0.13%/side | one-bar | past_only")
    print("Features: S plus current body size/close quantile from past only. Direction model is empirical next-price sign by (S,size_bin).")
    print("Decision thresholds are fixed diagnostics: p>=0.55 and p>=0.60; no Aug fitting.")
    for sym in SYMBOLS:
        rows=load(args.input, sym)
        body_hist=deque(maxlen=1000)
        stats=defaultdict(lambda:[0,0])
        total_rows=0
        print(f"\n{sym}")
        for prob_cut in (0.55,0.60):
            equity=1000.0; trades=wins=0; correct=0
            gross_sum=net_sum=0.0
            monthly={m:[1000.0,0,0,0.0,0.0] for m in MONTHS}
            body_hist.clear(); stats=defaultdict(lambda:[0,0])
            seen=0
            for i in range(len(rows)-1):
                cur=rows[i]; nxt=rows[i+1]
                body_pct=abs(cur[5]-cur[2])/cur[5] if cur[5] else 0.0
                st=s(cur)
                if len(body_hist)>=MIN_HISTORY:
                    cuts=quantile_cuts(list(body_hist))
                    b=qbin(body_pct, cuts)
                    n,p=stats[(st,b)]
                    prob=(p/n) if n else 0.5
                    if prob>=prob_cut:
                        side=1
                    elif prob<=1.0-prob_cut:
                        side=-1
                    else:
                        side=0
                    actual=sign(nxt[5]-cur[5])
                    if side:
                        gross=side*(nxt[5]-cur[5])/cur[5]*100
                        net=gross-2*FEE_SIDE*100
                        trades+=1; wins+=int(net>0); correct+=int(side==actual)
                        gross_sum+=gross; net_sum+=net
                        z=monthly[cur[0]]; z[1]+=1; z[2]+=int(side==actual); z[3]+=gross; z[4]+=net
                body_hist.append(body_pct)
                actual_next_body=sign(nxt[5]-nxt[2])
                if side if False else False:
                    pass
                # Update empirical table after observing next candle. The key is the current candle's features.
                if len(body_hist)>=MIN_HISTORY:
                    cuts2=quantile_cuts(list(body_hist)[:-1])
                    if cuts2 is not None:
                        b2=qbin(body_pct, cuts2)
                        z=stats[(st,b2)]
                        z[0]+=1; z[1]+=int(actual_next_body>0)
                else:
                    # Still build history before model becomes active.
                    pass
            print(f"  P>={prob_cut:.2f} trades={trades} acc={100*correct/trades if trades else 0:.1f}% avg_gross={gross_sum/trades if trades else 0:+.4f}% avg_net={net_sum/trades if trades else 0:+.4f}% net_sum={net_sum:+.2f}% win={100*wins/trades if trades else 0:.1f}%")
            print("   "+" ".join(f"{m}:{monthly[m][4]:+.2f}%({int(monthly[m][1])})" for m in MONTHS))
        # Simple static diagnostic of (S,size-bin) counts and direction probabilities built over whole history is intentionally omitted.

if __name__=="__main__":
    main()
