from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
CAPITAL = 1000.0
MIN_BODY_RATIO = 0.0001


def load(path: Path, symbol: str):
    out = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("symbol") != symbol:
                continue
            try:
                out.append((str(r["month"]), int(float(r["timestamp"])), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(out, key=lambda x: x[1])


def S(r) -> int:
    _, _, o, h, l, c = r
    j = c-o; k = h-c; m = h-o; ll = l-c
    return int(k > j) + int(k > m) + int(ll > j)


def side(s: int) -> int:
    return 1 if s in (2, 3) else -1


def run(rows, month: str, mode: str):
    rs = [r for r in rows if r[0] == month]
    hist = defaultdict(list); global_hist = []
    equity = CAPITAL; position = None; entry_i = -1
    pnl = 0.0; trades = wins = 0; changed = 0; entries = 0; ratio_diffs = []

    for i in range(len(rs)-1):
        r = rs[i]; s = S(r); sd = side(s)
        body = abs(r[5]-r[2]); valid = r[5] != 0 and body/abs(r[5]) >= MIN_BODY_RATIO
        global_ratio = statistics.median(global_hist) if global_hist else 2.0
        state_ratio = statistics.median(hist[s]) if hist[s] else global_ratio

        if position is None:
            entries += 1
            if state_ratio != global_ratio:
                changed += 1
            ratio_diffs.append(abs(state_ratio-global_ratio))
            ratio = state_ratio if mode == "state" else global_ratio
            effective_body = max(body, abs(r[5])*MIN_BODY_RATIO)
            tp_dist = effective_body * ratio; sl_dist = tp_dist*0.5
            target = r[5]+tp_dist if sd == 1 else r[5]-tp_dist
            stop = r[5]-sl_dist if sd == 1 else r[5]+sl_dist
            position = (sd, r[5], target, stop, ratio)
            entry_i = i

        if position is not None and i > entry_i:
            ps, en, target, stop, _ = position
            tp = False
            if ps == 1:
                if r[5] >= target: tp=True
                elif r[5] <= stop: tp=False
                else: tp=None
            else:
                if r[5] <= target: tp=True
                elif r[5] >= stop: tp=False
                else: tp=None
            if tp is not None:
                if tp:
                    exit_price = target
                else:
                    exit_price = stop
                tr_pnl = ps*(exit_price-en)/en*100.0
                pnl += tr_pnl
                trades += 1; wins += int(tr_pnl > 0)
                position = None; entry_i=-1

        if valid:
            ratio = abs(rs[i+1][5]-r[5])/body
            hist[s].append(ratio); global_hist.append(ratio)

    if position is not None:
        ps,en,_,_,_ = position
        tr_pnl = ps*(rs[-1][5]-en)/en*100.0
        pnl += tr_pnl; trades += 1; wins += int(tr_pnl > 0)

    return pnl, entries, changed, statistics.mean(ratio_diffs) if ratio_diffs else 0.0, trades, wins


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('reports/prepared_price_action_1h.csv')); args=ap.parse_args()
    print('EXCEL_TPSL_SANITY | exact S0-S3 | close_only | past_only | fixed_trade_capital=$1000')
    for sym in SYMBOLS:
        rows=load(args.input,sym); print('\n'+sym)
        for mode in ('global','state'):
            vals=[run(rows,m,mode) for m in MONTHS]
            pnl=sum(v[0] for v in vals); entries=sum(v[1] for v in vals); changed=sum(v[2] for v in vals); md=sum(v[3]*v[1] for v in vals)/entries; trades=sum(v[4] for v in vals); wins=sum(v[5] for v in vals)
            print(f'{mode.upper()} pnl_sum={pnl:+.4f}% entries={entries} changed_ratio_entries={changed} diff_pct={(100*changed/entries if entries else 0):.1f}% mean_abs_ratio_diff={md:.4f} trades={trades} win={(100*wins/trades if trades else 0):.2f}%')
        print('NOTE: GLOBAL and STATE should diverge whenever changed_ratio_entries > 0; this test computes both from the same replay logic.')

if __name__=='__main__':
    main()
