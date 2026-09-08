from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
CAPITAL = 1000.0
MIN_BODY_RATIO = 0.0001


def load_rows(path: Path, symbol: str):
    out = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for rec in csv.DictReader(f):
            if rec.get("symbol") != symbol:
                continue
            try:
                out.append(
                    (str(rec["month"]), int(float(rec["timestamp"])),
                     float(rec["open"]), float(rec["high"]),
                     float(rec["low"]), float(rec["close"]))
                )
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(out, key=lambda x: x[1])


def excel_s(row) -> int:
    _, _, o, h, _, c = row
    j = c - o
    k = h - c
    m = h - o
    l = row[4] - c
    return int(k > j) + int(k > m) + int(l > j)


def side_from_s(s: int) -> int:
    return 1 if s in (2, 3) else -1


def pnl_pct(side: int, entry: float, price: float) -> float:
    return side * (price - entry) / entry * 100.0 if entry > 0 else 0.0


def run_month(rows, month: str, mode: str):
    rs = [r for r in rows if r[0] == month]
    if len(rs) < 2:
        return {"pnl": 0.0, "trades": 0, "wins": 0, "dd": 0.0}

    # Each mode uses only history before the current row:
    # state = state-specific median; global = one median for all states;
    # fixed2 = original 2x-body baseline.
    state_hist = defaultdict(list)
    global_hist = []
    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    position = None
    entry_i = -1
    trades = wins = 0

    for i, cur in enumerate(rs[:-1]):
        s = excel_s(cur)
        side = side_from_s(s)
        _, _, o, _, _, c = cur
        body = abs(c - o)
        valid = c != 0.0 and body / abs(c) >= MIN_BODY_RATIO

        if mode == "fixed2":
            ratio = 2.0
        elif valid:
            ratio = (statistics.median(state_hist[s]) if state_hist[s]
                     else (statistics.median(global_hist) if global_hist else 2.0))
        else:
            ratio = 2.0

        # Enter one trade at each available flat point.
        if position is None:
            effective_body = max(body, abs(c) * MIN_BODY_RATIO)
            tp_dist = effective_body * ratio
            sl_dist = tp_dist * 0.5
            target = c + tp_dist if side == 1 else c - tp_dist
            stop = c - sl_dist if side == 1 else c + sl_dist
            position = (side, c, target, stop)
            entry_i = i

        if position is not None and i > entry_i:
            ps, entry, target, stop = position
            trade_pnl = None
            if ps == 1:
                if c >= target:
                    trade_pnl = (target - entry) / entry * 100.0
                elif c <= stop:
                    trade_pnl = (stop - entry) / entry * 100.0
            else:
                if c <= target:
                    trade_pnl = (entry - target) / entry * 100.0
                elif c >= stop:
                    trade_pnl = (entry - stop) / entry * 100.0
            if trade_pnl is not None:
                equity += CAPITAL * trade_pnl / 100.0
                trades += 1
                wins += int(trade_pnl > 0)
                position = None
                entry_i = -1

        # Make current row's target available only after it has been evaluated.
        if valid:
            _, _, _, _, _, nc = rs[i + 1]
            move_ratio = abs(nc - c) / body
            if move_ratio == move_ratio:
                state_hist[s].append(move_ratio)
                global_hist.append(move_ratio)

        mtm = equity
        if position is not None:
            ps, entry, _, _ = position
            mtm += CAPITAL * pnl_pct(ps, entry, c) / 100.0
        peak = max(peak, mtm)
        max_dd = max(max_dd, (peak - mtm) / peak if peak else 0.0)

    if position is not None:
        ps, entry, _, _ = position
        ret = pnl_pct(ps, entry, rs[-1][5])
        equity += CAPITAL * ret / 100.0
        trades += 1
        wins += int(ret > 0)

    return {"pnl": (equity - CAPITAL) / CAPITAL * 100.0, "trades": trades,
            "wins": wins, "dd": max_dd * 100.0}


def compound(xs):
    v = 1.0
    for x in xs:
        v *= 1.0 + x / 100.0
    return (v - 1.0) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("reports/prepared_price_action_1h.csv"))
    args = ap.parse_args()
    print("EXCEL_TPSL_ABLATION | tf=1h | fee=0 | exact S0-S3 | fixed trade capital=$1000 | close_only | past_only")
    print("state=TP from S-state median | global=TP from one global median | fixed2=TP=2xbody | all SL=0.5xTP")

    for sym in SYMBOLS:
        rows = load_rows(args.input, sym)
        print(f"\n{sym}")
        results = {}
        for mode in ("fixed2", "global", "state"):
            monthly = []
            tt = tw = 0
            dd = 0.0
            for month in MONTHS:
                r = run_month(rows, month, mode)
                monthly.append(r["pnl"])
                tt += r["trades"]
                tw += r["wins"]
                dd = max(dd, r["dd"])
            results[mode] = (monthly, tt, tw, dd)
            label = {"fixed2": "FIXED2", "global": "GLOBAL", "state": "STATE"}[mode]
            print(f"{label} 4M={sum(monthly):+.2f}% compound={compound(monthly):+.2f}% trades={tt} win={(100*tw/tt if tt else 0):.1f}% DD={dd:.2f}%")
        state = results["state"][0]
        global_ = results["global"][0]
        fixed = results["fixed2"][0]
        print(f"DELTA STATE-GLOBAL(sum)={sum(state)-sum(global_):+.2f}pp STATE-FIXED2(sum)={sum(state)-sum(fixed):+.2f}pp")


if __name__ == "__main__":
    main()
