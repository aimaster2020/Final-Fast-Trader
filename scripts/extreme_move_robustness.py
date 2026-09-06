from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Result:
    window: int
    move_bars: int
    multiplier: float
    horizon: int
    split: str
    side: str
    symbol: str
    trades: int
    wins: int
    gross_pnl: float
    fees: float
    net_pnl: float


def budget(rows: list[dict], i: int, bars: int) -> float:
    if bars <= 0 or i < bars:
        return float("inf")
    return sum(abs(float(rows[j]["close"]) - float(rows[j - 1]["close"])) for j in range(i - bars, i))


def signal(rows: list[dict], i: int, window: int, move_bars: int, multiplier: float) -> str:
    if i < max(window, move_bars):
        return ""
    prev = rows[i - window:i]
    prev_low = min(float(r["low"]) for r in prev)
    prev_high = max(float(r["high"]) for r in prev)
    b = budget(rows, i, move_bars) * multiplier
    low = prev_low - float(rows[i]["low"]) >= b and float(rows[i]["low"]) < prev_low
    high = float(rows[i]["high"]) - prev_high >= b and float(rows[i]["high"]) > prev_high
    if low and high:
        return ""
    return "LOW" if low else "HIGH" if high else ""


def evaluate(rows: list[dict], window: int, move_bars: int, multiplier: float, horizon: int, side_filter: str = "ALL"):
    out = []
    start = max(window, move_bars)
    for i in range(start, len(rows) - horizon - 1):
        sig = signal(rows, i, window, move_bars, multiplier)
        if sig not in ("LOW", "HIGH"):
            continue
        if side_filter != "ALL" and sig != side_filter:
            continue
        entry = float(rows[i + 1]["open"])
        exit_price = float(rows[i + 1 + horizon]["close"])
        side = 1 if sig == "LOW" else -1
        ret = (exit_price / entry - 1.0) * side
        out.append((sig, ret))
    return out


def summarize(items, fee_rt: float, allocation: float):
    trades = len(items)
    wins = sum(r > 0 for _, r in items)
    gross = sum(r for _, r in items)
    fees = trades * allocation * fee_rt / 100.0
    net = allocation * gross - fees
    return trades, wins, gross, fees, net


def main():
    ap = argparse.ArgumentParser(description="Robustness/side-split validation for extreme move signal.")
    ap.add_argument("--prepared-file", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--months", default="2026-05,2026-06,2026-07,2026-08")
    ap.add_argument("--windows", default="10,15,20")
    ap.add_argument("--move-bars", default="4,5,6")
    ap.add_argument("--multipliers", default="1.5,2.0,2.5")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--allocation", type=float, default=0.30)
    ap.add_argument("--fee-roundtrip-pct", type=float, default=0.26)
    ap.add_argument("--output", default="reports/extreme_move_robustness.csv")
    args = ap.parse_args()

    path = Path(args.prepared_file)
    if not path.is_absolute():
        path = ROOT / path
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    months = [x.strip() for x in args.months.split(",") if x.strip()]
    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    move_bars = [int(x) for x in args.move_bars.split(",") if x.strip()]
    multipliers = [float(x) for x in args.multipliers.split(",") if x.strip()]

    grouped = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["symbol"].upper() in symbols and r["month"] in months:
                grouped[r["symbol"].upper()].append(r)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["timestamp"]))

    mid = len(months) // 2
    first = set(months[:mid])
    second = set(months[mid:])
    results = []

    for w in windows:
        for m in move_bars:
            for x in multipliers:
                for split_name, split_months in (("ALL", set(months)), ("FIRST", first), ("SECOND", second)):
                    for side in ("ALL", "LOW", "HIGH"):
                        items = []
                        for symbol in symbols:
                            rows = [r for r in grouped[symbol] if r["month"] in split_months]
                            ev = evaluate(rows, w, m, x, args.horizon, side)
                            items.extend(ev)
                            t, wins, gross, fees, net = summarize(ev, args.fee_roundtrip_pct, args.allocation)
                            results.append(Result(w, m, x, args.horizon, split_name, side, symbol, t, wins, gross, fees, net))

    print(f"EXTREME_MOVE_ROBUSTNESS H={args.horizon} fee={args.fee_roundtrip_pct:.2f}% alloc={args.allocation:.0%}")
    print("W M X SPLIT SIDE SYMBOL T WIN NET")
    print("-------------------------------------------------------")
    for r in results:
        if r.split == "ALL" and r.side in ("ALL", "LOW", "HIGH") and r.symbol == symbols[0]:
            pass

    # Compact aggregate report for parameter sets.
    aggs = []
    for w in windows:
        for m in move_bars:
            for x in multipliers:
                for split_name in ("ALL", "FIRST", "SECOND"):
                    for side in ("ALL", "LOW", "HIGH"):
                        rs = [r for r in results if r.window == w and r.move_bars == m and r.multiplier == x and r.split == split_name and r.side == side]
                        t = sum(r.trades for r in rs)
                        wins = sum(r.wins for r in rs)
                        net = sum(r.net_pnl for r in rs)
                        aggs.append((w, m, x, split_name, side, t, wins / t * 100 if t else 0.0, net))

    # Rank by second-half net with a minimum sample size.
    ranked = [a for a in aggs if a[3] == "SECOND" and a[4] == "ALL" and a[5] >= 10]
    ranked.sort(key=lambda a: a[7], reverse=True)
    print("TOP SECOND-HALF")
    for a in ranked[:15]:
        w,m,x,split,side,t,win,net = a
        first_match = next(v for v in aggs if v[:5] == (w,m,x,"FIRST","ALL"))
        print(f"W={w:2d} M={m:2d} X={x:.2f} FIRST_T={first_match[5]:3d} FIRST_NET={first_match[7]:+7.2f} SECOND_T={t:3d} SECOND_NET={net:+7.2f} WIN={win:5.1f}%")

    # Side split for canonical candidate.
    print("CANONICAL W=15 M=5 X=2")
    for split in ("ALL", "FIRST", "SECOND"):
        for side in ("LOW", "HIGH"):
            a = next(v for v in aggs if v[:5] == (15,5,2.0,split,side))
            print(f"{split:6} {side:4} T={a[5]:3d} WIN={a[6]:5.1f}% NET={a[7]:+7.2f}")

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["window","move_bars","multiplier","horizon","split","side","symbol","trades","wins","win_rate_pct","gross_return_sum","fees","net_pnl_on_1000_alloc"])
        for r in results:
            wri.writerow([r.window,r.move_bars,r.multiplier,r.horizon,r.split,r.side,r.symbol,r.trades,r.wins,(r.wins/r.trades*100 if r.trades else 0),r.gross_pnl,r.fees,r.net_pnl])
    print(f"SAVED {out.relative_to(ROOT)} rows={len(results)}")


if __name__ == "__main__":
    main()
