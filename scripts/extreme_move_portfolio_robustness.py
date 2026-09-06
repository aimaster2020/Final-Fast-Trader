from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepared_market_loader import load_prepared


@dataclass
class Position:
    symbol: str
    signal: str
    entry_ts: int
    exit_ts: int
    entry: float
    exit: float
    notional: float
    fee: float

    @property
    def side(self) -> int:
        return 1 if self.signal == "LOW" else -1

    @property
    def gross_ret(self) -> float:
        return ((self.exit / self.entry) - 1.0) * self.side

    @property
    def pnl(self) -> float:
        return self.notional * self.gross_ret - self.fee


def movement_budget(rows: list, i: int, bars: int) -> float:
    if bars <= 0 or i < bars:
        return float("inf")
    return sum(abs(rows[j].close - rows[j - 1].close) for j in range(i - bars, i))


def extreme_signal(rows: list, i: int, window: int, move_bars: int, multiplier: float) -> str:
    if i < max(window, move_bars):
        return ""
    previous = rows[i - window:i]
    prev_low = min(r.low for r in previous)
    prev_high = max(r.high for r in previous)
    budget = movement_budget(rows, i, move_bars) * multiplier
    low = rows[i].low < prev_low and (prev_low - rows[i].low) >= budget
    high = rows[i].high > prev_high and (rows[i].high - prev_high) >= budget
    if low and high:
        return ""
    return "LOW" if low else "HIGH" if high else ""


def build_candidates(symbol: str, rows: list, window: int, move_bars: int, multiplier: float, horizon: int, side: str) -> list[dict]:
    out = []
    start = max(window, move_bars)
    for i in range(start, len(rows) - horizon - 1):
        sig = extreme_signal(rows, i, window, move_bars, multiplier)
        if sig not in ("LOW", "HIGH"):
            continue
        if side != "ALL" and sig != side:
            continue
        entry_i = i + 1
        exit_i = i + 1 + horizon
        out.append({
            "symbol": symbol,
            "signal": sig,
            "entry_ts": rows[entry_i].timestamp,
            "exit_ts": rows[exit_i].timestamp,
            "entry": rows[entry_i].open,
            "exit": rows[exit_i].close,
        })
    return out


def run(candidates: list[dict], split_ts: int, initial: float, allocation: float, fee_pct: float, max_positions: int):
    candidates.sort(key=lambda x: (x["entry_ts"], x["symbol"]))
    capital = initial
    peak = initial
    dd = 0.0
    active: dict[str, Position] = {}
    closed: list[Position] = []

    def close(pos: Position) -> None:
        nonlocal capital, peak, dd
        capital += pos.pnl
        peak = max(peak, capital)
        dd = max(dd, (peak - capital) / peak * 100.0 if peak else 0.0)
        closed.append(pos)

    for c in candidates:
        for symbol, pos in list(active.items()):
            if pos.exit_ts <= c["entry_ts"]:
                close(pos)
                del active[symbol]
        if c["symbol"] in active or len(active) >= max_positions:
            continue
        notional = capital * allocation
        fee = notional * fee_pct / 100.0
        active[c["symbol"]] = Position(c["symbol"], c["signal"], c["entry_ts"], c["exit_ts"], c["entry"], c["exit"], notional, fee)

    for pos in sorted(active.values(), key=lambda p: p.exit_ts):
        close(pos)

    first = [p for p in closed if p.exit_ts < split_ts]
    second = [p for p in closed if p.exit_ts >= split_ts]

    def metrics(items):
        wins = [p for p in items if p.pnl > 0]
        losses = [p for p in items if p.pnl < 0]
        gp = sum(p.pnl for p in wins)
        gl = -sum(p.pnl for p in losses)
        pf = gp / gl if gl else (float("inf") if gp else 0.0)
        return len(items), len(wins) / len(items) * 100 if items else 0.0, sum(p.pnl for p in items), pf

    return {
        "total": metrics(closed),
        "first": metrics(first),
        "second": metrics(second),
        "final": capital,
        "ret": (capital / initial - 1.0) * 100.0 if initial else 0.0,
        "dd": dd,
        "fees": sum(p.fee for p in closed),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Continuous single-account robustness sweep for the extreme move strategy.")
    ap.add_argument("--prepared-file", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--months", default="2026-05,2026-06,2026-07,2026-08")
    ap.add_argument("--windows", default="10,15,20")
    ap.add_argument("--move-bars", default="4,5,6")
    ap.add_argument("--multipliers", default="1.5,2.0,2.5")
    ap.add_argument("--horizons", default="3,6,12")
    ap.add_argument("--side", choices=("ALL", "LOW", "HIGH"), default="LOW")
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--allocation", type=float, default=0.30)
    ap.add_argument("--max-positions", type=int, default=3)
    ap.add_argument("--fee-roundtrip-pct", type=float, default=0.26)
    ap.add_argument("--output", default="reports/extreme_move_portfolio_robustness.csv")
    args = ap.parse_args()

    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    months = [x.strip() for x in args.months.split(",") if x.strip()]
    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    move_bars = [int(x) for x in args.move_bars.split(",") if x.strip()]
    multipliers = [float(x) for x in args.multipliers.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    prepared = load_prepared(Path(args.prepared_file))
    split_month = months[len(months) // 2]
    split_ts = min((rows[0].timestamp for (symbol, month), rows in prepared.items() if symbol in symbols and month == split_month and rows), default=0)

    grouped = {}
    for symbol in symbols:
        rows = []
        for month in months:
            rows.extend(prepared.get((symbol, month), []))
        rows.sort(key=lambda c: c.timestamp)
        grouped[symbol] = rows

    print(f"EXTREME_MOVE_PORTFOLIO_ROBUSTNESS SIDE={args.side} alloc={args.allocation:.0%} maxpos={args.max_positions} fee={args.fee_roundtrip_pct:.2f}%")
    print("CONTINUOUS MONTHS | WINDOW USES PRIOR MONTH HISTORY | SPLIT=JUL 2026")
    print("W M X H T TOTAL_PNL FIRST_PNL SECOND_PNL SECOND_T SECOND_WIN RET PF DD")
    print("----------------------------------------------------------------------------")

    results = []
    for w in windows:
        for m in move_bars:
            for x in multipliers:
                for h in horizons:
                    candidates = []
                    for symbol in symbols:
                        candidates.extend(build_candidates(symbol, grouped[symbol], w, m, x, h, args.side))
                    r = run(candidates, split_ts, args.initial_capital, args.allocation, args.fee_roundtrip_pct, args.max_positions)
                    results.append((w, m, x, h, r))

    eligible = [z for z in results if z[4]["second"][0] >= 5]
    eligible.sort(key=lambda z: (z[4]["second"][2], z[4]["second"][3], z[4]["second"][0]), reverse=True)
    for w, m, x, h, r in eligible[:20]:
        t, win, pnl, pf = r["total"]
        ft, _, fpnl, _ = r["first"]
        st, sw, spnl, _ = r["second"]
        print(f"W={w:2d} M={m:2d} X={x:.2f} H={h:2d} {t:3d} {pnl:+9.2f} {fpnl:+9.2f} {spnl:+10.2f} {st:3d} {sw:6.1f}% {r['ret']:+5.2f}% PF={pf:.2f} DD={r['dd']:.2f}%")

    print("CANONICAL W=15 M=5 X=2")
    for h in horizons:
        r = next(rr for w,m,x,hh,rr in results if (w,m,x,hh) == (15,5,2.0,h))
        t, win, pnl, pf = r["total"]
        ft, fw, fpnl, fpf = r["first"]
        st, sw, spnl, spf = r["second"]
        print(f"H={h:2d} T={t:3d} WIN={win:5.1f}% PNL={pnl:+7.2f} FIRST={fpnl:+7.2f} SECOND={spnl:+7.2f} SECOND_T={st:3d} SECOND_WIN={sw:5.1f}% PF={pf:.2f} DD={r['dd']:.2f}%")

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["window","move_bars","multiplier","horizon","side","total_trades","total_win_pct","total_pnl","first_trades","first_win_pct","first_pnl","second_trades","second_win_pct","second_pnl","total_return_pct","total_pf","max_dd_pct","fees"])
        for w,m,x,h,r in results:
            t,win,pnl,pf = r["total"]
            ft,fw,fpnl,fpf = r["first"]
            st,sw,spnl,spf = r["second"]
            wri.writerow([w,m,x,h,args.side,t,win,pnl,ft,fw,fpnl,st,sw,spnl,r["ret"],pf,r["dd"],r["fees"]])
    print(f"SAVED {out.relative_to(ROOT)} rows={len(results)}")


if __name__ == "__main__":
    main()
