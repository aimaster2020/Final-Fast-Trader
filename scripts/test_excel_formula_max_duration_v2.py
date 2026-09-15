from __future__ import annotations

import argparse
import csv
from pathlib import Path

FIELDS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def sign(v, up, down):
    if v > up:
        return 1
    if v < -down:
        return -1
    return 0


def load(path, year):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rd = csv.reader(fh)
        first = next(rd, None)
        if first is None:
            return []
        if {str(x).strip().lower() for x in first} >= {"timestamp", "open", "high", "low", "close"}:
            headers = [str(x).strip() for x in first]
            raw = [{k: row[i] if i < len(row) else "" for i, k in enumerate(headers)} for row in rd]
        else:
            raw = [{k: row[i] if i < len(row) else "" for i, k in enumerate(FIELDS)} for row in [first] + list(rd)]
    rows = [r for r in raw if str(r.get("Timestamp", "")).strip().startswith(str(year))]
    rows.sort(key=lambda r: r.get("Timestamp", ""))
    return rows


def build_pt(rows, up, down):
    out = []
    for i in range(len(rows) - 1):
        r = rows[i]
        o, c = f(r.get("Open")), f(r.get("Close"))
        pt = None
        if i >= 5 and o is not None and c is not None:
            w = rows[i - 5:i + 1]
            closes = [f(x.get("Close")) for x in w]
            highs = [f(x.get("High")) for x in w]
            if all(x is not None for x in closes + highs):
                p = sum(closes) / 6.0 if c > o else sum(highs) / 6.0 if c < o else c
                pt = sign(p - c, up, down)
        out.append({"timestamp": r.get("Timestamp", ""), "close": c, "pt": pt})
    return out


def backtest(rows, max_duration, fee):
    trades = []
    pos = 0
    entry = None
    entry_i = None
    forced = 0
    entries = exits = 0

    for i, r in enumerate(rows):
        if r["pt"] is None or r["close"] is None:
            continue
        pt = r["pt"]

        if pos and i - entry_i >= max_duration:
            exit_price = r["close"]
            gross = (exit_price - entry) / entry if pos == 1 else (entry - exit_price) / entry
            net = gross - 2 * fee
            trades.append({"direction": "LONG" if pos == 1 else "SHORT", "bars": i - entry_i, "gross": gross, "net": net, "win": net > 0, "reason": "MAX_DURATION"})
            exits += 1
            forced += 1
            pos = 0
            entry = None
            entry_i = None

        if pos == 0:
            if pt in (-1, 1):
                pos = pt
                entry = r["close"]
                entry_i = i
                entries += 1
            continue

        if pt in (0, pos):
            continue

        exit_price = r["close"]
        gross = (exit_price - entry) / entry if pos == 1 else (entry - exit_price) / entry
        net = gross - 2 * fee
        trades.append({"direction": "LONG" if pos == 1 else "SHORT", "bars": i - entry_i, "gross": gross, "net": net, "win": net > 0, "reason": "SIGNAL_REVERSE"})
        exits += 1
        pos = pt
        entry = r["close"]
        entry_i = i
        entries += 1

    open_trade = None
    if pos and entry is not None:
        last = rows[-1]
        last_price = last["close"]
        unreal = (last_price - entry) / entry if pos == 1 else (entry - last_price) / entry
        open_trade = {"direction": "LONG" if pos == 1 else "SHORT", "gross": unreal, "net_after_entry_fee": unreal - fee, "bars": len(rows) - 1 - entry_i}

    return trades, open_trade, entries, exits, forced


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--ct-up", type=float, default=600)
    p.add_argument("--ct-down", type=float, default=800)
    p.add_argument("--max-duration", type=int, default=6)
    p.add_argument("--commission", type=float, default=0.0013)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    source = load(Path(a.input), a.year)
    pt = build_pt(source, a.ct_up, a.ct_down)
    trades, open_trade, entries, exits, forced = backtest(pt, a.max_duration, a.commission)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["trade_id", "direction", "bars", "gross", "net", "win", "reason"]
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i, t in enumerate(trades, 1):
            w.writerow({"trade_id": i, **t})

    print("=" * 100)
    print("EXACT EXCEL FORMULA / MAX DURATION BACKTEST")
    print("=" * 100)
    print(f"YEAR={a.year} CT_UP={a.ct_up:g} CT_DOWN={a.ct_down:g} MAX_DURATION={a.max_duration}")
    print(f"COMMISSION_PER_SIDE={a.commission * 100:.4f}%")
    print(f"input_rows={len(source)} frames={len(pt)}")
    print()
    print(f"entries={entries} exits={exits} forced_max_duration={forced}")
    print(f"commission_total={(entries + exits) * a.commission:.10f}")
    print()
    total_gross = sum(t["gross"] for t in trades)
    total_net = sum(t["net"] for t in trades)
    for side in ("LONG", "SHORT"):
        xs = [t for t in trades if t["direction"] == side]
        wins = sum(t["win"] for t in xs)
        print(f"{side} trades={len(xs)} wins={wins} losses={len(xs)-wins}")
        print(f"{side} win_rate={(100*wins/len(xs)) if xs else 0:.3f}% avg_net={(sum(t['net'] for t in xs)/len(xs)) if xs else 0:.10f} sum_gross={sum(t['gross'] for t in xs):.10f} sum_net={sum(t['net'] for t in xs):.10f}")
    print()
    print(f"TOTAL completed_trades={len(trades)} wins={sum(t['win'] for t in trades)} losses={sum(not t['win'] for t in trades)}")
    print(f"gross_return_sum={total_gross:.10f}")
    print(f"commission_total={(entries + exits) * a.commission:.10f}")
    print(f"net_return_sum={total_net:.10f}")
    if open_trade:
        print(f"OPEN_TRADE={open_trade['direction']} gross_unrealized={open_trade['gross']:.10f} net_after_entry_fee={open_trade['net_after_entry_fee']:.10f} bars={open_trade['bars']}")
    else:
        print("OPEN_TRADE=none")
    durations = [t["bars"] for t in trades]
    print()
    print(f"duration_min={min(durations) if durations else 0} duration_max={max(durations) if durations else 0} duration_avg={(sum(durations)/len(durations)) if durations else 0:.2f}")
    print(f"output={out}")


if __name__ == "__main__":
    main()
