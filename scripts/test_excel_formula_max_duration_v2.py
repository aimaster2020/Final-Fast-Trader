from __future__ import annotations

import argparse
import csv
from pathlib import Path

RAW_FIELDS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sign_asymmetric(value: float, up: float, down: float) -> int:
    if value > up:
        return 1
    if value < -down:
        return -1
    return 0


def looks_like_header(row):
    s = {str(x).strip().lower() for x in row}
    return {"timestamp", "open", "high", "low", "close"}.issubset(s)


def load_rows(path: Path, year: int):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return []
        if looks_like_header(first):
            fields = [str(x).strip() for x in first]
            raw = [{k: row[i] if i < len(row) else "" for i, k in enumerate(fields)} for row in reader]
        else:
            if len(first) < 5:
                raise ValueError("Expected Timestamp,Open,High,Low,Close[,Volume]")
            raw = [{k: row[i] if i < len(row) else "" for i, k in enumerate(RAW_FIELDS)} for row in [first] + list(reader)]
    rows = [r for r in raw if str(r.get("Timestamp", "")).strip().startswith(str(year))]
    rows.sort(key=lambda r: r.get("Timestamp", ""))
    return rows


def make_pt(rows, up, down):
    out = []
    for i in range(len(rows) - 1):
        r = rows[i]
        o, c = num(r.get("Open")), num(r.get("Close"))
        pt = None
        if i >= 5 and o is not None and c is not None:
            window = rows[i - 5:i + 1]
            closes = [num(x.get("Close")) for x in window]
            highs = [num(x.get("High")) for x in window]
            if all(v is not None for v in closes + highs):
                if c > o:
                    predict = sum(closes) / 6.0
                elif c < o:
                    predict = sum(highs) / 6.0
                else:
                    predict = c
                pt = sign_asymmetric(predict - c, up, down)
        out.append({"timestamp": r.get("Timestamp", ""), "close": c, "pt": pt})
    return out


def backtest(pt_rows, max_duration, commission):
    completed = []
    position = 0
    entry_price = None
    entry_time = None
    entry_bar = None
    forced = 0
    entries = exits = 0
    long_entries = short_entries = long_exits = short_exits = 0

    def close_trade(row, bar, reason):
        nonlocal position, entry_price, entry_time, entry_bar, exits, long_exits, short_exits, forced
        exit_price = row["close"]
        direction = "LONG" if position == 1 else "SHORT"
        if position == 1:
            gross = (exit_price - entry_price) / entry_price
            long_exits += 1
        else:
            gross = (entry_price - exit_price) / entry_price
            short_exits += 1
        net = gross - 2 * commission
        bars = bar - entry_bar
        completed.append({
            "direction": direction,
            "gross": gross,
            "net": net,
            "bars": bars,
            "reason": reason,
            "gross_return": gross,
            "net_return": net,
            "win": net > 0,
        })
        exits += 1
        if reason == "MAX_DURATION":
            forced += 1
        position = 0
        entry_price = entry_time = entry_bar = None

    for bar, row in enumerate(pt_rows):
        if row["pt"] is None or row["close"] is None:
            continue

        if position != 0 and bar - entry_bar >= max_duration:
            close_trade(row, bar, "MAX_DURATION")

        if position == 0:
            pt = row["pt"]
            if pt in (-1, 1):
                position = pt
                entry_price = row["close"]
                entry_time = row["timestamp"]
                entry_bar = bar
                entries += 1
                if position == 1:
                    long_entries += 1
                else:
                    short_entries += 1
            continue

        pt = row["pt"]
        if pt == 0 or pt == position:
            continue
        close_trade(row, bar, "SIGNAL_REVERSE")
        position = pt
        entry_price = row["close"]
        entry_time = row["timestamp"]
        entry_bar = bar
        entries += 1
        if position == 1:
            long_entries += 1
        else:
            short_entries += 1

    open_trade = None
    if position != 0 and entry_price is not None:
        last = pt_rows[-1]
        if position == 1:
            unrealized = (last["close"] - entry_price) / entry_price
            direction = "LONG"
        else:
            unrealized = (entry_price - last["close"]) / entry_price
            direction = "SHORT"
        open_trade = {
            "direction": direction,
            "gross": unrealized,
            "net_after_entry_fee": unrealized - commission,
            "bars": len(pt_rows) - 1 - entry_bar,
        }

    return completed, open_trade, {
        "entries": entries,
        "exits": exits,
        "long_entries": long_entries,
        "short_entries": short_entries,
        "long_exits": long_exits,
        "short_exits": short_exits,
        "forced": forced,
    }


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

    rows = load_rows(Path(a.input), a.year)
    pt_rows = make_pt(rows, a.ct_up, a.ct_down)
    trades, open_trade, stats = backtest(pt_rows, a.max_duration, a.commission)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["trade_id", "direction", "bars", "gross", "net", "gross_return", "net_return", "win", "reason"]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, t in enumerate(trades, 1):
            w.writerow({"trade_id": i, **t})

    long_t = [t for t in trades if t["direction"] == "LONG"]
    short_t = [t for t in trades if t["direction"] == "SHORT"]
    total_commission = (stats["entries"] + stats["exits"]) * a.commission
    gross = sum(t["gross"] for t in trades)
    net = sum(t["net"] for t in trades)

    print("=" * 100)
    print("EXACT EXCEL FORMULA / MAX DURATION BACKTEST")
    print("=" * 100)
    print(f"YEAR={a.year} CT_UP={a.ct_up:g} CT_DOWN={a.ct_down:g} MAX_DURATION={a.max_duration}")
    print(f"COMMISSION_PER_SIDE={a.commission * 100:.4f}%")
    print(f"input_rows={len(rows)} frames={len(pt_rows)}")
    print()
    print("POSITION STATS")
    print(f"entries={stats['entries']} exits={stats['exits']} forced_max_duration={stats['forced']}")
    print(f"long_entries={stats['long_entries']} short_entries={stats['short_entries']}")
    print(f"long_exits={stats['long_exits']} short_exits={stats['short_exits']}")
    print(f"commission_total={total_commission:.10f}")
    print()
    for name, side in (("LONG", long_t), ("SHORT", short_t)):
        wins = [t for t in side if t["win"]]
        losses = [t for t in side if not t["win"]]
        print(f"{name} trades={len(side)} wins={len(wins)} losses={len(losses)}")
        print(f"{name} win_rate={(len(wins)/len(side)*100) if side else 0:.3f}% avg_net={(sum(t['net'] for t in side)/len(side)) if side else 0:.10f} sum_gross={sum(t['gross'] for t in side):.10f} sum_net={sum(t['net'] for t in side):.10f}")
    print()
    print("TOTAL")
    print(f"completed_trades={len(trades)} wins={sum(t['win'] for t in trades)} losses={sum(not t['win'] for t in trades)}")
    print(f"gross_return_sum={gross:.10f}")
    print(f"commission_total={total_commission:.10f}")
    print(f"net_return_sum={net:.10f}")
    if open_trade:
        print(f"OPEN_TRADE={open_trade['direction']} gross_unrealized={open_trade['gross']:.10f} net_after_entry_fee={open_trade['net_after_entry_fee']:.10f} bars={open_trade['bars']}")
    else:
        print("OPEN_TRADE=none")
    print()
    durations = [t["bars"] for t in trades]
    print("DURATION")
    print(f"min={min(durations) if durations else 0} max={max(durations) if durations else 0} avg={(sum(durations)/len(durations)) if durations else 0:.2f}")
    print(f"max_duration_forced={stats['forced']}")
    print(f"output={out}")


if __name__ == "__main__":
    main()
