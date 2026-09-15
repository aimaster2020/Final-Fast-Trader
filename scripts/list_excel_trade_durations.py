from __future__ import annotations

import argparse
import csv
from pathlib import Path

RAW_FIELDS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]


def num(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sign_asymmetric(value: float, up_threshold: float, down_threshold: float) -> int:
    if value > up_threshold:
        return 1
    if value < -down_threshold:
        return -1
    return 0


def looks_like_header(row: list[str]) -> bool:
    normalized = [str(x).strip().lower() for x in row]
    return {"timestamp", "open", "high", "low", "close"}.issubset(normalized)


def is_year(value: str, year: int) -> bool:
    return str(value).strip().startswith(str(year))


def load_rows(path: Path, year: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return []

        if looks_like_header(first):
            fields = [str(x).strip() for x in first]
            raw_rows = [
                {k: raw[idx] if idx < len(raw) else "" for idx, k in enumerate(fields)}
                for raw in reader
            ]
        else:
            if len(first) < 5:
                raise ValueError("Expected headerless order: Timestamp,Open,High,Low,Close[,Volume].")
            raw_rows = [
                {k: raw[idx] if idx < len(raw) else "" for idx, k in enumerate(RAW_FIELDS)}
                for raw in [first] + list(reader)
            ]

    rows = [row for row in raw_rows if is_year(row.get("Timestamp", ""), year)]
    rows.sort(key=lambda row: row.get("Timestamp", ""))
    return rows


def calculate_pt(rows, ct_up: float, ct_down: float):
    result = []
    for i in range(len(rows) - 1):
        current = rows[i]
        nxt = rows[i + 1]
        o = num(current.get("Open"))
        h = num(current.get("High"))
        c = num(current.get("Close"))
        next_c = num(nxt.get("Close"))
        pt = None
        if None not in (o, h, c, next_c) and i >= 5:
            window = rows[i - 5 : i + 1]
            closes = [num(x.get("Close")) for x in window]
            highs = [num(x.get("High")) for x in window]
            if all(v is not None for v in closes + highs):
                body = c - o
                if body > 0:
                    predict = sum(closes) / 6.0
                elif body < 0:
                    predict = sum(highs) / 6.0
                else:
                    predict = c
                pt = sign_asymmetric(predict - c, ct_up, ct_down)

        result.append({
            "index": i,
            "timestamp": current.get("Timestamp", ""),
            "close": c,
            "pt": pt,
        })
    return result


def extract_trades(pt_rows):
    trades = []
    position = 0
    entry = None
    trade_id = 0

    for row in pt_rows:
        pt = row["pt"]
        desired = pt if pt in (-1, 1) else 0

        if position == 0:
            if desired != 0:
                position = desired
                entry = row
                trade_id += 1
        elif position == 1:
            if desired == -1:
                trades.append(make_trade(trade_id, position, entry, row))
                position = -1
                entry = row
                trade_id += 1
        elif position == -1:
            if desired == 1:
                trades.append(make_trade(trade_id, position, entry, row))
                position = 1
                entry = row
                trade_id += 1

    open_trade = None
    if position != 0 and entry is not None:
        open_trade = make_trade(trade_id, position, entry, None)
    return trades, open_trade


def make_trade(trade_id, direction, entry, exit_row):
    entry_index = entry["index"]
    end_index = exit_row["index"] if exit_row is not None else None
    duration = (end_index - entry_index) if end_index is not None else None
    entry_close = entry["close"]
    exit_close = exit_row["close"] if exit_row is not None else None

    gross_return = None
    if exit_close is not None and entry_close:
        if direction == 1:
            gross_return = (exit_close - entry_close) / entry_close
        else:
            gross_return = (entry_close - exit_close) / exit_close if exit_close else 0.0

    return {
        "trade_id": trade_id,
        "direction": "LONG" if direction == 1 else "SHORT",
        "entry_timestamp": entry["timestamp"],
        "exit_timestamp": exit_row["timestamp"] if exit_row else "OPEN",
        "entry_index": entry_index,
        "exit_index": end_index,
        "duration_candles": duration,
        "entry_close": entry_close,
        "exit_close": exit_close,
        "gross_return": gross_return,
        "status": "CLOSED" if exit_row else "OPEN",
    }


def main():
    p = argparse.ArgumentParser(description="List actual trade lifetimes for the exact Excel formula")
    p.add_argument("--input", required=True)
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--ct-up", type=float, default=600)
    p.add_argument("--ct-down", type=float, default=800)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    rows = load_rows(Path(args.input), args.year)
    pt_rows = calculate_pt(rows, args.ct_up, args.ct_down)
    trades, open_trade = extract_trades(pt_rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["trade_id", "direction", "entry_timestamp", "exit_timestamp", "entry_index", "exit_index", "duration_candles", "entry_close", "exit_close", "gross_return", "status"]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trades)
        if open_trade:
            writer.writerow(open_trade)

    print("=" * 100)
    print("EXACT EXCEL FORMULA / TRADE DURATION LIST")
    print("=" * 100)
    print(f"YEAR={args.year} CT_UP={args.ct_up:g} CT_DOWN={args.ct_down:g}")
    print(f"input_rows={len(rows)}")
    print(f"completed_trades={len(trades)}")
    print(f"open_trade={'YES' if open_trade else 'NO'}")
    print()
    print("TRADE LIST")
    print("id,direction,entry,exit,duration_candles,status")
    for t in trades:
        print(f"{t['trade_id']},{t['direction']},{t['entry_timestamp']},{t['exit_timestamp']},{t['duration_candles']},CLOSED")
    if open_trade:
        print(f"{open_trade['trade_id']},{open_trade['direction']},{open_trade['entry_timestamp']},OPEN,{open_trade['duration_candles']},OPEN")

    durations = [t["duration_candles"] for t in trades if t["duration_candles"] is not None]
    if durations:
        print()
        print("DURATION SUMMARY")
        print(f"min={min(durations)}")
        print(f"max={max(durations)}")
        print(f"avg={sum(durations) / len(durations):.2f}")
        print(f"median={sorted(durations)[len(durations)//2]:.2f}")
        print(f"one_candle={sum(d == 1 for d in durations)}")
        print(f"2_to_3={sum(2 <= d <= 3 for d in durations)}")
        print(f"4_to_6={sum(4 <= d <= 6 for d in durations)}")
        print(f"7_to_12={sum(7 <= d <= 12 for d in durations)}")
        print(f"13_plus={sum(d >= 13 for d in durations)}")
    print(f"output={out}")


if __name__ == "__main__":
    main()
