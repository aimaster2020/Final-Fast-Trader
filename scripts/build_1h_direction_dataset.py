from __future__ import annotations

import argparse
import csv
import gzip
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATE_FMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"


def parse_rows(reader):
    out = []
    for r in reader:
        if len(r) < 6:
            continue
        try:
            ts = int(float(r[0]))
            if ts > 10_000_000_000_000:
                ts //= 1_000_000
            elif ts > 10_000_000_000:
                ts //= 1_000
            out.append((ts, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
        except (ValueError, TypeError):
            continue
    return out


def load_binance_archive(path: Path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.endswith("/")]
            if not names:
                return []
            with z.open(names[0]) as raw:
                return parse_rows(csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")))
    if path.name.lower().endswith(".csv.gz"):
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as f:
            return parse_rows(csv.reader(f))
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return parse_rows(csv.reader(f))


def candidate_files(input_dir: Path, symbol: str):
    su = symbol.upper()
    return sorted(
        [
            p for p in input_dir.rglob("*")
            if p.is_file()
            and su in p.name.upper()
            and (p.suffix.lower() == ".zip" or p.suffix.lower() == ".csv" or p.name.lower().endswith(".csv.gz"))
        ],
        key=lambda p: str(p).upper(),
    )


def load_range(input_dir: Path, symbol: str, start_ts: int, end_ts: int):
    merged = {}
    used = []
    for path in candidate_files(input_dir, symbol):
        try:
            raw = load_binance_archive(path)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            print(f"SKIP {symbol} {path.name}: {exc}")
            continue
        matched = [r for r in raw if start_ts <= r[0] < end_ts]
        if matched:
            used.append(path.name)
            for r in matched:
                merged[r[0]] = r
    return [merged[k] for k in sorted(merged)], used


def resample_1h(rows):
    buckets = {}
    for ts, o, h, l, c, v in rows:
        hour = (ts // 3600) * 3600
        buckets.setdefault(hour, []).append((ts, o, h, l, c, v))
    out = []
    for hour in sorted(buckets):
        x = sorted(buckets[hour])
        out.append({
            "timestamp": hour,
            "open": x[0][1],
            "high": max(r[2] for r in x),
            "low": min(r[3] for r in x),
            "close": x[-1][4],
            "volume": sum(r[5] for r in x),
        })
    return out


def direction(o: float, c: float) -> str:
    return "UP" if c > o else "DOWN" if c < o else "FLAT"


def code(d: str) -> str:
    return "U" if d == "UP" else "D" if d == "DOWN" else "F"


def fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime(DATE_FMT)


def build_rows(symbol: str, candles):
    dirs = [direction(c["open"], c["close"]) for c in candles]
    rows = []
    for i in range(4, len(candles) - 4):
        c = candles[i]
        prev = dirs[i - 4:i]
        future = dirs[i + 1:i + 5]
        current = dirs[i]
        past5 = prev + [current]
        future5 = [current] + future
        rows.append({
            "symbol": symbol,
            "time_utc": fmt_ts(c["timestamp"]),
            "open": c["open"],
            "high": c["high"],
            "low": c["low"],
            "close": c["close"],
            "volume": c["volume"],
            "direction": current,
            "prev_4": "".join(code(x) for x in reversed(prev)),
            "prev_d4": prev[0],
            "prev_d3": prev[1],
            "prev_d2": prev[2],
            "prev_d1": prev[3],
            "seq_5_past_to_current": ">".join(past5),
            "seq_5_past_code": "".join(code(x) for x in past5),
            "next_d1": future[0],
            "next_d2": future[1],
            "next_d3": future[2],
            "next_d4": future[3],
            "seq_5_current_to_future": ">".join(future5),
            "seq_5_future_code": "".join(code(x) for x in future5),
            "future_1h_return_pct": (candles[i + 1]["close"] / c["close"] - 1) * 100,
            "future_2h_return_pct": (candles[i + 2]["close"] / c["close"] - 1) * 100,
            "future_3h_return_pct": (candles[i + 3]["close"] / c["close"] - 1) * 100,
            "future_4h_return_pct": (candles[i + 4]["close"] / c["close"] - 1) * 100,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Build 1H direction-sequence dataset and preview rows in console.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--end-date", default="2026-09-01", help="Exclusive UTC date")
    ap.add_argument("--output", default=None)
    ap.add_argument("--show", type=int, default=50, help="Rows to print to console")
    ap.add_argument("--view", choices=["head", "tail"], default="tail")
    ap.add_argument("--symbol-view", default=None, help="Only print this symbol, e.g. BTCUSDT")
    args = ap.parse_args()

    start = int(datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end = int(datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    all_rows = []

    print(f"BUILD_1H_DIRECTION_DATASET range={args.start_date}..{args.end_date} exclusive")
    print("U=UP(close>open) D=DOWN(close<open) F=FLAT(close=open)")
    print("PAST sequence = previous 4 candles -> current | FUTURE sequence = current -> next 4 candles")

    for symbol in symbols:
        raw, used = load_range(Path(args.input_dir), symbol, start, end)
        candles = resample_1h(raw)
        rows = build_rows(symbol, candles)
        all_rows.extend(rows)
        print(f"{symbol}: raw_1m={len(raw)} 1h={len(candles)} dataset_rows={len(rows)} files={len(used)}")

    if not all_rows:
        raise RuntimeError("No rows generated.")

    all_rows.sort(key=lambda r: (r["time_utc"], r["symbol"]))
    out = Path(args.output) if args.output else ROOT / "reports" / f"direction_dataset_1h_{args.start_date}_{args.end_date}.csv"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    view_rows = [r for r in all_rows if not args.symbol_view or r["symbol"] == args.symbol_view]
    sample = view_rows[:args.show] if args.view == "head" else view_rows[-args.show:]
    print(f"SAVED {out.relative_to(ROOT)} rows={len(all_rows)}")
    print(f"CONSOLE_VIEW {args.view} rows={len(sample)} symbol={args.symbol_view or 'ALL'}")
    print("SYMBOL | UTC TIME           | DIR | PAST4 | CURRENT | FUTURE4 | 5-CODE(PAST->NOW) | 5-CODE(NOW->FUT) | R1H | R4H")
    print("-" * 125)
    for r in sample:
        print(
            f"{r['symbol']:7} | {r['time_utc']} | {r['direction']:4} | "
            f"{r['prev_4']:5} | {code(r['direction']):7} | {''.join(code(r['x']) for x in []) if False else r['seq_5_future_code'][1:]:7} | "
            f"{r['seq_5_past_code']:17} | {r['seq_5_future_code']:17} | "
            f"{r['future_1h_return_pct']:+6.2f}% | {r['future_4h_return_pct']:+6.2f}%"
        )


if __name__ == "__main__":
    main()
