from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_pattern_trader.models import Candle
from scripts.july_r_composite_walkforward import resample

DATE_RE = re.compile(r"(?<!\d)(20\d{2})[-_](0[1-9]|1[0-2])(?:[-_](0[1-9]|[12]\d|3[01]))?(?!\d)")
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"


def _parse_rows(reader) -> list[Candle]:
    rows: list[Candle] = []
    for r in reader:
        if len(r) < 6:
            continue
        try:
            ts = int(float(r[0]))
            if ts > 10_000_000_000_000:
                ts //= 1_000_000
            elif ts > 10_000_000_000:
                ts //= 1_000
            rows.append(Candle(ts, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
        except (ValueError, TypeError):
            continue
    return rows


def load_binance_archive(path: Path) -> list[Candle]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".csv") and not n.endswith("/")]
            if not members:
                return []
            with zf.open(members[0]) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                return _parse_rows(csv.reader(text))
    if ".gz" in [s.lower() for s in path.suffixes]:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as f:
            return _parse_rows(csv.reader(f))
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return _parse_rows(csv.reader(f))


def month_of_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")


def candidate_files(input_dir: Path, symbol: str) -> list[Path]:
    su = symbol.upper()
    out: list[Path] = []
    for p in input_dir.rglob("*"):
        if not p.is_file():
            continue
        nu = p.name.upper()
        if su not in nu:
            continue
        if nu.endswith(".CSV") or nu.endswith(".CSV.GZ") or nu.endswith(".ZIP"):
            out.append(p)
    return sorted(out, key=lambda x: str(x).upper())


def load_range(input_dir: Path, symbol: str, start_ts: int, end_ts: int) -> tuple[list[Candle], list[str]]:
    candles: list[Candle] = []
    used: list[str] = []
    for path in candidate_files(input_dir, symbol):
        try:
            raw = load_binance_archive(path)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            print(f"SKIP {symbol} {path.name}: {exc}")
            continue
        if not raw:
            continue
        matched = [c for c in raw if start_ts <= c.timestamp < end_ts]
        if matched:
            candles.extend(matched)
            used.append(path.name)
    by_ts = {c.timestamp: c for c in candles}
    return [by_ts[k] for k in sorted(by_ts)], used


def fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def direction(c: Candle) -> str:
    if c.close > c.open:
        return "UP"
    if c.close < c.open:
        return "DOWN"
    return "FLAT"


def signed_dir(c: Candle) -> int:
    if c.close > c.open:
        return 1
    if c.close < c.open:
        return -1
    return 0


def build_rows(symbol: str, candles_1h: list[Candle], lookback: int = 4, lookforward: int = 4) -> list[dict]:
    dirs = [direction(c) for c in candles_1h]
    sdirs = [signed_dir(c) for c in candles_1h]
    rows: list[dict] = []
    start = lookback
    stop = len(candles_1h) - lookforward
    for i in range(start, stop):
        c = candles_1h[i]
        prev_dirs = dirs[i - lookback:i]
        next_dirs = dirs[i + 1:i + 1 + lookforward]
        prev_signed = sdirs[i - lookback:i]
        next_signed = sdirs[i + 1:i + 1 + lookforward]
        four_past = ">".join(prev_dirs + [dirs[i]])
        four_next = ">".join([dirs[i]] + next_dirs)
        past_code = "".join("U" if x == 1 else "D" if x == -1 else "F" for x in prev_signed + [sdirs[i]])
        next_code = "".join("U" if x == 1 else "D" if x == -1 else "F" for x in [sdirs[i]] + next_signed)
        body = abs(c.close - c.open)
        rng = c.high - c.low
        close_pos = (c.close - c.low) / rng if rng > 0 else 0.5
        rows.append({
            "timestamp_utc": fmt_ts(c.timestamp),
            "timestamp_ms": c.timestamp,
            "symbol": symbol,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "direction": dirs[i],
            "direction_code": sdirs[i],
            "prev_dir_1": prev_dirs[-1],
            "prev_dir_2": prev_dirs[-2],
            "prev_dir_3": prev_dirs[-3],
            "prev_dir_4": prev_dirs[-4],
            "four_direction_past": four_past,
            "four_direction_past_code": past_code,
            "next_dir_1": next_dirs[0],
            "next_dir_2": next_dirs[1],
            "next_dir_3": next_dirs[2],
            "next_dir_4": next_dirs[3],
            "four_direction_future": four_next,
            "four_direction_future_code": next_code,
            "body_pct": (body / c.open * 100.0) if c.open else 0.0,
            "range_pct": (rng / c.open * 100.0) if c.open else 0.0,
            "close_position": close_pos,
            "future_1_return_pct": (candles_1h[i + 1].close / c.close - 1.0) * 100.0 if c.close else 0.0,
            "future_2_return_pct": (candles_1h[i + 2].close / c.close - 1.0) * 100.0 if c.close else 0.0,
            "future_3_return_pct": (candles_1h[i + 3].close / c.close - 1.0) * 100.0 if c.close else 0.0,
            "future_4_return_pct": (candles_1h[i + 4].close / c.close - 1.0) * 100.0 if c.close else 0.0,
            "future_4h_direction": next_dirs[3],
        })
    return rows


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a 1H OHLC direction/sequence dataset from Binance 1m archives.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--end-date", default="2026-09-01", help="Exclusive UTC date")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if end <= start:
        raise ValueError("end-date must be after start-date")

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    input_dir = Path(args.input_dir)
    all_rows: list[dict] = []
    print(f"BUILD_1H_DIRECTION_DATASET range={args.start_date}..{args.end_date} exclusive symbols={','.join(symbols)}")
    print("SCHEMA: OHLC + current direction + previous 4 directions + next 4 directions + sequence codes + forward returns")

    for symbol in symbols:
        raw, used = load_range(input_dir, symbol, int(start.timestamp()), int(end.timestamp()))
        if len(raw) < 20:
            print(f"MISSING {symbol} raw_1m={len(raw)}")
            continue
        candles_1h = resample(raw, 60)
        rows = build_rows(symbol, candles_1h)
        all_rows.extend(rows)
        print(f"OK {symbol} raw_1m={len(raw)} candles_1h={len(candles_1h)} rows={len(rows)} files={len(used)}")

    if not all_rows:
        raise RuntimeError("No rows generated.")

    all_rows.sort(key=lambda r: (r["timestamp_ms"], r["symbol"]))
    out = Path(args.output) if args.output else ROOT / "reports" / f"direction_dataset_1h_{args.start_date}_{args.end_date}.csv"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    print(f"SAVED {out.relative_to(ROOT)} rows={len(all_rows)}")
    print("DIRECTION_CODES U=close>open D=close<open F=close==open")
    print("4-DIRECTION ORDER prev_dir_4,prev_dir_3,prev_dir_2,prev_dir_1,direction")
    print("FUTURE ORDER direction,next_dir_1,next_dir_2,next_dir_3,next_dir_4")


if __name__ == "__main__":
    main()
