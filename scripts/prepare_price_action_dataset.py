from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_pattern_trader.models import Candle
from scripts.build_1h_direction_dataset import load_range
from scripts.july_r_composite_walkforward import resample

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_MONTHS = "2026-05,2026-06,2026-07,2026-08"


def month_bounds(month: str) -> tuple[int, int]:
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return int(start.timestamp()), int(end.timestamp())


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a compact reusable OHLC dataset from Binance archives.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--months", default=DEFAULT_MONTHS)
    ap.add_argument("--timeframe", type=int, default=60, help="Target timeframe in minutes, e.g. 5 or 60.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    months = [x.strip() for x in args.months.split(",") if x.strip()]
    input_dir = Path(args.input_dir)
    rows: list[dict] = []

    if args.timeframe <= 0:
        ap.error("--timeframe must be positive")

    output = args.output or f"reports/prepared_price_action_{args.timeframe}m.csv"

    for month in months:
        start_ts, end_ts = month_bounds(month)
        for symbol in symbols:
            raw, used = load_range(input_dir, symbol, start_ts, end_ts)
            candles = resample([Candle(ts, o, h, l, c, v) for ts, o, h, l, c, v in raw], args.timeframe)
            print(f"{symbol} {month}: raw_1m={len(raw):,} tf={args.timeframe}m={len(candles):,} files={len(used)}")
            for c in candles:
                rows.append({
                    "symbol": symbol,
                    "month": month,
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                })

    rows.sort(key=lambda r: (r["timestamp"], r["symbol"]))
    out = Path(output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["symbol", "month", "timestamp", "open", "high", "low", "close", "volume"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"PREPARED {out.relative_to(ROOT)} rows={len(rows):,} timeframe={args.timeframe}m")


if __name__ == "__main__":
    main()
