from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path

from .models import Candle


def _timestamp(value: str) -> int:
    text = value.strip()
    try:
        return int(float(text))
    except ValueError:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())


def _header_map(headers: list[str]) -> dict[str, str]:
    aliases = {
        "timestamp": "timestamp", "time": "timestamp", "datetime": "timestamp", "date": "timestamp",
        "open": "open", "high": "high", "low": "low", "close": "close",
        "volume": "volume", "vol": "volume",
    }
    out: dict[str, str] = {}
    for header in headers:
        key = re.sub(r"[^a-z0-9]+", "", header.strip().lower())
        if key in aliases:
            out[aliases[key]] = header
    return out


def validate_candles(candles: list[Candle]) -> list[Candle]:
    ordered = sorted(candles, key=lambda c: c.timestamp)
    out: list[Candle] = []
    previous = None
    for c in ordered:
        if previous is not None and c.timestamp <= previous:
            continue
        if min(c.open, c.high, c.low, c.close) <= 0:
            continue
        if c.low > c.high or c.close < c.low or c.close > c.high:
            continue
        out.append(c)
        previous = c.timestamp
    if not out:
        raise ValueError("historical dataset contains no valid candles")
    return out


def load_csv(path: str | Path) -> list[Candle]:
    p = Path(path)
    with p.open("r", newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            rows = list(csv.reader(f, dialect))
        except csv.Error:
            f.seek(0)
            lines = [x.strip() for x in f if x.strip()]
            rows = [re.split(r"\s+", x) for x in lines]
    if not rows:
        raise ValueError("CSV is empty")
    headers, data = rows[0], rows[1:]
    mapping = _header_map(headers)
    required = {"timestamp", "open", "high", "low", "close"}
    if not required.issubset(mapping):
        raise ValueError("CSV must contain timestamp/open/high/low/close columns")
    index = {k: headers.index(v) for k, v in mapping.items()}
    candles = []
    for row in data:
        try:
            candles.append(Candle(
                _timestamp(row[index["timestamp"]]),
                float(row[index["open"]]), float(row[index["high"]]),
                float(row[index["low"]]), float(row[index["close"]]),
                float(row[index["volume"]]) if "volume" in index and row[index["volume"]] else 0.0,
            ))
        except (ValueError, TypeError, IndexError):
            continue
    return validate_candles(candles)
