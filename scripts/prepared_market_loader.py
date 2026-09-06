from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from fast_pattern_trader.models import Candle


def load_prepared(path: Path) -> dict[tuple[str, str], list[Candle]]:
    grouped: dict[tuple[str, str], list[Candle]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            grouped[(r["symbol"].upper(), r["month"])].append(
                Candle(
                    int(r["timestamp"]),
                    float(r["open"]),
                    float(r["high"]),
                    float(r["low"]),
                    float(r["close"]),
                    float(r["volume"]),
                )
            )
    for candles in grouped.values():
        candles.sort(key=lambda c: c.timestamp)
    return dict(grouped)
