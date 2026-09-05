from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum

class Signal(IntEnum):
    SELL = -1
    HOLD = 0
    BUY = 1

@dataclass(frozen=True)
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
