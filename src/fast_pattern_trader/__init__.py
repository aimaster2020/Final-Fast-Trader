"""Final Fast Trader core package."""

from .models import Candle, Signal
from .ohlc_rule_strategy import STRATEGY_NAME, decide, decide_movement

__all__ = ["Candle", "Signal", "STRATEGY_NAME", "decide", "decide_movement"]
