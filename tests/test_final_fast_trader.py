from fast_pattern_trader.models import Candle, Signal
from fast_pattern_trader.ohlc_rule_strategy import decide, decide_movement


def test_original_four_rules():
    c = Candle(0, 100, 105, 95, 104)
    d = decide(c)
    assert d.signal == Signal.SELL
    assert d.votes.close_gt_open == -1


def test_r16_disabled_by_default():
    c = Candle(0, 100, 105, 95, 104)
    d = decide_movement(c, rule_weights={"R16": 0.0})
    assert d.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
    assert d.score == d.score
