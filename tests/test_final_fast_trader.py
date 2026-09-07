from fast_pattern_trader.candle_formula_strategy import decide, evaluate_rules, is_range
from fast_pattern_trader.models import Candle, Signal


def candle(open_: float, high: float, low: float, close: float) -> Candle:
    return Candle(timestamp=0, open=open_, high=high, low=low, close=close)


def test_fall_rules_score_two_of_three_produces_sell():
    result = decide(candle(100, 110, 99, 108))
    assert result.fall_score == 2
    assert result.rise_score == 1
    assert result.signal == Signal.SELL


def test_rise_rules_score_three_of_three_produces_buy():
    result = decide(candle(100, 110, 95, 90))
    assert result.rise_score == 3
    assert result.fall_score == 0
    assert result.signal == Signal.BUY


def test_below_two_votes_is_hold():
    result = decide(candle(100, 105, 95, 100))
    assert result.fall_score < 2
    assert result.rise_score < 2
    assert result.signal == Signal.HOLD


def test_range_uses_close_minus_open_between_minus_100_and_100_inclusive():
    assert is_range(candle(100, 150, 50, 0))
    assert is_range(candle(100, 150, 50, 200))
    assert not is_range(candle(100, 250, 50, 201))
    assert not is_range(candle(200, 250, 50, 99))


def test_rules_are_based_only_on_current_ohlc():
    result = evaluate_rules(candle(100, 110, 90, 105))
    assert result.fall_rule_1 is False
    assert result.fall_rule_2 is True
    assert result.fall_rule_3 is True
    assert result.rise_rule_1 is False
    assert result.rise_rule_2 is False
    assert result.rise_rule_3 is False
