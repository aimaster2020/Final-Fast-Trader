from __future__ import annotations

from dataclasses import dataclass

from .models import Candle, Signal

STRATEGY_NAME = "candle_formula_weighted_v2"
RANGE_MIN_BODY = -100.0
RANGE_MAX_BODY = 100.0
DECISION_SCORE = 3.0

# The formulas themselves stay exactly as defined by the user.
# Their direction labels are aligned to the Excel ground truth:
# current Close > previous Close = UP
# current Close < previous Close = DOWN
#
# The original FALL group is therefore an UP group, and the original RISE
# group is therefore a DOWN group for prediction purposes.
UP_WEIGHT_1 = 2.0  # H-C < C-O
UP_WEIGHT_2 = 1.0  # L-C < C-O
UP_WEIGHT_3 = 1.0  # H-C < H-O
DOWN_WEIGHT_1 = 2.0  # H-C > C-O
DOWN_WEIGHT_2 = 1.0  # L-C > C-O
DOWN_WEIGHT_3 = 1.0  # H-C > H-O


@dataclass(frozen=True)
class FormulaRules:
    """The six current-OHLC rules with direction-aligned weights."""

    up_rule_1: bool
    up_rule_2: bool
    up_rule_3: bool
    down_rule_1: bool
    down_rule_2: bool
    down_rule_3: bool

    @property
    def up_score(self) -> float:
        return (
            float(self.up_rule_1) * UP_WEIGHT_1
            + float(self.up_rule_2) * UP_WEIGHT_2
            + float(self.up_rule_3) * UP_WEIGHT_3
        )

    @property
    def down_score(self) -> float:
        return (
            float(self.down_rule_1) * DOWN_WEIGHT_1
            + float(self.down_rule_2) * DOWN_WEIGHT_2
            + float(self.down_rule_3) * DOWN_WEIGHT_3
        )


@dataclass(frozen=True)
class FormulaDecision:
    signal: Signal
    up_score: float
    down_score: float
    is_range: bool
    body: float
    rules: FormulaRules


def evaluate_rules(candle: Candle) -> FormulaRules:
    """Evaluate all six user formulas on the current candle only.

    Variables:
        H = high, L = low, C = close, O = open

    UP formulas:
        1) H-C < C-O : 2
        2) L-C < C-O : 1
        3) H-C < H-O : 1

    DOWN formulas:
        1) H-C > C-O : 2
        2) L-C > C-O : 1
        3) H-C > H-O : 1
    """

    high_minus_close = candle.high - candle.close
    close_minus_open = candle.close - candle.open
    low_minus_close = candle.low - candle.close
    high_minus_open = candle.high - candle.open

    return FormulaRules(
        up_rule_1=high_minus_close < close_minus_open,
        up_rule_2=low_minus_close < close_minus_open,
        up_rule_3=high_minus_close < high_minus_open,
        down_rule_1=high_minus_close > close_minus_open,
        down_rule_2=low_minus_close > close_minus_open,
        down_rule_3=high_minus_close > high_minus_open,
    )


def is_range(candle: Candle) -> bool:
    """Return True when Close - Open is between -100 and +100 inclusive."""

    body = candle.close - candle.open
    return RANGE_MIN_BODY <= body <= RANGE_MAX_BODY


def decide(candle: Candle) -> FormulaDecision:
    """Return BUY/SELL at the weighted 3-point threshold; otherwise HOLD."""

    rules = evaluate_rules(candle)
    up_score = rules.up_score
    down_score = rules.down_score

    if up_score >= DECISION_SCORE:
        signal = Signal.BUY
    elif down_score >= DECISION_SCORE:
        signal = Signal.SELL
    else:
        signal = Signal.HOLD

    return FormulaDecision(
        signal=signal,
        up_score=up_score,
        down_score=down_score,
        is_range=is_range(candle),
        body=candle.close - candle.open,
        rules=rules,
    )
