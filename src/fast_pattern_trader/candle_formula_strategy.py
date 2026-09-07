from __future__ import annotations

from dataclasses import dataclass

from .models import Candle, Signal

STRATEGY_NAME = "candle_formula_weighted_v1"
RANGE_MIN_BODY = -100.0
RANGE_MAX_BODY = 100.0
DECISION_SCORE = 2.0

FALL_WEIGHT_1 = 2.0  # H-C < C-O
FALL_WEIGHT_2 = 1.0  # L-C < C-O
FALL_WEIGHT_3 = 1.0  # H-C < H-O
RISE_WEIGHT_1 = 2.0  # H-C > C-O
RISE_WEIGHT_2 = 1.0  # L-C > C-O
RISE_WEIGHT_3 = 1.0  # H-C > H-O


@dataclass(frozen=True)
class FormulaRules:
    """The six current-OHLC rules with user-defined weights."""

    fall_rule_1: bool
    fall_rule_2: bool
    fall_rule_3: bool
    rise_rule_1: bool
    rise_rule_2: bool
    rise_rule_3: bool

    @property
    def fall_score(self) -> float:
        return (
            float(self.fall_rule_1) * FALL_WEIGHT_1
            + float(self.fall_rule_2) * FALL_WEIGHT_2
            + float(self.fall_rule_3) * FALL_WEIGHT_3
        )

    @property
    def rise_score(self) -> float:
        return (
            float(self.rise_rule_1) * RISE_WEIGHT_1
            + float(self.rise_rule_2) * RISE_WEIGHT_2
            + float(self.rise_rule_3) * RISE_WEIGHT_3
        )


@dataclass(frozen=True)
class FormulaDecision:
    signal: Signal
    fall_score: float
    rise_score: float
    is_range: bool
    body: float
    rules: FormulaRules


def evaluate_rules(candle: Candle) -> FormulaRules:
    """Evaluate all six user rules on the current candle only.

    Variables:
        H = high, L = low, C = close, O = open

    Fall weights:
        1) H-C < C-O : 2
        2) L-C < C-O : 1
        3) H-C < H-O : 1

    Rise weights:
        1) H-C > C-O : 2
        2) L-C > C-O : 1
        3) H-C > H-O : 1
    """

    high_minus_close = candle.high - candle.close
    close_minus_open = candle.close - candle.open
    low_minus_close = candle.low - candle.close
    high_minus_open = candle.high - candle.open

    return FormulaRules(
        fall_rule_1=high_minus_close < close_minus_open,
        fall_rule_2=low_minus_close < close_minus_open,
        fall_rule_3=high_minus_close < high_minus_open,
        rise_rule_1=high_minus_close > close_minus_open,
        rise_rule_2=low_minus_close > close_minus_open,
        rise_rule_3=high_minus_close > high_minus_open,
    )


def is_range(candle: Candle) -> bool:
    """Return True when Close - Open is between -100 and +100 inclusive."""

    body = candle.close - candle.open
    return RANGE_MIN_BODY <= body <= RANGE_MAX_BODY


def decide(candle: Candle) -> FormulaDecision:
    """Return BUY/SELL at the weighted 2-point threshold; otherwise HOLD."""

    rules = evaluate_rules(candle)
    fall_score = rules.fall_score
    rise_score = rules.rise_score

    if fall_score >= DECISION_SCORE:
        signal = Signal.SELL
    elif rise_score >= DECISION_SCORE:
        signal = Signal.BUY
    else:
        signal = Signal.HOLD

    return FormulaDecision(
        signal=signal,
        fall_score=fall_score,
        rise_score=rise_score,
        is_range=is_range(candle),
        body=candle.close - candle.open,
        rules=rules,
    )
