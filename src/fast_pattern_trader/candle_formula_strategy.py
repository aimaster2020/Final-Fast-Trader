from __future__ import annotations

from dataclasses import dataclass

from .models import Candle, Signal

STRATEGY_NAME = "candle_formula_weighted_v3"
RANGE_MIN_BODY = -100.0
RANGE_MAX_BODY = 100.0
DECISION_SCORE = 3.0

# Exact Excel formula groups. Do not relabel these algebraically.
# RISE/UP group:
#   H-C > C-O
#   H-C > H-O
#   L-C > C-O
# FALL/DOWN group:
#   H-C < C-O
#   H-C < H-O
#   L-C < C-O
UP_WEIGHT_1 = 2.0
UP_WEIGHT_2 = 1.0
UP_WEIGHT_3 = 1.0
DOWN_WEIGHT_1 = 2.0
DOWN_WEIGHT_2 = 1.0
DOWN_WEIGHT_3 = 1.0


@dataclass(frozen=True)
class FormulaRules:
    """The six Excel OHLC rules with their original direction semantics."""

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
    """Evaluate the exact six Excel rules using only the current candle.

    H = high, L = low, C = close, O = open.
    Rule order matches the spreadsheet's weighted logic.
    """
    high_minus_close = candle.high - candle.close
    close_minus_open = candle.close - candle.open
    low_minus_close = candle.low - candle.close
    high_minus_open = candle.high - candle.open

    return FormulaRules(
        # Original RISE/UP formulas: >
        up_rule_1=high_minus_close > close_minus_open,
        up_rule_2=high_minus_open < high_minus_close,  # equivalent to H-C > H-O
        up_rule_3=low_minus_close > close_minus_open,
        # Original FALL/DOWN formulas: <
        down_rule_1=high_minus_close < close_minus_open,
        down_rule_2=high_minus_open > high_minus_close,  # equivalent to H-C < H-O
        down_rule_3=low_minus_close < close_minus_open,
    )


def is_range(candle: Candle) -> bool:
    """Return True when Close - Open is between -100 and +100 inclusive."""
    body = candle.close - candle.open
    return RANGE_MIN_BODY <= body <= RANGE_MAX_BODY


def decide(candle: Candle) -> FormulaDecision:
    """Return BUY for weighted UP=3, SELL for weighted DOWN=3, else HOLD."""
    rules = evaluate_rules(candle)
    up_score = rules.up_score
    down_score = rules.down_score

    if up_score >= DECISION_SCORE and down_score < DECISION_SCORE:
        signal = Signal.BUY
    elif down_score >= DECISION_SCORE and up_score < DECISION_SCORE:
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
