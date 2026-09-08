from __future__ import annotations

from dataclasses import dataclass

from .models import Candle, Signal

STRATEGY_NAME = "candle_formula_binary_s_v2"
RANGE_MIN_BODY = -100.0
RANGE_MAX_BODY = 100.0
DECISION_SCORE = 2

# Exact Excel formula groups.
# UP group:
#   R1: H-C > C-O
#   R2: H-C > H-O
#   R3: L-C > C-O
# DOWN group:
#   R4: H-C < C-O
#   R5: H-C < H-O
#   R6: L-C < C-O
#
# Updated binary decision logic:
#   S = R1 + R2 + R3
#   S = 3 or 2 -> BUY
#   S = 1 or 0 -> SELL
#   No HOLD is produced by the formula decision.


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
    def up_score(self) -> int:
        """Exact Excel S score: R1 + R2 + R3, no weighting."""
        return (
            int(self.up_rule_1)
            + int(self.up_rule_2)
            + int(self.up_rule_3)
        )

    @property
    def down_score(self) -> int:
        """Diagnostic complement score: R4 + R5 + R6, no weighting."""
        return (
            int(self.down_rule_1)
            + int(self.down_rule_2)
            + int(self.down_rule_3)
        )


@dataclass(frozen=True)
class FormulaDecision:
    signal: Signal
    up_score: int
    down_score: int
    is_range: bool
    body: float
    rules: FormulaRules


def evaluate_rules(candle: Candle) -> FormulaRules:
    """Evaluate the exact six Excel rules using only the current candle.

    H = high, L = low, C = close, O = open.
    Rule order matches the spreadsheet.
    """
    high_minus_close = candle.high - candle.close
    close_minus_open = candle.close - candle.open
    low_minus_close = candle.low - candle.close
    high_minus_open = candle.high - candle.open

    return FormulaRules(
        # Original UP formulas: >
        up_rule_1=high_minus_close > close_minus_open,
        up_rule_2=high_minus_open < high_minus_close,
        up_rule_3=low_minus_close > close_minus_open,
        # Original DOWN formulas: <
        down_rule_1=high_minus_close < close_minus_open,
        down_rule_2=high_minus_open > high_minus_close,
        down_rule_3=low_minus_close < close_minus_open,
    )


def is_range(candle: Candle) -> bool:
    """Return True when Close - Open is between -100 and +100 inclusive."""
    body = candle.close - candle.open
    return RANGE_MIN_BODY <= body <= RANGE_MAX_BODY


def decide(candle: Candle) -> FormulaDecision:
    """Return BUY for S=3/2 and SELL for S=0/1."""
    rules = evaluate_rules(candle)
    up_score = rules.up_score
    down_score = rules.down_score

    if up_score >= DECISION_SCORE:
        signal = Signal.BUY
    else:
        signal = Signal.SELL

    return FormulaDecision(
        signal=signal,
        up_score=up_score,
        down_score=down_score,
        is_range=is_range(candle),
        body=candle.close - candle.open,
        rules=rules,
    )
