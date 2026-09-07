from __future__ import annotations

from dataclasses import dataclass

from .models import Candle, Signal

STRATEGY_NAME = "candle_formula_3x3_v1_no_lc"
RANGE_MIN_BODY = -100.0
RANGE_MAX_BODY = 100.0
DECISION_SCORE = 2


@dataclass(frozen=True)
class FormulaRules:
    """Current-OHLC rules with the L-C rules disabled."""

    fall_rule_1: bool
    fall_rule_2: bool
    fall_rule_3: bool
    rise_rule_1: bool
    rise_rule_2: bool
    rise_rule_3: bool

    @property
    def fall_score(self) -> int:
        # Rule 2 (L-C) is intentionally disabled and does not contribute.
        return int(self.fall_rule_1) + int(self.fall_rule_3)

    @property
    def rise_score(self) -> int:
        # Rule 2 (L-C) is intentionally disabled and does not contribute.
        return int(self.rise_rule_1) + int(self.rise_rule_3)


@dataclass(frozen=True)
class FormulaDecision:
    signal: Signal
    fall_score: int
    rise_score: int
    is_range: bool
    body: float
    rules: FormulaRules


def evaluate_rules(candle: Candle) -> FormulaRules:
    """Evaluate the active current-candle OHLC rules only.

    Variables:
        H = high, L = low, C = close, O = open

    Active Fall:
        1) H-C < C-O
        3) H-C < H-O

    Active Rise:
        1) H-C > C-O
        3) H-C > H-O

    Disabled:
        2) L-C < C-O
        2) L-C > C-O
    """

    high_minus_close = candle.high - candle.close
    close_minus_open = candle.close - candle.open
    high_minus_open = candle.high - candle.open

    # L-C rules are deliberately kept as explicit FALSE values in the
    # output CSV so the audit trail shows that they were disabled.
    return FormulaRules(
        fall_rule_1=high_minus_close < close_minus_open,
        fall_rule_2=False,
        fall_rule_3=high_minus_close < high_minus_open,
        rise_rule_1=high_minus_close > close_minus_open,
        rise_rule_2=False,
        rise_rule_3=high_minus_close > high_minus_open,
    )


def is_range(candle: Candle) -> bool:
    """Return True when Close - Open is between -100 and +100 inclusive."""

    body = candle.close - candle.open
    return RANGE_MIN_BODY <= body <= RANGE_MAX_BODY


def decide(candle: Candle) -> FormulaDecision:
    """Return BUY/SELL when both active rules agree; otherwise HOLD.

    There are now 2 active rules per side because the L-C rules are disabled.
    Range detection remains informational and does not change the signal.
    """

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
