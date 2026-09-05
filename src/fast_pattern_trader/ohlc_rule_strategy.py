from __future__ import annotations
from dataclasses import dataclass
from .models import Candle, Signal

STRATEGY_NAME = "ohlc_rules_r1_r16_weighted_one_candle"
DEFAULT_RANGE_THRESHOLD = 0.995
DEFAULT_MOVEMENT_SCORE_THRESHOLD = 1.0
DEFAULT_MIN_PREDICTED_MOVE_PCT = 0.0
DEFAULT_R16_WEIGHT = 0.0
DEFAULT_RULE_WEIGHTS = {f"R{i}": 0.5 for i in range(1, 17)}
DEFAULT_RULE_WEIGHTS.update({"R2": 1.5, "R4": 1.5, "R16": DEFAULT_R16_WEIGHT})

@dataclass(frozen=True)
class RuleVotes:
    close_gt_open: int
    open_gt_close: int
    close_eq_high: int
    close_eq_low: int
    @property
    def score(self) -> int:
        return self.close_gt_open + self.open_gt_close + self.close_eq_high + self.close_eq_low

@dataclass(frozen=True)
class R16Prediction:
    direction: Signal
    predicted_move: float
    predicted_move_pct: float

@dataclass(frozen=True)
class OpenBehaviorState:
    open_high_similarity: float
    open_low_similarity: float
    directional_vote: float

@dataclass(frozen=True)
class CloseBehaviorState:
    close_high_similarity: float
    close_low_similarity: float
    directional_vote: float

@dataclass(frozen=True)
class WickBehaviorState:
    lower_wick_strength: float
    upper_wick_strength: float
    directional_vote: float

@dataclass(frozen=True)
class BodyPositionState:
    body_midpoint: float
    range_midpoint: float
    bullish_position: float
    bearish_position: float
    directional_vote: float

@dataclass(frozen=True)
class BodyStrengthState:
    body_strength: float
    bullish_strength: float
    bearish_strength: float
    directional_vote: float

@dataclass(frozen=True)
class RangeState:
    score: float
    close_high_distance_pct: float
    open_low_distance_pct: float
    projected_direction: Signal
    projected_target: float
    projected_move: float
    directional_vote: float

@dataclass(frozen=True)
class MovementDecision:
    signal: Signal
    score: float
    base_score: int
    range_vote: float
    open_behavior_vote: float
    close_behavior_vote: float
    wick_behavior_vote: float
    body_position_vote: float
    body_strength_vote: float
    movement_strength: float
    current_range_pct: float
    projected_move_pct: float
    projected_move: float
    projected_next_close: float
    votes: RuleVotes
    range: RangeState
    open_behavior: OpenBehaviorState
    close_behavior: CloseBehaviorState
    wick_behavior: WickBehaviorState
    body_position: BodyPositionState
    body_strength: BodyStrengthState

@dataclass(frozen=True)
class StrategyDecision:
    signal: Signal
    score: int
    votes: RuleVotes
    range: RangeState | None = None


def evaluate_rules(candle: Candle) -> RuleVotes:
    return RuleVotes(
        close_gt_open=-1 if candle.close > candle.open else 0,
        open_gt_close=1 if candle.open > candle.close else 0,
        close_eq_high=-1 if candle.close == candle.high else 0,
        close_eq_low=1 if candle.close == candle.low else 0,
    )


def evaluate_r16(candle: Candle) -> R16Prediction:
    body = candle.close - candle.open
    predicted_move = body if body > 0 else candle.close - candle.high
    direction = Signal.BUY if predicted_move > 0 else Signal.SELL if predicted_move < 0 else Signal.HOLD
    pct = abs(predicted_move / candle.close * 100) if candle.close else 0.0
    return R16Prediction(direction, predicted_move, pct)


def evaluate_open_behavior(candle: Candle) -> OpenBehaviorState:
    hi, lo = abs(candle.high-candle.open), abs(candle.open-candle.low)
    total = hi + lo
    high_sim, low_sim = (0.5, 0.5) if total <= 1e-12 else (lo/total, hi/total)
    return OpenBehaviorState(high_sim, low_sim, max(-1, min(1, high_sim-low_sim)))


def evaluate_close_behavior(candle: Candle) -> CloseBehaviorState:
    hi, lo = abs(candle.high-candle.close), abs(candle.close-candle.low)
    total = hi + lo
    high_sim, low_sim = (0.5, 0.5) if total <= 1e-12 else (lo/total, hi/total)
    return CloseBehaviorState(high_sim, low_sim, max(-1, min(1, high_sim-low_sim)))


def evaluate_wick_behavior(candle: Candle) -> WickBehaviorState:
    r = max(candle.high-candle.low, 1e-12)
    upper = max(candle.high-max(candle.open,candle.close), 0)/r
    lower = max(min(candle.open,candle.close)-candle.low, 0)/r
    return WickBehaviorState(max(0,min(1,lower)), max(0,min(1,upper)), max(-1,min(1,lower-upper)))


def evaluate_body_position(candle: Candle) -> BodyPositionState:
    r = max(candle.high-candle.low, 1e-12)
    bm, rm = (candle.open+candle.close)/2, (candle.high+candle.low)/2
    offset = max(-0.5, min(0.5, (bm-rm)/r))
    vote = offset*2
    return BodyPositionState(bm, rm, max(0,vote), max(0,-vote), max(-1,min(1,vote)))


def evaluate_body_strength(candle: Candle) -> BodyStrengthState:
    r = max(candle.high-candle.low, 1e-12)
    body = candle.close-candle.open
    strength = min(1, abs(body)/r)
    bull, bear = (strength,0) if body>0 else (0,strength) if body<0 else (0,0)
    return BodyStrengthState(strength,bull,bear,bull-bear)


def evaluate_range(candle: Candle, *, threshold: float=DEFAULT_RANGE_THRESHOLD) -> RangeState:
    if not 0 <= threshold <= 1: raise ValueError("range threshold must be in [0, 1]")
    midpoint=max((candle.open+candle.high+candle.low+candle.close)/4,1e-12)
    ch=abs(candle.high-candle.close)/midpoint
    ol=abs(candle.open-candle.low)/midpoint
    score=max(0,min(1,1-(ch+ol)/2))
    total=ch+ol
    vote=0 if total<=1e-12 else max(-1,min(1,(ol-ch)/total))
    direction=Signal.BUY if vote>0 else Signal.SELL if vote<0 else Signal.BUY if candle.close>candle.open else Signal.SELL if candle.open>candle.close else Signal.HOLD
    move=max(candle.high-candle.close,0) if direction==Signal.BUY else max(candle.open-candle.low,0) if direction==Signal.SELL else 0
    target=candle.close+move if direction==Signal.BUY else candle.close-move if direction==Signal.SELL else candle.close
    return RangeState(score,ch*100,ol*100,direction,target,move,vote*score)


def decide(candle: Candle, *, min_score: int=1) -> StrategyDecision:
    if min_score not in (1,2): raise ValueError("min_score must be 1 or 2")
    votes=evaluate_rules(candle)
    signal=Signal.BUY if votes.score>=min_score else Signal.SELL if votes.score<=-min_score else Signal.HOLD
    return StrategyDecision(signal,votes.score,votes)


def decide_movement(candle: Candle, *, score_threshold: float=DEFAULT_MOVEMENT_SCORE_THRESHOLD, range_threshold: float=DEFAULT_RANGE_THRESHOLD, min_predicted_move_pct: float=DEFAULT_MIN_PREDICTED_MOVE_PCT, rule_weights: dict[str,float]|None=None) -> MovementDecision:
    if score_threshold<0 or min_predicted_move_pct<0: raise ValueError("thresholds must be non-negative")
    w=dict(DEFAULT_RULE_WEIGHTS); w.update(rule_weights or {})
    votes=evaluate_rules(candle); r16=evaluate_r16(candle); rs=evaluate_range(candle,threshold=range_threshold)
    ob=evaluate_open_behavior(candle); cb=evaluate_close_behavior(candle); wb=evaluate_wick_behavior(candle); bp=evaluate_body_position(candle); bs=evaluate_body_strength(candle)
    c={
        "R1":w["R1"]*votes.close_gt_open,"R2":w["R2"]*votes.open_gt_close,"R3":w["R3"]*votes.close_eq_high,"R4":w["R4"]*votes.close_eq_low,
        "R5":w["R5"]*rs.directional_vote,"R6":w["R6"]*ob.open_high_similarity,"R7":-w["R7"]*ob.open_low_similarity,
        "R8":w["R8"]*cb.close_high_similarity,"R9":-w["R9"]*cb.close_low_similarity,"R10":w["R10"]*wb.lower_wick_strength,"R11":-w["R11"]*wb.upper_wick_strength,
        "R12":w["R12"]*bp.bullish_position,"R13":-w["R13"]*bp.bearish_position,"R14":w["R14"]*bs.bullish_strength,"R15":-w["R15"]*bs.bearish_strength,"R16":w["R16"]*int(r16.direction)}
    total=float(sum(c.values())); signal=Signal.BUY if total>=score_threshold else Signal.SELL if total<=-score_threshold else Signal.HOLD
    tw=max(sum(abs(float(x)) for x in w.values()),1e-12); strength=max(0,min(1,abs(total)/tw)); rng=max(candle.high-candle.low,0); rng_pct=rng/max(abs(candle.close),1e-12)*100; move_pct=rng_pct*strength; move=abs(candle.close)*move_pct/100
    if move_pct<min_predicted_move_pct and signal!=Signal.HOLD: signal=Signal.HOLD
    next_close=candle.close+move if signal==Signal.BUY else candle.close-move if signal==Signal.SELL else candle.close
    return MovementDecision(signal,total,votes.score,c["R5"],c["R6"]+c["R7"],c["R8"]+c["R9"],c["R10"]+c["R11"],c["R12"]+c["R13"],c["R14"]+c["R15"],strength,rng_pct,move_pct,move,next_close,votes,rs,ob,cb,wb,bp,bs)
