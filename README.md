# Final Fast Trader

Final Fast Trader is the **default project** for our Fast Trader work.

## Current strategy

The active strategy is a pure current-candle formula engine. It evaluates only the current OHLC values and has no knowledge database, historical pattern lookup, indicators, ML/DL, or online learning in the decision path.

Let:

- `H = high`
- `L = low`
- `C = close`
- `O = open`

### Fall / SELL rules

1. `H - C < C - O`
2. `L - C < C - O`
3. `H - C < H - O`

### Rise / BUY rules

1. `H - C > C - O`
2. `L - C > C - O`
3. `H - C > H - O`

Each satisfied rule gives one point to its direction. A direction is selected when it reaches **2 of 3 points**. Otherwise the strategy returns `HOLD`.

### Range detection

A candle is marked as range when:

`-100 <= C - O <= 100`

Range detection is returned as a separate state and does not currently override BUY/SELL.

## Core module

`src/fast_pattern_trader/candle_formula_strategy.py`

The public package exports the strategy through `fast_pattern_trader.decide`, plus `evaluate_rules` and `is_range`.

## Validation

`tests/test_final_fast_trader.py` covers the six formula rules, 2-of-3 decisions, and the inclusive range boundaries.

Historical R1-R16 and other earlier strategy experiments remain only as repository history/analysis artifacts and are not part of the active decision engine.

## Default workflow

All future Fast Trader changes, tests, backtests and live-trading preparation should target this repository first:

https://github.com/aimaster2020/Final-Fast-Trader
