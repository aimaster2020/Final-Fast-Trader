# Final Fast Trader

Final Fast Trader is now the **default project** for our Fast Trader work.

## Current engine

The project contains the finalized OHLC rule-based trading engine transferred from `mapn-fast-pattern-trader` branch `feat/mapn-fast-pattern-integration` (source commit `ab49746`).

Core model:
- Numeric OHLC only.
- No indicators.
- No ML/DL.
- R1-R16 weighted candle decision engine.
- R16 is disabled in the current trading configuration (`weight = 0`).
- Continuous-account backtesting.
- 30% capital allocation per entry.
- Maximum 2 concurrent positions.
- Opposite-signal confirmation count N controls exit/reversal.
- Nobitex taker fee model: 0.1% per side.
- Binance Vision monthly multi-symbol sweep.

## Current candidate baseline

| Timeframe | N |
|---|---:|
| 5m | 5 |
| 30m | 4 |

These are **baseline candidates**, not yet approved for live trading.

## Latest stability finding

The six-month Binance sweep for February-July 2026 showed that the fixed 5m:N5 and 30m:N4 candidates were not stable across all symbols. Therefore the next optimization stage is an N sweep (N=1..10) with after-fee evaluation and out-of-sample validation.

## Main scripts

- `scripts/ohlc_continuous_account_candidates_365d.py` — continuous-account backtest.
- `scripts/ohlc_binance_multimonth_sweep.py` — multi-month Binance stability sweep.
- `src/fast_pattern_trader/ohlc_rule_strategy.py` — R1-R16 engine.
- `src/fast_pattern_trader/data_loader.py` — OHLC CSV loader.

## Default workflow

All future Fast Trader changes, tests, backtests and live-trading preparation should target this repository first:

https://github.com/aimaster2020/Final-Fast-Trader

The old `mapn-fast-pattern-trader` repository is treated as the source/history repository unless explicitly requested otherwise.
