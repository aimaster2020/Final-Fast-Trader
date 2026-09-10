# Final-Fast-Trader Baseline

## Official baseline

**Branch:** `baseline/progressive-exit-entry-magnitude-filter`

**Baseline strategy file:** `scripts/backtest_progressive_exit_entry_magnitude_filter.py`

**Source branch:** `feat/new-candle-formula-strategy`

**Baseline branch starting commit:** `1738cc7be00bbeb907b5e19ee181d6e42536999f`

## Strategy definition

This baseline is the canonical starting point for new experiments. Do not replace or silently alter its execution logic when testing a new idea. New experiments should branch from this baseline and preserve it unchanged.

### Signal

- Score 0/1 -> SHORT
- Score 2/3 -> LONG
- Ambiguous cases are removed using the formula-direction filter.

### Magnitude

- LONG expected movement: `|High - Open|`
- SHORT expected movement: `|Open - Low|`
- Entry filter: expected movement divided by current candle close must be at least the configured `entry_min_fraction`.

### Execution

- Entry price: current candle **close**.
- The position then follows the progressive opposite-signal exit logic.
- Same-direction signals keep the trade open.
- First opposite signal is treated as noise when sufficiently weak.
- Second opposite signal requires at least 2x the first opposite strength and a second noise threshold.
- Third opposite signal is a mandatory exit.
- Any remaining open position is force-closed at the end of the test window.

### Capital and fees

- Initial capital: `$1000` per symbol.
- Commission reference: `0.0013` per side = `0.13%` per side = `0.26%` round trip.
- The script reports both zero-fee and configured-fee results.

### Baseline test command

```powershell
python .\scripts\backtest_progressive_exit_entry_magnitude_filter.py `
  --data-dir ".\reports\1h" `
  --symbols "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT" `
  --initial-capital 1000 `
  --fee 0.0013 `
  --entry-fractions "0.0075,0.0076,0.0077,0.0078,0.0079,0.0080,0.0081,0.0082,0.0083,0.0084,0.0085,0.0086,0.0087,0.0088,0.0089,0.0090"
```

## Continuity rule for future chats

When continuing this project in a new chat, use this branch as the canonical baseline unless a newer baseline is explicitly declared in a newer `BASELINE.md` commit. Do not infer a new baseline from an isolated positive test.
