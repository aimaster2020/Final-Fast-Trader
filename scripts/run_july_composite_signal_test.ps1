$ErrorActionPreference = "Stop"

python .\scripts\july_composite_signal_test.py `
  --input-dir "C:\Users\cgart\Downloads\mapn-fast-pattern-trader-live\data\binance\vision" `
  --symbols ALL `
  --month 2026-07 `
  --timeframes "5,15,30,60" `
  --scores "reports/july_r_behavior_scores.csv"
