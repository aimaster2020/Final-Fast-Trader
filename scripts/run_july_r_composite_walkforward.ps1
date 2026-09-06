$ErrorActionPreference = "Stop"

python .\scripts\july_r_composite_walkforward.py `
  --input-dir "C:\Users\cgart\Downloads\mapn-fast-pattern-trader-live\data\binance\vision" `
  --train-month "2026-06" `
  --test-month "2026-07" `
  --symbols ALL `
  --timeframes "5,15,30,60" `
  --gate 0.25 `
  --output "reports/july_r_composite_walkforward.csv" `
  --weights-output "reports/july_r_composite_walkforward_weights.csv"
