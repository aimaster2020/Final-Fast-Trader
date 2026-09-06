$ErrorActionPreference = "Stop"

python .\scripts\july_r_precision_sweep.py `
  --input-dir "C:\Users\cgart\Downloads\mapn-fast-pattern-trader-live\data\binance\vision" `
  --train-month "2026-06" `
  --test-month "2026-07" `
  --symbols ALL `
  --timeframes "5,15,30,60" `
  --gates "0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.90,0.95" `
  --min-confirms "2,3,4,5,6,7,8" `
  --min-signals 30 `
  --output "reports/july_r_precision_sweep.csv"
