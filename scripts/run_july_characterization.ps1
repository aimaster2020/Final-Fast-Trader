$ErrorActionPreference = 'Stop'

python .\scripts\characterize_july_rules.py `
  --input-dir "C:\Users\cgart\Downloads\mapn-fast-pattern-trader-live\data\binance\vision" `
  --symbols ALL `
  --month 2026-07 `
  --timeframes "5,15,30,60" `
  --output "reports/july_r_behavior_by_asset_tf.csv" `
  --matrix-output "reports/july_r_behavior_matrix.csv"

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
