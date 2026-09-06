$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python .\scripts\july_r_predictive_edge_wf.py `
  --input-dir "$env:USERPROFILE\Downloads\mapn-fast-pattern-trader-live\data\binance\vision" `
  --train-month 2026-06 `
  --test-month 2026-07 `
  --symbols ALL `
  --timeframes 5,15,30,60 `
  --min-train-signals 100 `
  --top-n 20
