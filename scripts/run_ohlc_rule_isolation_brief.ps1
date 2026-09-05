param(
    [string]$InputDir = "C:\Users\cgart\Downloads\mapn-fast-pattern-trader-live\data\binance\vision",
    [string]$Symbols = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT",
    [string]$Month = "2026-07",
    [string]$Timeframes = "5,30",
    [double]$InitialCapital = 1000,
    [double]$TradeAllocation = 0.10,
    [double]$Leverage = 10,
    [double]$FeePerSide = 0.013,
    [double]$Threshold = 1.0,
    [string]$Output = ".\reports\ohlc_rule_direction_july_2026.csv"
)

python .\scripts\ohlc_rule_isolation_monthly.py `
  --input-dir $InputDir `
  --symbols $Symbols `
  --month $Month `
  --timeframes $Timeframes `
  --initial-capital $InitialCapital `
  --trade-allocation $TradeAllocation `
  --leverage $Leverage `
  --fee-per-side $FeePerSide `
  --threshold $Threshold `
  --output $Output |
  Select-String '^[A-Z0-9]+\s+\d+m\s+(BOTH|LONG_ONLY|SHORT_ONLY)\s+R\d+:' |
  ForEach-Object {
      $_.Line -replace '^([A-Z0-9]+)\s+(\d+)m\s+(BOTH|LONG_ONLY|SHORT_ONLY)\s+(R\d+):\s+return=', '$1|$2m|$3|$4|return='
  }

Write-Host "Saved: $Output"
Write-Host "Saved ranked: $($Output -replace '\.csv$','_ranked.csv')"
Write-Host "Saved direction summary: $($Output -replace '\.csv$','_direction_summary.csv')"
