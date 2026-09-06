$ErrorActionPreference = "Stop"

python .\scripts\score_july_rules.py `
  --input "reports\july_r_behavior_matrix.csv" `
  --output "reports\july_r_behavior_scores.csv"
