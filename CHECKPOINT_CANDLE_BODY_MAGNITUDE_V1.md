# CHECKPOINT: CANDLE-BODY-MAGNITUDE-V1

## Project checkpoint name
**CANDLE-BODY-MAGNITUDE-V1**

## Repository
`aimaster2020/Final-Fast-Trader`

## Branch
`feat/new-candle-formula-strategy`

## Purpose
This checkpoint is the fixed baseline for the new candle-body movement strategy. Previous R1-R16 strategies are not the basis of this project.

## Confirmed formula
For each current candle:

- `N = Close - Open`  (c-o)
- `O = High - Close`  (h-c)
- `P = Low - Close`   (l-c)
- `Q = High - Open`   (h-o)

Three binary rules:

- `R1 = (O > N)`
- `R2 = (O > Q)`
- `R3 = (P > N)`

Score:

`AD = R1 + R2 + R3`

Strong-state targets:

- `AD = 3` predicts `N_next > N_current`
- `AD = 0` predicts `N_next < N_current`

Actual next-body direction is computed as:

`M = sign(N_next - N_current)`

## Verified 1H results on current project data
- BTCUSDT: AD=3 accuracy `80.75%`; AD=0 accuracy `83.40%`
- ETHUSDT: AD=3 accuracy `82.31%`; AD=0 accuracy `86.48%`
- SOLUSDT: AD=3 accuracy `81.53%`; AD=0 accuracy `82.89%`
- XRPUSDT: AD=3 accuracy `82.53%`; AD=0 accuracy `82.19%`

Combined strong-state accuracy across the four 1H datasets: approximately `82.83%`.

## Current interpretation
We have NOT yet predicted the exact numeric size of the next candle body. We have established a strong directional relationship for the **change in body size**:

`N_next - N_current`

The next research stage is to quantify the magnitude of that change and determine whether it is large enough, stable enough, and economically useful after trading costs.

## Do not alter this baseline when continuing research
Treat this checkpoint as the rollback/reference point for this research line. Any later experiment should be compared against this exact formula and these baseline measurements.
