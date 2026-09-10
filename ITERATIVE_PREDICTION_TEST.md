# Iterative predicted-frame test

Purpose: test multi-candle horizons using recursively generated predicted candles instead of peeking at observed future candles to construct intermediate frames.

- Signal on the observed current candle: score 0 = SHORT, score 3 = LONG; scores 1/2 ignored.
- The first predicted frame starts from the current observed close.
- Its magnitude uses the same directional geometry: LONG = |High-Open|, SHORT = |Open-Low|.
- The synthetic frame becomes the input to the next prediction.
- This recursion continues until horizon H.
- Evaluation uses the actual observed close at +H.
- No entry magnitude filter is included in this prediction experiment; it is intended to isolate recursive prediction behavior.
