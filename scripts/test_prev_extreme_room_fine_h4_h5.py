from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
LOOKBACK = 1
ROOMS = [0.0200, 0.0225, 0.0250, 0.0275, 0.0300, 0.0325, 0.0350, 0.0375, 0.0400]
TARGET = 0.020
STOP = 0.030
HORIZONS = [4, 5]
FEE_PER_SIDE = 0.0013


def load_symbol(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"{path}: missing {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def score_direction(o: float, h: float, l: float, c: float) -> int:
    body = c - o
    hc = h - c
    ho = h - o
    lc = l - c
    score = int(hc > body) + int(ho < hc) + int(lc > body)
    return -1 if score == 0 else 1 if score == 3 else 0


def evaluate(df: pd.DataFrame, horizon: int, room_threshold: float) -> tuple[int, float, float, float, float, int, int, int]:
    returns: list[float] = []
    target_hits = stop_hits = time_exits = 0

    for i in range(LOOKBACK, len(df) - 1):
        o, h, l, c = map(float, df.loc[i, ["open", "high", "low", "close"]])
        direction = score_direction(o, h, l, c)
        if direction == 0 or c <= 0:
            continue

        prev_high = float(df.iloc[i - LOOKBACK:i]["high"].max())
        prev_low = float(df.iloc[i - LOOKBACK:i]["low"].min())
        room = (prev_high - c) / c if direction == 1 else (c - prev_low) / c
        if room < room_threshold:
            continue

        target_price = c * (1.0 + TARGET) if direction == 1 else c * (1.0 - TARGET)
        stop_price = c * (1.0 - STOP) if direction == 1 else c * (1.0 + STOP)
        end = min(i + horizon, len(df) - 1)

        result: float | None = None
        reason = "TIME"
        for j in range(i + 1, end + 1):
            hj = float(df.loc[j, "high"])
            lj = float(df.loc[j, "low"])
            if direction == 1:
                hit_target = hj >= target_price
                hit_stop = lj <= stop_price
            else:
                hit_target = lj <= target_price
                hit_stop = hj >= stop_price

            if hit_stop and hit_target:
                result = -STOP - 2 * FEE_PER_SIDE
                reason = "STOP"
                break
            if hit_stop:
                result = -STOP - 2 * FEE_PER_SIDE
                reason = "STOP"
                break
            if hit_target:
                result = TARGET - 2 * FEE_PER_SIDE
                reason = "TARGET"
                break

        if result is None:
            final_c = float(df.loc[end, "close"])
            gross = (final_c - c) / c if direction == 1 else (c - final_c) / c
            result = gross - 2 * FEE_PER_SIDE

        returns.append(result)
        target_hits += reason == "TARGET"
        stop_hits += reason == "STOP"
        time_exits += reason == "TIME"

    if not returns:
        return 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0

    s = pd.Series(returns)
    pos = float(s[s > 0].sum())
    neg = abs(float(s[s < 0].sum()))
    return (
        len(s),
        float(s.sum() * 100),
        float(s.mean() * 100),
        float((s > 0).mean() * 100),
        pos / neg if neg else 0.0,
        target_hits,
        stop_hits,
        time_exits,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=SYMBOLS)
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    root = Path(args.data_dir)
    loaded = {s: load_symbol(root / f"{s}_1h.csv") for s in symbols}

    print("=" * 104)
    print("PREVIOUS EXTREME ROOM FINE SCAN — H4/H5")
    print("Lookback=1 | Target=2% | Stop=3% | Entry=Close | Fee=0.13%/side")
    print("Same-candle target+stop => STOP")
    print("=" * 104)

    for horizon in HORIZONS:
        print(f"\nH{horizon}")
        print(" ROOM>=      N   RETURN%   AVG%  WIN%    PF  TARGET STOP TIME")
        for room in ROOMS:
            parts = [evaluate(loaded[s], horizon, room) for s in symbols]
            n = sum(x[0] for x in parts)
            if n == 0:
                continue
            ret = sum(x[1] for x in parts)
            avg = sum(x[2] * x[0] for x in parts) / n
            win = sum(x[3] * x[0] / 100 for x in parts) / n * 100
            gross_profit = sum((x[4] if x[4] > 0 else 0.0) * 0 for x in parts)
            # Recompute pooled PF from trade-level aggregates is not retained; use summed returns as a screen.
            # The detailed room scan above already reports PF. This fine scan focuses on stability versus sample size.
            t = sum(x[5] for x in parts)
            st = sum(x[6] for x in parts)
            tm = sum(x[7] for x in parts)
            print(f" {room*100:6.2f}% {n:6d} {ret:9.2f} {avg:6.3f} {win:6.1f} {'-':>5} {t:6d} {st:5d} {tm:4d}")


if __name__ == "__main__":
    main()
