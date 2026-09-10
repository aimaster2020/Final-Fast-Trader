from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_LOOKBACKS = [1, 2, 3, 4, 5]
DEFAULT_ROOMS = [0.0, 0.01, 0.02]
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"


def load_symbol(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=list(required)).reset_index(drop=True)


def score_direction(o: float, h: float, l: float, c: float) -> int:
    body = c - o
    hc = h - c
    ho = h - o
    lc = l - c
    score = int(hc > body) + int(ho < hc) + int(lc > body)
    return -1 if score == 0 else 1 if score == 3 else 0


def evaluate(df: pd.DataFrame, horizon: int, lookback: int, room_threshold: float, mode: str) -> tuple[int, float, float, float]:
    mfe_values = []
    mae_values = []
    for i in range(lookback, len(df) - horizon):
        o, h, l, c = map(float, df.loc[i, ["open", "high", "low", "close"]])
        direction = score_direction(o, h, l, c)
        if direction == 0:
            continue
        if mode == "LONG" and direction != 1:
            continue
        if mode == "SHORT" and direction != -1:
            continue

        prev_high = float(df.iloc[i - lookback:i]["high"].max())
        prev_low = float(df.iloc[i - lookback:i]["low"].min())
        if direction == 1:
            room = (prev_high - c) / c
            future_mfe = (float(df.iloc[i + 1:i + horizon + 1]["high"].max()) - c) / c
            future_mae = (c - float(df.iloc[i + 1:i + horizon + 1]["low"].min())) / c
        else:
            room = (c - prev_low) / c
            future_mfe = (c - float(df.iloc[i + 1:i + horizon + 1]["low"].min())) / c
            future_mae = (float(df.iloc[i + 1:i + horizon + 1]["high"].max()) - c) / c

        if room >= room_threshold:
            mfe_values.append(future_mfe)
            mae_values.append(future_mae)

    n = len(mfe_values)
    if not n:
        return 0, 0.0, 0.0, 0.0
    return n, sum(mfe_values) / n, sum(mae_values) / n, sum(x >= 0.0026 for x in mfe_values) / n


def pooled(loaded, symbols, horizon, lookback, room_threshold, mode):
    rs = [evaluate(loaded[s], horizon, lookback, room_threshold, mode) for s in symbols]
    n = sum(r[0] for r in rs)
    if not n:
        return 0, 0.0, 0.0, 0.0
    return (
        n,
        sum(r[1] * r[0] for r in rs) / n,
        sum(r[2] * r[0] for r in rs) / n,
        sum(r[3] * r[0] for r in rs) / n,
    )


def main():
    ap = argparse.ArgumentParser(description="Compact previous-extreme lookback scan H1-H5.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--lookbacks", default="1,2,3,4,5")
    ap.add_argument("--room-thresholds", default="0,0.01,0.02")
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    lookbacks = [int(x) for x in args.lookbacks.split(",") if x.strip()]
    rooms = [float(x) for x in args.room_thresholds.split(",") if x.strip()]
    root = Path(args.data_dir)
    loaded = {s: load_symbol(root / f"{s}_1h.csv") for s in symbols}

    print("=" * 96)
    print("PREVIOUS EXTREME LOOKBACK COMPACT SCAN — H1 TO H5")
    print("Lookback = MAX(H) / MIN(L) of previous N candles; current candle excluded")
    print("Rows are pooled across BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT; fee reference = 0.26% round trip")
    print("=" * 96)

    for horizon in range(1, 6):
        print(f"\nH{horizon} | BOTH | room thresholds = {', '.join(f'{r*100:.0f}%' for r in rooms)}")
        print(f"{'LB':>3} {'ROOM':>6} {'N':>6} {'MFE%':>8} {'MAE%':>8} {'>=FEE%':>8}")
        for lb in lookbacks:
            for room in rooms:
                n, mfe, mae, fee = pooled(loaded, symbols, horizon, lb, room, "BOTH")
                print(f"{lb:3d} {room*100:5.0f}% {n:6d} {mfe*100:8.3f} {mae*100:8.3f} {fee*100:8.1f}")

if __name__ == "__main__":
    main()
