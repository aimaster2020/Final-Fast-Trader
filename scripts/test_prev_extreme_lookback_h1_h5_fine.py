from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LOOKBACKS = [1, 2, 3, 4, 5]
ROOM_THRESHOLDS = [0.01, 0.02]
SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"


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


def evaluate(df: pd.DataFrame, horizon: int, lookback: int, threshold: float) -> tuple[int, float, float, float]:
    mfe: list[float] = []
    mae: list[float] = []
    for i in range(lookback, len(df) - horizon):
        o, h, l, c = map(float, df.loc[i, ["open", "high", "low", "close"]])
        direction = score_direction(o, h, l, c)
        if direction == 0 or c <= 0:
            continue
        prev_high = float(df.iloc[i - lookback:i]["high"].max())
        prev_low = float(df.iloc[i - lookback:i]["low"].min())
        if direction == 1:
            room = (prev_high - c) / c
            fmfe = (float(df.iloc[i + 1:i + horizon + 1]["high"].max()) - c) / c
            fmae = (c - float(df.iloc[i + 1:i + horizon + 1]["low"].min())) / c
        else:
            room = (c - prev_low) / c
            fmfe = (c - float(df.iloc[i + 1:i + horizon + 1]["low"].min())) / c
            fmae = (float(df.iloc[i + 1:i + horizon + 1]["high"].max()) - c) / c
        if room >= threshold:
            mfe.append(fmfe)
            mae.append(fmae)
    if not mfe:
        return 0, 0.0, 0.0, 0.0
    s = pd.Series(mfe)
    a = pd.Series(mae)
    return len(s), float(s.mean()), float(a.mean()), float((s >= 0.0026).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=SYMBOLS)
    ap.add_argument("--lookbacks", default="1,2,3,4,5")
    ap.add_argument("--room-thresholds", default="0.01,0.02")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    lookbacks = [int(x) for x in args.lookbacks.split(",") if x.strip()]
    thresholds = [float(x) for x in args.room_thresholds.split(",") if x.strip()]
    root = Path(args.data_dir)
    loaded = {s: load_symbol(root / f"{s}_1h.csv") for s in symbols}

    print("=" * 88)
    print("PREVIOUS EXTREME FINE LOOKBACK — H1 TO H5")
    print("Lookbacks: " + ", ".join(map(str, lookbacks)) + " | Room: " + ", ".join(f"{x*100:.0f}%" for x in thresholds))
    print("MAX(H) / MIN(L) uses previous candles only; current candle excluded; pooled BOTH signals")
    print("Fee reference = 0.26% round trip")
    print("=" * 88)

    for horizon in range(1, 6):
        print(f"\nH{horizon}")
        print(" LB  ROOM      N    MFE%    MAE%  >=FEE%")
        for lb in lookbacks:
            for th in thresholds:
                parts = [evaluate(loaded[s], horizon, lb, th) for s in symbols]
                n = sum(x[0] for x in parts)
                if n == 0:
                    print(f"{lb:3d} {th*100:5.0f}% {0:7d} {0:8.3f} {0:8.3f} {0:8.1f}")
                    continue
                avg_mfe = sum(x[1] * x[0] for x in parts) / n
                avg_mae = sum(x[2] * x[0] for x in parts) / n
                fee = sum(x[3] * x[0] for x in parts) / n
                print(f"{lb:3d} {th*100:5.0f}% {n:7d} {avg_mfe*100:8.3f} {avg_mae*100:8.3f} {fee*100:8.1f}")


if __name__ == "__main__":
    main()
