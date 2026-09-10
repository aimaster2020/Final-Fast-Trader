from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LOOKBACKS = [3, 5, 8, 10, 15, 20, 30, 50]
SIGNALS = (-1, 1)  # SHORT, LONG


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


def evaluate(df: pd.DataFrame, horizon: int, lookback: int, mode: str, room_threshold: float) -> dict[str, float | int]:
    rows: list[tuple[float, float]] = []
    start = lookback
    end = len(df) - horizon
    for i in range(start, end):
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

        if room < room_threshold:
            continue
        rows.append((future_mfe, future_mae))

    if not rows:
        return {"n": 0, "avg_mfe": 0.0, "med_mfe": 0.0, "fee": 0.0, "reach": 0.0, "avg_mae": 0.0}

    s = pd.DataFrame(rows, columns=["mfe", "mae"])
    # Fee reference = 0.26% round trip.
    return {
        "n": int(len(s)),
        "avg_mfe": float(s.mfe.mean()),
        "med_mfe": float(s.mfe.median()),
        "fee": float((s.mfe >= 0.0026).mean()),
        "reach": float((s.mfe >= 0.5 * room_threshold).mean()) if room_threshold > 0 else 0.0,
        "avg_mae": float(s.mae.mean()),
    }


def pooled(loaded: dict[str, pd.DataFrame], symbols: list[str], horizon: int, lookback: int, mode: str, room_threshold: float) -> dict[str, float | int]:
    rs = [evaluate(loaded[s], horizon, lookback, mode, room_threshold) for s in symbols]
    n = sum(int(r["n"]) for r in rs)
    if n == 0:
        return {"n": 0, "avg_mfe": 0.0, "med_mfe": 0.0, "fee": 0.0, "reach": 0.0, "avg_mae": 0.0}
    return {
        "n": n,
        "avg_mfe": sum(float(r["avg_mfe"]) * int(r["n"]) for r in rs) / n,
        "med_mfe": sum(float(r["med_mfe"]) * int(r["n"]) for r in rs) / n,
        "fee": sum(float(r["fee"]) * int(r["n"]) for r in rs) / n,
        "reach": sum(float(r["reach"]) * int(r["n"]) for r in rs) / n,
        "avg_mae": sum(float(r["avg_mae"]) * int(r["n"]) for r in rs) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan previous-extreme lookback lengths over H1-H5.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--lookbacks", default=",".join(map(str, LOOKBACKS)))
    ap.add_argument("--room-thresholds", default="0,0.005,0.01,0.015,0.02")
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    lookbacks = [int(x) for x in args.lookbacks.split(",") if x.strip()]
    thresholds = [float(x) for x in args.room_thresholds.split(",") if x.strip()]
    root = Path(args.data_dir)
    loaded = {s: load_symbol(root / f"{s}_1h.csv") for s in symbols}

    print("=" * 118)
    print("PREVIOUS EXTREME LOOKBACK SCAN — H1 TO H5")
    print("Reference high = MAX(High) of previous N candles; reference low = MIN(Low) of previous N candles")
    print("Current candle excluded. Signal = real Score 0 SHORT / Score 3 LONG. Fee reference = 0.26% round trip.")
    print()

    for horizon in range(1, 6):
        print(f"{'=' * 45} H{horizon} {'=' * 45}")
        for mode in ("SHORT", "LONG", "BOTH"):
            print(mode)
            print(f"{'LOOKBACK':>8}{'ROOM':>8}{'N':>7}{'AVG_MFE%':>11}{'MED_MFE%':>11}{'>=FEE%':>9}{'AVG_MAE%':>11}")
            for lookback in lookbacks:
                for room in thresholds:
                    r = pooled(loaded, symbols, horizon, lookback, mode, room)
                    print(
                        f"{lookback:8d}{room*100:7.2f}%{int(r['n']):7d}"
                        f"{float(r['avg_mfe'])*100:11.3f}{float(r['med_mfe'])*100:11.3f}"
                        f"{float(r['fee'])*100:9.1f}{float(r['avg_mae'])*100:11.3f}"
                    )
                print()
            print()

    print("Interpretation: use this scan to choose a stable lookback, not the single best in-sample row.")
    print("A useful lookback should show a consistent relationship between room size and future MFE across H1-H5.")


if __name__ == "__main__":
    main()
