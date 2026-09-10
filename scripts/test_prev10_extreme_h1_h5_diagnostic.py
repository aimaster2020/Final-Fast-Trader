from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
THRESHOLDS = (0.0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00)


def load_symbol(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"Missing OHLC column {c} in {path}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def score_direction(row: pd.Series) -> int:
    o, h, l, c = map(float, row[["open", "high", "low", "close"]])
    body = c - o
    hc = h - c
    ho = h - o
    lc = l - c
    score = int(hc > body) + int(ho < hc) + int(lc > body)
    return -1 if score == 0 else 1 if score == 3 else 0


def collect_rows(df: pd.DataFrame, horizon: int, lookback: int, direction: int) -> pd.DataFrame:
    rows = []
    for i in range(lookback, len(df) - horizon):
        signal = score_direction(df.iloc[i])
        if signal != direction:
            continue

        close = float(df.iloc[i].close)
        if close <= 0:
            continue

        prev_high = float(df.iloc[i - lookback:i].high.max())
        prev_low = float(df.iloc[i - lookback:i].low.min())
        future_high = float(df.iloc[i + 1:i + horizon + 1].high.max())
        future_low = float(df.iloc[i + 1:i + horizon + 1].low.min())

        if direction == 1:
            room = (prev_high - close) / close * 100.0
            mfe = (future_high - close) / close * 100.0
            adverse = (close - future_low) / close * 100.0
        else:
            room = (close - prev_low) / close * 100.0
            mfe = (close - future_low) / close * 100.0
            adverse = (future_high - close) / close * 100.0

        rows.append(
            {
                "room": room,
                "mfe": mfe,
                "adverse": adverse,
                "reaches_room": mfe >= max(room, 0.0),
                "mfe_ge_fee": mfe >= 0.26,
            }
        )
    return pd.DataFrame(rows)


def summarize(s: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    x = s[s["room"] >= threshold]
    if x.empty:
        return {"n": 0, "avg_mfe": 0.0, "med_mfe": 0.0, "win": 0.0, "fee": 0.0, "reach": 0.0, "avg_adv": 0.0}
    return {
        "n": len(x),
        "avg_mfe": float(x.mfe.mean()),
        "med_mfe": float(x.mfe.median()),
        "win": float((x.mfe > 0).mean() * 100.0),
        "fee": float((x.mfe >= 0.26).mean() * 100.0),
        "reach": float(x.reaches_room.mean() * 100.0),
        "avg_adv": float(x.adverse.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Previous-10 extreme diagnostic across H1-H5.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--lookback", type=int, default=10)
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    root = Path(args.data_dir)
    loaded = {s: load_symbol(root / f"{s}_1h.csv") for s in symbols}

    print("=" * 118)
    print("PREVIOUS-10 EXTREME DIAGNOSTIC — H1 TO H5")
    print("=" * 118)
    print(f"Reference ceiling = MAX(High) of previous {args.lookback} candles")
    print(f"Reference floor   = MIN(Low)  of previous {args.lookback} candles")
    print("Current signal: real Score 0=SHORT / Score 3=LONG")
    print("Future MFE/MAE: actual High/Low over the requested horizon")
    print("Fee reference: 0.26% round trip")
    print()

    for horizon in range(1, 6):
        print(f"{'=' * 46} H{horizon} {'=' * 46}")
        pooled = {"SHORT": [], "LONG": []}

        for symbol, df in loaded.items():
            for mode, direction in (("SHORT", -1), ("LONG", 1)):
                s = collect_rows(df, horizon, args.lookback, direction)
                pooled[mode].append(s)

        for mode in ("SHORT", "LONG"):
            s = pd.concat(pooled[mode], ignore_index=True) if pooled[mode] else pd.DataFrame()
            print(mode)
            print(f"{'MIN_ROOM%':>10}{'N':>7}{'AVG_MFE%':>11}{'MED_MFE%':>11}{'WIN%':>9}{'>=FEE%':>10}{'REACH_ROOM%':>14}{'AVG_MAE%':>11}")
            for threshold in THRESHOLDS:
                r = summarize(s, threshold)
                print(
                    f"{threshold:10.2f}{int(r['n']):7d}{r['avg_mfe']:11.3f}"
                    f"{r['med_mfe']:11.3f}{r['win']:9.1f}{r['fee']:10.1f}"
                    f"{r['reach']:14.1f}{r['avg_adv']:11.3f}"
                )
            print()

        all_rows = []
        for mode in ("SHORT", "LONG"):
            if pooled[mode]:
                all_rows.extend(pooled[mode])
        all_s = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        print("BOTH")
        print(f"{'MIN_ROOM%':>10}{'N':>7}{'AVG_MFE%':>11}{'MED_MFE%':>11}{'WIN%':>9}{'>=FEE%':>10}{'REACH_ROOM%':>14}{'AVG_MAE%':>11}")
        for threshold in THRESHOLDS:
            r = summarize(all_s, threshold)
            print(
                f"{threshold:10.2f}{int(r['n']):7d}{r['avg_mfe']:11.3f}"
                f"{r['med_mfe']:11.3f}{r['win']:9.1f}{r['fee']:10.1f}"
                f"{r['reach']:14.1f}{r['avg_adv']:11.3f}"
            )
        print()

    print("Interpretation: use this as a diagnostic only. No entry/exit threshold is frozen from this in-sample scan.")
    print("The key question is whether previous-10 room consistently separates future excursion across H1-H5.")


if __name__ == "__main__":
    main()
