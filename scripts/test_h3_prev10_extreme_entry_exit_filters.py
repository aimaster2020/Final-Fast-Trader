from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
FEE = 0.0013
ROUND_TRIP_FEE_PCT = 0.26


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    time_col = next((c for c in ("timestamp", "time", "datetime", "date") if c in df.columns), None)
    if time_col:
        df = df.sort_values(time_col)
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def score_direction(o: float, h: float, l: float, c: float) -> tuple[int, int]:
    body = c - o
    hc = h - c
    ho = h - o
    lc = l - c
    score = int(hc > body) + int(ho < hc) + int(lc > body)
    direction = -1 if score == 0 else 1 if score == 3 else 0
    return score, direction


def add_extreme_features(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    out = df.copy()
    # IMPORTANT: shift(1) makes the reference strictly ex-ante.
    out["prev10_high"] = out["high"].shift(1).rolling(lookback).max()
    out["prev10_low"] = out["low"].shift(1).rolling(lookback).min()
    out["signal_dir"] = [
        score_direction(float(r.open), float(r.high), float(r.low), float(r.close))[1]
        for r in out.itertuples()
    ]

    out["dist_to_prev10_high_pct"] = (out["prev10_high"] / out["close"] - 1.0) * 100.0
    out["dist_to_prev10_low_pct"] = (1.0 - out["prev10_low"] / out["close"]) * 100.0

    out["future3_high"] = out["high"].shift(-1).rolling(3).max().shift(-2)
    out["future3_low"] = out["low"].shift(-1).rolling(3).min().shift(-2)

    out["future3_up_pct"] = (out["future3_high"] / out["close"] - 1.0) * 100.0
    out["future3_down_pct"] = (1.0 - out["future3_low"] / out["close"]) * 100.0

    # Future excursion relative to the previous-10-candle extreme.
    out["up_room_pct"] = (out["prev10_high"] / out["close"] - 1.0) * 100.0
    out["down_room_pct"] = (1.0 - out["prev10_low"] / out["close"]) * 100.0
    return out


def evaluate_direction(
    df: pd.DataFrame,
    direction: int,
    lookback: int,
    min_room: float,
    target_frac: float,
    stop_frac: float,
    fee: float,
    initial_capital: float,
) -> dict[str, float | int]:
    capital = float(initial_capital)
    candidates = 0
    entries = 0
    wins = 0
    gross_sum = 0.0
    net_sum = 0.0

    n = len(df)
    for i in range(lookback, n - 3):
        r = df.iloc[i]
        if int(r.signal_dir) != direction:
            continue
        if not np.isfinite(r.prev10_high) or not np.isfinite(r.prev10_low):
            continue
        candidates += 1

        room = float(r.up_room_pct if direction == 1 else r.down_room_pct)
        if room < min_room:
            continue

        entry = float(r.close)
        if entry <= 0:
            continue

        future_high = float(df.iloc[i + 1 : i + 4]["high"].max())
        future_low = float(df.iloc[i + 1 : i + 4]["low"].min())

        # Target is a fraction of the available room to the prior 10-candle extreme.
        target_pct = room * target_frac
        stop_pct = room * stop_frac

        if direction == 1:
            target = entry * (1.0 + target_pct / 100.0)
            stop = entry * (1.0 - stop_pct / 100.0)
            hit_target = future_high >= target
            hit_stop = future_low <= stop
        else:
            target = entry * (1.0 - target_pct / 100.0)
            stop = entry * (1.0 + stop_pct / 100.0)
            hit_target = future_low <= target
            hit_stop = future_high >= stop

        # Conservative same-horizon labeling: if both are touched, count as stop first
        # because OHLC bars do not tell us the intrabar order.
        if hit_stop:
            exit_price = stop
            outcome = -stop_pct / 100.0
            wins += 0
        elif hit_target:
            exit_price = target
            outcome = target_pct / 100.0
            wins += 1
        else:
            exit_price = float(df.iloc[i + 3].close)
            outcome = (
                (exit_price - entry) / entry
                if direction == 1
                else (entry - exit_price) / entry
            )
            wins += int(outcome > 0)

        gross_sum += outcome
        net = (1.0 + outcome) * (1.0 - fee) ** 2 - 1.0
        net_sum += net
        capital *= max(1.0 + net, 0.0)
        entries += 1

    return {
        "candidates": candidates,
        "entries": entries,
        "win_rate": wins / entries * 100.0 if entries else 0.0,
        "avg_gross": gross_sum / entries * 100.0 if entries else 0.0,
        "avg_net": net_sum / entries * 100.0 if entries else 0.0,
        "final": capital,
        "return": (capital / initial_capital - 1.0) * 100.0,
    }


def descriptive_scan(df: pd.DataFrame, direction: int, thresholds: list[float]) -> None:
    label = "LONG" if direction == 1 else "SHORT"
    base = df[df.signal_dir == direction].copy()
    base = base[base.prev10_high.notna() & base.prev10_low.notna()]
    print(f"{label} | base={len(base)}")
    print(f"{'MIN_ROOM%':>10} {'N':>7} {'AVG_MFE%':>10} {'MED_MFE%':>10} {'WIN%':>8} {'>=FEE%':>9}")
    for threshold in thresholds:
        x = base[base["up_room_pct" if direction == 1 else "down_room_pct"] >= threshold]
        if x.empty:
            continue
        mfe = x["future3_up_pct" if direction == 1 else "future3_down_pct"]
        win = (mfe > 0).mean() * 100.0
        fee_cover = (mfe > ROUND_TRIP_FEE_PCT).mean() * 100.0
        print(
            f"{threshold:10.2f} {len(x):7d} {mfe.mean():10.3f} {mfe.median():10.3f} "
            f"{win:8.1f} {fee_cover:9.1f}"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="H3 previous-10-candle extreme diagnostic and entry/exit filter test")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=FEE)
    ap.add_argument("--room-thresholds", default="0,0.25,0.50,0.75,1.00,1.50,2.00")
    ap.add_argument("--target-frac", type=float, default=0.50)
    ap.add_argument("--stop-frac", type=float, default=0.50)
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    thresholds = [float(x) for x in args.room_thresholds.split(",")]
    loaded = {s: add_extreme_features(load(Path(args.data_dir) / f"{s}_1h.csv"), args.lookback) for s in symbols}

    print("=" * 110)
    print("H3 PREVIOUS-10 EXTREME — ENTRY / EXIT FILTER TEST")
    print("=" * 110)
    print(f"Reference ceiling: MAX(High) of previous {args.lookback} candles")
    print(f"Reference floor:   MIN(Low)  of previous {args.lookback} candles")
    print("Future evaluation: actual maximum High / minimum Low across t+1..t+3")
    print("Reference is strictly ex-ante: current candle is excluded")
    print(f"Fee: {args.fee*100:.2f}% per side | round trip: {args.fee*200:.2f}%")
    print()

    print("DIAGNOSTIC: does available room to the previous extreme separate future H3 excursion?")
    print()
    for symbol in symbols:
        print(f"[{symbol}]")
        for direction in (-1, 1):
            descriptive_scan(loaded[symbol], direction, thresholds)

    print("STRATEGY TEST: enter only when room >= threshold; target/stop are fractions of room")
    print(f"target_frac={args.target_frac:.2f} stop_frac={args.stop_frac:.2f}")
    print()
    for threshold in thresholds:
        print(f"MIN_ROOM = {threshold:.2f}%")
        for mode in ("SHORT", "LONG", "BOTH"):
            dirs = (-1,) if mode == "SHORT" else (1,) if mode == "LONG" else (-1, 1)
            results = [
                evaluate_direction(
                    loaded[s], d, args.lookback, threshold, args.target_frac, args.stop_frac,
                    args.fee, args.initial_capital
                )
                for s in symbols
                for d in dirs
            ]
            entries = sum(int(r["entries"]) for r in results)
            wins = sum(round(float(r["win_rate"]) * int(r["entries"]) / 100.0) for r in results)
            final = sum(float(r["final"]) for r in results)
            initial = args.initial_capital * len(results)
            avg_gross = (
                sum(float(r["avg_gross"]) * int(r["entries"]) for r in results) / entries
                if entries else 0.0
            )
            avg_net = (
                sum(float(r["avg_net"]) * int(r["entries"]) for r in results) / entries
                if entries else 0.0
            )
            print(
                f"  {mode:<5} entries={entries:5d} win={wins/entries*100 if entries else 0:5.1f}% "
                f"avg_gross={avg_gross:+.3f}% avg_net={avg_net:+.3f}% "
                f"final={final:9.2f} return={final/initial*100-100:+7.2f}%"
            )
        print()

    print("NOTE: this is an exploratory diagnostic. The selected thresholds/target/stop are not frozen yet.")
    print("The first decision should come from the diagnostic separation of future H3 max/min, not from PnL alone.")
    print("=" * 110)


if __name__ == "__main__":
    main()
