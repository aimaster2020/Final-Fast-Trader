from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
LOOKBACK = 1
ROOM_THRESHOLD = 0.02
TARGETS = [0.003, 0.005, 0.0075, 0.010]
STOPS = [0.005, 0.0075, 0.010, 0.015]
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


def one_trade(
    df: pd.DataFrame,
    i: int,
    horizon: int,
    target: float,
    stop: float,
) -> tuple[float, str] | None:
    o, h, l, c = map(float, df.loc[i, ["open", "high", "low", "close"]])
    direction = score_direction(o, h, l, c)
    if direction == 0 or c <= 0:
        return None

    prev_high = float(df.iloc[i - LOOKBACK:i]["high"].max())
    prev_low = float(df.iloc[i - LOOKBACK:i]["low"].min())
    room = (prev_high - c) / c if direction == 1 else (c - prev_low) / c
    if room < ROOM_THRESHOLD:
        return None

    end = min(i + horizon, len(df) - 1)
    for j in range(i + 1, end + 1):
        hj = float(df.loc[j, "high"])
        lj = float(df.loc[j, "low"])
        if direction == 1:
            target_price = c * (1.0 + target)
            stop_price = c * (1.0 - stop)
            hit_target = hj >= target_price
            hit_stop = lj <= stop_price
        else:
            target_price = c * (1.0 - target)
            stop_price = c * (1.0 + stop)
            hit_target = lj <= target_price
            hit_stop = hj >= stop_price

        # Conservative same-candle rule: stop is assumed first.
        if hit_stop and hit_target:
            return -stop - 2 * FEE_PER_SIDE, "STOP"
        if hit_stop:
            return -stop - 2 * FEE_PER_SIDE, "STOP"
        if hit_target:
            return target - 2 * FEE_PER_SIDE, "TARGET"

    final_c = float(df.loc[end, "close"])
    gross = (final_c - c) / c if direction == 1 else (c - final_c) / c
    return gross - 2 * FEE_PER_SIDE, "TIME"


def evaluate(
    df: pd.DataFrame,
    horizon: int,
    target: float,
    stop: float,
) -> tuple[int, float, float, float, float, float, int, int, int]:
    returns: list[float] = []
    target_hits = stop_hits = time_exits = 0
    for i in range(LOOKBACK, len(df) - 1):
        result = one_trade(df, i, horizon, target, stop)
        if result is None:
            continue
        ret, reason = result
        returns.append(ret)
        target_hits += reason == "TARGET"
        stop_hits += reason == "STOP"
        time_exits += reason == "TIME"

    if not returns:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0

    s = pd.Series(returns)
    positive = float(s[s > 0].sum())
    negative = float(s[s < 0].sum())
    return (
        len(s),
        float(s.sum() * 100),
        float((s > 0).mean() * 100),
        float(s.mean() * 100),
        positive * 100,
        abs(negative) * 100,
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

    print("=" * 112)
    print("PREVIOUS EXTREME PATH TEST — H1 TO H5")
    print("Lookback=1 | Room>=2% | Entry at current Close | Fee=0.13%/side")
    print("Target/stop are fixed % from entry; same-candle target+stop => STOP (conservative)")
    print("=" * 112)

    for horizon in range(1, 6):
        print(f"\nH{horizon}")
        print(" TARGET  STOP     N   RETURN%   WIN%  AVG%    PF  TARGET STOP TIME")
        for target in TARGETS:
            for stop in STOPS:
                parts = [evaluate(loaded[s], horizon, target, stop) for s in symbols]
                n = sum(x[0] for x in parts)
                if n == 0:
                    continue
                total_return = sum(x[1] for x in parts)
                wins = sum(x[2] * x[0] / 100 for x in parts)
                avg = sum(x[3] * x[0] for x in parts) / n
                gross_profit = sum(x[4] for x in parts)
                gross_loss = sum(x[5] for x in parts)
                pf = gross_profit / gross_loss if gross_loss > 0 else 0.0
                t_hits = sum(x[6] for x in parts)
                s_hits = sum(x[7] for x in parts)
                tm_hits = sum(x[8] for x in parts)
                print(f" {target*100:5.2f}% {stop*100:5.2f}% {n:6d} {total_return:9.2f} {wins/n*100:6.1f} {avg:6.3f} {pf:5.2f} {t_hits:6d} {s_hits:5d} {tm_hits:4d}")


if __name__ == "__main__":
    main()
