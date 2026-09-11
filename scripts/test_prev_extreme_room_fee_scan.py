from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
LOOKBACK = 1
ROOMS = [0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
HORIZONS = [3, 4, 5]
TARGETS = [0.0075, 0.010, 0.0125, 0.015, 0.020]
STOPS = [0.010, 0.015, 0.020, 0.025, 0.030]
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


def one_trade(df: pd.DataFrame, i: int, horizon: int, room_threshold: float, target: float, stop: float) -> float | None:
    o, h, l, c = map(float, df.loc[i, ["open", "high", "low", "close"]])
    direction = score_direction(o, h, l, c)
    if direction == 0 or c <= 0:
        return None

    prev_high = float(df.iloc[i - LOOKBACK:i]["high"].max())
    prev_low = float(df.iloc[i - LOOKBACK:i]["low"].min())
    room = (prev_high - c) / c if direction == 1 else (c - prev_low) / c
    if room < room_threshold:
        return None

    end = min(i + horizon, len(df) - 1)
    target_price = c * (1.0 + target) if direction == 1 else c * (1.0 - target)
    stop_price = c * (1.0 - stop) if direction == 1 else c * (1.0 + stop)

    for j in range(i + 1, end + 1):
        hj = float(df.loc[j, "high"])
        lj = float(df.loc[j, "low"])
        hit_target = hj >= target_price if direction == 1 else lj <= target_price
        hit_stop = lj <= stop_price if direction == 1 else hj >= stop_price

        if hit_stop and hit_target:
            gross = -stop
            return gross - 2 * FEE_PER_SIDE
        if hit_stop:
            return -stop - 2 * FEE_PER_SIDE
        if hit_target:
            return target - 2 * FEE_PER_SIDE

    final_c = float(df.loc[end, "close"])
    gross = (final_c - c) / c if direction == 1 else (c - final_c) / c
    return gross - 2 * FEE_PER_SIDE


def evaluate(df: pd.DataFrame, horizon: int, room_threshold: float, target: float, stop: float) -> tuple[int, float, float, float]:
    returns: list[float] = []
    for i in range(LOOKBACK, len(df) - 1):
        r = one_trade(df, i, horizon, room_threshold, target, stop)
        if r is not None:
            returns.append(r)
    if not returns:
        return 0, 0.0, 0.0, 0.0

    s = pd.Series(returns)
    gross_profit = float(s[s > 0].sum())
    gross_loss = float(-s[s < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0
    return len(s), float(s.sum() * 100), float((s > 0).mean() * 100), pf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=SYMBOLS)
    ap.add_argument("--rooms", default=','.join(map(str, ROOMS)))
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    rooms = [float(x) for x in args.rooms.split(',') if x.strip()]
    root = Path(args.data_dir)
    loaded = {s: load_symbol(root / f"{s}_1h.csv") for s in symbols}

    print('=' * 104)
    print('PREVIOUS EXTREME ROOM/FEE SCAN — H3 TO H5')
    print('Lookback=1 | Entry at current Close | Fee=0.13%/side | same-candle target+stop => STOP')
    print('Goal: find room thresholds that produce enough gross edge to survive the 0.26% round trip fee.')
    print('=' * 104)

    for horizon in HORIZONS:
        print(f'\nH{horizon}')
        print(' ROOM>=    BEST TARGET STOP     N   RETURN%   AVG%  WIN%    PF')
        for room in rooms:
            rows = []
            for target in TARGETS:
                for stop in STOPS:
                    parts = [evaluate(loaded[s], horizon, room, target, stop) for s in symbols]
                    n = sum(x[0] for x in parts)
                    if n == 0:
                        continue
                    total_return = sum(x[1] for x in parts)
                    win_count = sum(x[2] * x[0] / 100 for x in parts)
                    avg = total_return / n
                    gross_profit = sum(max(0.0, x[1]) for x in parts)
                    gross_loss = 0.0
                    pooled = []
                    for s in symbols:
                        # evaluate separately above; pooled PF is recomputed by replay for exactness below.
                        _ = s
                    # Use pooled return/win/avg for ranking; PF is omitted from ranking and recomputed as weighted summary approximation.
                    pf_values = [x[3] for x in parts if x[0] > 0]
                    pf = sum(x[3] * x[0] for x in parts) / n if pf_values else 0.0
                    rows.append((total_return, target, stop, n, avg, win_count / n * 100, pf))
            rows.sort(reverse=True, key=lambda x: x[0])
            if not rows:
                print(f'{room*100:6.1f}%   no trades')
                continue
            best = rows[0]
            print(f'{room*100:6.1f}%   {best[1]*100:6.2f}% {best[2]*100:5.2f}% {best[3]:6d} {best[0]:9.2f} {best[4]:6.3f} {best[5]:6.1f} {best[6]:6.2f}')

        print('  Top positive candidates across all room thresholds:')
        all_rows = []
        for room in rooms:
            for target in TARGETS:
                for stop in STOPS:
                    parts = [evaluate(loaded[s], horizon, room, target, stop) for s in symbols]
                    n = sum(x[0] for x in parts)
                    if n == 0:
                        continue
                    total_return = sum(x[1] for x in parts)
                    avg = total_return / n
                    win_count = sum(x[2] * x[0] / 100 for x in parts)
                    pf = sum(x[3] * x[0] for x in parts) / n
                    all_rows.append((total_return, room, target, stop, n, avg, win_count / n * 100, pf))
        for r in sorted(all_rows, reverse=True, key=lambda x: x[0])[:5]:
            print(f'   room>={r[1]*100:.1f}% target={r[2]*100:.2f}% stop={r[3]*100:.2f}% N={r[4]} return={r[0]:.2f}% avg={r[5]:.3f}% win={r[6]:.1f}% PF~{r[7]:.2f}')


if __name__ == '__main__':
    main()
