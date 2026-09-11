from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
LOOKBACK = 1
HORIZON = 5
ROOM_THRESHOLD = 0.035
TARGET = 0.02
STOP = 0.03
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


def trade_result(df: pd.DataFrame, i: int) -> tuple[float, str] | None:
    o, h, l, c = map(float, df.loc[i, ["open", "high", "low", "close"]])
    direction = score_direction(o, h, l, c)
    if direction == 0 or c <= 0:
        return None

    prev_high = float(df.iloc[i - LOOKBACK:i]["high"].max())
    prev_low = float(df.iloc[i - LOOKBACK:i]["low"].min())
    room = (prev_high - c) / c if direction == 1 else (c - prev_low) / c
    if room < ROOM_THRESHOLD:
        return None

    end = min(i + HORIZON, len(df) - 1)
    target_price = c * (1.0 + TARGET) if direction == 1 else c * (1.0 - TARGET)
    stop_price = c * (1.0 - STOP) if direction == 1 else c * (1.0 + STOP)

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
            return -STOP - 2 * FEE_PER_SIDE, "STOP"
        if hit_stop:
            return -STOP - 2 * FEE_PER_SIDE, "STOP"
        if hit_target:
            return TARGET - 2 * FEE_PER_SIDE, "TARGET"

    final_c = float(df.loc[end, "close"])
    gross = (final_c - c) / c if direction == 1 else (c - final_c) / c
    return gross - 2 * FEE_PER_SIDE, "TIME"


def run_oos(df: pd.DataFrame, split: float, initial_capital: float) -> tuple[float, int, float, int, int, int, int, float, int]:
    n = len(df)
    start = max(LOOKBACK, int(n * (1.0 - split)))
    capital = initial_capital
    peak = capital
    max_dd = 0.0
    trades = wins = targets = stops = times = 0
    i = start

    while i < n - 1:
        result = trade_result(df, i)
        if result is None:
            i += 1
            continue
        ret, reason = result
        capital *= 1.0 + ret
        trades += 1
        wins += ret > 0
        targets += reason == "TARGET"
        stops += reason == "STOP"
        times += reason == "TIME"
        peak = max(peak, capital)
        dd = (peak - capital) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        i += HORIZON + 1

    return capital, trades, (wins / trades * 100 if trades else 0.0), targets, stops, times, start, max_dd * 100, n - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--oos-fraction", type=float, default=0.30)
    args = ap.parse_args()

    if not 0 < args.oos_fraction < 1:
        raise SystemExit("--oos-fraction must be between 0 and 1")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    root = Path(args.data_dir)

    print("=" * 108)
    print("FIXED-PARAMETER H5 OUT-OF-SAMPLE BACKTEST")
    print("H5 | Lookback=1 | Room>=3.5% | Target=2% | Stop=3% | Fee=0.13%/side")
    print(f"OOS = last {args.oos_fraction*100:.0f}% of each symbol; parameters are not re-selected")
    print("One position/symbol; same-candle target+stop => STOP")
    print("=" * 108)

    pooled_initial = args.initial_capital * len(symbols)
    pooled_final = 0.0
    pooled_trades = 0
    pooled_wins = 0.0
    pooled_targets = pooled_stops = pooled_times = 0
    pooled_dd = 0.0

    for symbol in symbols:
        df = load_symbol(root / f"{symbol}_1h.csv")
        capital, trades, win, targets, stops, times, start, dd, oos_n = run_oos(df, args.oos_fraction, args.initial_capital)
        pooled_final += capital
        pooled_trades += trades
        pooled_wins += trades * win / 100.0
        pooled_targets += targets
        pooled_stops += stops
        pooled_times += times
        pooled_dd = max(pooled_dd, dd)
        oos_return = (capital / args.initial_capital - 1.0) * 100
        print(f"{symbol}: OOS rows={oos_n:4d} capital=${capital:,.2f} return={oos_return:7.2f}% trades={trades:4d} win={win:5.1f}% target={targets:3d} stop={stops:3d} time={times:3d} DD={dd:6.2f}%")

    pooled_return = (pooled_final / pooled_initial - 1.0) * 100 if pooled_initial else 0.0
    pooled_win = pooled_wins / pooled_trades * 100 if pooled_trades else 0.0
    print("-" * 108)
    print(f"POOLED OOS: ${pooled_final:,.2f} from ${pooled_initial:,.2f} return={pooled_return:7.2f}% trades={pooled_trades:4d} win={pooled_win:5.1f}% target={pooled_targets:3d} stop={pooled_stops:3d} time={pooled_times:3d} max_symbol_DD={pooled_dd:6.2f}%")


if __name__ == "__main__":
    main()
