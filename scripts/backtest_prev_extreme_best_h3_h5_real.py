from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
INITIAL_CAPITAL = 1000.0
FEE_PER_SIDE = 0.0013

SETUPS = {
    3: {"room": 0.040, "target": 0.020, "stop": 0.030},
    4: {"room": 0.035, "target": 0.020, "stop": 0.030},
    5: {"room": 0.035, "target": 0.020, "stop": 0.030},
}


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


def run_symbol(df: pd.DataFrame, horizon: int, room_threshold: float, target: float, stop: float, capital: float) -> tuple[float, int, int, int, int, float, float]:
    i = 1
    trades = wins = targets = stops = timeouts = 0
    equity = capital
    max_equity = equity
    max_dd = 0.0

    while i < len(df) - 1:
        o, h, l, c = map(float, df.loc[i, ["open", "high", "low", "close"]])
        direction = score_direction(o, h, l, c)
        if direction == 0 or c <= 0:
            i += 1
            continue

        prev_high = float(df.loc[i - 1, "high"])
        prev_low = float(df.loc[i - 1, "low"])
        room = (prev_high - c) / c if direction == 1 else (c - prev_low) / c
        if room < room_threshold:
            i += 1
            continue

        entry = c
        end = min(i + horizon, len(df) - 1)
        result = None
        exit_idx = end
        for j in range(i + 1, end + 1):
            hj = float(df.loc[j, "high"])
            lj = float(df.loc[j, "low"])
            if direction == 1:
                target_price = entry * (1.0 + target)
                stop_price = entry * (1.0 - stop)
                hit_target = hj >= target_price
                hit_stop = lj <= stop_price
            else:
                target_price = entry * (1.0 - target)
                stop_price = entry * (1.0 + stop)
                hit_target = lj <= target_price
                hit_stop = hj >= stop_price

            # Conservative assumption when both levels are touched in the same candle.
            if hit_stop and hit_target:
                result = -stop
                stops += 1
                exit_idx = j
                break
            if hit_stop:
                result = -stop
                stops += 1
                exit_idx = j
                break
            if hit_target:
                result = target
                targets += 1
                exit_idx = j
                break

        if result is None:
            final_c = float(df.loc[end, "close"])
            result = (final_c - entry) / entry if direction == 1 else (entry - final_c) / entry
            timeouts += 1
            exit_idx = end

        net = result - 2 * FEE_PER_SIDE
        equity *= 1.0 + net
        trades += 1
        wins += int(net > 0)

        max_equity = max(max_equity, equity)
        drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0.0
        max_dd = max(max_dd, drawdown)

        # One position at a time: resume after the exit candle.
        i = exit_idx + 1

    return equity, trades, wins, targets, stops, timeouts, max_dd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    root = Path(args.data_dir)
    loaded = {s: load_symbol(root / f"{s}_1h.csv") for s in symbols}

    print("=" * 108)
    print("REAL CAPITAL BACKTEST — BEST PREVIOUS EXTREME SETUPS")
    print("Direction = exact score_direction | Entry=Close | Fee=0.13%/side | One position/symbol")
    print("Same-candle target+stop => STOP | Fixed capital is compounded trade-to-trade")
    print("=" * 108)

    for horizon, setup in SETUPS.items():
        print(f"\nH{horizon}: Room>={setup['room']*100:.1f}%  Target={setup['target']*100:.1f}%  Stop={setup['stop']*100:.1f}%")
        pooled_start = args.initial_capital * len(symbols)
        pooled_end = 0.0
        pooled_trades = pooled_wins = pooled_targets = pooled_stops = pooled_timeouts = 0
        pooled_dd = 0.0
        for symbol in symbols:
            result = run_symbol(loaded[symbol], horizon, setup["room"], setup["target"], setup["stop"], args.initial_capital)
            equity, trades, wins, targets, stops, timeouts, dd = result
            pooled_end += equity
            pooled_trades += trades
            pooled_wins += wins
            pooled_targets += targets
            pooled_stops += stops
            pooled_timeouts += timeouts
            pooled_dd = max(pooled_dd, dd)
            ret = (equity / args.initial_capital - 1.0) * 100
            print(f"  {symbol}: ${equity:,.2f}  return={ret:8.2f}%  trades={trades:4d}  win={wins/trades*100 if trades else 0:5.1f}%  target={targets:4d} stop={stops:4d} time={timeouts:4d} DD={dd*100:6.2f}%")

        pooled_return = (pooled_end / pooled_start - 1.0) * 100
        pooled_win = pooled_wins / pooled_trades * 100 if pooled_trades else 0.0
        print(f"  POOLED:  ${pooled_end:,.2f} from ${pooled_start:,.2f}  return={pooled_return:8.2f}%  trades={pooled_trades:4d}  win={pooled_win:5.1f}%  target={pooled_targets:4d} stop={pooled_stops:4d} time={pooled_timeouts:4d} max_symbol_DD={pooled_dd*100:6.2f}%")


if __name__ == "__main__":
    main()
