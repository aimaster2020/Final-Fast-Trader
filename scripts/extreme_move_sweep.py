from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_pattern_trader.models import Candle
from scripts.build_1h_direction_dataset import load_range
from scripts.july_r_composite_walkforward import resample

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_MONTHS = "2026-05,2026-06,2026-07,2026-08"


@dataclass
class Signal:
    symbol: str
    month: str
    signal: str
    idx: int
    entry: float
    future_close: float
    future_ret_pct: float


def month_bounds(month: str) -> tuple[int, int]:
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return int(start.timestamp()), int(end.timestamp())


def load_month_candles(input_dir: Path, symbol: str, month: str) -> list[Candle]:
    start_ts, end_ts = month_bounds(month)
    raw, used = load_range(input_dir, symbol, start_ts, end_ts)
    print(f"{symbol} {month}: raw_1m={len(raw):,} source_files={len(used)}")
    return [Candle(ts, o, h, l, c, v) for ts, o, h, l, c, v in raw]


def close_change_sum(candles: list[Candle], i: int, bars: int) -> float:
    """Sum of absolute close-to-close price changes over the M bars ending at i-1."""
    if bars <= 0 or i < bars + 1:
        return math.inf
    total = 0.0
    for j in range(i - bars, i):
        total += abs(candles[j].close - candles[j - 1].close)
    return total


def signal_extreme(candles: list[Candle], i: int, window: int, move_bars: int, multiplier: float) -> str:
    """
    LONG when the current low is the lowest low of the previous+current window
    and the downside excursion from the previous close is at least
    multiplier * the sum of absolute close changes over move_bars.

    SHORT is the symmetric high condition.
    """
    if i < max(window - 1, move_bars) + 1:
        return ""

    xs = candles[i - window + 1 : i + 1]
    prior_close = candles[i - 1].close
    movement_budget = close_change_sum(candles, i, move_bars) * multiplier
    if not math.isfinite(movement_budget):
        return ""

    lowest = min(c.low for c in xs)
    highest = max(c.high for c in xs)

    is_low_extreme = candles[i].low <= lowest and (prior_close - candles[i].low) >= movement_budget
    is_high_extreme = candles[i].high >= highest and (candles[i].high - prior_close) >= movement_budget

    if is_low_extreme and is_high_extreme:
        return "BOTH"
    if is_low_extreme:
        return "LOW"
    if is_high_extreme:
        return "HIGH"
    return ""


def pct_return(entry: float, exit_price: float, side: int) -> float:
    if entry <= 0:
        return 0.0
    return ((exit_price / entry) - 1.0) * side * 100.0


def evaluate_symbol(symbol: str, month: str, candles: list[Candle], window: int, move_bars: int, multiplier: float, horizon: int) -> list[Signal]:
    out: list[Signal] = []
    start = max(window - 1, move_bars) + 1
    for i in range(start, len(candles) - horizon):
        sig = signal_extreme(candles, i, window, move_bars, multiplier)
        if sig in ("LOW", "HIGH"):
            side = 1 if sig == "LOW" else -1
            entry = candles[i].close
            future = candles[i + horizon].close
            out.append(
                Signal(
                    symbol=symbol,
                    month=month,
                    signal=sig,
                    idx=i,
                    entry=entry,
                    future_close=future,
                    future_ret_pct=pct_return(entry, future, side),
                )
            )
    return out


def summarize(signals: list[Signal]) -> tuple[int, float, float, float]:
    n = len(signals)
    if not n:
        return 0, 0.0, 0.0, 0.0
    wins = sum(s.future_ret_pct > 0 for s in signals)
    avg = sum(s.future_ret_pct for s in signals) / n
    gross_up = sum(s.future_ret_pct for s in signals if s.future_ret_pct > 0)
    gross_down = -sum(s.future_ret_pct for s in signals if s.future_ret_pct < 0)
    pf = gross_up / gross_down if gross_down else (float("inf") if gross_up else 0.0)
    return n, wins / n * 100.0, avg, pf


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep extreme low/high signals over window and movement-memory sizes.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--months", default=DEFAULT_MONTHS)
    ap.add_argument("--timeframe", type=int, default=60)
    ap.add_argument("--windows", default="5,10,15,20,30,40,60")
    ap.add_argument("--move-bars", default="1,2,3,4,5,6,8,10")
    ap.add_argument("--multiplier", type=float, default=1.0)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--output", default="reports/extreme_move_sweep.csv")
    args = ap.parse_args()

    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    move_bars = [int(x) for x in args.move_bars.split(",") if x.strip()]
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    months = [x.strip() for x in args.months.split(",") if x.strip()]
    cache: dict[tuple[str, str], list[Candle]] = {}

    print(
        f"EXTREME_MOVE_SWEEP tf={args.timeframe}m windows={windows} "
        f"move_bars={move_bars} multiplier={args.multiplier:.2f} horizon={args.horizon}"
    )
    print("LOW = current low is window minimum + downside excursion >= multiplier * sum(abs(close changes), M bars)")
    print("HIGH = current high is window maximum + upside excursion >= multiplier * sum(abs(close changes), M bars)")

    results = []
    for month in months:
        for symbol in symbols:
            candles = resample(load_month_candles(Path(args.input_dir), symbol, month), args.timeframe)
            cache[(symbol, month)] = candles

    for window in windows:
        for bars in move_bars:
            all_signals: list[Signal] = []
            for (symbol, month), candles in cache.items():
                all_signals.extend(evaluate_symbol(symbol, month, candles, window, bars, args.multiplier, args.horizon))
            n, win, avg, pf = summarize(all_signals)
            lows = sum(s.signal == "LOW" for s in all_signals)
            highs = sum(s.signal == "HIGH" for s in all_signals)
            results.append({
                "window": window,
                "move_bars": bars,
                "signals": n,
                "low_signals": lows,
                "high_signals": highs,
                "win_rate_pct": win,
                "avg_forward_return_pct": avg,
                "profit_factor": pf,
            })
            print(
                f"W={window:2d} M={bars:2d} T={n:5d} LOW={lows:5d} HIGH={highs:5d} "
                f"WIN={win:5.1f}% AVG={avg:+.4f}% PF={pf:.2f}"
            )

    results.sort(key=lambda r: (r["avg_forward_return_pct"], r["profit_factor"]), reverse=True)
    print("TOP10")
    for r in results[:10]:
        print(
            f"W={r['window']:2d} M={r['move_bars']:2d} T={r['signals']:5d} "
            f"WIN={r['win_rate_pct']:5.1f}% AVG={r['avg_forward_return_pct']:+.4f}% PF={r['profit_factor']:.2f}"
        )

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        fields = list(results[0].keys()) if results else ["window", "move_bars", "signals", "low_signals", "high_signals", "win_rate_pct", "avg_forward_return_pct", "profit_factor"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"SAVED {out.relative_to(ROOT)} rows={len(results)}")


if __name__ == "__main__":
    main()
