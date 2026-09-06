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
    """Sum absolute close-to-close changes over the M completed bars before i."""
    if bars <= 0 or i < bars:
        return math.inf
    # IMPORTANT: only completed bars BEFORE the signal candle.
    return sum(abs(candles[j].close - candles[j - 1].close) for j in range(i - bars, i))


def signal_extreme(
    candles: list[Candle],
    i: int,
    window: int,
    move_bars: int,
    multiplier: float,
) -> str:
    """
    LOW:
      - current low is below every low in the previous W completed bars
      - excursion below the previous-window minimum is >= multiplier * movement budget

    HIGH is the exact symmetric condition.

    Movement budget is the sum of absolute close-to-close changes over the M
    completed bars immediately before the signal candle.
    """
    if window <= 0 or move_bars <= 0 or multiplier <= 0:
        return ""
    if i < max(window, move_bars):
        return ""

    previous = candles[i - window : i]
    movement_budget = close_change_sum(candles, i, move_bars) * multiplier
    if not math.isfinite(movement_budget):
        return ""

    prev_low = min(c.low for c in previous)
    prev_high = max(c.high for c in previous)

    downside_excursion = prev_low - candles[i].low
    upside_excursion = candles[i].high - prev_high

    is_low_extreme = candles[i].low < prev_low and downside_excursion >= movement_budget
    is_high_extreme = candles[i].high > prev_high and upside_excursion >= movement_budget

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


def evaluate_symbol(
    symbol: str,
    month: str,
    candles: list[Candle],
    window: int,
    move_bars: int,
    multiplier: float,
    horizon: int,
) -> list[Signal]:
    out: list[Signal] = []
    start = max(window, move_bars)
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
    ap = argparse.ArgumentParser(description="Sweep new extreme low/high signals over window, movement-memory and threshold multiplier.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--months", default=DEFAULT_MONTHS)
    ap.add_argument("--timeframe", type=int, default=60)
    ap.add_argument("--windows", default="5,10,15,20,30,40,60")
    ap.add_argument("--move-bars", default="1,2,3,4,5,6,8,10")
    ap.add_argument("--multipliers", default="0.5,0.75,1.0,1.25,1.5,2.0")
    ap.add_argument("--multiplier", type=float, default=None, help="Legacy single multiplier override")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--output", default="reports/extreme_move_sweep.csv")
    args = ap.parse_args()

    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    move_bars = [int(x) for x in args.move_bars.split(",") if x.strip()]
    multipliers = [args.multiplier] if args.multiplier is not None else [float(x) for x in args.multipliers.split(",") if x.strip()]
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    months = [x.strip() for x in args.months.split(",") if x.strip()]
    cache: dict[tuple[str, str], list[Candle]] = {}

    print(
        f"EXTREME_MOVE_SWEEP tf={args.timeframe}m windows={windows} "
        f"move_bars={move_bars} multipliers={multipliers} horizon={args.horizon}"
    )
    print("LOW = current low breaks previous-window minimum + excursion >= multiplier * sum(abs(close changes), M bars)")
    print("HIGH = current high breaks previous-window maximum + excursion >= multiplier * sum(abs(close changes), M bars)")
    print("Movement sum uses ONLY completed candles before the signal candle.")

    results = []
    for month in months:
        for symbol in symbols:
            candles = resample(load_month_candles(Path(args.input_dir), symbol, month), args.timeframe)
            cache[(symbol, month)] = candles

    for window in windows:
        for bars in move_bars:
            for multiplier in multipliers:
                all_signals: list[Signal] = []
                for (symbol, month), candles in cache.items():
                    all_signals.extend(evaluate_symbol(symbol, month, candles, window, bars, multiplier, args.horizon))
                n, win, avg, pf = summarize(all_signals)
                lows = sum(s.signal == "LOW" for s in all_signals)
                highs = sum(s.signal == "HIGH" for s in all_signals)
                results.append({
                    "window": window,
                    "move_bars": bars,
                    "multiplier": multiplier,
                    "signals": n,
                    "low_signals": lows,
                    "high_signals": highs,
                    "win_rate_pct": win,
                    "avg_forward_return_pct": avg,
                    "profit_factor": pf,
                })
                print(
                    f"W={window:2d} M={bars:2d} X={multiplier:4.2f} T={n:5d} LOW={lows:5d} HIGH={highs:5d} "
                    f"WIN={win:5.1f}% AVG={avg:+.4f}% PF={pf:.2f}"
                )

    results.sort(key=lambda r: (r["avg_forward_return_pct"], r["profit_factor"], r["signals"]), reverse=True)
    print("TOP15")
    for r in results[:15]:
        print(
            f"W={r['window']:2d} M={r['move_bars']:2d} X={r['multiplier']:4.2f} T={r['signals']:5d} "
            f"WIN={r['win_rate_pct']:5.1f}% AVG={r['avg_forward_return_pct']:+.4f}% PF={r['profit_factor']:.2f}"
        )

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        fields = list(results[0].keys()) if results else [
            "window", "move_bars", "multiplier", "signals", "low_signals", "high_signals",
            "win_rate_pct", "avg_forward_return_pct", "profit_factor"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"SAVED {out.relative_to(ROOT)} rows={len(results)}")


if __name__ == "__main__":
    main()
