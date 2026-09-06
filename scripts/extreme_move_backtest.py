from __future__ import annotations

import argparse
import csv
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
from scripts.prepared_market_loader import load_prepared

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_MONTHS = "2026-05,2026-06,2026-07,2026-08"

@dataclass
class Trade:
    symbol: str
    month: str
    signal: str
    side: int
    signal_idx: int
    entry_idx: int
    exit_idx: int
    entry: float
    exit: float
    gross_ret_pct: float
    fee: float
    pnl: float

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

def movement_budget(candles: list[Candle], i: int, bars: int) -> float:
    if bars <= 0 or i < bars:
        return float("inf")
    return sum(abs(candles[j].close - candles[j - 1].close) for j in range(i - bars, i))

def extreme_signal(candles: list[Candle], i: int, window: int, move_bars: int, multiplier: float) -> str:
    if i < max(window, move_bars):
        return ""
    previous = candles[i - window:i]
    prev_low = min(c.low for c in previous)
    prev_high = max(c.high for c in previous)
    budget = movement_budget(candles, i, move_bars) * multiplier
    down = prev_low - candles[i].low
    up = candles[i].high - prev_high
    low = candles[i].low < prev_low and down >= budget
    high = candles[i].high > prev_high and up >= budget
    if low and high:
        return ""
    if low:
        return "LOW"
    if high:
        return "HIGH"
    return ""

def run_symbol(symbol: str, month: str, candles: list[Candle], window: int, move_bars: int, multiplier: float, horizon: int, capital: float, allocation: float, fee_roundtrip_pct: float) -> tuple[list[Trade], float, float]:
    trades: list[Trade] = []
    peak = capital
    max_dd = 0.0
    active_until = -1
    for i in range(max(window, move_bars), len(candles) - horizon - 1):
        if i <= active_until:
            continue
        sig = extreme_signal(candles, i, window, move_bars, multiplier)
        if sig not in ("LOW", "HIGH"):
            continue
        entry_idx = i + 1
        exit_idx = i + 1 + horizon
        if exit_idx >= len(candles):
            continue
        side = 1 if sig == "LOW" else -1
        entry = candles[entry_idx].open
        exit_price = candles[exit_idx].close
        if entry <= 0:
            continue
        gross_ret = ((exit_price / entry) - 1.0) * side
        notional = capital * allocation
        fee = notional * fee_roundtrip_pct / 100.0
        pnl = notional * gross_ret - fee
        capital += pnl
        peak = max(peak, capital)
        dd = (peak - capital) / peak * 100.0 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        trades.append(Trade(symbol, month, sig, side, i, entry_idx, exit_idx, entry, exit_price, gross_ret * 100.0, fee, pnl))
        active_until = exit_idx
    return trades, capital, max_dd

def summary(trades: list[Trade], initial: float, final: float, max_dd: float) -> dict:
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    gp = sum(t.pnl for t in wins)
    gl = -sum(t.pnl for t in losses)
    pf = gp / gl if gl else (float("inf") if gp else 0.0)
    return {"trades": len(trades), "wins": len(wins), "win_rate": len(wins) / len(trades) * 100 if trades else 0.0, "pnl": final - initial, "return_pct": (final / initial - 1) * 100 if initial else 0.0, "final": final, "pf": pf, "fees": sum(t.fee for t in trades), "max_dd": max_dd}

def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest extreme-low/high reversal signals with next-open entry and fixed horizon exit.")
    ap.add_argument("--input-dir", required=False)
    ap.add_argument("--prepared-file", default=None)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--months", default=DEFAULT_MONTHS)
    ap.add_argument("--timeframe", type=int, default=60)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--move-bars", type=int, default=5)
    ap.add_argument("--multiplier", type=float, default=2.0)
    ap.add_argument("--horizons", default="1,2,3,4,6,12")
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--allocation", type=float, default=0.30)
    ap.add_argument("--fee-roundtrip-pct", type=float, default=0.26)
    ap.add_argument("--output", default="reports/extreme_move_backtest_trades.csv")
    args = ap.parse_args()

    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    months = [x.strip() for x in args.months.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    if args.prepared_file:
        prepared = load_prepared(Path(args.prepared_file))
        cache = {(symbol, month): prepared.get((symbol, month), []) for symbol in symbols for month in months}
        source_mode = "prepared"
    elif args.input_dir:
        cache = {}
        input_dir = Path(args.input_dir)
        for month in months:
            for symbol in symbols:
                cache[(symbol, month)] = resample(load_month_candles(input_dir, symbol, month), args.timeframe)
        source_mode = "archives"
    else:
        ap.error("provide --prepared-file or --input-dir")
        return

    print(f"EXTREME_MOVE_BACKTEST source={source_mode} tf={args.timeframe}m W={args.window} M={args.move_bars} X={args.multiplier:.2f} alloc={args.allocation:.0%} fee={args.fee_roundtrip_pct:.2f}%")
    print("LOW -> LONG | HIGH -> SHORT | entry=next candle open | exit=close after H bars | one active trade per symbol")

    all_rows: list[dict] = []
    for horizon in horizons:
        all_trades: list[Trade] = []
        symbol_finals = []
        max_dd = 0.0
        for symbol in symbols:
            symbol_trades: list[Trade] = []
            symbol_final = args.initial_capital
            symbol_dd = 0.0
            for month in months:
                candles = cache[(symbol, month)]
                if not candles:
                    continue
                trades, symbol_final, dd = run_symbol(symbol, month, candles, args.window, args.move_bars, args.multiplier, horizon, symbol_final, args.allocation, args.fee_roundtrip_pct)
                symbol_trades.extend(trades)
                all_trades.extend(trades)
                symbol_dd = max(symbol_dd, dd)
            symbol_finals.append(symbol_final)
            max_dd = max(max_dd, symbol_dd)
            r = summary(symbol_trades, args.initial_capital, symbol_final, symbol_dd)
            print(f"H={horizon:2d} {symbol:7} T={r['trades']:3d} WIN={r['win_rate']:5.1f}% PNL={r['pnl']:+8.2f} FINAL={r['final']:8.2f} PF={r['pf']:.2f} DD={r['max_dd']:.2f}% FEES={r['fees']:.2f}")
        combined_initial = args.initial_capital * len(symbols)
        combined_final = sum(symbol_finals)
        r = summary(all_trades, combined_initial, combined_final, max_dd)
        print(f"H={horizon:2d} TOTAL   T={r['trades']:3d} WIN={r['win_rate']:5.1f}% PNL={r['pnl']:+8.2f} RET={r['return_pct']:+6.2f}% FINAL={r['final']:8.2f} PF={r['pf']:.2f} DD={r['max_dd']:.2f}% FEES={r['fees']:.2f}")
        print("-")
        for t in all_trades:
            all_rows.append({"horizon": horizon, "symbol": t.symbol, "month": t.month, "signal": t.signal, "side": "LONG" if t.side == 1 else "SHORT", "signal_idx": t.signal_idx, "entry_idx": t.entry_idx, "exit_idx": t.exit_idx, "entry": t.entry, "exit": t.exit, "gross_ret_pct": t.gross_ret_pct, "fee": t.fee, "pnl": t.pnl})

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["horizon", "symbol", "month", "signal", "side", "signal_idx", "entry_idx", "exit_idx", "entry", "exit", "gross_ret_pct", "fee", "pnl"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"SAVED {out.relative_to(ROOT)} rows={len(all_rows)}")

if __name__ == "__main__":
    main()
