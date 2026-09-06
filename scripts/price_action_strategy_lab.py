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

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_MONTHS = "2026-05,2026-06,2026-07,2026-08"
STRATEGIES = ("BREAKOUT_RETEST", "FAILED_BREAKOUT", "IMPULSE_PULLBACK")


@dataclass
class Trade:
    symbol: str
    month: str
    strategy: str
    side: int
    entry_idx: int
    exit_idx: int
    entry: float
    exit: float
    ret_pct: float
    pnl: float
    fee: float


def month_of(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")


def month_bounds(month: str) -> tuple[int, int]:
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
    year = start.year + (1 if start.month == 12 else 0)
    nxt_month = 1 if start.month == 12 else start.month + 1
    end = start.replace(year=year, month=nxt_month)
    return int(start.timestamp()), int(end.timestamp())


def load_month_candles(input_dir: Path, symbol: str, month: str) -> list[Candle]:
    start_ts, end_ts = month_bounds(month)
    raw, used = load_range(input_dir, symbol, start_ts, end_ts)
    print(f"{symbol} {month}: raw_1m={len(raw):,} source_files={len(used)}")
    return [Candle(ts, o, h, l, c, v) for ts, o, h, l, c, v in raw]


def prior_range(candles, end_idx: int, lookback: int):
    if end_idx < lookback:
        return None
    xs = candles[end_idx - lookback:end_idx]
    return max(c.high for c in xs), min(c.low for c in xs)


def candle_body_pct(c) -> float:
    return abs(c.close - c.open) / c.open * 100.0 if c.open > 0 else 0.0


def signal_breakout_retest(candles, i: int, lookback: int, retest_bars: int) -> int:
    if i < lookback + 2:
        return 0
    start = max(lookback, i - retest_bars)
    for b in range(i - 1, start - 1, -1):
        level = prior_range(candles, b, lookback)
        if level is None:
            continue
        hi, lo = level
        breakout = candles[b]
        if breakout.close > hi and candles[i].low <= hi and candles[i].close > hi:
            return 1
        if breakout.close < lo and candles[i].high >= lo and candles[i].close < lo:
            return -1
    return 0


def signal_failed_breakout(candles, i: int, lookback: int) -> int:
    if i < lookback + 1:
        return 0
    level = prior_range(candles, i - 1, lookback)
    if level is None:
        return 0
    hi, lo = level
    cur = candles[i]
    if cur.high > hi and cur.close < hi:
        return -1
    if cur.low < lo and cur.close > lo:
        return 1
    return 0


def signal_impulse_pullback(candles, i: int, impulse_pct: float, pullback_max: float) -> int:
    if i < 3:
        return 0
    impulse, pullback, confirm = candles[i - 2], candles[i - 1], candles[i]
    impulse_body = candle_body_pct(impulse)
    if impulse_body < impulse_pct:
        return 0

    if impulse.close > impulse.open:
        impulse_size = impulse.close - impulse.open
        retrace = max(0.0, impulse.close - pullback.close) / impulse_size
        if retrace <= pullback_max and pullback.low <= impulse.close and confirm.close > pullback.high:
            return 1
    elif impulse.close < impulse.open:
        impulse_size = impulse.open - impulse.close
        retrace = max(0.0, pullback.close - impulse.close) / impulse_size
        if retrace <= pullback_max and pullback.high >= impulse.close and confirm.close < pullback.low:
            return -1
    return 0


def signal_for(strategy: str, candles, i: int, args) -> int:
    if strategy == "BREAKOUT_RETEST":
        return signal_breakout_retest(candles, i, args.lookback, args.retest_bars)
    if strategy == "FAILED_BREAKOUT":
        return signal_failed_breakout(candles, i, args.lookback)
    return signal_impulse_pullback(candles, i, args.impulse_pct, args.pullback_max)


def run_symbol(symbol: str, candles, strategy: str, args) -> list[Trade]:
    trades: list[Trade] = []
    capital = args.initial_capital
    position = None
    last_entry = -10**9

    for i in range(args.lookback + 3, len(candles) - 1):
        if position is not None:
            age = i - position["entry_idx"]
            sig = signal_for(strategy, candles, i, args)
            should_exit = age >= args.hold_bars or (sig and sig != position["side"])
            if should_exit:
                exit_i = i
                entry = position["entry"]
                exit_price = candles[exit_i].close
                side = position["side"]
                gross_ret = (exit_price / entry - 1.0) * side
                notional = capital * args.allocation
                fee = notional * args.fee_roundtrip_pct / 100.0
                pnl = notional * gross_ret - fee
                capital += pnl
                trades.append(Trade(symbol, month_of(position["ts"]), strategy, side, position["entry_idx"], exit_i, entry, exit_price, gross_ret * 100.0, pnl, fee))
                position = None

        if position is None:
            sig = signal_for(strategy, candles, i, args)
            if sig and i - last_entry >= args.cooldown_bars:
                position = {
                    "entry_idx": i + 1,
                    "entry": candles[i + 1].open,
                    "side": sig,
                    "ts": candles[i + 1].timestamp,
                }
                last_entry = i + 1

    if position is not None:
        exit_i = len(candles) - 1
        entry = position["entry"]
        exit_price = candles[exit_i].close
        side = position["side"]
        gross_ret = (exit_price / entry - 1.0) * side
        notional = capital * args.allocation
        fee = notional * args.fee_roundtrip_pct / 100.0
        pnl = notional * gross_ret - fee
        capital += pnl
        trades.append(Trade(symbol, month_of(position["ts"]), strategy, side, position["entry_idx"], exit_i, entry, exit_price, gross_ret * 100.0, pnl, fee))

    return trades


def summarize(trades: list[Trade], initial: float) -> dict:
    pnl = sum(t.pnl for t in trades)
    wins = sum(t.pnl > 0 for t in trades)
    gp = sum(t.pnl for t in trades if t.pnl > 0)
    gl = -sum(t.pnl for t in trades if t.pnl < 0)
    pf = gp / gl if gl else (float("inf") if gp else 0.0)
    return {
        "trades": len(trades),
        "win": wins / len(trades) * 100 if trades else 0.0,
        "pnl": pnl,
        "return": pnl / initial * 100 if initial else 0.0,
        "final": initial + pnl,
        "pf": pf,
        "fees": sum(t.fee for t in trades),
    }


def main():
    ap = argparse.ArgumentParser(description="Price-action strategy lab: breakout-retest, failed-breakout, impulse-pullback.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--months", default=DEFAULT_MONTHS)
    ap.add_argument("--timeframe", type=int, default=60)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--retest-bars", type=int, default=3)
    ap.add_argument("--hold-bars", type=int, default=4)
    ap.add_argument("--cooldown-bars", type=int, default=1)
    ap.add_argument("--impulse-pct", type=float, default=0.50)
    ap.add_argument("--pullback-max", type=float, default=0.60)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--allocation", type=float, default=0.30)
    ap.add_argument("--fee-roundtrip-pct", type=float, default=0.26)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    months = [x.strip() for x in args.months.split(",") if x.strip()]
    all_trades: list[Trade] = []

    print(f"PRICE_ACTION_LAB tf={args.timeframe}m lookback={args.lookback} retest={args.retest_bars} hold={args.hold_bars} impulse>={args.impulse_pct:.2f}% pullback<={args.pullback_max:.2f} fee={args.fee_roundtrip_pct:.2f}%")
    for month in months:
        print(f"MONTH {month}")
        month_rows: list[Trade] = []
        for symbol in symbols:
            candles = resample(load_month_candles(input_dir, symbol, month), args.timeframe)
            if len(candles) < args.lookback + 10:
                print(f"INSUFFICIENT {symbol} {month} complete_{args.timeframe}m={len(candles)}")
                continue
            for strategy in STRATEGIES:
                ts = run_symbol(symbol, candles, strategy, args)
                month_rows.extend(ts)
                all_trades.extend(ts)
        for strategy in STRATEGIES:
            r = summarize([t for t in month_rows if t.strategy == strategy], args.initial_capital * len(symbols))
            print(f"{strategy:18} T={r['trades']:3d} WIN={r['win']:5.1f}% PNL={r['pnl']:+7.2f} PF={r['pf']:.2f} FEES={r['fees']:.2f}")

    print("TOTAL")
    base = args.initial_capital * len(symbols) * len(months)
    for strategy in STRATEGIES:
        ts = [t for t in all_trades if t.strategy == strategy]
        r = summarize(ts, base)
        print(f"{strategy:18} T={r['trades']:4d} WIN={r['win']:5.1f}% PNL={r['pnl']:+8.2f} RET={r['return']:+6.2f}% PF={r['pf']:.2f} FEES={r['fees']:.2f}")

    out = ROOT / "reports" / "price_action_strategy_lab_trades.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        fields = ["symbol", "month", "strategy", "side", "entry_idx", "exit_idx", "entry", "exit", "ret_pct", "pnl", "fee"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in all_trades:
            w.writerow({"symbol": t.symbol, "month": t.month, "strategy": t.strategy, "side": "LONG" if t.side == 1 else "SHORT", "entry_idx": t.entry_idx, "exit_idx": t.exit_idx, "entry": t.entry, "exit": t.exit, "ret_pct": t.ret_pct, "pnl": t.pnl, "fee": t.fee})
    print(f"SAVED {out.relative_to(ROOT)} rows={len(all_trades)}")


if __name__ == "__main__":
    main()
