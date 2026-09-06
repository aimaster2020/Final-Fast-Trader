from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.july_r_composite_walkforward import find_month_file, load_binance, resample

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_MONTHS = "2026-05,2026-06,2026-07,2026-08"


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
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")


def prior_range(candles, i: int, lookback: int):
    if i < lookback:
        return None
    xs = candles[i - lookback:i]
    return max(c.high for c in xs), min(c.low for c in xs)


def signal_breakout_retest(candles, i: int, lookback: int, retest_bars: int) -> int:
    if i < lookback + 1 or i - retest_bars < lookback:
        return 0
    hi, lo = prior_range(candles, i - 1, lookback) or (0.0, 0.0)
    prev = candles[i - 1]
    cur = candles[i]
    # Detect fresh breakout on previous candle; enter on current candle when it
    # retests the broken level and closes back in breakout direction.
    if prev.close > hi and cur.low <= hi and cur.close > hi:
        return 1
    if prev.close < lo and cur.high >= lo and cur.close < lo:
        return -1
    return 0


def signal_failed_breakout(candles, i: int, lookback: int) -> int:
    if i < lookback + 1:
        return 0
    hi, lo = prior_range(candles, i - 1, lookback) or (0.0, 0.0)
    cur = candles[i]
    if cur.high > hi and cur.close < hi:
        return -1
    if cur.low < lo and cur.close > lo:
        return 1
    return 0


def signal_impulse_pullback(candles, i: int, lookback: int, impulse_pct: float, pullback_max: float) -> int:
    if i < 3 or i < lookback:
        return 0
    a, b, cur = candles[i - 2], candles[i - 1], candles[i]
    if a.open <= 0:
        return 0
    impulse = abs(b.close - b.open) / b.open * 100.0
    if impulse < impulse_pct:
        return 0
    # Require a directional impulse followed by a small pullback on current bar.
    if b.close > b.open:
        pullback = max(0.0, b.close - cur.close) / max(1e-12, b.close - b.open)
        if 0.0 <= pullback <= pullback_max and cur.close > cur.open:
            return 1
    elif b.close < b.open:
        pullback = max(0.0, cur.close - b.close) / max(1e-12, b.open - b.close)
        if 0.0 <= pullback <= pullback_max and cur.close < cur.open:
            return -1
    return 0


def signal_for(strategy: str, candles, i: int, args) -> int:
    if strategy == "BREAKOUT_RETEST":
        return signal_breakout_retest(candles, i, args.lookback, args.retest_bars)
    if strategy == "FAILED_BREAKOUT":
        return signal_failed_breakout(candles, i, args.lookback)
    return signal_impulse_pullback(candles, i, args.lookback, args.impulse_pct, args.pullback_max)


def run_symbol(symbol: str, candles, strategy: str, args) -> list[Trade]:
    trades: list[Trade] = []
    capital = args.initial_capital
    position = None
    last_entry = -10**9
    for i in range(args.lookback + 2, len(candles)):
        c = candles[i]
        if position is not None:
            if i - position["entry_idx"] >= args.hold_bars:
                exit_i = i
            else:
                sig = signal_for(strategy, candles, i, args)
                exit_i = i if sig and sig != position["side"] else None
            if exit_i is not None:
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
        if position is None and i < len(candles) - 1:
            sig = signal_for(strategy, candles, i, args)
            if sig and i - last_entry >= args.cooldown_bars:
                position = {"entry_idx": i + 1, "entry": candles[i + 1].open, "side": sig, "ts": candles[i + 1].timestamp}
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
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = -sum(t.pnl for t in trades if t.pnl < 0)
    pf = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0)
    capital = initial + pnl
    return {"trades": len(trades), "wins": wins, "win": wins / len(trades) * 100 if trades else 0.0, "pnl": pnl, "return": pnl / initial * 100 if initial else 0.0, "final": capital, "pf": pf, "fees": sum(t.fee for t in trades)}


def main():
    ap = argparse.ArgumentParser(description="Simple price-action strategy lab: breakout-retest, failed-breakout, impulse-pullback.")
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

    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    months = [x.strip() for x in args.months.split(",") if x.strip()]
    all_trades: list[Trade] = []

    print(f"PRICE_ACTION_LAB tf={args.timeframe}m lookback={args.lookback} hold={args.hold_bars} alloc={args.allocation*100:.0f}% fee={args.fee_roundtrip_pct:.2f}%")
    for month in months:
        print(f"MONTH {month}")
        month_rows = []
        for symbol in symbols:
            path = find_month_file(Path(args.input_dir), symbol, month)
            if not path:
                continue
            raw = load_binance(path)
            candles = resample(raw, args.timeframe)
            if len(candles) < args.lookback + 5:
                continue
            for strategy in ("BREAKOUT_RETEST", "FAILED_BREAKOUT", "IMPULSE_PULLBACK"):
                ts = run_symbol(symbol, candles, strategy, args)
                month_rows.extend(ts)
                all_trades.extend(ts)
        for strategy in ("BREAKOUT_RETEST", "FAILED_BREAKOUT", "IMPULSE_PULLBACK"):
            r = summarize([t for t in month_rows if t.strategy == strategy], args.initial_capital)
            print(f"{strategy:18} T={r['trades']:3d} WIN={r['win']:5.1f}% PNL={r['pnl']:+7.2f} PF={r['pf']:.2f} FEES={r['fees']:.2f}")

    print("TOTAL")
    for strategy in ("BREAKOUT_RETEST", "FAILED_BREAKOUT", "IMPULSE_PULLBACK"):
        ts = [t for t in all_trades if t.strategy == strategy]
        r = summarize(ts, args.initial_capital * len(symbols))
        print(f"{strategy:18} T={r['trades']:4d} WIN={r['win']:5.1f}% PNL={r['pnl']:+8.2f} RET={r['return']:+6.2f}% PF={r['pf']:.2f} FEES={r['fees']:.2f}")

    out = ROOT / "reports" / "price_action_strategy_lab_trades.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["symbol","month","strategy","side","entry_idx","exit_idx","entry","exit","ret_pct","pnl","fee"])
        w.writeheader()
        for t in all_trades:
            w.writerow({"symbol":t.symbol,"month":t.month,"strategy":t.strategy,"side":"LONG" if t.side == 1 else "SHORT","entry_idx":t.entry_idx,"exit_idx":t.exit_idx,"entry":t.entry,"exit":t.exit,"ret_pct":t.ret_pct,"pnl":t.pnl,"fee":t.fee})
    print(f"SAVED {out.relative_to(ROOT)} rows={len(all_trades)}")


if __name__ == "__main__":
    main()
