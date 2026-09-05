from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fast_pattern_trader.models import Candle

FEE_PER_SIDE = 0.013          # 1.3% of leveraged notional per side
LEVERAGE = 10.0
DEFAULT_ALLOCATION = 0.10     # 10% of currently free capital as margin
DEFAULT_CAPITAL = 1000.0


def resample(candles: list[Candle], minutes: int) -> list[Candle]:
    if minutes == 1:
        return candles
    bucket = minutes * 60
    groups: dict[int, list[Candle]] = {}
    for c in candles:
        groups.setdefault((c.timestamp // bucket) * bucket, []).append(c)
    out: list[Candle] = []
    for key, group in sorted(groups.items()):
        group = sorted(group, key=lambda x: x.timestamp)
        ids = sorted({x.timestamp // 60 for x in group})
        if len(ids) != minutes or ids[-1] - ids[0] + 1 != minutes:
            continue
        out.append(Candle(key, group[0].open, max(x.high for x in group), min(x.low for x in group), group[-1].close))
    return out


def load_binance(path: Path) -> list[Candle]:
    rows: list[Candle] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) < 6:
                continue
            try:
                ts = int(float(r[0]))
                if ts > 10_000_000_000_000:       # microseconds
                    ts //= 1_000_000
                elif ts > 10_000_000_000:         # milliseconds
                    ts //= 1_000
                rows.append(Candle(ts, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
            except (ValueError, TypeError):
                continue
    return sorted(rows, key=lambda x: x.timestamp)


def rule_components(c: Candle) -> dict[str, float]:
    from fast_pattern_trader.ohlc_rule_strategy import (
        evaluate_body_position,
        evaluate_body_strength,
        evaluate_close_behavior,
        evaluate_open_behavior,
        evaluate_range,
        evaluate_r16,
        evaluate_rules,
        evaluate_wick_behavior,
    )

    v = evaluate_rules(c)
    r = evaluate_range(c)
    ob = evaluate_open_behavior(c)
    cb = evaluate_close_behavior(c)
    wb = evaluate_wick_behavior(c)
    bp = evaluate_body_position(c)
    bs = evaluate_body_strength(c)
    r16 = evaluate_r16(c)
    return {
        "R1": float(v.close_gt_open),
        "R2": float(v.open_gt_close),
        "R3": float(v.close_eq_high),
        "R4": float(v.close_eq_low),
        "R5": float(r.directional_vote),
        "R6": float(ob.open_high_similarity),
        "R7": float(-ob.open_low_similarity),
        "R8": float(cb.close_high_similarity),
        "R9": float(-cb.close_low_similarity),
        "R10": float(wb.lower_wick_strength),
        "R11": float(-wb.upper_wick_strength),
        "R12": float(bp.bullish_position),
        "R13": float(-bp.bearish_position),
        "R14": float(bs.bullish_strength),
        "R15": float(-bs.bearish_strength),
        "R16": float(int(r16.direction)),
    }


def signal_for_rule(c: Candle, rule: str, threshold: float) -> int:
    value = rule_components(c)[rule]
    return 1 if value >= threshold else -1 if value <= -threshold else 0


def _close_position(p: dict, price: float, capital: float, fee: float, leverage: float):
    gross = (price / p["entry"] - 1.0) if p["side"] == 1 else (p["entry"] / price - 1.0)
    pnl = p["margin"] * leverage * gross
    close_fee = p["notional"] * fee
    # Return reserved margin + leveraged PnL - closing commission.
    capital += p["margin"] + pnl - close_fee
    return capital, (pnl - close_fee) / p["margin"] if p["margin"] else 0.0, close_fee


def evaluate_rule(candles: list[Candle], rule: str, initial_capital: float, allocation: float, threshold: float, fee: float, leverage: float):
    free_capital = initial_capital
    positions: list[dict] = []
    trade_returns: list[float] = []
    fees = 0.0
    peak = initial_capital
    max_dd = 0.0
    last_signal = 0
    signals = buys = sells = holds = 0

    for c in candles:
        sig = signal_for_rule(c, rule, threshold)
        if sig:
            signals += 1
            if sig == 1:
                buys += 1
            else:
                sells += 1
        else:
            holds += 1

        # Opposite signal closes every currently open position, then reverses.
        if sig and last_signal and sig != last_signal and positions:
            for p in positions:
                free_capital, tr, cf = _close_position(p, c.close, free_capital, fee, leverage)
                trade_returns.append(tr)
                fees += cf
            positions = []

        # Each same-side signal adds a new 10%-of-free-capital margin position.
        # There is deliberately no max-position parameter: free capital is the constraint.
        if sig and free_capital > 0:
            margin = free_capital * allocation
            notional = margin * leverage
            entry_fee = notional * fee
            if margin + entry_fee <= free_capital:
                free_capital -= margin + entry_fee
                fees += entry_fee
                positions.append({
                    "side": sig,
                    "entry": c.close,
                    "margin": margin,
                    "notional": notional,
                    "opened_at": c.timestamp,
                })
            last_signal = sig

        equity = free_capital
        for p in positions:
            gross = (c.close / p["entry"] - 1.0) if p["side"] == 1 else (p["entry"] / c.close - 1.0)
            equity += p["margin"] + p["margin"] * leverage * gross
        peak = max(peak, equity)
        if peak:
            max_dd = max(max_dd, (peak - equity) / peak * 100.0)

    # Force-close all positions at the last candle so the monthly result is realized.
    for p in positions:
        free_capital, tr, cf = _close_position(p, candles[-1].close, free_capital, fee, leverage)
        trade_returns.append(tr)
        fees += cf

    final_capital = free_capital
    wins = sum(x > 0 for x in trade_returns)
    days = max((candles[-1].timestamp - candles[0].timestamp) / 86400.0, 1 / 24)
    return {
        "rule": rule,
        "candles": len(candles),
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "return_pct": (final_capital / initial_capital - 1.0) * 100.0,
        "signals": signals,
        "buy_signals": buys,
        "sell_signals": sells,
        "hold_candles": holds,
        "completed_trades": len(trade_returns),
        "wins": wins,
        "losses": len(trade_returns) - wins,
        "win_rate_pct": wins / len(trade_returns) * 100.0 if trade_returns else 0.0,
        "commission_paid": fees,
        "max_drawdown_pct": max_dd,
        "calendar_days": days,
    }


def find_month_file(input_dir: Path, symbol: str, month: str) -> Path | None:
    exact = [
        input_dir / f"{symbol}-1m-{month}.csv",
        input_dir / f"{symbol}_1m_{month}.csv",
        input_dir / f"{symbol}-{month}.csv",
    ]
    for p in exact:
        if p.exists():
            return p
    found = list(input_dir.rglob(f"*{symbol}*{month}*.csv"))
    return found[0] if found else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Isolate R1-R16 one at a time on one monthly Binance dataset.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--month", required=True, help="YYYY-MM")
    ap.add_argument("--timeframes", default="5,30")
    ap.add_argument("--initial-capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--trade-allocation", type=float, default=DEFAULT_ALLOCATION)
    ap.add_argument("--leverage", type=float, default=LEVERAGE)
    ap.add_argument("--fee-per-side", type=float, default=FEE_PER_SIDE, help="decimal; 1.3%% = 0.013")
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--output", default="reports/ohlc_rule_isolation_monthly.csv")
    args = ap.parse_args()

    rows: list[dict] = []
    for symbol in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        path = find_month_file(Path(args.input_dir), symbol, args.month)
        if path is None:
            print(f"MISSING {symbol} {args.month}")
            continue
        raw = load_binance(path)
        print(f"{symbol} {args.month}: loaded {len(raw):,} 1m candles from {path.name}")
        for tf in [int(x) for x in args.timeframes.split(",")]:
            candles = resample(raw, tf)
            if not candles:
                print(f"{symbol} {tf}m: no complete candles")
                continue
            for i in range(1, 17):
                rule = f"R{i}"
                s = evaluate_rule(candles, rule, args.initial_capital, args.trade_allocation, args.threshold, args.fee_per_side, args.leverage)
                s.update({
                    "symbol": symbol,
                    "month": args.month,
                    "timeframe_min": tf,
                    "fee_per_side_pct": args.fee_per_side * 100.0,
                    "allocation_pct": args.trade_allocation * 100.0,
                    "leverage": args.leverage,
                    "start_utc": candles[0].timestamp,
                    "end_utc": candles[-1].timestamp,
                })
                rows.append(s)
                print(f"{symbol} {tf}m {rule}: return={s['return_pct']:.2f}% trades={s['completed_trades']} win={s['win_rate_pct']:.1f}% DD={s['max_drawdown_pct']:.2f}% fees={s['commission_paid']:.2f}")

    if not rows:
        raise SystemExit("No input data found")
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ranked = sorted(rows, key=lambda x: (float(x["return_pct"]), float(x["win_rate_pct"]), -float(x["max_drawdown_pct"])), reverse=True)
    rank_path = p.with_name(p.stem + "_ranked.csv")
    with rank_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ranked)
    print(f"Saved: {p}")
    print(f"Saved ranked: {rank_path}")


if __name__ == "__main__":
    main()
