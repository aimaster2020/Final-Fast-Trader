from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from ohlc_rule_isolation_monthly import (
    DEFAULT_ALLOCATION,
    DEFAULT_CAPITAL,
    DEFAULT_MAX_HOLD_BARS,
    FEE_PER_SIDE,
    LEVERAGE,
    TESTS,
    evaluate_rule,
    find_month_file,
    load_binance,
    resample,
)


def discover_symbols(input_dir: Path, month: str) -> list[str]:
    symbols: set[str] = set()
    token = month.replace("-", "")
    for p in input_dir.rglob("*.csv"):
        name = p.name.upper().replace("-", "").replace("_", "")
        if token not in name:
            continue
        stem = p.stem.upper()
        parts = stem.replace("_", "-").split("-")
        if parts:
            symbols.add(parts[0])
    return sorted(symbols)


def main() -> None:
    ap = argparse.ArgumentParser(description="R1-R16 cross-asset monthly characterization backtest.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--symbols", default="ALL", help="comma-separated symbols or ALL")
    ap.add_argument("--month", default="2026-07")
    ap.add_argument("--timeframes", default="5,15,30,60")
    ap.add_argument("--directions", default="BOTH,LONG_ONLY,SHORT_ONLY")
    ap.add_argument("--initial-capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--trade-allocation", type=float, default=DEFAULT_ALLOCATION)
    ap.add_argument("--fee-per-side", type=float, default=FEE_PER_SIDE)
    ap.add_argument("--max-hold-bars", type=int, default=DEFAULT_MAX_HOLD_BARS)
    ap.add_argument("--output", default="reports/july_r_characterization.csv")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if args.symbols.strip().upper() == "ALL":
        symbols = discover_symbols(input_dir, args.month)
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    timeframes = [int(x.strip()) for x in args.timeframes.split(",") if x.strip()]
    directions = [x.strip().upper() for x in args.directions.split(",") if x.strip()]

    allowed = {"BOTH", "LONG_ONLY", "SHORT_ONLY"}
    if any(x not in allowed for x in directions):
        raise SystemExit("Invalid --directions")
    if not symbols:
        raise SystemExit(f"No symbols found for {args.month}")

    rows: list[dict] = []
    print(f"MONTH={args.month} assets={len(symbols)} TFs={','.join(map(str,timeframes))} rules=16 dirs={','.join(directions)} fee={args.fee_per_side*100:.2f}% alloc={args.trade_allocation*100:.1f}% hold={args.max_hold_bars}")
    print(f"ASSETS={','.join(symbols)}")

    for symbol in symbols:
        path = find_month_file(input_dir, symbol, args.month)
        if path is None:
            print(f"MISSING {symbol}")
            continue
        raw = load_binance(path)
        for tf in timeframes:
            candles = resample(raw, tf)
            if not candles:
                continue
            for direction in directions:
                for i in range(1, 17):
                    rule = f"R{i}"
                    s = evaluate_rule(candles, rule, direction, args.initial_capital, args.trade_allocation, TESTS[rule], args.fee_per_side, LEVERAGE, args.max_hold_bars)
                    rows.append({
                        "symbol": symbol,
                        "month": args.month,
                        "timeframe_min": tf,
                        "direction": direction,
                        "rule": rule,
                        "return_pct": s["return_pct"],
                        "final_capital": s["final_capital"],
                        "completed_trades": s["completed_trades"],
                        "win_rate_pct": s["win_rate_pct"],
                        "profit_factor": s["profit_factor"],
                        "max_drawdown_pct": s["max_drawdown_pct"],
                        "commission_paid": s["commission_paid"],
                        "signals": s["signals"],
                        "accepted_signals": s["accepted_signals"],
                        "buy_signals": s["buy_signals"],
                        "sell_signals": s["sell_signals"],
                        "time_exits": s["time_exits"],
                        "signal_exits": s["signal_exits"],
                    })

    if not rows:
        raise SystemExit("No input data found")

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\nSUMMARY: R | TF | DIR | MED_RET | AVG_WR | AVG_TRADES | POS_ASSETS | MED_DD")
    grouped: dict[tuple[int, str, str], list[dict]] = {}
    for r in rows:
        grouped.setdefault((int(r["timeframe_min"]), r["direction"], r["rule"]), []).append(r)

    for tf in timeframes:
        for direction in directions:
            for i in range(1, 17):
                rule = f"R{i}"
                group = grouped.get((tf, direction, rule), [])
                if not group:
                    continue
                returns = [float(x["return_pct"]) for x in group]
                wins = [float(x["win_rate_pct"]) for x in group]
                trades = [int(x["completed_trades"]) for x in group]
                dds = [float(x["max_drawdown_pct"]) for x in group]
                positive = sum(x > 0 for x in returns)
                print(f"{rule:>3} | {tf:>2}m | {direction[:5]:<5} | {statistics.median(returns):>8.2f}% | {statistics.mean(wins):>6.1f}% | {statistics.mean(trades):>9.1f} | {positive:>3}/{len(group):<3} | {statistics.median(dds):>6.2f}%")

    print(f"\nSAVED {p} rows={len(rows)}")


if __name__ == "__main__":
    main()
