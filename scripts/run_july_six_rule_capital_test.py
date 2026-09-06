from __future__ import annotations

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Independent monthly capital backtest for R1-R16 across timeframes.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--month", default="2026-07")
    ap.add_argument("--timeframes", default="5,15,30,60", help="comma-separated minutes")
    ap.add_argument("--direction", choices=["LONG_ONLY", "SHORT_ONLY", "BOTH"], default="LONG_ONLY")
    ap.add_argument("--initial-capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--trade-allocation", type=float, default=DEFAULT_ALLOCATION)
    ap.add_argument("--fee-per-side", type=float, default=FEE_PER_SIDE)
    ap.add_argument("--max-hold-bars", type=int, default=DEFAULT_MAX_HOLD_BARS)
    args = ap.parse_args()

    path = find_month_file(Path(args.input_dir), args.symbol, args.month)
    if path is None:
        raise SystemExit(f"MISSING {args.symbol} {args.month}")

    raw = load_binance(path)
    timeframes = [int(x.strip()) for x in args.timeframes.split(",") if x.strip()]

    print(
        f"{args.symbol} {args.month} direction={args.direction} "
        f"capital={args.initial_capital:.0f} leverage=1x fee={args.fee_per_side*100:.2f}%/side "
        f"exit={args.max_hold_bars}bars"
    )

    for tf in timeframes:
        candles = resample(raw, tf)
        if not candles:
            print(f"TF={tf}m NO_COMPLETE_CANDLES")
            continue

        print(f"\nTF={tf}m candles={len(candles)}")
        for rule, threshold in TESTS.items():
            s = evaluate_rule(
                candles,
                rule,
                args.direction,
                args.initial_capital,
                args.trade_allocation,
                threshold,
                args.fee_per_side,
                LEVERAGE,
                args.max_hold_bars,
            )
            pf = s["profit_factor"]
            pf_text = f"{pf:.2f}" if pf != float("inf") else "INF"
            print(
                f"{rule} capital={s['final_capital']:.2f} return={s['return_pct']:.2f}% "
                f"trades={s['completed_trades']} win={s['win_rate_pct']:.1f}% "
                f"PF={pf_text} DD={s['max_drawdown_pct']:.2f}% fees={s['commission_paid']:.2f}"
            )


if __name__ == "__main__":
    main()
