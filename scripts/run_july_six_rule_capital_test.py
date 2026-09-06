from __future__ import annotations

import argparse
from pathlib import Path

from ohlc_rule_isolation_monthly import (
    DEFAULT_ALLOCATION,
    DEFAULT_CAPITAL,
    FEE_PER_SIDE,
    LEVERAGE,
    evaluate_rule,
    find_month_file,
    load_binance,
    resample,
)

TESTS = {
    "R10": 0.50,
    "R12": 0.50,
    "R14": 0.75,
    "R15": 0.75,
    "R6": 0.50,
    "R2": 0.25,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Independent July 2026 capital backtest for six R1-R16 rules.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--month", default="2026-07")
    ap.add_argument("--timeframe", type=int, default=30)
    ap.add_argument("--direction", choices=["LONG_ONLY", "SHORT_ONLY", "BOTH"], default="LONG_ONLY")
    ap.add_argument("--initial-capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--trade-allocation", type=float, default=DEFAULT_ALLOCATION)
    ap.add_argument("--leverage", type=float, default=LEVERAGE)
    ap.add_argument("--fee-per-side", type=float, default=FEE_PER_SIDE)
    args = ap.parse_args()

    path = find_month_file(Path(args.input_dir), args.symbol, args.month)
    if path is None:
        raise SystemExit(f"MISSING {args.symbol} {args.month}")

    candles = resample(load_binance(path), args.timeframe)
    if not candles:
        raise SystemExit("NO_COMPLETE_CANDLES")

    for rule, threshold in TESTS.items():
        s = evaluate_rule(
            candles,
            rule,
            args.direction,
            args.initial_capital,
            args.trade_allocation,
            threshold,
            args.fee_per_side,
            args.leverage,
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
