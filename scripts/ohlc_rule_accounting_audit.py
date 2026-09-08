from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fast_pattern_trader.models import Candle
from ohlc_rule_isolation_monthly import (
    DEFAULT_ALLOCATION,
    DEFAULT_CAPITAL,
    FEE_PER_SIDE,
    LEVERAGE,
    find_month_file,
    load_binance,
    resample,
    signal_for_rule,
)

RULE_ROLE = {
    "R1": "Bullish body: close > open",
    "R2": "Bearish body: open > close",
    "R3": "Close at high",
    "R4": "Close at low",
    "R5": "Range direction",
    "R6": "Open near high",
    "R7": "Open near low",
    "R8": "Close near high",
    "R9": "Close near low",
    "R10": "Lower-wick strength",
    "R11": "Upper-wick strength",
    "R12": "Bullish body position in range",
    "R13": "Bearish body position in range",
    "R14": "Bullish body strength",
    "R15": "Bearish body strength",
    "R16": "Final candle/range direction",
}

RULE_USE = {
    "R1": "body direction / bullish-vs-bearish state",
    "R2": "body direction / bearish-vs-bullish state",
    "R3": "strong close-at-high / buying-pressure confirmation",
    "R4": "strong close-at-low / selling-pressure confirmation",
    "R5": "range direction / broad candle state",
    "R6": "opening-location structure",
    "R7": "opening-location structure",
    "R8": "closing-location strength / buying-pressure context",
    "R9": "closing-location weakness / selling-pressure context",
    "R10": "lower-price rejection / possible buying-pressure context",
    "R11": "upper-price rejection / possible selling-pressure context",
    "R12": "where the bullish body sits inside the range",
    "R13": "where the bearish body sits inside the range",
    "R14": "bullish body strength / impulse magnitude",
    "R15": "bearish body strength / impulse magnitude",
    "R16": "aggregate candle direction state",
}


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    x = sorted(values)
    n = len(x)
    m = n // 2
    return x[m] if n % 2 else (x[m - 1] + x[m]) / 2.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    xs, ys = xs[:n], ys[:n]
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    den = (dx * dy) ** 0.5
    return num / den if den else 0.0


def _forward_metrics(candles: list[Candle], i: int, side: int, horizon: int) -> tuple[float, float, float]:
    if i + horizon >= len(candles):
        return 0.0, 0.0, 0.0
    base = candles[i].close
    if base <= 0:
        return 0.0, 0.0, 0.0
    end = candles[i + horizon].close
    signed_close = ((end / base - 1.0) * 100.0) * side
    highs = [c.high for c in candles[i + 1:i + horizon + 1]]
    lows = [c.low for c in candles[i + 1:i + horizon + 1]]
    favorable = ((max(highs) / base - 1.0) * 100.0) if side == 1 else ((base - min(lows)) / base * 100.0)
    adverse = ((base - min(lows)) / base * 100.0) if side == 1 else ((max(highs) / base - 1.0) * 100.0)
    return signed_close, max(0.0, favorable), max(0.0, adverse)


def evaluate_rule_diagnostic(
    candles: list[Candle],
    rule: str,
    threshold: float,
) -> dict:
    """Pure rule diagnostic.

    No capital, leverage, fees, positions, wins, profit factor or ranking are
    used to judge the rule. Each rule is treated as an observable numeric
    feature and its conditional relationship with future candles is measured.
    """
    values: list[float] = []
    signed_values: list[float] = []
    next_returns: list[float] = []
    signal_returns: list[float] = []
    signal_favorable_1: list[float] = []
    signal_adverse_1: list[float] = []
    signal_favorable_3: list[float] = []
    signal_adverse_3: list[float] = []
    signal_favorable_6: list[float] = []
    signal_adverse_6: list[float] = []
    signal_hits_1: list[bool] = []
    signal_hits_3: list[bool] = []
    signal_hits_6: list[bool] = []
    signal_abs_values: list[float] = []
    all_abs_values: list[float] = []

    positive = negative = zero = 0
    signal_count = 0
    buy_count = sell_count = 0

    for i, c in enumerate(candles):
        value = float(__import__("ohlc_rule_isolation_monthly", fromlist=["rule_components"]).rule_components(c)[rule])
        values.append(value)
        all_abs_values.append(abs(value))

        if value > 0:
            positive += 1
        elif value < 0:
            negative += 1
        else:
            zero += 1

        if i + 1 < len(candles) and c.close > 0:
            nr = (candles[i + 1].close / c.close - 1.0) * 100.0
            next_returns.append(nr)

        if value >= threshold:
            side = 1
            buy_count += 1
        elif value <= -threshold:
            side = -1
            sell_count += 1
        else:
            continue

        if i + 1 >= len(candles):
            continue

        signal_count += 1
        signal_abs_values.append(abs(value))
        m1, f1, a1 = _forward_metrics(candles, i, side, 1)
        signal_returns.append(m1)
        signal_favorable_1.append(f1)
        signal_adverse_1.append(a1)
        if m1 > 0:
            signal_hits_1.append(True)
        else:
            signal_hits_1.append(False)

        for horizon, favs, advs, hits in (
            (3, signal_favorable_3, signal_adverse_3, signal_hits_3),
            (6, signal_favorable_6, signal_adverse_6, signal_hits_6),
        ):
            if i + horizon < len(candles):
                m, f, a = _forward_metrics(candles, i, side, horizon)
                favs.append(f)
                advs.append(a)
                hits.append(m > 0)

    # Feature-vs-future relationship uses the signed rule value, not a trading
    # account. This is intentionally independent of threshold.
    usable = min(len(values) - 1, len(next_returns))
    corr = _pearson(values[:usable], next_returns[:usable]) if usable else 0.0

    samples = len(signal_returns)
    hit1 = _mean([1.0 if x else 0.0 for x in signal_hits_1]) * 100.0
    hit3 = _mean([1.0 if x else 0.0 for x in signal_hits_3]) * 100.0
    hit6 = _mean([1.0 if x else 0.0 for x in signal_hits_6]) * 100.0

    if signal_count == 0:
        observed = "NO_SIGNAL_AT_THRESHOLD"
    elif samples < 30:
        observed = "INSUFFICIENT_CONDITIONAL_SAMPLE"
    elif corr >= 0.05:
        observed = "POSITIVE_ASSOCIATION"
    elif corr <= -0.05:
        observed = "NEGATIVE_ASSOCIATION"
    elif hit1 >= 55.0 or hit3 >= 55.0 or hit6 >= 55.0:
        observed = "DIRECTIONAL_ASSOCIATION"
    elif max(hit1, hit3, hit6) <= 45.0:
        observed = "CONTRARIAN_ASSOCIATION"
    else:
        observed = "NO_CLEAR_DIRECTIONAL_ASSOCIATION"

    continuation = (
        "FOLLOW_THROUGH_OBSERVED"
        if hit3 >= 55.0 or hit6 >= 55.0
        else "NO_CLEAR_FOLLOW_THROUGH"
    )
    excursion = (
        "FAVORABLE_EXCURSION_DOMINATES"
        if _mean(signal_favorable_3) > _mean(signal_adverse_3) * 1.10
        else "ADVERSE_EXCURSION_DOMINATES"
        if _mean(signal_adverse_3) > _mean(signal_favorable_3) * 1.10
        else "BALANCED_EXCURSION"
    )

    return {
        "rule": rule,
        "rule_role": RULE_ROLE[rule],
        "what_it_is_good_for": RULE_USE[rule],
        "candles": len(candles),
        "threshold": threshold,
        "positive_value_candles": positive,
        "negative_value_candles": negative,
        "zero_value_candles": zero,
        "mean_value": _mean(values),
        "median_value": _median(values),
        "mean_abs_value": _mean(all_abs_values),
        "signal_count": signal_count,
        "signal_coverage_pct": signal_count / max(1, len(candles)) * 100.0,
        "buy_signal_count": buy_count,
        "sell_signal_count": sell_count,
        "conditional_samples": samples,
        "conditional_hit_rate_1_pct": hit1,
        "conditional_hit_rate_3_pct": hit3,
        "conditional_hit_rate_6_pct": hit6,
        "mean_next_return_all_pct": _mean(next_returns),
        "median_next_return_all_pct": _median(next_returns),
        "mean_signed_return_1_pct": _mean(signal_returns),
        "median_signed_return_1_pct": _median(signal_returns),
        "mean_favorable_1_pct": _mean(signal_favorable_1),
        "mean_adverse_1_pct": _mean(signal_adverse_1),
        "mean_favorable_3_pct": _mean(signal_favorable_3),
        "mean_adverse_3_pct": _mean(signal_adverse_3),
        "mean_favorable_6_pct": _mean(signal_favorable_6),
        "mean_adverse_6_pct": _mean(signal_adverse_6),
        "feature_next_return_correlation": corr,
        "observed_behavior": observed,
        "follow_through_observed": continuation,
        "excursion_behavior": excursion,
        "standalone_trading_value": "NOT_TESTED_BY_PROFIT",
    }


def evaluate_rule_audited(
    candles: list[Candle],
    rule: str,
    direction: str,
    initial_capital: float,
    allocation: float,
    threshold: float,
    fee: float,
    leverage: float,
):
    """Compatibility wrapper.

    The old account model is deliberately no longer used as a utility test.
    Direction is retained only so existing callers can still request the three
    legacy views; the returned diagnostics are identical for all directions.
    """
    d = evaluate_rule_diagnostic(candles, rule, threshold)
    d.update({
        "direction": direction,
        "initial_capital": initial_capital,
        "final_capital": initial_capital,
        "return_pct": 0.0,
        "accepted_signals": d["signal_count"],
        "buy_signals": d["buy_signal_count"],
        "sell_signals": d["sell_signal_count"],
        "hold_candles": len(candles) - d["signal_count"],
        "completed_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate_pct": 0.0,
        "gross_profit_factor": 0.0,
        "net_profit_factor": 0.0,
        "commission_paid": 0.0,
        "liquidations": 0,
        "max_drawdown_pct": 0.0,
        "sum_trade_net_pnl": 0.0,
        "account_pnl": 0.0,
        "reconciliation_error": 0.0,
        "reconciliation_ok": True,
    })
    return d


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Neutral OHLC R1-R16 feature diagnostics. No winners, ranking or profit scoring."
    )
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--month", required=True)
    ap.add_argument("--timeframes", default="5,30")
    ap.add_argument("--initial-capital", type=float, default=DEFAULT_CAPITAL,
                    help="retained for CLI compatibility; not used to score rules")
    ap.add_argument("--trade-allocation", type=float, default=DEFAULT_ALLOCATION,
                    help="retained for CLI compatibility; not used to score rules")
    ap.add_argument("--leverage", type=float, default=LEVERAGE,
                    help="retained for CLI compatibility; not used to score rules")
    ap.add_argument("--fee-per-side", type=float, default=FEE_PER_SIDE,
                    help="retained for CLI compatibility; not used to score rules")
    ap.add_argument("--threshold", type=float, default=0.5, help="single threshold; used when --thresholds is omitted")
    ap.add_argument("--thresholds", default=None, help="comma-separated thresholds, e.g. 0.25,0.5,0.75,1.0")
    ap.add_argument("--output", default="reports/ohlc_rule_diagnostics.csv")
    args = ap.parse_args()

    thresholds = (
        [float(x) for x in args.thresholds.split(",") if x.strip()]
        if args.thresholds else [args.threshold]
    )
    rows: list[dict] = []
    for symbol in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        path = find_month_file(Path(args.input_dir), symbol, args.month)
        if path is None:
            print(f"MISSING {symbol} {args.month}")
            continue
        raw = load_binance(path)
        for tf in [int(x) for x in args.timeframes.split(",")]:
            candles = resample(raw, tf)
            if not candles:
                print(f"{symbol}|{tf}m|NO_COMPLETE_CANDLES")
                continue

            for threshold in thresholds:
                for i in range(1, 17):
                    rule = f"R{i}"
                    d = evaluate_rule_diagnostic(candles, rule, threshold)
                    d.update({
                        "symbol": symbol,
                        "month": args.month,
                        "timeframe_min": tf,
                    })
                    rows.append(d)
                    print(
                        f"{symbol}|{tf}m|threshold={threshold:g}|{rule}|"
                        f"purpose={d['what_it_is_good_for']}|"
                        f"coverage={d['signal_coverage_pct']:.2f}%|"
                        f"samples={d['conditional_samples']}|"
                        f"h1={d['conditional_hit_rate_1_pct']:.1f}%|"
                        f"h3={d['conditional_hit_rate_3_pct']:.1f}%|"
                        f"h6={d['conditional_hit_rate_6_pct']:.1f}%|"
                        f"corr={d['feature_next_return_correlation']:.4f}|"
                        f"obs={d['observed_behavior']}|"
                        f"follow={d['follow_through_observed']}|"
                        f"excursion={d['excursion_behavior']}"
                    )

    if not rows:
        raise SystemExit("No input data found")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # A second file groups each rule by timeframe without ranking it.
    summary: list[dict] = []
    for symbol in sorted({r["symbol"] for r in rows}):
        for tf in sorted({int(r["timeframe_min"]) for r in rows if r["symbol"] == symbol}):
            subset = [r for r in rows if r["symbol"] == symbol and int(r["timeframe_min"]) == tf]
            for r in subset:
                summary.append({
                    "symbol": symbol,
                    "month": args.month,
                    "timeframe_min": tf,
                    "rule": r["rule"],
                    "threshold": r["threshold"],
                    "rule_role": r["rule_role"],
                    "what_it_is_good_for": r["what_it_is_good_for"],
                    "observed_behavior": r["observed_behavior"],
                    "follow_through_observed": r["follow_through_observed"],
                    "excursion_behavior": r["excursion_behavior"],
                    "signal_coverage_pct": r["signal_coverage_pct"],
                    "conditional_samples": r["conditional_samples"],
                    "conditional_hit_rate_1_pct": r["conditional_hit_rate_1_pct"],
                    "conditional_hit_rate_3_pct": r["conditional_hit_rate_3_pct"],
                    "conditional_hit_rate_6_pct": r["conditional_hit_rate_6_pct"],
                    "mean_signed_return_1_pct": r["mean_signed_return_1_pct"],
                    "mean_favorable_3_pct": r["mean_favorable_3_pct"],
                    "mean_adverse_3_pct": r["mean_adverse_3_pct"],
                    "feature_next_return_correlation": r["feature_next_return_correlation"],
                    "standalone_trading_value": r["standalone_trading_value"],
                })

    summary_path = out.with_name(out.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print(f"Saved diagnostics: {out}")
    print(f"Saved neutral summary: {summary_path}")
    print("IMPORTANT: no rule winner/rank, capital return, win rate or profit factor is used as utility.")

if __name__ == "__main__":
    main()
