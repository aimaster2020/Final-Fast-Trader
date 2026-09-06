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
    "R1": "بدنه: close > open",
    "R2": "بدنه: open > close",
    "R3": "بسته‌شدن روی سقف: close = high",
    "R4": "بسته‌شدن روی کف: close = low",
    "R5": "جهت کلی رنج کندل",
    "R6": "شباهت open به high",
    "R7": "شباهت open به low",
    "R8": "شباهت close به high",
    "R9": "شباهت close به low",
    "R10": "قدرت سایه پایین",
    "R11": "قدرت سایه بالا",
    "R12": "موقعیت صعودی بدنه در رنج",
    "R13": "موقعیت نزولی بدنه در رنج",
    "R14": "قدرت صعودی بدنه",
    "R15": "قدرت نزولی بدنه",
    "R16": "جهت نهایی کندل/رنج",
}


def _mark_to_market(position: dict, price: float, leverage: float) -> float:
    gross = (
        price / position["entry"] - 1.0
        if position["side"] == 1
        else position["entry"] / price - 1.0
    )
    return position["margin"] + position["margin"] * leverage * gross


def _liquidation_price(position: dict, leverage: float) -> float:
    if leverage <= 0:
        return float("inf")
    if position["side"] == 1:
        return position["entry"] * (1.0 - 1.0 / leverage)
    return position["entry"] * (1.0 + 1.0 / leverage)


def _close_position(
    position: dict,
    price: float,
    cash: float,
    fee: float,
    leverage: float,
    liquidation: bool = False,
) -> tuple[float, float, float, float]:
    gross_return = (
        price / position["entry"] - 1.0
        if position["side"] == 1
        else position["entry"] / price - 1.0
    )
    gross_pnl = position["margin"] * leverage * gross_return
    close_fee = position["notional"] * fee
    if liquidation:
        gross_pnl = -position["margin"]

    total_trade_fees = position["entry_fee"] + close_fee
    net_pnl = gross_pnl - total_trade_fees

    cash += position["margin"] + gross_pnl - close_fee
    cash = max(0.0, cash)
    return cash, gross_pnl, net_pnl, close_fee


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
    """Neutral rule diagnostic.

    The purpose of this report is NOT to find a trading winner.
    It measures what each R-rule observes/predicts on the next candle:
      - coverage: how often the rule emits a directional signal
      - next-candle directional agreement
      - mean/median next-candle return
      - mean favorable/adverse excursion
      - optional accounting check for the existing execution model

    Trading performance is retained only as an audit trail. It must not be
    used as the rule's utility label.
    """
    # The direction argument is retained for compatibility with the existing
    # 3-way output. Prediction diagnostics themselves are based on the raw
    # rule signal, so LONG_ONLY/SHORT_ONLY cannot manufacture an edge.
    cash = initial_capital
    positions: list[dict] = []
    trade_records: list[dict] = []
    fees_paid = 0.0
    liquidations = 0
    peak_equity = initial_capital
    max_dd = 0.0
    last_signal = 0
    signals = buys = sells = holds = accepted_signals = 0

    next_returns: list[float] = []
    directional_hits: list[bool] = []
    favorable_moves: list[float] = []
    adverse_moves: list[float] = []

    def record_close(p: dict, price: float, liquidation: bool = False) -> None:
        nonlocal cash, fees_paid, liquidations
        cash, gross_pnl, net_pnl, close_fee = _close_position(
            p, price, cash, fee, leverage, liquidation
        )
        trade_records.append(
            {
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "entry_fee": p["entry_fee"],
                "exit_fee": close_fee,
                "fee": p["entry_fee"] + close_fee,
                "liquidation": liquidation,
            }
        )
        fees_paid += close_fee
        if liquidation:
            liquidations += 1

    for i, c in enumerate(candles):
        raw_sig = signal_for_rule(c, rule, threshold)

        if raw_sig:
            signals += 1
            if raw_sig == 1:
                buys += 1
            else:
                sells += 1
        else:
            holds += 1

        if direction == "LONG_ONLY" and raw_sig != 1:
            sig = 0
        elif direction == "SHORT_ONLY" and raw_sig != -1:
            sig = 0
        else:
            sig = raw_sig

        if sig:
            accepted_signals += 1

        # Prediction diagnostic: signal at candle i is evaluated only against
        # candle i+1. The current candle is never allowed to see its future.
        if raw_sig and i + 1 < len(candles):
            nxt = candles[i + 1]
            base = c.close
            if base > 0:
                next_ret = (nxt.close / base - 1.0) * 100.0
                next_returns.append(next_ret)
                directional_hits.append(
                    (raw_sig == 1 and nxt.close > base)
                    or (raw_sig == -1 and nxt.close < base)
                )
                if raw_sig == 1:
                    favorable_moves.append(max(0.0, (nxt.high / base - 1.0) * 100.0))
                    adverse_moves.append(max(0.0, (base - nxt.low) / base * 100.0))
                else:
                    favorable_moves.append(max(0.0, (base - nxt.low) / base * 100.0))
                    adverse_moves.append(max(0.0, (nxt.high - base) / base * 100.0))

        # Existing execution model is kept only to expose accounting/risk
        # behavior; it is NOT used to decide whether a rule is useful.
        if direction == "BOTH" and sig and last_signal and sig != last_signal and positions:
            for p in positions:
                record_close(p, c.close)
            positions = []

        if sig and cash > 0:
            margin = cash * allocation
            notional = margin * leverage
            entry_fee = notional * fee
            if margin > 0 and margin + entry_fee <= cash:
                cash -= margin + entry_fee
                fees_paid += entry_fee
                positions.append(
                    {
                        "side": sig,
                        "entry": c.close,
                        "margin": margin,
                        "notional": notional,
                        "entry_fee": entry_fee,
                        "opened_at": c.timestamp,
                    }
                )
            last_signal = sig

        still_open: list[dict] = []
        for p in positions:
            liq = _liquidation_price(p, leverage)
            hit = (c.low <= liq) if p["side"] == 1 else (c.high >= liq)
            if hit and c.timestamp > p["opened_at"]:
                record_close(p, liq, liquidation=True)
            else:
                still_open.append(p)
        positions = still_open

        equity = cash + sum(_mark_to_market(p, c.close, leverage) for p in positions)
        equity = max(0.0, equity)
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            max_dd = max(max_dd, (peak_equity - equity) / peak_equity * 100.0)

    for p in positions:
        record_close(p, candles[-1].close)

    final_capital = max(0.0, cash)

    gross_profits = [x["gross_pnl"] for x in trade_records if x["gross_pnl"] > 0]
    gross_losses = [-x["gross_pnl"] for x in trade_records if x["gross_pnl"] < 0]
    net_profits = [x["net_pnl"] for x in trade_records if x["net_pnl"] > 0]
    net_losses = [-x["net_pnl"] for x in trade_records if x["net_pnl"] < 0]
    wins = len(net_profits)
    losses = len(net_losses)
    gross_profit = sum(gross_profits)
    gross_loss = sum(gross_losses)
    net_profit = sum(net_profits)
    net_loss = sum(net_losses)
    sum_trade_net = sum(x["net_pnl"] for x in trade_records)
    reconciliation_error = sum_trade_net - (final_capital - initial_capital)
    reconciliation_ok = abs(reconciliation_error) <= max(
        1e-8, initial_capital * 1e-10
    )

    gross_pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (float("inf") if gross_profit > 0 else 0.0)
    )
    net_pf = (
        net_profit / net_loss
        if net_loss > 0
        else (float("inf") if net_profit > 0 else 0.0)
    )

    def median(values: list[float]) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        n = len(values)
        mid = n // 2
        return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2.0

    prediction_count = len(next_returns)
    prediction_hit_rate = (
        sum(directional_hits) / prediction_count * 100.0
        if prediction_count else 0.0
    )
    mean_next_return = (
        sum(next_returns) / prediction_count if prediction_count else 0.0
    )
    median_next_return = median(next_returns)
    mean_favorable = (
        sum(favorable_moves) / prediction_count if prediction_count else 0.0
    )
    mean_adverse = (
        sum(adverse_moves) / prediction_count if prediction_count else 0.0
    )
    signal_coverage_pct = (
        signals / max(1, len(candles)) * 100.0
    )

    # Neutral interpretation. There is deliberately no "winner" or ranking.
    # A rule is only considered directionally informative when its signal is
    # sufficiently observable and the next-candle conditional return agrees
    # with the sign of the rule. Otherwise we describe the observed behavior.
    if signals == 0:
        diagnostic = "INACTIVE_NO_SIGNAL"
    elif prediction_count < 30:
        diagnostic = "LOW_SAMPLE"
    elif prediction_hit_rate >= 50.0 and mean_next_return > 0:
        diagnostic = "UPWARD_EDGE_OBSERVED"
    elif prediction_hit_rate >= 50.0 and mean_next_return < 0:
        diagnostic = "DOWNWARD_EDGE_OBSERVED"
    elif prediction_hit_rate < 50.0 and mean_next_return > 0:
        diagnostic = "MIXED_CONTRARIAN"
    elif prediction_hit_rate < 50.0 and mean_next_return < 0:
        diagnostic = "MIXED_CONTRARIAN"
    else:
        diagnostic = "NO_CLEAR_EDGE"

    return {
        "direction": direction,
        "rule": rule,
        "rule_role": RULE_ROLE[rule],
        "candles": len(candles),
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "return_pct": (final_capital / initial_capital - 1.0) * 100.0,
        "signals": signals,
        "accepted_signals": accepted_signals,
        "buy_signals": buys,
        "sell_signals": sells,
        "hold_candles": holds,
        "signal_coverage_pct": signal_coverage_pct,
        "prediction_samples": prediction_count,
        "directional_hit_rate_pct": prediction_hit_rate,
        "mean_next_candle_return_pct": mean_next_return,
        "median_next_candle_return_pct": median_next_return,
        "mean_favorable_excursion_pct": mean_favorable,
        "mean_adverse_excursion_pct": mean_adverse,
        "diagnostic": diagnostic,
        "completed_trades": len(trade_records),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": wins / len(trade_records) * 100.0 if trade_records else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "gross_profit_factor": gross_pf,
        "net_profit": net_profit,
        "net_loss": net_loss,
        "net_profit_factor": net_pf,
        "commission_paid": fees_paid,
        "liquidations": liquidations,
        "max_drawdown_pct": max_dd,
        "sum_trade_net_pnl": sum_trade_net,
        "account_pnl": final_capital - initial_capital,
        "reconciliation_error": reconciliation_error,
        "reconciliation_ok": reconciliation_ok,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="OHLC R1-R16 accounting audit and rule utility diagnostics.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--month", required=True)
    ap.add_argument("--timeframes", default="5,30")
    ap.add_argument("--initial-capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--trade-allocation", type=float, default=DEFAULT_ALLOCATION)
    ap.add_argument("--leverage", type=float, default=LEVERAGE)
    ap.add_argument("--fee-per-side", type=float, default=FEE_PER_SIDE)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--output", default="reports/ohlc_rule_accounting_audit.csv")
    args = ap.parse_args()

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
                continue
            for direction in ("BOTH", "LONG_ONLY", "SHORT_ONLY"):
                for i in range(1, 17):
                    s = evaluate_rule_audited(
                        candles, f"R{i}", direction,
                        args.initial_capital, args.trade_allocation,
                        args.threshold, args.fee_per_side, args.leverage,
                    )
                    s.update({
                        "symbol": symbol,
                        "month": args.month,
                        "timeframe_min": tf,
                        "fee_per_side_pct": args.fee_per_side * 100.0,
                        "allocation_pct": args.trade_allocation * 100.0,
                        "leverage": args.leverage,
                        "threshold": args.threshold,
                    })
                    rows.append(s)
                    gpf = s["gross_profit_factor"]
                    npf = s["net_profit_factor"]
                    gpf_text = "inf" if gpf == float("inf") else f"{gpf:.2f}"
                    npf_text = "inf" if npf == float("inf") else f"{npf:.2f}"
                    print(
                        f"{symbol}|{tf}m|{direction}|{s['rule']}|"
                        f"role={s['utility_assessment']} return={s['return_pct']:.2f}% "
                        f"trades={s['completed_trades']} win={s['win_rate_pct']:.1f}% "
                        f"NPF={npf_text} DD={s['max_drawdown_pct']:.2f}% "
                        f"fees={s['commission_paid']:.2f} liq={s['liquidations']} "
                        f"recon={'OK' if s['reconciliation_ok'] else 'FAIL'}"
                    )

    if not rows:
        raise SystemExit("No input data found")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Human-readable rule diagnostics. No ranking and no "winner" field.
    # This report explains the observed behavior of each rule instead.
    groups: dict[tuple[str, int, str], dict[str, dict]] = {}
    for row in rows:
        groups.setdefault(
            (row["symbol"], int(row["timeframe_min"]), row["rule"]), {}
        )[row["direction"]] = row

    diag_rows: list[dict] = []
    for (symbol, tf, rule), group in groups.items():
        both = group.get("BOTH")
        if not both:
            continue

        diag_rows.append({
            "symbol": symbol,
            "month": args.month,
            "timeframe_min": tf,
            "rule": rule,
            "rule_role": RULE_ROLE[rule],
            "diagnostic": both["diagnostic"],
            "what_it_is_good_for": (
                "UP/DOWN directional signal"
                if both["diagnostic"] in {"UPWARD_EDGE_OBSERVED", "DOWNWARD_EDGE_OBSERVED"}
                else "NO_STANDALONE_USE_FOUND"
            ),
            "signals": int(both["signals"]),
            "signal_coverage_pct": float(both["signal_coverage_pct"]),
            "prediction_samples": int(both["prediction_samples"]),
            "directional_hit_rate_pct": float(both["directional_hit_rate_pct"]),
            "mean_next_candle_return_pct": float(both["mean_next_candle_return_pct"]),
            "median_next_candle_return_pct": float(both["median_next_candle_return_pct"]),
            "mean_favorable_excursion_pct": float(both["mean_favorable_excursion_pct"]),
            "mean_adverse_excursion_pct": float(both["mean_adverse_excursion_pct"]),
            "buy_signals": int(both["buy_signals"]),
            "sell_signals": int(both["sell_signals"]),
            "hold_candles": int(both["hold_candles"]),
            "accounting_reconciliation_ok": bool(both["reconciliation_ok"]),
            "execution_return_pct_reference_only": float(both["return_pct"]),
            "execution_max_drawdown_pct_reference_only": float(both["max_drawdown_pct"]),
            "execution_liquidations_reference_only": int(both["liquidations"]),
        })

    diag_path = out.with_name(out.stem + "_diagnostics.csv")
    if diag_rows:
        with diag_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(diag_rows[0]))
            writer.writeheader()
            writer.writerows(diag_rows)



if __name__ == "__main__":
    main()
