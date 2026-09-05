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

    # Isolated margin: liquidation consumes the remaining margin.
    if liquidation:
        gross_pnl = -position["margin"]

    net_pnl = gross_pnl - close_fee
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
    """Audited accounting model for the existing OHLC signal semantics.

    Accounting:
    - Dollar PnL is calculated from actual isolated-margin positions.
    - Gross PF is before closing fees.
    - Net PF is after entry and exit fees.
    - Each isolated-margin position cannot lose more than its margin.
    - Intrabar liquidation is detected from candle high/low.
    - Equity is marked to market and cannot become negative.
    """
    cash = initial_capital
    positions: list[dict] = []
    trade_records: list[dict] = []
    fees_paid = 0.0
    liquidations = 0
    peak_equity = initial_capital
    max_dd = 0.0
    last_signal = 0

    signals = buys = sells = holds = accepted_signals = 0

    def record_close(p: dict, price: float, liquidation: bool = False) -> None:
        nonlocal cash, fees_paid, liquidations
        cash, gross_pnl, net_pnl, close_fee = _close_position(
            p, price, cash, fee, leverage, liquidation
        )
        trade_records.append(
            {
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "fee": close_fee,
                "liquidation": liquidation,
            }
        )
        fees_paid += close_fee
        if liquidation:
            liquidations += 1

    for c in candles:
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
                        "opened_at": c.timestamp,
                    }
                )
            last_signal = sig

        # Detect isolated-margin liquidation from the candle's adverse excursion.
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

    # Realize remaining positions at the final close.
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

    gross_pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    net_pf = net_profit / net_loss if net_loss > 0 else (float("inf") if net_profit > 0 else 0.0)

    avg_win = net_profit / wins if wins else 0.0
    avg_loss = net_loss / losses if losses else 0.0

    return {
        "direction": direction,
        "rule": rule,
        "candles": len(candles),
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "return_pct": (final_capital / initial_capital - 1.0) * 100.0,
        "signals": signals,
        "accepted_signals": accepted_signals,
        "buy_signals": buys,
        "sell_signals": sells,
        "hold_candles": holds,
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
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_per_trade": sum(x["net_pnl"] for x in trade_records) / len(trade_records) if trade_records else 0.0,
        "commission_paid": fees_paid,
        "liquidations": liquidations,
        "max_drawdown_pct": max_dd,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audited accounting backtest for OHLC R1-R16.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--month", required=True, help="YYYY-MM")
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
                        candles,
                        f"R{i}",
                        direction,
                        args.initial_capital,
                        args.trade_allocation,
                        args.threshold,
                        args.fee_per_side,
                        args.leverage,
                    )
                    s.update(
                        {
                            "symbol": symbol,
                            "month": args.month,
                            "timeframe_min": tf,
                            "fee_per_side_pct": args.fee_per_side * 100.0,
                            "allocation_pct": args.trade_allocation * 100.0,
                            "leverage": args.leverage,
                            "threshold": args.threshold,
                        }
                    )
                    rows.append(s)
                    gpf = s["gross_profit_factor"]
                    npf = s["net_profit_factor"]
                    gpf_text = "inf" if gpf == float("inf") else f"{gpf:.2f}"
                    npf_text = "inf" if npf == float("inf") else f"{npf:.2f}"
                    print(
                        f"{symbol}|{tf}m|{direction}|{s['rule']}|"
                        f"return={s['return_pct']:.2f}% trades={s['completed_trades']} "
                        f"win={s['win_rate_pct']:.1f}% GPF={gpf_text} NPF={npf_text} "
                        f"avgW={s['avg_win']:.4f} avgL={s['avg_loss']:.4f} "
                        f"DD={s['max_drawdown_pct']:.2f}% fees={s['commission_paid']:.2f} "
                        f"liq={s['liquidations']}"
                    )

    if not rows:
        raise SystemExit("No input data found")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
