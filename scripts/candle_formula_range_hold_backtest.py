from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal


RANGE_MIN_BODY = -100.0
RANGE_MAX_BODY = 100.0


@dataclass
class Position:
    side: int  # +1 long, -1 short
    entry_price: float
    entry_timestamp: int
    entry_index: int
    allocated_capital: float


def load_month(path: Path, month: str, symbol: str) -> list[Candle]:
    candles: list[Candle] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("month") != month or row.get("symbol") != symbol:
                continue
            try:
                candles.append(
                    Candle(
                        int(float(row["timestamp"])),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(candles, key=lambda x: x.timestamp)


def body_in_hold_range(candle: Candle) -> bool:
    body = candle.close - candle.open
    return RANGE_MIN_BODY <= body <= RANGE_MAX_BODY


def signal_side(candle: Candle) -> int:
    signal = decide(candle).signal
    if signal == Signal.BUY:
        return 1
    if signal == Signal.SELL:
        return -1
    return 0


def pnl_pct(side: int, entry: float, exit_price: float) -> float:
    if entry <= 0:
        return 0.0
    return side * (exit_price - entry) / entry


def close_position(
    position: Position,
    exit_price: float,
    equity: float,
    commission: float,
) -> tuple[float, float, float, float]:
    """Return (new_equity, gross_pnl, net_pnl, fees) for one round-trip trade."""
    trade_return = pnl_pct(position.side, position.entry_price, exit_price)
    gross = position.allocated_capital * trade_return
    fees = position.allocated_capital * commission * 2.0
    net = gross - fees
    return equity + net, gross, net, fees


def mark_to_market_equity(equity: float, position: Position | None, price: float) -> float:
    """Equity marked to the current close while the position remains open."""
    if position is None:
        return equity
    unrealized = position.allocated_capital * pnl_pct(position.side, position.entry_price, price)
    return equity + unrealized


def run_timeframe(
    path: Path,
    timeframe: str,
    month: str,
    symbol: str,
    initial_capital: float,
    commission: float,
    allocation: float,
    output: Path,
) -> dict[str, float | int | str]:
    candles = load_month(path, month, symbol)
    if len(candles) < 2:
        raise SystemExit(f"No usable {timeframe} candles for {symbol} {month}: {path}")
    if initial_capital <= 0:
        raise SystemExit("initial_capital must be > 0")
    if not 0.0 < allocation <= 1.0:
        raise SystemExit("allocation must be in (0, 1]")
    if commission < 0:
        raise SystemExit("commission must be >= 0")

    equity = initial_capital
    position: Position | None = None
    trades = 0
    wins = 0
    losses = 0
    flats = 0
    gross_pnl = 0.0
    net_pnl = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    total_fees = 0.0
    peak_equity = initial_capital
    max_drawdown = 0.0
    forced_close = 0
    held_through_opposite = 0
    switches = 0
    entries_long = 0
    entries_short = 0

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "symbol", "timeframe", "month", "timestamp", "open", "high", "low", "close",
        "body", "in_range", "signal", "position_before", "action", "position_after",
        "trade_pnl_pct", "trade_net_pnl", "equity", "mtm_equity", "drawdown_pct",
    ]
    sample_rows: list[dict[str, object]] = []

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, candle in enumerate(candles):
            sig = signal_side(candle)
            body = candle.close - candle.open
            in_range = body_in_hold_range(candle)
            position_before = position.side if position else 0
            action = "HOLD"
            trade_return_pct = ""
            trade_net = ""

            if position is None:
                if sig != 0:
                    allocated = equity * allocation
                    position = Position(sig, candle.close, candle.timestamp, i, allocated)
                    if sig > 0:
                        entries_long += 1
                        action = "OPEN_LONG"
                    else:
                        entries_short += 1
                        action = "OPEN_SHORT"
            else:
                if sig == 0:
                    action = "HOLD_NO_SIGNAL"
                elif sig == position.side:
                    action = "HOLD_SAME"
                elif not in_range:
                    held_through_opposite += 1
                    action = "HOLD_OUTSIDE_RANGE"
                else:
                    equity, gross, net, fees = close_position(
                        position, candle.close, equity, commission
                    )
                    trade_return_pct = f"{pnl_pct(position.side, position.entry_price, candle.close) * 100.0:.8f}"
                    trade_net = f"{net:.10f}"
                    gross_pnl += gross
                    net_pnl += net
                    total_fees += fees
                    gross_profit += max(gross, 0.0)
                    gross_loss += min(gross, 0.0)
                    trades += 1
                    if net > 0:
                        wins += 1
                    elif net < 0:
                        losses += 1
                    else:
                        flats += 1
                    switches += 1

                    new_side = sig
                    new_allocated = equity * allocation
                    position = Position(
                        new_side, candle.close, candle.timestamp, i, new_allocated
                    )
                    if new_side > 0:
                        entries_long += 1
                        action = "SWITCH_LONG"
                    else:
                        entries_short += 1
                        action = "SWITCH_SHORT"

            mtm_equity = mark_to_market_equity(equity, position, candle.close)
            peak_equity = max(peak_equity, mtm_equity)
            drawdown = (peak_equity - mtm_equity) / peak_equity if peak_equity else 0.0
            max_drawdown = max(max_drawdown, drawdown)

            record = {
                "symbol": symbol,
                "timeframe": timeframe,
                "month": month,
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "body": body,
                "in_range": int(in_range),
                "signal": "UP" if sig > 0 else "DOWN" if sig < 0 else "HOLD",
                "position_before": position_before,
                "action": action,
                "position_after": position.side if position else 0,
                "trade_pnl_pct": trade_return_pct,
                "trade_net_pnl": trade_net,
                "equity": f"{equity:.10f}",
                "mtm_equity": f"{mtm_equity:.10f}",
                "drawdown_pct": f"{drawdown * 100.0:.10f}",
            }
            writer.writerow(record)
            if len(sample_rows) < 3:
                sample_rows.append(record)

    if position is not None:
        candle = candles[-1]
        equity, gross, net, fees = close_position(position, candle.close, equity, commission)
        gross_pnl += gross
        net_pnl += net
        total_fees += fees
        gross_profit += max(gross, 0.0)
        gross_loss += min(gross, 0.0)
        trades += 1
        forced_close = 1
        if net > 0:
            wins += 1
        elif net < 0:
            losses += 1
        else:
            flats += 1
        final_drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0.0
        max_drawdown = max(max_drawdown, final_drawdown)

    total_closed = wins + losses + flats
    win_rate = wins / total_closed * 100.0 if total_closed else 0.0
    final_return = (equity / initial_capital - 1.0) * 100.0
    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else float("inf") if gross_profit > 0 else 0.0

    return {
        "symbol": symbol,
        "month": month,
        "timeframe": timeframe,
        "candles": len(candles),
        "initial": initial_capital,
        "final": equity,
        "return_pct": final_return,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "flats": flats,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "fees": total_fees,
        "max_drawdown_pct": max_drawdown * 100.0,
        "long_entries": entries_long,
        "short_entries": entries_short,
        "switches": switches,
        "held_through_opposite": held_through_opposite,
        "forced_close": forced_close,
        "commission": commission,
        "allocation": allocation,
        "sample_rows": sample_rows,
        "fields": fields,
        "output": str(output),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Range-hold backtest for the exact Excel candle formula strategy.")
    ap.add_argument("--month", default="2026-07")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--commission", type=float, default=0.0, help="One-way commission rate; e.g. 0.001 = 0.1%%")
    ap.add_argument("--allocation", type=float, default=1.0, help="Fraction of equity used per position")
    ap.add_argument("--input-5m", default="reports/prepared_price_action_5m.csv")
    ap.add_argument("--input-15m", default="reports/prepared_price_action_15m.csv")
    ap.add_argument("--input-1h", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--output-dir", default="reports/candle_formula_range_hold_backtest")
    args = ap.parse_args()

    jobs = (
        ("5m", Path(args.input_5m)),
        ("15m", Path(args.input_15m)),
        ("1h", Path(args.input_1h)),
    )

    results = []
    for timeframe, path in jobs:
        output = Path(args.output_dir) / f"{args.symbol}_{timeframe}_{args.month}.csv"
        result = run_timeframe(
            path,
            timeframe,
            args.month,
            args.symbol,
            args.initial_capital,
            args.commission,
            args.allocation,
            output,
        )
        results.append(result)
        pf = result["profit_factor"]
        pf_text = "INF" if pf == float("inf") else f"{pf:.2f}"
        print(
            f"{timeframe} | candles={result['candles']} | "
            f"initial={result['initial']:.2f} final={result['final']:.2f} "
            f"return={result['return_pct']:.2f}% | trades={result['trades']} "
            f"win={result['win_rate']:.2f}% PF={pf_text} | DD_MTM={result['max_drawdown_pct']:.2f}% | "
            f"fees={result['fees']:.4f} | switches={result['switches']} "
            f"held_opp={result['held_through_opposite']}"
        )
        print(f"  COLUMNS={','.join(result['fields'])}")
        for n, row in enumerate(result["sample_rows"], 1):
            print(f"  ROW{n}=" + " | ".join(f"{field}={row[field]}" for field in result["fields"]))

    summary = Path(args.output_dir) / f"summary_{args.symbol}_{args.month}.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "symbol", "month", "timeframe", "candles", "initial", "final", "return_pct",
            "trades", "wins", "losses", "flats", "win_rate", "profit_factor", "gross_pnl", "net_pnl", "fees",
            "max_drawdown_pct", "long_entries", "short_entries", "switches", "held_through_opposite",
            "forced_close", "commission", "allocation",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({field: r[field] for field in fields})

    print(f"SUMMARY={summary}")


if __name__ == "__main__":
    main()
