from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal


DEFAULT_RANGE_MIN = -100.0
DEFAULT_RANGE_MAX = 100.0
DEFAULT_COMMISSION = 0.0013  # 0.13% per side


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


def body_in_range(candle: Candle, range_min: float, range_max: float) -> bool:
    body = candle.close - candle.open
    return range_min <= body <= range_max


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
    """Return (new_equity, gross_pnl, net_pnl, total_round_trip_fees)."""
    trade_return = pnl_pct(position.side, position.entry_price, exit_price)
    gross = position.allocated_capital * trade_return
    fees = position.allocated_capital * commission * 2.0
    net = gross - fees
    return equity + net, gross, net, fees


def mark_to_market_equity(equity: float, position: Position | None, price: float) -> float:
    if position is None:
        return equity
    unrealized = position.allocated_capital * pnl_pct(
        position.side, position.entry_price, price
    )
    return equity + unrealized


def run_timeframe(
    path: Path,
    timeframe: str,
    month: str,
    symbol: str,
    initial_capital: float,
    commission: float,
    allocation: float,
    range_min: float,
    range_max: float,
    output: Path,
) -> dict[str, object]:
    candles = load_month(path, month, symbol)
    if len(candles) < 1:
        raise SystemExit(f"No usable {timeframe} candles for {symbol} {month}: {path}")
    if initial_capital <= 0:
        raise SystemExit("initial_capital must be > 0")
    if not 0.0 < allocation <= 1.0:
        raise SystemExit("allocation must be in (0, 1]")
    if commission < 0:
        raise SystemExit("commission must be >= 0")
    if range_min > range_max:
        raise SystemExit("range_min must be <= range_max")

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
    switches = 0
    held_through_opposite = 0
    ignored_outside_entry_signals = 0
    entries_long = 0
    entries_short = 0
    entry_signals = 0

    fields = [
        "symbol", "timeframe", "month", "timestamp", "open", "high", "low", "close",
        "body", "in_range", "signal", "position_before", "action", "position_after",
        "trade_pnl_pct", "trade_net_pnl", "equity", "mtm_equity", "drawdown_pct",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    sample_rows: list[dict[str, object]] = []

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, candle in enumerate(candles):
            sig = signal_side(candle)
            body = candle.close - candle.open
            in_range = range_min <= body <= range_max
            position_before = position.side if position else 0
            action = "HOLD"
            trade_return_pct = ""
            trade_net = ""

            if position is None:
                # ENTRY: first valid signal only when body is inside the configurable range.
                if sig != 0 and in_range:
                    allocated = equity * allocation
                    position = Position(sig, candle.close, candle.timestamp, i, allocated)
                    entry_signals += 1
                    if sig > 0:
                        entries_long += 1
                        action = "OPEN_LONG"
                    else:
                        entries_short += 1
                        action = "OPEN_SHORT"
                elif sig != 0:
                    ignored_outside_entry_signals += 1
                    action = "IGNORE_ENTRY_OUTSIDE_RANGE"
            else:
                if sig == 0:
                    action = "HOLD_NO_SIGNAL"
                elif sig == position.side:
                    action = "HOLD_SAME"
                elif not in_range:
                    held_through_opposite += 1
                    action = "HOLD_OPPOSITE_OUTSIDE_RANGE"
                else:
                    equity, gross, net, fees = close_position(
                        position, candle.close, equity, commission
                    )
                    trade_return_pct = (
                        f"{pnl_pct(position.side, position.entry_price, candle.close) * 100.0:.8f}"
                    )
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
                    entry_signals += 1
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
        equity, gross, net, fees = close_position(
            position, candle.close, equity, commission
        )
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
        final_drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0.0
        max_drawdown = max(max_drawdown, final_drawdown)

    total_closed = wins + losses + flats
    win_rate = wins / total_closed * 100.0 if total_closed else 0.0
    final_return = (equity / initial_capital - 1.0) * 100.0
    profit_factor = (
        gross_profit / abs(gross_loss)
        if gross_loss < 0
        else float("inf") if gross_profit > 0 else 0.0
    )

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
        "entry_signals": entry_signals,
        "switches": switches,
        "held_through_opposite": held_through_opposite,
        "ignored_outside_entry_signals": ignored_outside_entry_signals,
        "forced_close": 1 if position is not None else 0,
        "commission": commission,
        "allocation": allocation,
        "range_min": range_min,
        "range_max": range_max,
        "sample_rows": sample_rows,
        "fields": fields,
        "output": str(output),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Trade exact Excel candle signals with configurable body range and 0.13% default per side."
    )
    ap.add_argument("--month", default="2026-05")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--commission", type=float, default=DEFAULT_COMMISSION, help="One-way commission rate; default 0.0013 = 0.13%%")
    ap.add_argument("--allocation", type=float, default=1.0)
    ap.add_argument("--range-min", type=float, default=DEFAULT_RANGE_MIN)
    ap.add_argument("--range-max", type=float, default=DEFAULT_RANGE_MAX)
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
            args.range_min,
            args.range_max,
            output,
        )
        results.append(result)
        pf = result["profit_factor"]
        pf_text = "INF" if pf == float("inf") else f"{pf:.2f}"
        print(
            f"{timeframe} | candles={result['candles']} | "
            f"range=[{result['range_min']},{result['range_max']}] | "
            f"commission/side={result['commission'] * 100.0:.2f}% | "
            f"initial={result['initial']:.2f} final={result['final']:.2f} "
            f"return={result['return_pct']:.2f}% | trades={result['trades']} "
            f"win={result['win_rate']:.2f}% PF={pf_text} | DD_MTM={result['max_drawdown_pct']:.2f}% | "
            f"fees={result['fees']:.4f} | ignored_entry={result['ignored_outside_entry_signals']} "
            f"held_opp={result['held_through_opposite']}"
        )
        print(f"  COLUMNS={','.join(result['fields'])}")
        for n, row in enumerate(result["sample_rows"], 1):
            print(
                f"  ROW{n}="
                + " | ".join(f"{field}={row[field]}" for field in result["fields"])
            )

    summary = Path(args.output_dir) / f"summary_{args.symbol}_{args.month}.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "symbol", "month", "timeframe", "candles", "initial", "final", "return_pct",
            "trades", "wins", "losses", "flats", "win_rate", "profit_factor", "gross_pnl",
            "net_pnl", "fees", "max_drawdown_pct", "long_entries", "short_entries",
            "entry_signals", "switches", "held_through_opposite", "ignored_outside_entry_signals",
            "forced_close", "commission", "allocation", "range_min", "range_max",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({field: r[field] for field in fields})

    print(f"SUMMARY={summary}")


if __name__ == "__main__":
    main()
