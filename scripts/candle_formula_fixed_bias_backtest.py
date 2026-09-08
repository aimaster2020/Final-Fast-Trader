from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal

MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
CAPITAL = 1000.0
DEFAULT_COMMISSION = 0.0013
BIAS_LOW = -1.0
BIAS_HIGH = 1.0


@dataclass
class Position:
    side: int
    entry: float
    capital: float


def load(path: Path, month: str, symbol: str) -> list[Candle]:
    out: list[Candle] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("month") != month or row.get("symbol") != symbol:
                continue
            try:
                out.append(
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
    return sorted(out, key=lambda c: c.timestamp)


def signal(candle: Candle) -> int:
    decision = decide(candle)
    if decision.signal == Signal.BUY:
        return 1
    if decision.signal == Signal.SELL:
        return -1
    return 0


def body_direction(candle: Candle) -> int:
    body = candle.close - candle.open
    if body > 0:
        return 1
    if body < 0:
        return -1
    return 0


def in_bias(body: float) -> bool:
    return BIAS_LOW <= body <= BIAS_HIGH


def pnl(side: int, entry: float, price: float) -> float:
    return side * (price - entry) / entry if entry > 0 else 0.0


def run(candles: list[Candle], commission: float) -> dict[str, float | int]:
    equity = CAPITAL
    position: Position | None = None
    previous_prediction = 0
    trades = wins = entries = exits = 0
    gross_profit = gross_loss = 0.0
    peak = CAPITAL
    max_dd = 0.0

    for candle in candles:
        current_signal = signal(candle)
        body = candle.close - candle.open
        previous_correct = (
            previous_prediction != 0
            and previous_prediction == body_direction(candle)
        )

        if position is None:
            if (
                current_signal != 0
                and not in_bias(body)
                and previous_correct
            ):
                position = Position(current_signal, candle.close, equity)
                entries += 1
        elif in_bias(body):
            gross = position.capital * pnl(
                position.side, position.entry, candle.close
            )
            net = gross - position.capital * commission * 2.0
            equity += net
            trades += 1
            wins += int(net > 0)
            gross_profit += max(gross, 0.0)
            gross_loss += min(gross, 0.0)
            position = None
            exits += 1

        mtm = equity
        if position is not None:
            mtm += position.capital * pnl(
                position.side, position.entry, candle.close
            )
        peak = max(peak, mtm)
        max_dd = max(max_dd, (peak - mtm) / peak if peak else 0.0)
        previous_prediction = current_signal

    if position is not None and candles:
        gross = position.capital * pnl(
            position.side, position.entry, candles[-1].close
        )
        net = gross - position.capital * commission * 2.0
        equity += net
        trades += 1
        wins += int(net > 0)
        gross_profit += max(gross, 0.0)
        gross_loss += min(gross, 0.0)

    return {
        "return": (equity / CAPITAL - 1.0) * 100.0,
        "final": equity,
        "trades": trades,
        "wins": wins,
        "win_rate": wins / trades * 100.0 if trades else 0.0,
        "pf": gross_profit / abs(gross_loss) if gross_loss < 0 else (float("inf") if gross_profit else 0.0),
        "dd": max_dd * 100.0,
        "entries": entries,
        "exits": exits,
    }


def compound(returns: list[float]) -> float:
    factor = 1.0
    for value in returns:
        factor *= 1.0 + value / 100.0
    return (factor - 1.0) * 100.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fixed candle-formula backtest with bias range [-1,+1]"
    )
    ap.add_argument("--input", default="reports/prepared_price_action_1h.csv")
    ap.add_argument(
        "--commission",
        type=float,
        default=DEFAULT_COMMISSION,
        help="Commission per side as decimal; e.g. 0.0013 = 0.13%%",
    )
    args = ap.parse_args()
    path = Path(args.input)

    fee_pct = args.commission * 100.0
    print(
        f"CANDLE_FORMULA_FIXED_BIAS | tf=1h | bias=[-1,+1] | "
        f"previous-prediction-confirmed-entry | same-bias-exit | "
        f"fee={fee_pct:.2f}%/side | capital=1000"
    )

    for symbol in SYMBOLS:
        monthly_returns: list[float] = []
        print(f"\n{symbol}")
        for month in MONTHS:
            candles = load(path, month, symbol)
            result = run(candles, args.commission)
            monthly_returns.append(float(result["return"]))
            print(
                f"{month}={float(result['return']):+.2f}% "
                f"final={float(result['final']):.2f} "
                f"trades={int(result['trades'])} "
                f"win={float(result['win_rate']):.1f}% "
                f"PF={'INF' if result['pf'] == float('inf') else f'{float(result[\"pf\"]):.2f}'} "
                f"DD={float(result['dd']):.2f}% "
                f"entries={int(result['entries'])} exits={int(result['exits'])}"
            )
        print(
            f"4M_COMPOUND={compound(monthly_returns):+.2f}% "
            f"avg={sum(monthly_returns) / len(monthly_returns):+.2f}% "
            f"worst={min(monthly_returns):+.2f}%"
        )


if __name__ == "__main__":
    main()
