from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal

DEFAULT_MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
DEFAULT_COMMISSION = 0.0013

# Bias ranges selected in the earlier multi-month bias tests.
DEFAULT_BIAS = {
    "BTCUSDT": (-200.0, -125.0),
    "ETHUSDT": (-200.0, -175.0),
    "SOLUSDT": (25.0, 50.0),
    "XRPUSDT": (-200.0, -175.0),
}


@dataclass
class Position:
    side: int
    entry_price: float
    capital: float


def load_month(path: Path, month: str, symbol: str) -> list[Candle]:
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


def signal_side(candle: Candle) -> int:
    signal = decide(candle).signal
    if signal == Signal.BUY:
        return 1
    if signal == Signal.SELL:
        return -1
    return 0


def body_side(candle: Candle) -> int:
    body = candle.close - candle.open
    if body > 0:
        return 1
    if body < 0:
        return -1
    return 0


def pnl(side: int, entry: float, price: float) -> float:
    return side * (price - entry) / entry if entry > 0 else 0.0


def run(
    candles: list[Candle],
    lower: float,
    upper: float,
    commission: float,
) -> dict[str, float | int]:
    equity = 1000.0
    position: Position | None = None
    trades = wins = long_entries = short_entries = 0
    gross_profit = gross_loss = 0.0
    peak = 1000.0
    max_dd = 0.0
    previous_prediction = 0

    for candle in candles:
        sig = signal_side(candle)
        body = candle.close - candle.open

        # The previous candle's prediction must have correctly predicted
        # the current candle body direction before a new position may open.
        entry_confirmed = previous_prediction != 0 and previous_prediction == body_side(candle)

        if position is None:
            if sig != 0 and not (lower <= body <= upper) and entry_confirmed:
                position = Position(sig, candle.close, equity)
                long_entries += int(sig > 0)
                short_entries += int(sig < 0)
        # No opposite-signal exit. Position remains open until the end of
        # the month/test window, where it is force-closed below.

        mtm = equity
        if position is not None:
            mtm += position.capital * pnl(
                position.side,
                position.entry_price,
                candle.close,
            )
        peak = max(peak, mtm)
        max_dd = max(max_dd, (peak - mtm) / peak if peak else 0.0)

        previous_prediction = sig

    if position is not None and candles:
        side = position.side
        entry = position.entry_price
        capital = position.capital
        gross = capital * pnl(side, entry, candles[-1].close)
        fee = capital * commission * 2.0
        net = gross - fee
        equity += net
        trades += 1
        wins += int(net > 0)
        gross_profit += max(gross, 0.0)
        gross_loss += min(gross, 0.0)

    return {
        "final": equity,
        "return_pct": (equity / 1000.0 - 1.0) * 100.0,
        "trades": trades,
        "win_rate": wins / trades * 100.0 if trades else 0.0,
        "pf": gross_profit / abs(gross_loss) if gross_loss < 0 else (float("inf") if gross_profit else 0.0),
        "dd": max_dd * 100.0,
        "long": long_entries,
        "short": short_entries,
    }


def compound(returns: list[float]) -> float:
    factor = 1.0
    for r in returns:
        factor *= 1.0 + r / 100.0
    return (factor - 1.0) * 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description="1h candle formula strategy with prior bias and previous-prediction confirmation.")
    ap.add_argument("--timeframe", default="1h", choices=("1h",))
    ap.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    ap.add_argument("--input-1h", default="reports/prepared_price_action_1h.csv")
    args = ap.parse_args()

    months = tuple(x.strip() for x in args.months.split(",") if x.strip())
    symbols = tuple(x.strip() for x in args.symbols.split(",") if x.strip())
    path = Path(args.input_1h)

    print(
        f"CANDLE_FORMULA | tf={args.timeframe} | prior-bias | previous-prediction-confirmed-entry | "
        f"no-opposite-exit | fee={args.commission * 100:.2f}%/side | "
        f"roundtrip={args.commission * 200:.2f}% | capital=1000"
    )
    for symbol in symbols:
        lower, upper = DEFAULT_BIAS.get(symbol, (-100.0, 100.0))
        month_results: list[float] = []
        print(f"\n{symbol} | bias=[{lower:.0f},{upper:.0f}]")
        for month in months:
            candles = load_month(path, month, symbol)
            if not candles:
                print(f"{month} | NO_DATA")
                continue
            r = run(candles, lower, upper, args.commission)
            month_results.append(float(r["return_pct"]))
            pf = "INF" if r["pf"] == float("inf") else f"{float(r['pf']):.2f}"
            print(
                f"{month} | return={float(r['return_pct']):+.2f}% final={float(r['final']):.2f} "
                f"trades={int(r['trades'])} win={float(r['win_rate']):.1f}% PF={pf} "
                f"DD={float(r['dd']):.2f}% L={int(r['long'])} S={int(r['short'])}"
            )
        if month_results:
            print(
                f"{symbol} | 4M_COMPOUND={compound(month_results):+.2f}% | "
                f"AVG_MONTH={sum(month_results) / len(month_results):+.2f}% | "
                f"WORST={min(month_results):+.2f}%"
            )


if __name__ == "__main__":
    main()
