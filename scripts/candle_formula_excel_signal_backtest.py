from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from fast_pattern_trader.models import Candle

MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
CAPITAL = 1000.0


@dataclass
class Position:
    side: int
    entry: float
    capital: float
    abs_body: float
    target: float
    stop: float


def load_all(path: Path, symbol: str) -> list[tuple[str, Candle]]:
    rows: list[tuple[str, Candle]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("symbol") != symbol:
                continue
            try:
                rows.append(
                    (
                        row["month"],
                        Candle(
                            int(float(row["timestamp"])),
                            float(row["open"]),
                            float(row["high"]),
                            float(row["low"]),
                            float(row["close"]),
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(rows, key=lambda x: x[1].timestamp)


def excel_columns(c: Candle) -> tuple[float, float, float, float, float, float, int, int, int, int]:
    j = c.close - c.open
    k = c.high - c.close
    l = c.low - c.close
    m = c.high - c.open
    n = c.low - c.open
    o = c.high - c.low
    p = int(k > j)
    q = int(k > m)
    r = int(l > j)
    s = p + q + r
    return j, k, l, m, n, o, p, q, r, s


def excel_signal(s: int) -> int:
    if s == 3:
        return 1
    if s == 0:
        return -1
    return 0


def excel_i_for_row(j_current: float, j_next: float | None) -> int | None:
    if j_next is None:
        return None
    if j_next > j_current:
        return 1
    if j_next < j_current:
        return -1
    return 0


def pnl_pct(side: int, entry: float, price: float) -> float:
    if entry <= 0:
        return 0.0
    return side * (price - entry) / entry * 100.0


def position_levels(close: float, open_: float, side: int) -> tuple[float, float, float]:
    """A = abs(C-O); TP uses 2A, SL uses half of 2A = A."""
    a = abs(close - open_)
    if side == 1:
        return a, close + 2.0 * a, close - a
    return a, close - 2.0 * a, close + a


def run_month(rows: list[tuple[str, Candle]], month: str) -> dict[str, float | int]:
    month_indices = [i for i, (m, _) in enumerate(rows) if m == month]
    if not month_indices:
        return {"return": 0.0, "trades": 0, "wins": 0, "predictions": 0, "correct": 0, "dd": 0.0}

    equity = CAPITAL
    position: Position | None = None
    peak = CAPITAL
    max_dd = 0.0
    predictions = correct = trades = wins = 0

    for idx in month_indices:
        _, candle = rows[idx]
        j, *_rest, s = excel_columns(candle)
        signal = excel_signal(s)

        next_candle = rows[idx + 1][1] if idx + 1 < len(rows) else None
        next_j = None if next_candle is None else next_candle.close - next_candle.open
        actual = excel_i_for_row(j, next_j)
        current_correct = signal != 0 and actual is not None and signal == actual

        if signal != 0:
            predictions += 1
            correct += int(current_correct)

        # All trading decisions use CLOSED candle prices only.
        # First close an existing position, then optionally open a new one
        # from the same confirmed close if a new signal is present.
        closed_this_candle = False
        if position is not None:
            close_price = candle.close
            trade_pnl: float | None = None
            if position.side == 1:
                if close_price >= position.target:
                    trade_pnl = 2.0 * position.abs_body / position.entry * 100.0
                elif close_price <= position.stop:
                    trade_pnl = -position.abs_body / position.entry * 100.0
            else:
                if close_price <= position.target:
                    trade_pnl = 2.0 * position.abs_body / position.entry * 100.0
                elif close_price >= position.stop:
                    trade_pnl = -position.abs_body / position.entry * 100.0

            if trade_pnl is not None:
                equity += position.capital * trade_pnl / 100.0
                trades += 1
                wins += int(trade_pnl > 0)
                position = None
                closed_this_candle = True

        if position is None and signal != 0 and not closed_this_candle:
            abs_body, target, stop = position_levels(candle.close, candle.open, signal)
            position = Position(
                side=signal,
                entry=candle.close,
                capital=equity,
                abs_body=abs_body,
                target=target,
                stop=stop,
            )

        # Force close at the final CLOSED candle of each month if TP/SL was not hit.
        next_month = rows[idx + 1][0] if idx + 1 < len(rows) else None
        if position is not None and next_month != month:
            trade_pnl = pnl_pct(position.side, position.entry, candle.close)
            equity += position.capital * trade_pnl / 100.0
            trades += 1
            wins += int(trade_pnl > 0)
            position = None

        mtm = equity
        if position is not None:
            floating_pct = pnl_pct(position.side, position.entry, candle.close)
            mtm += position.capital * floating_pct / 100.0
        peak = max(peak, mtm)
        max_dd = max(max_dd, (peak - mtm) / peak if peak else 0.0)

    return {
        "return": (equity / CAPITAL - 1.0) * 100.0,
        "trades": trades,
        "wins": wins,
        "predictions": predictions,
        "correct": correct,
        "dd": max_dd * 100.0,
    }


def compound(values: list[float]) -> float:
    x = 1.0
    for value in values:
        x *= 1.0 + value / 100.0
    return (x - 1.0) * 100.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Exact Excel candle formula backtest; closed-candle entry/exit; TP=2*abs(C-O), SL=abs(C-O); no commission."
    )
    ap.add_argument("--input", default="reports/prepared_price_action_1h.csv")
    args = ap.parse_args()

    path = Path(args.input)
    print(
        "EXCEL_EXACT_BACKTEST | tf=1h | fee=0 | entry=CLOSE_SIGNAL | "
        "TP=2xABS(C-O) | SL=ABS(C-O) | close_only | "
        "signal=S3_BUY/S0_SELL/S1,S2_HOLD | "
        "I(row)=IF(J_next>J_current,1,IF(J_next<J_current,-1,0))"
    )

    for symbol in SYMBOLS:
        rows = load_all(path, symbol)
        monthly_returns: list[float] = []
        total_trades = total_wins = total_predictions = total_correct = 0
        max_dd = 0.0
        print(f"\n{symbol}")
        for month in MONTHS:
            r = run_month(rows, month)
            ret = float(r["return"])
            monthly_returns.append(ret)
            trades = int(r["trades"])
            wins = int(r["wins"])
            predictions = int(r["predictions"])
            correct = int(r["correct"])
            acc = correct / predictions * 100.0 if predictions else 0.0
            total_trades += trades
            total_wins += wins
            total_predictions += predictions
            total_correct += correct
            max_dd = max(max_dd, float(r["dd"]))
            print(
                f"{month} return={ret:+.2f}% trades={trades} "
                f"win={(wins / trades * 100.0 if trades else 0.0):.1f}% "
                f"pred={predictions} acc={acc:.1f}% DD={float(r['dd']):.2f}%"
            )

        print(
            f"4M compound={compound(monthly_returns):+.2f}% "
            f"avg={sum(monthly_returns)/len(monthly_returns):+.2f}% "
            f"worst={min(monthly_returns):+.2f}% trades={total_trades} "
            f"win={(total_wins/total_trades*100.0 if total_trades else 0.0):.1f}% "
            f"pred={total_predictions} acc={(total_correct/total_predictions*100.0 if total_predictions else 0.0):.1f}% "
            f"DD={max_dd:.2f}%"
        )


if __name__ == "__main__":
    main()
