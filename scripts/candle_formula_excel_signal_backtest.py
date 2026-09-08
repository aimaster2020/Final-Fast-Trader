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
    j = c.close - c.open       # J = C-O
    k = c.high - c.close       # K = H-C
    l = c.low - c.close        # L = Low-C
    m = c.high - c.open        # M = H-O
    n = c.low - c.open          # N = Low-O
    o = c.high - c.low          # O = H-L
    p = int(k > j)               # P = IF(K>J,1,0)
    q = int(k > m)               # Q = IF(K>M,1,0)
    r = int(l > j)               # R = IF(L>J,1,0)
    s = p + q + r                # S = SUM(P:R)
    return j, k, l, m, n, o, p, q, r, s


def excel_signal(s: int) -> int:
    # Expanded direction: S=3/2 -> BUY, S=0/1 -> SELL; no HOLD.
    if s in (2, 3):
        return 1
    return -1


def excel_i_for_row(j_current: float, j_next: float | None) -> int | None:
    # Exact Excel placement: I(row) = compare J(next row) vs J(current row).
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
    """Closed-entry-candle levels from A=abs(C-O).

    BUY : Entry=C, TP=C+2A, SL=C-A
    SELL: Entry=C, TP=C-2A, SL=C+A
    """
    a = abs(close - open_)
    if side == 1:
        return a, close + 2.0 * a, close - a
    return a, close - 2.0 * a, close + a


def run_month(rows: list[tuple[str, Candle]], month: str) -> dict[str, float | int]:
    month_indices = [i for i, (m, _) in enumerate(rows) if m == month]
    if not month_indices:
        return {
            "return": 0.0,
            "trades": 0,
            "wins": 0,
            "predictions": 0,
            "correct": 0,
            "dd": 0.0,
        }

    equity = CAPITAL
    position: Position | None = None
    peak = CAPITAL
    max_dd = 0.0
    predictions = correct = trades = wins = 0
    entry_idx: int | None = None

    for idx in month_indices:
        _, candle = rows[idx]
        j, *_rest, s = excel_columns(candle)
        signal = excel_signal(s)

        next_candle = rows[idx + 1][1] if idx + 1 < len(rows) else None
        next_j = None if next_candle is None else next_candle.close - next_candle.open
        actual = excel_i_for_row(j, next_j)
        current_correct = actual is not None and signal == actual

        predictions += 1
        correct += int(current_correct)

        # Entry is based only on the current CLOSED candle direction signal.
        if position is None:
            abs_body, target, stop = position_levels(candle.close, candle.open, signal)
            position = Position(
                side=signal,
                entry=candle.close,
                capital=equity,
                abs_body=abs_body,
                target=target,
                stop=stop,
            )
            entry_idx = idx

        # Exit is checked only on a later CLOSED candle.
        if position is not None and entry_idx is not None and idx > entry_idx:
            close_price = candle.close
            trade_pnl: float | None = None
            if position.side == 1:
                if close_price >= position.target:
                    trade_pnl = (position.target - position.entry) / position.entry * 100.0
                elif close_price <= position.stop:
                    trade_pnl = (position.stop - position.entry) / position.entry * 100.0
            else:
                if close_price <= position.target:
                    trade_pnl = (position.entry - position.target) / position.entry * 100.0
                elif close_price >= position.stop:
                    trade_pnl = (position.entry - position.stop) / position.entry * 100.0

            if trade_pnl is not None:
                equity += position.capital * trade_pnl / 100.0
                trades += 1
                wins += int(trade_pnl > 0)
                position = None
                entry_idx = None

        # Force-close at the last CLOSED candle of the month if TP/SL was not hit.
        next_month = rows[idx + 1][0] if idx + 1 < len(rows) else None
        if position is not None and next_month != month:
            trade_pnl = pnl_pct(position.side, position.entry, candle.close)
            equity += position.capital * trade_pnl / 100.0
            trades += 1
            wins += int(trade_pnl > 0)
            position = None
            entry_idx = None

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
        description="Excel candle formula backtest with expanded direction: S=3/2 BUY and S=0/1 SELL."
    )
    ap.add_argument("--input", default="reports/prepared_price_action_1h.csv")
    args = ap.parse_args()

    path = Path(args.input)
    print(
        "EXCEL_DIRECTION_TEST | tf=1h | fee=0 | signal=S3/S2_BUY + S0/S1_SELL | no_hold | "
        "I(row)=IF(J_next>J_current,1,IF(J_next<J_current,-1,0)) | "
        "TP=+/-2xABS(C-O) | SL=+/-1xABS(C-O) | close_only"
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
