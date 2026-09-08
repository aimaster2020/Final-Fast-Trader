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
    """Exact user Excel J:O and P:R formulas."""
    j = c.close - c.open       # J = G-D = C-O
    k = c.high - c.close      # K = E-G = H-C
    l = c.low - c.close       # L = F-G = Low-C
    m = c.high - c.open       # M = E-D = H-O
    n = c.low - c.open        # N = F-D = Low-O
    o = c.high - c.low        # O = E-F = H-L
    p = int(k > j)             # P = IF(K>J,1,0)
    q = int(k > m)             # Q = IF(K>M,1,0)
    r = int(l > j)             # R = IF(L>J,1,0)
    s = p + q + r              # S = SUM(P:R)
    return j, k, l, m, n, o, p, q, r, s


def excel_signal(s: int) -> int:
    """Signal implied by S: 3=BUY, 0=SELL, 1/2=HOLD."""
    if s == 3:
        return 1
    if s == 0:
        return -1
    return 0


def excel_i_for_row(j_current: float, j_next: float | None) -> int | None:
    """Exact Excel I placement: in row N, compare J(N+1) with J(N)."""
    if j_next is None:
        return None
    if j_next > j_current:
        return 1
    if j_next < j_current:
        return -1
    return 0


def in_bias(body: float, width: float) -> bool:
    return -width <= body <= width


def pnl(side: int, entry: float, price: float) -> float:
    return side * (price - entry) / entry if entry > 0 else 0.0


def run_symbol(rows: list[tuple[str, Candle]], width: float) -> dict[str, dict[str, float | int]]:
    """Run exact formula timing without look-ahead in the trading layer.

    Excel row N:
      signal_N = f(S_N)
      I_N = direction(J_(N+1), J_N)
      T/U_N = whether signal_N correctly predicts I_N

    A trade at row N may use only information known by row N:
      - previous row's prediction correctness (signal_(N-1) vs I_(N-1))
      - current row's signal_N
      - current row's body/bias state
    """
    equity = CAPITAL
    position: Position | None = None
    peak = CAPITAL
    max_dd = 0.0
    previous_prediction = 0
    previous_prediction_correct = False
    results: dict[str, dict[str, float | int]] = {}

    for i, (month, candle) in enumerate(rows):
        if month not in MONTHS:
            continue

        j, *_rest, s = excel_columns(candle)
        current_signal = excel_signal(s)

        next_candle = rows[i + 1][1] if i + 1 < len(rows) else None
        next_month = rows[i + 1][0] if i + 1 < len(rows) else None
        next_j = None if next_candle is None else next_candle.close - next_candle.open
        actual_direction = excel_i_for_row(j, next_j)
        current_correct = actual_direction is not None and current_signal != 0 and current_signal == actual_direction

        bucket = results.setdefault(month, {"return": 0.0, "trades": 0, "wins": 0, "predictions": 0, "correct": 0, "dd": 0.0})
        if current_signal != 0 and actual_direction is not None:
            bucket["predictions"] += 1
            bucket["correct"] += int(current_correct)

        body = j
        bias = in_bias(body, width)
        entered = False
        closed = False

        if position is None:
            if current_signal != 0 and not bias and previous_prediction_correct:
                position = Position(current_signal, candle.close, equity)
                entered = True
        elif bias:
            gross = position.capital * pnl(position.side, position.entry, candle.close)
            equity += gross
            bucket["trades"] += 1
            bucket["wins"] += int(gross > 0)
            position = None
            closed = True

        # Force-close at month end, matching prior test convention.
        if position is not None and next_month != month:
            gross = position.capital * pnl(position.side, position.entry, candle.close)
            equity += gross
            bucket["trades"] += 1
            bucket["wins"] += int(gross > 0)
            position = None
            closed = True

        mtm = equity
        if position is not None:
            mtm += position.capital * pnl(position.side, position.entry, candle.close)
        peak = max(peak, mtm)
        max_dd = max(max_dd, (peak - mtm) / peak if peak else 0.0)
        bucket["dd"] = max(bucket["dd"], max_dd * 100.0)

        previous_prediction = current_signal
        previous_prediction_correct = bool(current_correct)
        _ = entered, closed, previous_prediction

    # Convert monthly equity changes to percentage returns requires replaying each month.
    # Re-run each month in isolation while preserving the exact Excel I look-ahead across rows.
    for month in MONTHS:
        month_rows = [(m, c) for m, c in rows if m == month]
        if not month_rows:
            results.setdefault(month, {"return": 0.0, "trades": 0, "wins": 0, "predictions": 0, "correct": 0, "dd": 0.0})
            continue
        # Keep the overall accumulated results above for diagnostics; return is filled below
        # from a dedicated month-level replay.
        results[month]["return"] = _run_month_return(rows, month, width)

    return results


def _run_month_return(rows: list[tuple[str, Candle]], month: str, width: float) -> float:
    month_indices = [i for i, (m, _) in enumerate(rows) if m == month]
    if not month_indices:
        return 0.0

    equity = CAPITAL
    position: Position | None = None
    previous_prediction_correct = False

    for idx in month_indices:
        _, candle = rows[idx]
        j, *_rest, s = excel_columns(candle)
        signal = excel_signal(s)

        next_candle = rows[idx + 1][1] if idx + 1 < len(rows) else None
        next_j = None if next_candle is None else next_candle.close - next_candle.open
        actual = excel_i_for_row(j, next_j)
        current_correct = actual is not None and signal != 0 and signal == actual

        if position is None:
            if signal != 0 and not in_bias(j, width) and previous_prediction_correct:
                position = Position(signal, candle.close, equity)
        elif in_bias(j, width):
            equity += position.capital * pnl(position.side, position.entry, candle.close)
            position = None

        next_month = rows[idx + 1][0] if idx + 1 < len(rows) else None
        if position is not None and next_month != month:
            equity += position.capital * pnl(position.side, position.entry, candle.close)
            position = None

        previous_prediction_correct = bool(current_correct)

    return (equity / CAPITAL - 1.0) * 100.0


def compound(values: list[float]) -> float:
    x = 1.0
    for value in values:
        x *= 1.0 + value / 100.0
    return (x - 1.0) * 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact user Excel candle formula backtest; no commission.")
    ap.add_argument("--input", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--width", type=float, default=1.0)
    args = ap.parse_args()

    path = Path(args.input)
    print(
        f"EXCEL_EXACT_BACKTEST | tf=1h | fee=0 | width={args.width:g} | "
        "I(row)=IF(J_next>J_current,1,IF(J_next<J_current,-1,0)) | "
        "signal=S3_BUY/S0_SELL/S1,S2_HOLD"
    )

    for symbol in SYMBOLS:
        rows = load_all(path, symbol)
        results = run_symbol(rows, args.width)
        monthly_returns: list[float] = []
        total_trades = total_wins = total_predictions = total_correct = 0
        max_dd = 0.0
        print(f"\n{symbol}")

        for month in MONTHS:
            r = results.get(month, {"return": 0.0, "trades": 0, "wins": 0, "predictions": 0, "correct": 0, "dd": 0.0})
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
                f"{month} return={ret:+.2f}% trades={trades} win={(wins/trades*100.0 if trades else 0.0):.1f}% "
                f"pred={predictions} acc={acc:.1f}% DD={float(r['dd']):.2f}%"
            )

        print(
            f"4M compound={compound(monthly_returns):+.2f}% avg={sum(monthly_returns)/len(monthly_returns):+.2f}% "
            f"worst={min(monthly_returns):+.2f}% trades={total_trades} "
            f"win={(total_wins/total_trades*100.0 if total_trades else 0.0):.1f}% "
            f"pred={total_predictions} acc={(total_correct/total_predictions*100.0 if total_predictions else 0.0):.1f}% "
            f"DD={max_dd:.2f}%"
        )


if __name__ == "__main__":
    main()
