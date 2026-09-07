from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal

TRAIN_MONTHS = ("2026-05", "2026-06", "2026-07")
VALIDATION_MONTH = "2026-08"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
COMMISSION = 0.0013
CAPITAL = 1000.0
STEP = 25.0
MIN_BIAS = -300.0
MAX_BIAS = 300.0

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
                out.append(Candle(int(float(row["timestamp"])), float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(out, key=lambda c: c.timestamp)


def sig(c: Candle) -> int:
    s = decide(c).signal
    return 1 if s == Signal.BUY else -1 if s == Signal.SELL else 0


def body_dir(c: Candle) -> int:
    d = c.close - c.open
    return 1 if d > 0 else -1 if d < 0 else 0


def in_bias(body: float, lo: float, hi: float) -> bool:
    return lo <= body <= hi


def pnl(side: int, entry: float, price: float) -> float:
    return side * (price - entry) / entry if entry > 0 else 0.0


def run(candles: list[Candle], lo: float, hi: float) -> dict[str, float | int]:
    equity = CAPITAL
    position: Position | None = None
    prev_pred = 0
    trades = wins = entries = exits = 0
    gross_profit = gross_loss = 0.0
    peak = CAPITAL
    max_dd = 0.0

    for c in candles:
        s = sig(c)
        body = c.close - c.open
        prev_correct = prev_pred != 0 and prev_pred == body_dir(c)

        if position is None:
            # Same production candidate logic: current signal + outside bias + previous prediction correct.
            if s != 0 and not in_bias(body, lo, hi) and prev_correct:
                position = Position(s, c.close, equity)
                entries += 1
        else:
            # Same production exit logic: body returns into the same bias range.
            if in_bias(body, lo, hi):
                gross = position.capital * pnl(position.side, position.entry, c.close)
                net = gross - position.capital * COMMISSION * 2.0
                equity += net
                trades += 1
                wins += int(net > 0)
                gross_profit += max(gross, 0.0)
                gross_loss += min(gross, 0.0)
                position = None
                exits += 1

        mtm = equity
        if position is not None:
            mtm += position.capital * pnl(position.side, position.entry, c.close)
        peak = max(peak, mtm)
        max_dd = max(max_dd, (peak - mtm) / peak if peak else 0.0)
        prev_pred = s

    if position is not None and candles:
        gross = position.capital * pnl(position.side, position.entry, candles[-1].close)
        net = gross - position.capital * COMMISSION * 2.0
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


def candidates() -> list[tuple[float, float]]:
    vals = [MIN_BIAS + i * STEP for i in range(int((MAX_BIAS - MIN_BIAS) / STEP) + 1)]
    out: list[tuple[float, float]] = []
    for lo in vals:
        for hi in vals:
            if lo >= hi:
                continue
            if lo <= 0 <= hi:
                # Include a compact set of cross-zero ranges; most earlier useful biases were same-sign.
                if lo not in (-50.0, -25.0) or hi not in (25.0, 50.0):
                    continue
            out.append((lo, hi))
    return out


def score(train_returns: list[float]) -> tuple[int, float, float, float]:
    positives = sum(r > 0 for r in train_returns)
    avg = sum(train_returns) / len(train_returns)
    worst = min(train_returns)
    compound = 1.0
    for r in train_returns:
        compound *= 1.0 + r / 100.0
    comp = (compound - 1.0) * 100.0
    # Prefer more positive months, then compound return, then average, then less-bad worst month.
    return positives, comp, avg, worst


def main() -> None:
    ap = argparse.ArgumentParser(description="1h walk-forward search for bias using prior prediction confirmation and bias-range exit")
    ap.add_argument("--input", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    path = Path(args.input)

    print("BIAS_WF | tf=1h | train=May-Jun-Jul | validation=Aug | fee=0.13%/side | entry=previous-prediction-correct | exit=same-bias")

    for symbol in SYMBOLS:
        train = [load(path, m, symbol) for m in TRAIN_MONTHS]
        valid = load(path, VALIDATION_MONTH, symbol)
        rows: list[tuple[tuple[int, float, float, float], float, float, dict[str, float | int]]] = []
        for lo, hi in candidates():
            rs = [run(cs, lo, hi) for cs in train]
            rets = [float(r["return"]) for r in rs]
            # Avoid selecting degenerate no-trade ranges unless no alternatives exist.
            total_trades = sum(int(r["trades"]) for r in rs)
            if total_trades == 0:
                continue
            rows.append((score(rets), lo, hi, run(valid, lo, hi)))

        rows.sort(key=lambda x: x[0], reverse=True)
        print(f"\n{symbol}")
        for rank, (sc, lo, hi, vr) in enumerate(rows[: args.top], 1):
            p = "INF" if vr["pf"] == float("inf") else f"{float(vr['pf']):.2f}"
            print(f"#{rank} bias=[{lo:.0f},{hi:.0f}] train_pos={sc[0]}/3 train_comp={sc[1]:+.2f}% train_avg={sc[2]:+.2f}% train_worst={sc[3]:+.2f}% | AUG={float(vr['return']):+.2f}% final={float(vr['final']):.2f} trades={int(vr['trades'])} win={float(vr['win_rate']):.1f}% PF={p} DD={float(vr['dd']):.2f}%")

if __name__ == "__main__":
    main()
