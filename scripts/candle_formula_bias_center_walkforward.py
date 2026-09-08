from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle

TRAIN_MONTHS = ("2026-05", "2026-06", "2026-07")
VALIDATION_MONTH = "2026-08"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
CAPITAL = 1000.0
DEFAULT_COMMISSION = 0.0013
STEP = 25.0
MIN_BIAS = -300.0
MAX_BIAS = 300.0
BIAS_HALF_WIDTH = 1.0


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
    d = decide(candle)
    # Main rule: S=3 or 2 -> BUY, S=0 or 1 -> SELL.
    return 1 if d.up_score >= 2 else -1


def body_direction(candle: Candle) -> int:
    body = candle.close - candle.open
    return 1 if body > 0 else -1 if body < 0 else 0


def in_bias(body: float, bias: float) -> bool:
    return (bias - BIAS_HALF_WIDTH) <= body <= (bias + BIAS_HALF_WIDTH)


def pnl(side: int, entry: float, price: float) -> float:
    return side * (price - entry) / entry if entry > 0 else 0.0


def run(candles: list[Candle], bias: float, commission: float) -> dict[str, float | int]:
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
        previous_correct = previous_prediction != 0 and previous_prediction == body_direction(candle)

        if position is None:
            if not in_bias(body, bias) and previous_correct:
                position = Position(current_signal, candle.close, equity)
                entries += 1
        elif in_bias(body, bias):
            gross = position.capital * pnl(position.side, position.entry, candle.close)
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
            mtm += position.capital * pnl(position.side, position.entry, candle.close)
        peak = max(peak, mtm)
        max_dd = max(max_dd, (peak - mtm) / peak if peak else 0.0)
        previous_prediction = current_signal

    if position is not None and candles:
        gross = position.capital * pnl(position.side, position.entry, candles[-1].close)
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
    }


def compound(returns: list[float]) -> float:
    factor = 1.0
    for value in returns:
        factor *= 1.0 + value / 100.0
    return (factor - 1.0) * 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward search of bias-center for a +/-1 body range")
    ap.add_argument("--input", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--min-trades", type=int, default=15)
    ap.add_argument("--min-positive-months", type=int, default=2)
    ap.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    ap.add_argument("--min-bias", type=float, default=MIN_BIAS)
    ap.add_argument("--max-bias", type=float, default=MAX_BIAS)
    ap.add_argument("--step", type=float, default=STEP)
    args = ap.parse_args()
    path = Path(args.input)

    fee_pct = args.commission * 100.0
    print(
        f"BIAS_CENTER_WF | tf=1h | range=[bias-1,bias+1] | "
        f"main=S3,2_BUY/S0,1_SELL | train=May-Jun-Jul | validation=Aug | "
        f"fee={fee_pct:.2f}%/side | min_trades={args.min_trades} | "
        f"min_positive_months={args.min_positive_months}"
    )

    centers: list[float] = []
    x = args.min_bias
    while x <= args.max_bias + 1e-9:
        centers.append(round(x, 10))
        x += args.step

    for symbol in SYMBOLS:
        train = [load(path, month, symbol) for month in TRAIN_MONTHS]
        valid = load(path, VALIDATION_MONTH, symbol)
        rows: list[tuple[tuple[int, float, float, float], float, dict[str, float | int]]] = []

        for bias in centers:
            results = [run(candles, bias, args.commission) for candles in train]
            returns = [float(r["return"]) for r in results]
            total_trades = sum(int(r["trades"]) for r in results)
            positive_months = sum(r > 0 for r in returns)
            if total_trades < args.min_trades or positive_months < args.min_positive_months:
                continue
            score = (positive_months, compound(returns), sum(returns) / 3.0, min(returns))
            rows.append((score, bias, run(valid, bias, args.commission)))

        rows.sort(key=lambda item: item[0], reverse=True)
        print(f"\n{symbol}")
        if not rows:
            print("NO_BIAS_PASSED_FILTER")
            continue

        for rank, (score, bias, vr) in enumerate(rows[: args.top], 1):
            pf = vr["pf"]
            pf_text = "INF" if pf == float("inf") else f"{float(pf):.2f}"
            lo = bias - BIAS_HALF_WIDTH
            hi = bias + BIAS_HALF_WIDTH
            print(
                f"#{rank} bias={bias:+.0f} range=[{lo:+.0f},{hi:+.0f}] "
                f"train_pos={score[0]}/3 train_comp={score[1]:+.2f}% "
                f"train_avg={score[2]:+.2f}% train_worst={score[3]:+.2f}% | "
                f"AUG={float(vr['return']):+.2f}% final={float(vr['final']):.2f} "
                f"trades={int(vr['trades'])} win={float(vr['win_rate']):.1f}% "
                f"PF={pf_text} DD={float(vr['dd']):.2f}%"
            )


if __name__ == "__main__":
    main()
