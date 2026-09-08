#!/usr/bin/env python3
"""Adaptive TP/SL backtest for the exact Excel S0-S3 candle formula.

Baseline: TP=2x entry candle body, SL=1x body.
Adaptive: predict the next close movement/body ratio using only prior
observations of the same S state; fallback to the past same-month global
median. TP=predicted ratio x effective body, SL=0.5 x TP (2:1 reward/risk).
Execution is close-only, matching the existing baseline backtest.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INPUT = Path("reports/prepared_price_action_1h.csv")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
CAPITAL = 1000.0
MIN_BODY_RATIO = 0.0001

@dataclass
class Row:
    symbol: str
    month: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float

@dataclass
class Position:
    side: int
    entry: float
    capital: float
    target: float
    stop: float


def excel_s(r: Row) -> int:
    j = r.close - r.open
    k = r.high - r.close
    m = r.high - r.open
    l = r.low - r.close
    return int(k > j) + int(k > m) + int(l > j)


def signal_from_s(s: int) -> int:
    return 1 if s in (2, 3) else -1


def load_rows(path: Path, symbol: str) -> list[Row]:
    out: list[Row] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for rec in csv.DictReader(f):
            if rec.get("symbol") != symbol:
                continue
            try:
                out.append(Row(
                    symbol,
                    str(rec["month"]),
                    int(float(rec["timestamp"])),
                    float(rec["open"]),
                    float(rec["high"]),
                    float(rec["low"]),
                    float(rec["close"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(out, key=lambda r: r.timestamp)


def pnl_pct(side: int, entry: float, price: float) -> float:
    return side * (price - entry) / entry * 100.0 if entry > 0 else 0.0


def close_trade(position: Position, price: float) -> float:
    return position.capital * pnl_pct(position.side, position.entry, price) / 100.0


def levels(close: float, side: int, body: float, tp_mult: float) -> tuple[float, float]:
    effective_body = max(body, abs(close) * MIN_BODY_RATIO)
    tp_dist = effective_body * tp_mult
    sl_dist = tp_dist * 0.5
    if side == 1:
        return close + tp_dist, close - sl_dist
    return close - tp_dist, close + sl_dist


def run_month(rows: list[Row], month: str, adaptive: bool) -> dict[str, float]:
    month_rows = [r for r in rows if r.month == month]
    if len(month_rows) < 2:
        return {"return": 0.0, "trades": 0, "wins": 0, "dd": 0.0, "pred": 0, "correct": 0}

    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    position: Position | None = None
    entry_index = -1
    trades = wins = predictions = correct = 0

    state_hist: dict[int, list[float]] = defaultdict(list)
    global_hist: list[float] = []

    for i, cur in enumerate(month_rows):
        s = excel_s(cur)
        side = signal_from_s(s)
        predictions += 1

        # Direction score is the same target used in the canonical test.
        if i + 1 < len(month_rows):
            j = cur.close - cur.open
            nj = month_rows[i + 1].close - month_rows[i + 1].open
            correct += int((side == 1 and nj > j) or (side == -1 and nj < j))

        body = abs(cur.close - cur.open)
        ratio_valid = cur.close != 0.0 and body / abs(cur.close) >= MIN_BODY_RATIO
        pred_ratio = None
        if ratio_valid:
            if state_hist[s]:
                pred_ratio = statistics.median(state_hist[s])
            elif global_hist:
                pred_ratio = statistics.median(global_hist)

        if position is None and i < len(month_rows) - 1:
            if adaptive:
                tp_mult = pred_ratio if pred_ratio is not None else 2.0
            else:
                tp_mult = 2.0
            target, stop = levels(cur.close, side, body, tp_mult)
            position = Position(side, cur.close, equity, target, stop)
            entry_index = i

        # Close-only exit, matching the canonical baseline.
        if position is not None and i > entry_index:
            trade_pnl: float | None = None
            if position.side == 1:
                if cur.close >= position.target:
                    trade_pnl = (position.target - position.entry) / position.entry * 100.0
                elif cur.close <= position.stop:
                    trade_pnl = (position.stop - position.entry) / position.entry * 100.0
            else:
                if cur.close <= position.target:
                    trade_pnl = (position.entry - position.target) / position.entry * 100.0
                elif cur.close >= position.stop:
                    trade_pnl = (position.entry - position.stop) / position.entry * 100.0
            if trade_pnl is not None:
                equity += position.capital * trade_pnl / 100.0
                trades += 1
                wins += int(trade_pnl > 0)
                position = None
                entry_index = -1

        # State target becomes available only after this row has been evaluated.
        if ratio_valid and i + 1 < len(month_rows):
            nxt = month_rows[i + 1]
            move_ratio = abs(nxt.close - cur.close) / body
            if move_ratio == move_ratio:
                state_hist[s].append(move_ratio)
                global_hist.append(move_ratio)

        if position is not None:
            mtm = equity + position.capital * pnl_pct(position.side, position.entry, cur.close) / 100.0
        else:
            mtm = equity
        peak = max(peak, mtm)
        max_dd = max(max_dd, (peak - mtm) / peak if peak else 0.0)

    if position is not None:
        equity += close_trade(position, month_rows[-1].close)
        trades += 1
        wins += int(pnl_pct(position.side, position.entry, month_rows[-1].close) > 0)

    return {
        "return": (equity / CAPITAL - 1.0) * 100.0,
        "trades": trades,
        "wins": wins,
        "dd": max_dd * 100.0,
        "pred": predictions,
        "correct": correct,
    }


def compound(xs: list[float]) -> float:
    v = 1.0
    for x in xs:
        v *= 1.0 + x / 100.0
    return (v - 1.0) * 100.0


def run_symbol(rows: list[Row], name: str, adaptive: bool) -> None:
    returns: list[float] = []
    total_trades = total_wins = total_pred = total_correct = 0
    max_dd = 0.0
    tag = "ADAPTIVE" if adaptive else "BASELINE"
    print(f"\n{name} {tag}")
    for month in MONTHS:
        r = run_month(rows, month, adaptive)
        returns.append(r["return"])
        total_trades += int(r["trades"])
        total_wins += int(r["wins"])
        total_pred += int(r["pred"])
        total_correct += int(r["correct"])
        max_dd = max(max_dd, float(r["dd"]))
        win = 100.0 * r["wins"] / r["trades"] if r["trades"] else 0.0
        acc = 100.0 * r["correct"] / r["pred"] if r["pred"] else 0.0
        print(f"{month} return={r['return']:+.2f}% trades={r['trades']} win={win:.1f}% acc={acc:.1f}% DD={r['dd']:.2f}%")
    win = 100.0 * total_wins / total_trades if total_trades else 0.0
    acc = 100.0 * total_correct / total_pred if total_pred else 0.0
    print(f"4M {tag.lower()}={compound(returns):+.2f}% avg={sum(returns)/len(returns):+.2f}% worst={min(returns):+.2f}% trades={total_trades} win={win:.1f}% acc={acc:.1f}% DD={max_dd:.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = ap.parse_args()
    print("EXCEL_ADAPTIVE_TPSL | tf=1h | fee=0 | exact S0-S3 | close_only | past_only")
    print("adaptive: TP=state_median(relative_move) x effective_body | SL=0.5xTP | baseline: TP=2xbody SL=1xbody")
    for symbol in SYMBOLS:
        rows = load_rows(args.input, symbol)
        run_symbol(rows, symbol, adaptive=False)
        run_symbol(rows, symbol, adaptive=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
