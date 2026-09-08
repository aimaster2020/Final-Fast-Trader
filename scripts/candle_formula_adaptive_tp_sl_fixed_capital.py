#!/usr/bin/env python3
"""Fixed-capital comparison for the exact Excel S0-S3 strategy.

Compares the existing baseline TP/SL with adaptive TP/SL while keeping the
trade capital fixed at $1,000 for every completed trade. This isolates the
exit-rule effect from compounding position size.

Baseline: TP=2x current candle body, SL=1x body.
Adaptive: TP=walk-forward median historical relative move for the same S
state x current body; SL=0.5x TP. No future data is used.

Execution is close-only, fee=0, same four symbols/months as the canonical test.
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
FIXED_CAPITAL = 1000.0
MIN_BODY_RATIO = 0.0001

@dataclass
class Row:
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
    target: float
    stop: float


def excel_s(r: Row) -> int:
    j = r.close - r.open
    k = r.high - r.close
    m = r.high - r.open
    l = r.low - r.close
    return int(k > j) + int(k > m) + int(l > j)


def side_from_s(s: int) -> int:
    return 1 if s in (2, 3) else -1


def load_rows(path: Path, symbol: str) -> list[Row]:
    out: list[Row] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for rec in csv.DictReader(f):
            if rec.get("symbol") != symbol:
                continue
            try:
                out.append(Row(
                    str(rec["month"]), int(float(rec["timestamp"])),
                    float(rec["open"]), float(rec["high"]),
                    float(rec["low"]), float(rec["close"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(out, key=lambda x: x.timestamp)


def pnl_pct(side: int, entry: float, price: float) -> float:
    return side * (price - entry) / entry * 100.0 if entry > 0 else 0.0


def levels(close: float, body: float, side: int, tp_mult: float) -> tuple[float, float]:
    effective_body = max(body, abs(close) * MIN_BODY_RATIO)
    tp_dist = effective_body * max(tp_mult, MIN_BODY_RATIO)
    sl_dist = tp_dist * 0.5
    if side == 1:
        return close + tp_dist, close - sl_dist
    return close - tp_dist, close + sl_dist


def trade_return_pct(position: Position, exit_price: float) -> float:
    return pnl_pct(position.side, position.entry, exit_price)


def run_month(rows: list[Row], month: str, adaptive: bool) -> dict[str, float]:
    data = [r for r in rows if r.month == month]
    if len(data) < 2:
        return {"ret": 0.0, "trades": 0, "wins": 0, "dd": 0.0}

    # Past-only history inside the test month, matching the prior adaptive test.
    history: dict[int, list[float]] = defaultdict(list)
    global_history: list[float] = []
    position: Position | None = None
    entry_index = -1
    realized_equity = FIXED_CAPITAL
    peak = FIXED_CAPITAL
    max_dd = 0.0
    total_pnl = 0.0
    trades = wins = 0

    for i, cur in enumerate(data):
        s = excel_s(cur)
        side = side_from_s(s)
        body = abs(cur.close - cur.open)
        valid = cur.close != 0.0 and body / abs(cur.close) >= MIN_BODY_RATIO

        pred_ratio = None
        if adaptive and valid:
            pred_ratio = statistics.median(history[s]) if history[s] else (
                statistics.median(global_history) if global_history else None
            )

        if position is None and i < len(data) - 1:
            tp_mult = (pred_ratio if pred_ratio is not None else 2.0) if adaptive else 2.0
            target, stop = levels(cur.close, body, side, tp_mult)
            position = Position(side, cur.close, target, stop)
            entry_index = i

        if position is not None and i > entry_index:
            exit_price = None
            if position.side == 1:
                if cur.close >= position.target:
                    exit_price = position.target
                elif cur.close <= position.stop:
                    exit_price = position.stop
            else:
                if cur.close <= position.target:
                    exit_price = position.target
                elif cur.close >= position.stop:
                    exit_price = position.stop

            if exit_price is not None:
                r = trade_return_pct(position, exit_price)
                pnl = FIXED_CAPITAL * r / 100.0
                total_pnl += pnl
                realized_equity += pnl
                trades += 1
                wins += int(pnl > 0)
                position = None
                entry_index = -1

        # The newly observed target is available only after this row's decision.
        if adaptive and valid and i + 1 < len(data):
            nxt = data[i + 1]
            ratio = abs(nxt.close - cur.close) / body
            if ratio == ratio:
                history[s].append(ratio)
                global_history.append(ratio)

        mtm = realized_equity
        if position is not None:
            mtm += FIXED_CAPITAL * pnl_pct(position.side, position.entry, cur.close) / 100.0
        peak = max(peak, mtm)
        max_dd = max(max_dd, (peak - mtm) / peak if peak else 0.0)

    if position is not None:
        r = trade_return_pct(position, data[-1].close)
        pnl = FIXED_CAPITAL * r / 100.0
        total_pnl += pnl
        realized_equity += pnl
        trades += 1
        wins += int(pnl > 0)

    # Return is expressed relative to a fresh $1,000 stake per trade:
    # total_pnl / fixed stake, not compounded equity.
    return {"ret": total_pnl / FIXED_CAPITAL * 100.0,
            "trades": trades, "wins": wins, "dd": max_dd * 100.0}


def compound(xs: list[float]) -> float:
    x = 1.0
    for v in xs:
        x *= 1.0 + v / 100.0
    return (x - 1.0) * 100.0


def run_symbol(rows: list[Row], symbol: str, adaptive: bool) -> None:
    tag = "ADAPTIVE" if adaptive else "BASELINE"
    vals: list[float] = []
    trades = wins = 0
    max_dd = 0.0
    print(f"\n{symbol} {tag} | fixed_trade_capital=$1000")
    for month in MONTHS:
        r = run_month(rows, month, adaptive)
        vals.append(r["ret"])
        trades += int(r["trades"])
        wins += int(r["wins"])
        max_dd = max(max_dd, r["dd"])
        win = 100.0 * r["wins"] / r["trades"] if r["trades"] else 0.0
        print(f"{month} pnl={r['ret']:+.2f}% trades={r['trades']} win={win:.1f}% DD={r['dd']:.2f}%")
    print(f"4M fixed_pnl={sum(vals):+.2f}% compound_reference={compound(vals):+.2f}% trades={trades} win={(100.0*wins/trades if trades else 0.0):.1f}% DD={max_dd:.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = ap.parse_args()
    print("EXCEL_ADAPTIVE_TPSL_FIXED | tf=1h | fee=0 | exact S0-S3 | close_only | past_only")
    print("fixed trade capital isolates exit effect; baseline TP=2xbody SL=1xbody; adaptive TP=state_median(relative_move)xbody SL=0.5xTP")
    for symbol in SYMBOLS:
        rows = load_rows(args.input, symbol)
        run_symbol(rows, symbol, adaptive=False)
        run_symbol(rows, symbol, adaptive=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
