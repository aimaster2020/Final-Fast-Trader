#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest previous-window formula with capital, commission, and prediction-range filter.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--f3", type=float, default=100.0)
    p.add_argument("--f4", type=float, default=-150.0)
    p.add_argument("--window", type=int, choices=range(1, 6), default=3)
    p.add_argument("--commission-per-side", type=float, default=0.0)
    p.add_argument("--min-prediction-pct", type=float, default=0.0)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--trades-output", default=None, help="Optional CSV path for completed trade-level commission/profit details.")
    return p.parse_args()


def year_of(v: str) -> Optional[int]:
    try:
        import datetime as dt
        ts = int(float(v))
        if ts > 10_000_000_000:
            ts //= 1000
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).year
    except Exception:
        return None


def load_rows(path: Path, year: Optional[int]):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError("Input CSV is empty")
    header = [x.strip().lower() for x in rows[0]]
    headered = all(x in header for x in ("timestamp", "open", "high", "close"))
    out = []
    if headered:
        idx = {name: i for i, name in enumerate(header)}
        for r in rows[1:]:
            if len(r) <= max(idx["timestamp"], idx["open"], idx["high"], idx["close"]):
                continue
            ts = r[idx["timestamp"]]
            if year is not None and year_of(ts) != year:
                continue
            out.append((ts, float(r[idx["open"]]), float(r[idx["high"]]), float(r[idx["close"]])))
    else:
        for r in rows:
            if len(r) < 5:
                continue
            ts = r[0]
            if year is not None and year_of(ts) != year:
                continue
            out.append((ts, float(r[1]), float(r[2]), float(r[4])))
    return out


def direction(move: float) -> int:
    return 1 if move > 0 else -1 if move < 0 else 0


def run(rows, initial_capital: float, f3: float, f4: float, window: int, commission: float, min_prediction_pct: float):
    if len(rows) <= window + 1:
        raise ValueError("Not enough rows")
    capital = initial_capital
    position = 0
    entry_price = None
    entry_capital = None
    entry_capital_before_fee = None
    entry_fee = 0.0
    entry_timestamp = None
    trades = []
    entries = exits = 0
    wins = losses = 0
    commission_paid = 0.0
    equity_rows = []

    def charge_fee() -> float:
        nonlocal capital, commission_paid
        fee = capital * commission
        capital -= fee
        commission_paid += fee
        return fee

    def close_trade(exit_timestamp: str, exit_price: float) -> None:
        nonlocal capital, position, entry_price, entry_capital
        nonlocal entry_capital_before_fee, entry_fee, entry_timestamp
        nonlocal exits, wins, losses
        if not position or entry_price is None or entry_capital is None or entry_capital_before_fee is None:
            raise RuntimeError("Missing open trade state")

        gross_return = ((exit_price - entry_price) / entry_price) if position == 1 else ((entry_price - exit_price) / entry_price)
        gross_pnl = entry_capital * gross_return
        capital_before_exit_fee = capital + gross_pnl
        capital = capital_before_exit_fee
        exit_fee = charge_fee()

        net_pnl_after_both_fees = capital - entry_capital_before_fee
        total_fees = entry_fee + exit_fee
        trades.append({
            "entry_timestamp": entry_timestamp,
            "exit_timestamp": exit_timestamp,
            "side": "LONG" if position == 1 else "SHORT",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "capital_before_entry": entry_capital_before_fee,
            "entry_fee": entry_fee,
            "entry_fee_pct": entry_fee / entry_capital_before_fee * 100.0,
            "capital_after_entry_fee": entry_capital,
            "gross_return_pct": gross_return * 100.0,
            "gross_pnl": gross_pnl,
            "capital_before_exit_fee": capital_before_exit_fee,
            "exit_fee": exit_fee,
            "exit_fee_pct": exit_fee / capital_before_exit_fee * 100.0 if capital_before_exit_fee else 0.0,
            "total_fees": total_fees,
            "total_fee_pct_of_entry_capital": total_fees / entry_capital_before_fee * 100.0,
            "net_pnl_after_fees": net_pnl_after_both_fees,
            "net_return_pct": net_pnl_after_both_fees / entry_capital_before_fee * 100.0,
        })
        exits += 1
        if gross_pnl > 0:
            wins += 1
        else:
            losses += 1

    for i in range(window, len(rows) - 1):
        ts, o, _, c = rows[i]
        body = c - o
        if body > f3:
            pred = sum(rows[j][3] for j in range(i - window, i)) / window
        elif body < f4:
            pred = sum(rows[j][2] for j in range(i - window, i)) / window
        else:
            pred = c

        predicted_move_pct = abs(pred - c) / c * 100.0 if c else 0.0
        pt = direction(pred - c) if predicted_move_pct >= min_prediction_pct else 0
        changed = False

        if position == 0 and pt:
            entry_capital_before_fee = capital
            entry_fee = charge_fee()
            position = pt
            entry_price = c
            entry_capital = capital
            entry_timestamp = ts
            entries += 1
            changed = True
        elif position and pt and pt != position:
            close_trade(ts, c)
            entry_capital_before_fee = capital
            entry_fee = charge_fee()
            position = pt
            entry_price = c
            entry_capital = capital
            entry_timestamp = ts
            entries += 1
            changed = True

        if position and entry_price is not None and entry_capital is not None:
            unrealized_return = ((c - entry_price) / entry_price) if position == 1 else ((entry_price - c) / entry_price)
            mark_equity = capital + entry_capital * unrealized_return
        else:
            mark_equity = capital

        equity_rows.append({
            "timestamp": ts,
            "close": c,
            "predicted_move_pct": predicted_move_pct,
            "pt": pt,
            "position": position,
            "capital": capital,
            "mark_equity": mark_equity,
            "changed": int(changed),
        })

    open_trade = None
    if position and entry_price is not None and entry_capital is not None:
        last_close = rows[-1][3]
        unrealized_return = ((last_close - entry_price) / entry_price) if position == 1 else ((entry_price - last_close) / entry_price)
        open_trade = {
            "side": "LONG" if position == 1 else "SHORT",
            "entry_timestamp": entry_timestamp,
            "entry_price": entry_price,
            "last_price": last_close,
            "unrealized_return": unrealized_return,
            "mark_equity": capital + entry_capital * unrealized_return,
        }

    return capital, entries, exits, wins, losses, commission_paid, trades, open_trade, equity_rows


def main() -> None:
    a = parse_args()
    rows = load_rows(Path(a.input), a.year)
    final_capital, entries, exits, wins, losses, fees, trades, open_trade, equity_rows = run(
        rows, a.initial_capital, a.f3, a.f4, a.window, a.commission_per_side, a.min_prediction_pct
    )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        fields = ["timestamp", "close", "predicted_move_pct", "pt", "position", "capital", "mark_equity", "changed"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(equity_rows)

    if a.trades_output:
        trades_out = Path(a.trades_output)
        trades_out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "entry_timestamp", "exit_timestamp", "side", "entry_price", "exit_price",
            "capital_before_entry", "entry_fee", "entry_fee_pct", "capital_after_entry_fee",
            "gross_return_pct", "gross_pnl", "capital_before_exit_fee", "exit_fee", "exit_fee_pct",
            "total_fees", "total_fee_pct_of_entry_capital", "net_pnl_after_fees", "net_return_pct",
        ]
        with trades_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(trades)
        print(f"trades_output={trades_out}")

    completed = len(trades)
    gross_sum = sum(float(t["gross_return_pct"]) for t in trades)
    final_return = final_capital / a.initial_capital - 1.0
    print(f"input={a.input}")
    print(f"output={out}")
    print(f"rows={len(rows)} frames={max(0, len(rows) - a.window - 1)}")
    print(f"initial_capital={a.initial_capital:.2f}")
    print(f"F3={a.f3:g} F4={a.f4:g} WINDOW={a.window}")
    print(f"min_prediction_pct={a.min_prediction_pct:g}%")
    print(f"commission_per_side={a.commission_per_side*100:.4f}%")
    print()
    print("RESULT")
    print(f"entries={entries} exits={exits} completed_trades={completed}")
    print(f"wins={wins} losses={losses} win_rate={(wins/completed*100 if completed else 0):.4f}%")
    print(f"commission_total={fees:.6f}")
    print(f"gross_return_sum={gross_sum:.4f}%")
    print(f"final_capital={final_capital:.6f}")
    print(f"net_return={final_return*100:.4f}%")
    if open_trade:
        print(f"open_trade={open_trade['side']} entry={open_trade['entry_price']:.8f} last={open_trade['last_price']:.8f} unrealized={open_trade['unrealized_return']*100:.4f}% mark_equity={open_trade['mark_equity']:.6f}")
    else:
        print("open_trade=NO")


if __name__ == "__main__":
    main()
