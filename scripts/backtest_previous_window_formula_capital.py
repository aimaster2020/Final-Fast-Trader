#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest 3-window previous-candle formula with capital and optional commission.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--f3", type=float, default=100.0)
    p.add_argument("--f4", type=float, default=-150.0)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--commission-per-side", type=float, default=0.0, help="Commission rate per entry/exit side, e.g. 0.0013")
    p.add_argument("--year", type=int, default=None)
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
    headered = any(x in header for x in ("timestamp", "open", "high", "low", "close"))
    out = []
    if headered:
        idx = {name: i for i, name in enumerate(header)}
        for req in ("timestamp", "open", "high", "close"):
            if req not in idx:
                raise ValueError(f"Missing column: {req}")
        for r in rows[1:]:
            if len(r) <= max(idx["timestamp"], idx["open"], idx["high"], idx["close"]):
                continue
            if year is not None and year_of(r[idx["timestamp"]]) != year:
                continue
            out.append((r[idx["timestamp"]], float(r[idx["open"]]), float(r[idx["high"]]), float(r[idx["close"]])))
    else:
        for r in rows:
            if len(r) < 5:
                continue
            if year is not None and year_of(r[0]) != year:
                continue
            out.append((r[0], float(r[1]), float(r[2]), float(r[4])))
    return out


def signal(pred: float, close: float) -> int:
    move = pred - close
    return 1 if move > 0 else -1 if move < 0 else 0


def run(rows, initial_capital: float, f3: float, f4: float, window: int, commission: float):
    if len(rows) <= window + 1:
        raise ValueError("Not enough rows")

    capital = initial_capital
    position = 0
    entry_price = None
    entry_equity = None
    entry_timestamp = None
    trades = []
    entries = exits = 0
    wins = losses = 0
    commission_paid = 0.0

    equity_rows = []

    def apply_commission(amount: float) -> float:
        nonlocal commission_paid, capital
        fee = amount * commission
        commission_paid += fee
        capital -= fee
        return fee

    for i in range(window, len(rows) - 1):
        ts, o, h, c = rows[i]
        next_c = rows[i + 1][3]

        body = c - o
        if body > f3:
            pred = sum(rows[j][3] for j in range(i - window, i)) / window
        elif body < f4:
            pred = sum(rows[j][2] for j in range(i - window, i)) / window
        else:
            pred = c

        pt = signal(pred, c)
        changed = False

        if position == 0:
            if pt != 0:
                position = pt
                entry_price = c
                entry_equity = capital
                entry_timestamp = ts
                if commission:
                    apply_commission(entry_equity)
                entries += 1
                changed = True
        elif pt != 0 and pt != position:
            # Exit old position at current close, then immediately enter new side at same close.
            exit_price = c
            gross_return = ((exit_price - entry_price) / entry_price) if position == 1 else ((entry_price - exit_price) / entry_price)
            gross_pnl = entry_equity * gross_return
            capital += gross_pnl
            if commission:
                apply_commission(entry_equity + gross_pnl)
            exits += 1
            wins += gross_pnl > 0
            losses += gross_pnl <= 0
            trades.append({
                "entry_timestamp": entry_timestamp,
                "exit_timestamp": ts,
                "side": "LONG" if position == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "gross_pnl": gross_pnl,
            })
            position = pt
            entry_price = c
            entry_equity = capital
            entry_timestamp = ts
            if commission:
                apply_commission(entry_equity)
            entries += 1
            changed = True

        # Mark current equity at close. Open P&L is based on current close.
        if position and entry_price is not None and entry_equity is not None:
            unrealized_return = ((c - entry_price) / entry_price) if position == 1 else ((entry_price - c) / entry_price)
            mark_equity = capital + entry_equity * unrealized_return
        else:
            mark_equity = capital
        equity_rows.append({"timestamp": ts, "close": c, "pt": pt, "position": position, "capital": capital, "mark_equity": mark_equity, "changed": int(changed)})

    open_trade = None
    if position and entry_price is not None and entry_equity is not None:
        last_close = rows[-1][3]
        unrealized_return = ((last_close - entry_price) / entry_price) if position == 1 else ((entry_price - last_close) / entry_price)
        mark_equity = capital + entry_equity * unrealized_return
        open_trade = {
            "side": "LONG" if position == 1 else "SHORT",
            "entry_timestamp": entry_timestamp,
            "entry_price": entry_price,
            "last_price": last_close,
            "unrealized_return": unrealized_return,
            "mark_equity": mark_equity,
        }

    return capital, entries, exits, wins, losses, commission_paid, trades, open_trade, equity_rows


def main():
    a = parse_args()
    rows = load_rows(Path(a.input), a.year)
    final_capital, entries, exits, wins, losses, fees, trades, open_trade, equity_rows = run(
        rows, a.initial_capital, a.f3, a.f4, a.window, a.commission_per_side
    )

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        fields = ["timestamp", "close", "pt", "position", "capital", "mark_equity", "changed"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(equity_rows)

    completed = len(trades)
    gross_sum = sum(float(t["gross_return"]) for t in trades)
    net_return = final_capital / a.initial_capital - 1.0
    print(f"input={a.input}")
    print(f"output={out}")
    print(f"rows={len(rows)}")
    print(f"initial_capital={a.initial_capital:.2f}")
    print(f"F3={a.f3:g} F4={a.f4:g} WINDOW={a.window}")
    print(f"commission_per_side={a.commission_per_side*100:.4f}%")
    print()
    print("RESULT")
    print(f"entries={entries} exits={exits} completed_trades={completed}")
    print(f"wins={wins} losses={losses} win_rate={(wins/completed*100 if completed else 0):.4f}%")
    print(f"commission_total={fees:.2f}")
    print(f"gross_return_sum={gross_sum*100:.4f}%")
    print(f"final_capital={final_capital:.2f}")
    print(f"net_return={net_return*100:.4f}%")
    if open_trade:
        print(f"open_trade={open_trade['side']} entry={open_trade['entry_price']:.8f} last={open_trade['last_price']:.8f} unrealized={open_trade['unrealized_return']*100:.4f}% mark_equity={open_trade['mark_equity']:.2f}")
    else:
        print("open_trade=NO")


if __name__ == "__main__":
    main()
