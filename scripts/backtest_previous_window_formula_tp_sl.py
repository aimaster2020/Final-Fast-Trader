#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest previous-window formula with prediction-based take-profit and stop-loss."
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--trades-output", default=None)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--f3", type=float, default=800.0)
    p.add_argument("--f4", type=float, default=-150.0)
    p.add_argument("--window", type=int, choices=range(1, 6), default=5)
    p.add_argument("--commission-per-side", type=float, default=0.0013)
    p.add_argument("--min-prediction-pct", type=float, default=1.0)
    p.add_argument("--tp-multiplier", type=float, default=1.0)
    p.add_argument("--sl-multiplier", type=float, default=1.0)
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
    headered = all(x in header for x in ("timestamp", "open", "high", "low", "close"))
    out = []

    if headered:
        idx = {name: i for i, name in enumerate(header)}
        needed = max(idx[k] for k in ("timestamp", "open", "high", "low", "close"))
        for r in rows[1:]:
            if len(r) <= needed:
                continue
            ts = r[idx["timestamp"]]
            if year is not None and year_of(ts) != year:
                continue
            out.append(
                (
                    ts,
                    float(r[idx["open"]]),
                    float(r[idx["high"]]),
                    float(r[idx["low"]]),
                    float(r[idx["close"]]),
                )
            )
    else:
        for r in rows:
            if len(r) < 5:
                continue
            ts = r[0]
            if year is not None and year_of(ts) != year:
                continue
            out.append((ts, float(r[1]), float(r[2]), float(r[3]), float(r[4])))

    if not out:
        raise ValueError("No usable OHLC rows found")
    return out


def direction(move: float) -> int:
    return 1 if move > 0 else -1 if move < 0 else 0


def predicted_price(rows, i: int, window: int, f3: float, f4: float) -> float:
    _, o, _, _, c = rows[i]
    body = c - o
    if body > f3:
        return sum(rows[j][4] for j in range(i - window, i)) / window
    if body < f4:
        return sum(rows[j][2] for j in range(i - window, i)) / window
    return c


def run(
    rows,
    initial_capital: float,
    f3: float,
    f4: float,
    window: int,
    commission: float,
    min_prediction_pct: float,
    tp_multiplier: float,
    sl_multiplier: float,
):
    if len(rows) <= window + 1:
        raise ValueError("Not enough rows")

    capital = initial_capital
    trades = []
    equity_rows = []
    commission_total = 0.0
    entries = exits = wins = losses = 0

    i = window
    while i < len(rows) - 1:
        ts, o, h, l, c = rows[i]
        pred = predicted_price(rows, i, window, f3, f4)
        predicted_move = pred - c
        predicted_move_pct = abs(predicted_move) / c * 100.0 if c else 0.0
        side = direction(predicted_move) if predicted_move_pct >= min_prediction_pct else 0

        equity_rows.append(
            {
                "timestamp": ts,
                "close": c,
                "predicted_price": pred,
                "predicted_move_pct": predicted_move_pct,
                "signal": side,
                "capital": capital,
            }
        )

        if not side:
            i += 1
            continue

        entry_timestamp = ts
        entry_price = c
        capital_before_entry = capital
        entry_fee = capital * commission
        capital -= entry_fee
        commission_total += entry_fee
        entry_capital = capital
        entries += 1

        distance = abs(predicted_move)
        if side == 1:
            target_price = entry_price + distance * tp_multiplier
            stop_price = entry_price - distance * sl_multiplier
        else:
            target_price = entry_price - distance * tp_multiplier
            stop_price = entry_price + distance * sl_multiplier

        exit_timestamp = None
        exit_price = None
        exit_reason = None

        # Entry happens at the current candle close. Only future candles can hit TP/SL.
        j = i + 1
        while j < len(rows):
            nts, no, nh, nl, nc = rows[j]
            if side == 1:
                hit_stop = nl <= stop_price
                hit_target = nh >= target_price
            else:
                hit_stop = nh >= stop_price
                hit_target = nl <= target_price

            # Conservative rule for an OHLC candle where both levels are touched:
            # assume the stop is hit first.
            if hit_stop:
                exit_timestamp = nts
                exit_price = stop_price
                exit_reason = "SL"
                break
            if hit_target:
                exit_timestamp = nts
                exit_price = target_price
                exit_reason = "TP"
                break
            j += 1

        if exit_price is None:
            # No future candle reached either level. Mark the trade to the final close,
            # without treating it as a TP/SL hit.
            nts, no, nh, nl, nc = rows[-1]
            exit_timestamp = nts
            exit_price = nc
            exit_reason = "EOD"
            j = len(rows) - 1

        if side == 1:
            gross_return = (exit_price - entry_price) / entry_price
        else:
            gross_return = (entry_price - exit_price) / entry_price

        gross_pnl = entry_capital * gross_return
        capital_before_exit_fee = entry_capital + gross_pnl
        capital = capital_before_exit_fee
        exit_fee = capital * commission
        capital -= exit_fee
        commission_total += exit_fee
        net_pnl = capital - capital_before_entry
        total_fees = entry_fee + exit_fee
        net_return_pct = net_pnl / capital_before_entry * 100.0

        exits += 1
        if net_pnl > 0:
            wins += 1
        else:
            losses += 1

        trades.append(
            {
                "entry_timestamp": entry_timestamp,
                "exit_timestamp": exit_timestamp,
                "side": "LONG" if side == 1 else "SHORT",
                "entry_price": entry_price,
                "predicted_price": pred,
                "target_price": target_price,
                "stop_price": stop_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "capital_before_entry": capital_before_entry,
                "entry_fee": entry_fee,
                "entry_fee_pct": entry_fee / capital_before_entry * 100.0,
                "capital_after_entry_fee": entry_capital,
                "gross_return_pct": gross_return * 100.0,
                "gross_pnl": gross_pnl,
                "capital_before_exit_fee": capital_before_exit_fee,
                "exit_fee": exit_fee,
                "exit_fee_pct": exit_fee / capital_before_exit_fee * 100.0 if capital_before_exit_fee else 0.0,
                "total_fees": total_fees,
                "total_fee_pct_of_entry_capital": total_fees / capital_before_entry * 100.0,
                "net_pnl_after_fees": net_pnl,
                "net_return_pct": net_return_pct,
                "bars_held": max(1, j - i),
            }
        )

        # Continue after the candle where the trade closed. This prevents overlapping trades.
        i = max(i + 1, j + 1)

    return capital, entries, exits, wins, losses, commission_total, trades, equity_rows


def main() -> None:
    a = parse_args()
    if a.tp_multiplier <= 0 or a.sl_multiplier <= 0:
        raise ValueError("TP and SL multipliers must be > 0")

    rows = load_rows(Path(a.input), a.year)
    final_capital, entries, exits, wins, losses, fees, trades, equity_rows = run(
        rows,
        a.initial_capital,
        a.f3,
        a.f4,
        a.window,
        a.commission_per_side,
        a.min_prediction_pct,
        a.tp_multiplier,
        a.sl_multiplier,
    )

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        fields = ["timestamp", "close", "predicted_price", "predicted_move_pct", "signal", "capital"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(equity_rows)

    if a.trades_output:
        trades_out = Path(a.trades_output)
        trades_out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "entry_timestamp",
            "exit_timestamp",
            "side",
            "entry_price",
            "predicted_price",
            "target_price",
            "stop_price",
            "exit_price",
            "exit_reason",
            "capital_before_entry",
            "entry_fee",
            "entry_fee_pct",
            "capital_after_entry_fee",
            "gross_return_pct",
            "gross_pnl",
            "capital_before_exit_fee",
            "exit_fee",
            "exit_fee_pct",
            "total_fees",
            "total_fee_pct_of_entry_capital",
            "net_pnl_after_fees",
            "net_return_pct",
            "bars_held",
        ]
        with trades_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(trades)
        print(f"trades_output={trades_out}")

    completed = len(trades)
    tp_count = sum(1 for t in trades if t["exit_reason"] == "TP")
    sl_count = sum(1 for t in trades if t["exit_reason"] == "SL")
    eod_count = sum(1 for t in trades if t["exit_reason"] == "EOD")
    final_return = final_capital / a.initial_capital - 1.0

    print(f"input={a.input}")
    print(f"output={out}")
    print(f"rows={len(rows)}")
    print(f"initial_capital={a.initial_capital:.2f}")
    print(f"F3={a.f3:g} F4={a.f4:g} WINDOW={a.window}")
    print(f"min_prediction_pct={a.min_prediction_pct:g}%")
    print(f"tp_multiplier={a.tp_multiplier:g} sl_multiplier={a.sl_multiplier:g}")
    print(f"commission_per_side={a.commission_per_side * 100:.4f}%")
    print()
    print("RESULT")
    print(f"entries={entries} exits={exits} completed_trades={completed}")
    print(f"tp={tp_count} sl={sl_count} eod={eod_count}")
    print(f"wins={wins} losses={losses} win_rate={(wins / completed * 100 if completed else 0):.4f}%")
    print(f"commission_total={fees:.6f}")
    print(f"final_capital={final_capital:.6f}")
    print(f"net_return={final_return * 100:.4f}%")


if __name__ == "__main__":
    main()
