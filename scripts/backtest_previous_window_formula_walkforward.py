#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Walk-forward OOS test for previous-window formula with prediction-size filter.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--commission-per-side", type=float, default=0.0013)
    p.add_argument("--window", type=int, choices=range(1, 6), default=3)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--min-train-ratio", type=float, default=0.5)
    p.add_argument("--f3-min", type=float, default=400)
    p.add_argument("--f3-max", type=float, default=1000)
    p.add_argument("--f3-step", type=float, default=50)
    p.add_argument("--f4-min", type=float, default=-500)
    p.add_argument("--f4-max", type=float, default=0)
    p.add_argument("--f4-step", type=float, default=50)
    p.add_argument("--pred-min", type=float, default=0)
    p.add_argument("--pred-max", type=float, default=1)
    p.add_argument("--pred-step", type=float, default=0.25)
    return p.parse_args()


def load(path: Path):
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
            need = max(idx["timestamp"], idx["open"], idx["high"], idx["close"])
            if len(r) <= need:
                continue
            out.append((r[idx["timestamp"]], float(r[idx["open"]]), float(r[idx["high"]]), float(r[idx["close"]])))
    else:
        for r in rows:
            if len(r) >= 5:
                out.append((r[0], float(r[1]), float(r[2]), float(r[4])))
    return out


def frange(start: float, stop: float, step: float):
    count = int(round((stop - start) / step))
    return [start + i * step for i in range(count + 1)]


def backtest(rows, initial, f3, f4, window, commission, min_prediction_pct):
    if len(rows) <= window + 1:
        return {"final_capital": initial, "net_return": 0.0, "trades": 0, "wins": 0, "losses": 0, "commission": 0.0}

    capital = initial
    position = 0
    entry_price = 0.0
    trades = wins = losses = 0
    commission_paid = 0.0

    for i in range(window, len(rows) - 1):
        _, o, _, c = rows[i]
        body = c - o
        if body > f3:
            pred = sum(rows[j][3] for j in range(i - window, i)) / window
        elif body < f4:
            pred = sum(rows[j][2] for j in range(i - window, i)) / window
        else:
            pred = c

        prediction_pct = abs(pred - c) / c * 100.0 if c else 0.0
        pt = 0 if prediction_pct < min_prediction_pct else (1 if pred > c else -1 if pred < c else 0)

        if position == 0 and pt:
            fee = capital * commission
            capital -= fee
            commission_paid += fee
            position = pt
            entry_price = c
        elif position and pt and pt != position:
            gross = ((c - entry_price) / entry_price) if position == 1 else ((entry_price - c) / entry_price)
            pnl = capital * gross
            capital += pnl
            wins += pnl > 0
            losses += pnl <= 0
            trades += 1

            fee = capital * commission
            capital -= fee
            commission_paid += fee
            position = pt
            entry_price = c

    return {
        "final_capital": capital,
        "net_return": (capital / initial - 1.0) * 100.0,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "commission": commission_paid,
    }


def main():
    a = parse_args()
    rows = load(Path(a.input))
    n = len(rows)
    train_start = max(a.window + 2, int(n * a.min_train_ratio))
    remaining = n - train_start
    step = max(1, remaining // a.folds)

    capital = a.initial_capital
    results = []
    f3_values = frange(a.f3_min, a.f3_max, a.f3_step)
    f4_values = frange(a.f4_min, a.f4_max, a.f4_step)
    pred_values = frange(a.pred_min, a.pred_max, a.pred_step)

    for fold in range(a.folds):
        test_start = train_start + fold * step
        test_end = n if fold == a.folds - 1 else min(n, test_start + step)
        if test_start >= n or test_end - test_start <= a.window + 1:
            break

        train = rows[:test_start]
        test = rows[test_start:test_end]
        best = None

        for f3 in f3_values:
            for f4 in f4_values:
                for pred_pct in pred_values:
                    train_result = backtest(train, 1000.0, f3, f4, a.window, a.commission_per_side, pred_pct)
                    rank = (
                        train_result["net_return"],
                        train_result["trades"],
                        train_result["wins"] / train_result["trades"] if train_result["trades"] else 0.0,
                    )
                    if best is None or rank > best[0]:
                        best = (rank, f3, f4, pred_pct, train_result)

        _, best_f3, best_f4, best_pred, train_result = best
        old_capital = capital
        test_result = backtest(
            test,
            capital,
            best_f3,
            best_f4,
            a.window,
            a.commission_per_side,
            best_pred,
        )
        capital = test_result["final_capital"]

        results.append({
            "fold": fold + 1,
            "train_rows": len(train),
            "test_rows": len(test),
            "f3": best_f3,
            "f4": best_f4,
            "min_prediction_pct": best_pred,
            "train_net_return_pct": train_result["net_return"],
            "test_trades": test_result["trades"],
            "test_wins": test_result["wins"],
            "test_losses": test_result["losses"],
            "test_win_rate_pct": (test_result["wins"] / test_result["trades"] * 100.0) if test_result["trades"] else 0.0,
            "test_commission": test_result["commission"],
            "capital_before": old_capital,
            "capital_after": capital,
            "test_net_return_pct": ((capital / old_capital) - 1.0) * 100.0 if old_capital else 0.0,
        })

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"input={a.input}")
    print(f"output={out}")
    print(f"rows={n} folds={len(results)} window={a.window}")
    print(f"commission_per_side={a.commission_per_side * 100:.4f}%")
    print("WALK-FORWARD OOS RESULTS")
    for r in results:
        print(
            f"fold={r['fold']} train={r['train_rows']} test={r['test_rows']} "
            f"F3={r['f3']:g} F4={r['f4']:g} min_pred={r['min_prediction_pct']:g}% "
            f"test_net={r['test_net_return_pct']:.4f}% trades={r['test_trades']} "
            f"win_rate={r['test_win_rate_pct']:.2f}% capital={r['capital_after']:.2f}"
        )
    print(f"FINAL_OOS_CAPITAL={capital:.6f} TOTAL_OOS_RETURN={(capital / a.initial_capital - 1.0) * 100.0:.4f}%")


if __name__ == "__main__":
    main()
