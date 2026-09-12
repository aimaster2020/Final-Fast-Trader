from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

DEFAULT_THRESHOLD = 0.73
DEFAULT_HORIZON = 27
DEFAULT_FEE_PCT = 0.26
DEFAULT_MIN_SCORE = 1
DEFAULT_FOLDS = 5
DEFAULT_INITIAL_TRAIN_FRAC = 0.50


def read_ohlc(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = {x.strip().lower(): x for x in (reader.fieldnames or [])}
        required = ["open", "high", "low", "close"]
        missing = [x for x in required if x not in fields]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        rows: list[dict[str, float]] = []
        for raw in reader:
            try:
                rows.append({k: float(raw[fields[k]]) for k in required})
            except (TypeError, ValueError):
                continue
    if len(rows) < 100:
        raise ValueError("Need at least 100 valid OHLC rows")
    return rows


def build_signals(rows: list[dict[str, float]]) -> list[dict[str, float | int | None]]:
    signals: list[dict[str, float | int | None]] = []
    for i, row in enumerate(rows):
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        co = c - o
        hc = h - c
        lc = c - l
        ho = h - o

        n = int(hc > co)
        oo = int(lc > co)
        p = int(hc > ho)
        r = int(hc < co)
        s = int(lc < co)
        t = int(hc < ho)

        ac = 2 * n + oo + p - 2 * r - s - t
        ad = 1 if ac > 0 else -1 if ac < 0 else 0

        if ad == 1:
            u = h - o + c
        elif ad == -1:
            u = l - o + c
        else:
            u = None

        x = abs(u - c) / abs(c) * 100.0 if u is not None and c != 0 else None
        signals.append({"index": i, "ac": ac, "ad": ad, "x": x})
    return signals


def collect_fold_trades(
    rows: list[dict[str, float]],
    signals: list[dict[str, float | int | None]],
    start: int,
    end: int,
    horizon: int,
    threshold: float,
    fee_pct: float,
    min_score: int,
) -> list[dict[str, float | int]]:
    trades: list[dict[str, float | int]] = []
    next_free = start

    for signal in signals:
        i = int(signal["index"])
        if i < start or i >= end or i < next_free:
            continue

        target_i = i + horizon
        if target_i >= end:
            continue

        ac = int(signal["ac"])
        ad = int(signal["ad"])
        predicted_move_pct = signal["x"]
        if ad == 0 or predicted_move_pct is None:
            continue
        if abs(ac) < min_score or float(predicted_move_pct) < threshold:
            continue

        entry = rows[i]["close"]
        exit_ = rows[target_i]["close"]
        if entry == 0:
            continue

        gross_pct = ad * (exit_ - entry) / entry * 100.0
        net_pct = gross_pct - fee_pct
        trades.append({
            "index": i,
            "gross_pct": gross_pct,
            "net_pct": net_pct,
            "predicted_move_pct": float(predicted_move_pct),
            "score": ac,
        })
        next_free = target_i + 1

    return trades


def metrics(trades: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not trades:
        return {
            "trades": 0, "win_rate": 0.0, "avg_net": 0.0, "sum_net": 0.0,
            "compounded_net": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0,
        }

    nets = [float(t["net_pct"]) for t in trades]
    wins = sum(1 for x in nets if x > 0)
    gains = sum(x for x in nets if x > 0)
    losses = -sum(x for x in nets if x < 0)

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for net in nets:
        equity *= 1.0 + net / 100.0
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)

    return {
        "trades": len(nets),
        "win_rate": wins / len(nets),
        "avg_net": sum(nets) / len(nets),
        "sum_net": sum(nets),
        "compounded_net": (equity - 1.0) * 100.0,
        "profit_factor": gains / losses if losses else float("inf"),
        "max_drawdown": max_dd * 100.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen-parameter walk-forward test for the exact Excel candle formula strategy")
    ap.add_argument("--input", required=True)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--fee-pct", type=float, default=DEFAULT_FEE_PCT)
    ap.add_argument("--min-score", type=int, default=DEFAULT_MIN_SCORE)
    ap.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    ap.add_argument("--initial-train-frac", type=float, default=DEFAULT_INITIAL_TRAIN_FRAC)
    args = ap.parse_args()

    if args.horizon < 1 or args.folds < 2 or not 0 < args.initial_train_frac < 1:
        raise ValueError("Invalid horizon/folds/initial-train-frac")

    rows = read_ohlc(Path(args.input))
    signals = build_signals(rows)
    n = len(rows)
    initial_train = int(n * args.initial_train_frac)
    oos_n = n - initial_train
    fold_size = oos_n // args.folds
    if fold_size < args.horizon + 1:
        raise ValueError("OOS folds are too small for the selected horizon")

    print("=" * 130)
    print("FROZEN PARAMETER WALK-FORWARD - EXACT EXCEL FORMULA")
    print("=" * 130)
    print(f"input={args.input}")
    print(f"rows={n} | initial_train={initial_train} | folds={args.folds}")
    print(f"FROZEN: horizon={args.horizon}h | threshold={args.threshold:.2f}% | min_score={args.min_score} | fee={args.fee_pct:.2f}% RT | NON_OVERLAP")
    print("Parameters are not re-optimized inside any test fold.")
    print("Signal: AC=2*N+O+P-2*R-S-T -> AD -> U -> X")
    print("=" * 130)

    all_oos: list[dict[str, float | int]] = []
    for fold in range(args.folds):
        start = initial_train + fold * fold_size
        end = initial_train + (fold + 1) * fold_size if fold < args.folds - 1 else n
        trades = collect_fold_trades(rows, signals, start, end, args.horizon, args.threshold, args.fee_pct, args.min_score)
        m = metrics(trades)
        all_oos.extend(trades)
        pf = "inf" if math.isinf(float(m["profit_factor"])) else f"{float(m['profit_factor']):.3f}"
        print(
            f"FOLD {fold + 1}: test_rows={start}:{end} | trades={int(m['trades']):>3} | "
            f"win={float(m['win_rate']) * 100:>6.2f}% | avg_net={float(m['avg_net']):>8.5f}% | "
            f"sum_net={float(m['sum_net']):>9.5f}% | compounded={float(m['compounded_net']):>9.5f}% | "
            f"PF={pf:>5} | maxDD={float(m['max_drawdown']):>7.3f}%"
        )

    overall = metrics(all_oos)
    pf = "inf" if math.isinf(float(overall["profit_factor"])) else f"{float(overall['profit_factor']):.3f}"
    print("-" * 130)
    print(
        f"ALL OOS: trades={int(overall['trades'])} | win={float(overall['win_rate']) * 100:.2f}% | "
        f"avg_net={float(overall['avg_net']):.5f}% | sum_net={float(overall['sum_net']):.5f}% | "
        f"compounded={float(overall['compounded_net']):.5f}% | PF={pf} | "
        f"maxDD={float(overall['max_drawdown']):.3f}%"
    )
    print("=" * 130)


if __name__ == "__main__":
    main()
