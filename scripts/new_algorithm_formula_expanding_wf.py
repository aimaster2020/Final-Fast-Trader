from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

DEFAULT_FEE_PCT = 0.26
DEFAULT_THRESHOLDS = "0.26,0.30,0.35,0.40,0.45,0.50,0.60,0.65,0.70,0.73,0.75,0.80,0.85,0.90,1.00,1.25,1.50,2.00"
DEFAULT_HORIZONS = "8,12,18,20,24,25,26,27,28,30,32,36"
DEFAULT_FOLDS = 5
DEFAULT_INITIAL_TRAIN_FRAC = 0.50
DEFAULT_MIN_TRAIN_TRADES = 10


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
    out: list[dict[str, float | int | None]] = []
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
        out.append({"index": i, "ac": ac, "ad": ad, "x": x})
    return out


def parse_list(value: str, cast):
    result = [cast(x.strip()) for x in value.split(",") if x.strip()]
    if not result:
        raise ValueError("List cannot be empty")
    return result


def collect_trades(
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

    for sig in signals:
        i = int(sig["index"])
        if i < start or i >= end or i < next_free:
            continue

        target = i + horizon
        if target >= end:
            continue

        ad = int(sig["ad"])
        ac = int(sig["ac"])
        pred = sig["x"]
        if ad == 0 or pred is None:
            continue
        if abs(ac) < min_score or float(pred) < threshold:
            continue

        entry = rows[i]["close"]
        exit_ = rows[target]["close"]
        if entry == 0:
            continue

        gross = ad * (exit_ - entry) / entry * 100.0
        net = gross - fee_pct
        trades.append({"index": i, "gross_pct": gross, "net_pct": net})
        next_free = target + 1

    return trades


def metrics(trades: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_net": 0.0,
            "sum_net": 0.0,
            "compounded_net": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
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


def rank_key(row: dict[str, float | int]) -> tuple[float, float, float, int]:
    return (
        float(row["compounded_net"]),
        float(row["sum_net"]),
        float(row["profit_factor"]),
        int(row["trades"]),
    )


def pf_text(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Expanding walk-forward: optimize only on past data, then freeze and test on the next block."
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--fee-pct", type=float, default=DEFAULT_FEE_PCT)
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--horizons", default=DEFAULT_HORIZONS)
    ap.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    ap.add_argument("--initial-train-frac", type=float, default=DEFAULT_INITIAL_TRAIN_FRAC)
    ap.add_argument("--min-train-trades", type=int, default=DEFAULT_MIN_TRAIN_TRADES)
    args = ap.parse_args()

    if args.folds < 2 or not 0.3 <= args.initial_train_frac < 1.0:
        raise ValueError("Invalid folds or initial-train-frac")

    rows = read_ohlc(Path(args.input))
    signals = build_signals(rows)
    thresholds = parse_list(args.thresholds, float)
    horizons = parse_list(args.horizons, int)
    n = len(rows)
    initial_train = int(n * args.initial_train_frac)
    oos_n = n - initial_train
    fold_size = oos_n // args.folds
    if fold_size <= max(horizons):
        raise ValueError("OOS fold too small for the selected horizon grid")

    print("=" * 155)
    print("EXPANDING WALK-FORWARD - EXACT EXCEL FORMULA")
    print("=" * 155)
    print(f"input={args.input}")
    print(f"rows={n} | initial_train={initial_train} | folds={args.folds} | fold_size={fold_size}")
    print(f"fee round-trip={args.fee_pct:.4f}%")
    print("Each fold optimizes only on data strictly before that fold, then freezes parameters for the next block.")
    print("Mode: NON_OVERLAP; every trade must open and close entirely within the relevant train/test block.")
    print("Signal: AC=2*N+O+P-2*R-S-T -> AD -> U -> X -> threshold")
    print("=" * 155)

    all_oos: list[dict[str, float | int]] = []
    chosen: list[dict[str, float | int]] = []

    for fold in range(args.folds):
        test_start = initial_train + fold * fold_size
        test_end = initial_train + (fold + 1) * fold_size if fold < args.folds - 1 else n
        train_end = test_start

        candidates: list[dict[str, float | int]] = []
        for horizon in horizons:
            for threshold in thresholds:
                for min_score in (1, 2, 3, 4):
                    train_trades = collect_trades(
                        rows, signals, 0, train_end, horizon, threshold, args.fee_pct, min_score
                    )
                    m = metrics(train_trades)
                    if int(m["trades"]) < args.min_train_trades:
                        continue
                    candidates.append(
                        {
                            "horizon": horizon,
                            "threshold": threshold,
                            "min_score": min_score,
                            **m,
                        }
                    )

        if not candidates:
            raise ValueError(f"Fold {fold + 1}: no training configuration reached min-train-trades")

        best = max(candidates, key=rank_key)
        h = int(best["horizon"])
        threshold = float(best["threshold"])
        min_score = int(best["min_score"])

        test_trades = collect_trades(
            rows, signals, test_start, test_end, h, threshold, args.fee_pct, min_score
        )
        test_m = metrics(test_trades)
        all_oos.extend(test_trades)
        chosen.append(best)

        print(
            f"FOLD {fold + 1}: train=0:{train_end} | test={test_start}:{test_end} | "
            f"chosen H={h} thr={threshold:.2f}% score>={min_score} | "
            f"trainTrades={int(best['trades'])} trainComp={float(best['compounded_net']):.3f}% | "
            f"OOS trades={int(test_m['trades']):>3} win={float(test_m['win_rate'])*100:>6.2f}% | "
            f"avgNet={float(test_m['avg_net']):>8.4f}% sumNet={float(test_m['sum_net']):>9.4f}% | "
            f"comp={float(test_m['compounded_net']):>8.4f}% PF={pf_text(float(test_m['profit_factor'])):>6} | "
            f"maxDD={float(test_m['max_drawdown']):>7.3f}%"
        )

    overall = metrics(all_oos)
    print("-" * 155)
    print(
        f"ALL OOS: trades={int(overall['trades'])} | win={float(overall['win_rate'])*100:.2f}% | "
        f"avgNet={float(overall['avg_net']):.5f}% | sumNet={float(overall['sum_net']):.5f}% | "
        f"compounded={float(overall['compounded_net']):.5f}% | "
        f"PF={pf_text(float(overall['profit_factor']))} | maxDD={float(overall['max_drawdown']):.3f}%"
    )
    print("=" * 155)


if __name__ == "__main__":
    main()
