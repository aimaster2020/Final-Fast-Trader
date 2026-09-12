from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

DEFAULT_FEE_PCT = 0.26
DEFAULT_THRESHOLDS = "0.26,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.75,0.80,0.90,1.00,1.25,1.50,2.00,2.50,3.00"
DEFAULT_HORIZONS = "1,2,3,4,5,6,8,12,18,24,27,30,36"


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
        if ad == 0 or pred is None or abs(ac) < min_score or float(pred) < threshold:
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
        return {"trades": 0, "win_rate": 0.0, "avg_net": 0.0, "sum_net": 0.0, "compounded_net": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}

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


def rank_key(r: dict[str, float | int]) -> tuple[float, float, float, int]:
    return (
        float(r["compounded_net"]),
        float(r["sum_net"]),
        float(r["profit_factor"]),
        int(r["trades"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train on first half, freeze best Excel-formula parameters, then test once on second half.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--fee-pct", type=float, default=DEFAULT_FEE_PCT)
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--horizons", default=DEFAULT_HORIZONS)
    ap.add_argument("--train-frac", type=float, default=0.50)
    ap.add_argument("--min-train-trades", type=int, default=10)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    if not 0.3 <= args.train_frac <= 0.7:
        raise ValueError("--train-frac should normally be between 0.30 and 0.70")

    rows = read_ohlc(Path(args.input))
    signals = build_signals(rows)
    thresholds = parse_list(args.thresholds, float)
    horizons = parse_list(args.horizons, int)
    if any(h < 1 for h in horizons):
        raise ValueError("All horizons must be >= 1")

    split = int(len(rows) * args.train_frac)
    if split <= max(horizons):
        raise ValueError("Training section is too small for the selected horizons")

    print("=" * 150)
    print("CLEAN HALF-SPLIT TRAIN -> FREEZE -> OOS TEST - EXACT EXCEL FORMULA")
    print("=" * 150)
    print(f"input={args.input}")
    print(f"rows={len(rows)} | train_rows=0:{split} | oos_rows={split}:{len(rows)}")
    print(f"fee round-trip={args.fee_pct:.4f}%")
    print("TRAIN: parameters may be optimized. OOS: parameters are frozen; no re-optimization.")
    print("Signal: AC=2*N+O+P-2*R-S-T -> AD -> U -> X -> threshold")
    print("Mode: NON_OVERLAP; trades must open and close entirely inside their section.")
    print("=" * 150)

    candidates: list[dict[str, float | int]] = []
    for horizon in horizons:
        for threshold in thresholds:
            for min_score in (1, 2, 3, 4):
                train_trades = collect_trades(rows, signals, 0, split, horizon, threshold, args.fee_pct, min_score)
                m = metrics(train_trades)
                if int(m["trades"]) < args.min_train_trades:
                    continue
                candidates.append({"horizon": horizon, "threshold": threshold, "min_score": min_score, **m})

    if not candidates:
        raise ValueError("No training configuration reached --min-train-trades")

    candidates.sort(key=rank_key, reverse=True)
    print(f"\nTOP {args.top} TRAIN CANDIDATES")
    for r in candidates[: args.top]:
        pf = "inf" if math.isinf(float(r["profit_factor"])) else f"{float(r['profit_factor']):.3f}"
        print(
            f"H={int(r['horizon']):>2} | thr={float(r['threshold']):.2f}% | score>={int(r['min_score'])} | "
            f"trades={int(r['trades']):>3} | win={float(r['win_rate'])*100:>6.2f}% | "
            f"avg_net={float(r['avg_net']):>8.5f}% | sum_net={float(r['sum_net']):>9.5f}% | "
            f"comp={float(r['compounded_net']):>9.5f}% | PF={pf} | DD={float(r['max_drawdown']):>7.3f}%"
        )

    best = candidates[0]
    h = int(best["horizon"])
    threshold = float(best["threshold"])
    min_score = int(best["min_score"])

    oos_trades = collect_trades(rows, signals, split, len(rows), h, threshold, args.fee_pct, min_score)
    oos = metrics(oos_trades)
    pf = "inf" if math.isinf(float(oos["profit_factor"])) else f"{float(oos['profit_factor']):.3f}"

    print("-" * 150)
    print("FROZEN PARAMETERS SELECTED ONLY FROM TRAIN")
    print(f"Horizon={h}h | Threshold={threshold:.2f}% | Min Score={min_score}")
    print(
        f"TRAIN: trades={int(best['trades'])} | win={float(best['win_rate'])*100:.2f}% | "
        f"compounded={float(best['compounded_net']):.5f}% | PF={('inf' if math.isinf(float(best['profit_factor'])) else f'{float(best['profit_factor']):.3f}')} | "
        f"maxDD={float(best['max_drawdown']):.3f}%"
    )
    print("=" * 150)
    print("PURE OUT-OF-SAMPLE TEST")
    print(
        f"OOS: trades={int(oos['trades'])} | win={float(oos['win_rate'])*100:.2f}% | "
        f"avg_net={float(oos['avg_net']):.5f}% | sum_net={float(oos['sum_net']):.5f}% | "
        f"compounded={float(oos['compounded_net']):.5f}% | PF={pf} | maxDD={float(oos['max_drawdown']):.3f}%"
    )
    print("=" * 150)


if __name__ == "__main__":
    main()
