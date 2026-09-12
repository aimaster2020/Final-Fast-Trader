from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

DEFAULT_FEE_PCT = 0.26
DEFAULT_THRESHOLD_GRID = (
    "0.26,0.30,0.35,0.40,0.45,0.50,0.60,0.75,0.90,1.00,1.25,1.50,2.00,2.50,3.00"
)


def norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def read_ohlc(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = {norm(x): x for x in (reader.fieldnames or [])}
        required = ["open", "high", "low", "close"]
        missing = [x for x in required if x not in fields]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        rows: list[dict[str, float]] = []
        for raw in reader:
            try:
                rows.append({x: float(raw[fields[x]]) for x in required})
            except (TypeError, ValueError):
                continue

    if len(rows) < 10:
        raise ValueError("Need at least 10 valid OHLC rows")
    return rows


def excel_signal(rows: list[dict[str, float]]) -> list[dict[str, float | int | None]]:
    """Rebuild the supplied workbook formulas exactly enough for live-safe trading.

    N = HC > CO
    O = LC > CO
    P = HC > HO
    R = HC < CO
    S = LC < CO
    T = HC < HO
    AC = 2*N + O + P - 2*R - S - T
    AD = sign(AC)
    U = H-O+C for AD=+1, L-O+C for AD=-1, blank for AD=0
    W = U-C
    X = abs(W)/C*100
    V = X >= threshold (threshold is applied later during the sweep)
    """
    out: list[dict[str, float | int | None]] = []

    for i, row in enumerate(rows):
        o = row["open"]
        h = row["high"]
        l = row["low"]
        c = row["close"]

        co = c - o
        hc = h - c
        lc = c - l
        ho = h - o

        n = 1 if hc > co else 0
        oo = 1 if lc > co else 0
        p = 1 if hc > ho else 0
        r = 1 if hc < co else 0
        s = 1 if lc < co else 0
        t = 1 if hc < ho else 0

        ac = 2 * n + oo + p - 2 * r - s - t
        ad = 1 if ac > 0 else -1 if ac < 0 else 0

        if ad == 1:
            u = h - o + c
        elif ad == -1:
            u = l - o + c
        else:
            u = None

        w = (u - c) if u is not None else None
        x = (abs(w) / abs(c) * 100.0) if w is not None and c != 0 else None

        out.append(
            {
                "index": i,
                "ac": ac,
                "ad": ad,
                "u": u,
                "w": w,
                "x": x,
            }
        )

    return out


def trade_metrics(
    rows: list[dict[str, float]],
    signals: list[dict[str, float | int | None]],
    horizon: int,
    threshold_pct: float,
    fee_pct: float,
    min_abs_score: int,
    allow_overlap: bool,
) -> dict[str, float | int]:
    trades: list[tuple[int, float, float, float, int]] = []
    next_free = 0

    for signal in signals:
        i = int(signal["index"])
        if not allow_overlap and i < next_free:
            continue

        target_i = i + horizon
        if target_i >= len(rows):
            continue

        ad = int(signal["ad"])
        ac = int(signal["ac"])
        pred_pct = signal["x"]
        if ad == 0 or pred_pct is None:
            continue
        if abs(ac) < min_abs_score:
            continue
        if float(pred_pct) < threshold_pct:
            continue

        entry = rows[i]["close"]
        exit_ = rows[target_i]["close"]
        if entry == 0:
            continue

        raw_return_pct = ad * (exit_ - entry) / entry * 100.0
        net_return_pct = raw_return_pct - fee_pct
        trades.append((i, raw_return_pct, net_return_pct, float(pred_pct), ac))

        if not allow_overlap:
            next_free = target_i + 1

    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_gross": 0.0,
            "avg_net": 0.0,
            "sum_net": 0.0,
            "compounded_net": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_predicted_move": 0.0,
        }

    wins = sum(1 for _, _, net, _, _ in trades if net > 0)
    gross_sum = sum(gross for _, gross, _, _, _ in trades)
    net_sum = sum(net for _, _, net, _, _ in trades)
    avg_net = net_sum / len(trades)
    gains = sum(net for _, _, net, _, _ in trades if net > 0)
    losses = -sum(net for _, _, net, _, _ in trades if net < 0)
    profit_factor = gains / losses if losses else float("inf")
    avg_predicted = sum(p for _, _, _, p, _ in trades) / len(trades)

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for _, _, net, _, _ in trades:
        equity *= 1.0 + net / 100.0
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak
        max_dd = max(max_dd, drawdown)

    return {
        "trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades),
        "avg_gross": gross_sum / len(trades),
        "avg_net": avg_net,
        "sum_net": net_sum,
        "compounded_net": (equity - 1.0) * 100.0,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd * 100.0,
        "avg_predicted_move": avg_predicted,
    }


def parse_floats(value: str) -> list[float]:
    result = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not result:
        raise ValueError("Grid cannot be empty")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optimize net profit using the exact candle formulas from candle_direction_magnitude_rules_only.xlsx"
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--fee-pct", type=float, default=DEFAULT_FEE_PCT)
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLD_GRID)
    ap.add_argument("--horizons", default="1,2,3,4,5,6,8,12,24")
    ap.add_argument("--min-trades", type=int, default=20)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    rows = read_ohlc(Path(args.input))
    signals = excel_signal(rows)
    thresholds = parse_floats(args.thresholds)
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    if any(h < 1 for h in horizons):
        raise ValueError("All horizons must be >= 1")

    results: list[dict[str, float | int | str]] = []

    for overlap in (False, True):
        mode = "NON_OVERLAP" if not overlap else "OVERLAP"
        for horizon in horizons:
            for threshold in thresholds:
                for min_score in (1, 2, 3, 4):
                    m = trade_metrics(
                        rows,
                        signals,
                        horizon,
                        threshold,
                        args.fee_pct,
                        min_score,
                        overlap,
                    )
                    if int(m["trades"]) == 0:
                        continue
                    results.append(
                        {
                            "mode": mode,
                            "horizon": horizon,
                            "threshold": threshold,
                            "min_score": min_score,
                            **m,
                        }
                    )

    print("=" * 150)
    print("EXACT EXCEL FORMULA - PROFIT OPTIMIZER")
    print("=" * 150)
    print(f"input={args.input}")
    print(f"OHLC rows={len(rows)}")
    print(f"fee round-trip={args.fee_pct:.4f}%")
    print("Signal uses only N,O,P,R,S,T -> AC -> AD -> U -> W -> X -> threshold.")
    print("AC = 2*N + O + P - 2*R - S - T")
    print("U = H-O+C for AD=+1; U = L-O+C for AD=-1")
    print("No future values are used for entry.")
    print("=" * 150)

    def sort_key(r: dict[str, float | int | str]) -> tuple[float, float, float, int]:
        return (
            float(r["compounded_net"]),
            float(r["sum_net"]),
            float(r["profit_factor"]),
            int(r["trades"]),
        )

    for mode in ("NON_OVERLAP", "OVERLAP"):
        subset = [r for r in results if r["mode"] == mode]
        print(f"\nTOP {args.top} - {mode} (ranked by compounded net return)")
        print(
            "Mode        H  Thr%  Score  Trades  Win%   AvgNet%  SumNet%  "
            "CompNet%  PF     MaxDD%  AvgPred%"
        )
        for r in sorted(subset, key=sort_key, reverse=True)[: args.top]:
            pf = r["profit_factor"]
            pf_text = "inf" if math.isinf(float(pf)) else f"{float(pf):.2f}"
            print(
                f"{str(r['mode']):<11} "
                f"{int(r['horizon']):>2}  "
                f"{float(r['threshold']):>4.2f}  "
                f"{int(r['min_score']):>5}  "
                f"{int(r['trades']):>6}  "
                f"{float(r['win_rate'])*100:>5.1f}  "
                f"{float(r['avg_net']):>8.4f}  "
                f"{float(r['sum_net']):>8.4f}  "
                f"{float(r['compounded_net']):>9.4f}  "
                f"{pf_text:>5}  "
                f"{float(r['max_drawdown']):>7.3f}  "
                f"{float(r['avg_predicted_move']):>8.4f}"
            )

        stable = [r for r in subset if int(r["trades"]) >= args.min_trades]
        print(f"\nBEST {mode} WITH >= {args.min_trades} TRADES")
        for r in sorted(stable, key=sort_key, reverse=True)[:10]:
            pf = r["profit_factor"]
            pf_text = "inf" if math.isinf(float(pf)) else f"{float(pf):.3f}"
            print(
                f"H={int(r['horizon'])} | threshold={float(r['threshold']):.2f}% | "
                f"score>={int(r['min_score'])} | trades={int(r['trades'])} | "
                f"win={float(r['win_rate'])*100:.2f}% | "
                f"avg_net={float(r['avg_net']):.5f}% | "
                f"sum_net={float(r['sum_net']):.5f}% | "
                f"compounded={float(r['compounded_net']):.5f}% | "
                f"PF={pf_text} | "
                f"maxDD={float(r['max_drawdown']):.3f}%"
            )

    print("=" * 150)


if __name__ == "__main__":
    main()
