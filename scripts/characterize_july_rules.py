from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_pattern_trader.models import Candle
from scripts.ohlc_rule_isolation_monthly import (
    TESTS,
    find_month_file,
    load_binance,
    resample,
    signal_for_rule,
)

RULES = [f"R{i}" for i in range(1, 17)]


def discover_symbols(input_dir: Path, month: str) -> list[str]:
    symbols: set[str] = set()
    for p in input_dir.rglob(f"*{month}*.csv"):
        name = p.name.upper()
        for suffix in (f"-1M-{month}.CSV", f"_1M_{month}.CSV", f"-{month}.CSV"):
            if name.endswith(suffix):
                symbols.add(name[: -len(suffix)])
                break
    return sorted(symbols)


def summarize_signals(candles: list[Candle], rule: str, threshold: float) -> dict[str, float]:
    signals = [signal_for_rule(c, rule, threshold) for c in candles]
    n = len(signals)
    nonzero = [s for s in signals if s]
    long_count = sum(s == 1 for s in signals)
    short_count = sum(s == -1 for s in signals)

    runs: list[int] = []
    current = 0
    previous = 0
    for s in signals:
        if s and s == previous:
            current += 1
        elif s:
            if current:
                runs.append(current)
            current = 1
        else:
            if current:
                runs.append(current)
                current = 0
        previous = s
    if current:
        runs.append(current)

    reversals = sum(a != b for a, b in zip(nonzero, nonzero[1:]))
    transitions = max(len(nonzero) - 1, 0)
    bias = (long_count - short_count) / len(nonzero) if nonzero else 0.0

    return {
        "candles": n,
        "signal_count": len(nonzero),
        "signal_rate_pct": len(nonzero) / n * 100.0 if n else 0.0,
        "long_signals": long_count,
        "short_signals": short_count,
        "hold_candles": n - len(nonzero),
        "hold_rate_pct": (n - len(nonzero)) / n * 100.0 if n else 0.0,
        "long_pct_of_signals": long_count / len(nonzero) * 100.0 if nonzero else 0.0,
        "short_pct_of_signals": short_count / len(nonzero) * 100.0 if nonzero else 0.0,
        "directional_bias": bias,
        "avg_persistence_bars": sum(runs) / len(runs) if runs else 0.0,
        "max_persistence_bars": max(runs) if runs else 0,
        "reversal_rate_pct": reversals / transitions * 100.0 if transitions else 0.0,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    x = sorted(values)
    m = len(x) // 2
    return x[m] if len(x) % 2 else (x[m - 1] + x[m]) / 2.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Characterize R1-R16 signal behavior on July Binance data.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--month", default="2026-07")
    ap.add_argument("--timeframes", default="5,15,30,60")
    ap.add_argument("--output", default="reports/july_r_behavior_by_asset_tf.csv")
    ap.add_argument("--matrix-output", default="reports/july_r_behavior_matrix.csv")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    symbols = discover_symbols(input_dir, args.month) if args.symbols.strip().upper() == "ALL" else [x.strip() for x in args.symbols.split(",") if x.strip()]
    tfs = [int(x) for x in args.timeframes.split(",") if x.strip()]

    detail_rows: list[dict] = []
    for symbol in symbols:
        path = find_month_file(input_dir, symbol, args.month)
        if path is None:
            continue
        raw = load_binance(path)
        for tf in tfs:
            candles = resample(raw, tf)
            if not candles:
                continue
            for rule in RULES:
                m = summarize_signals(candles, rule, TESTS[rule])
                detail_rows.append({"month": args.month, "symbol": symbol, "timeframe_min": tf, "rule": rule, **m})

    if not detail_rows:
        raise SystemExit("No input data found")

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        fields = list(detail_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(detail_rows)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in detail_rows:
        grouped[row["rule"]].append(row)

    matrix_rows: list[dict] = []
    for rule in RULES:
        rows = grouped[rule]
        by_tf: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            by_tf[int(r["timeframe_min"])].append(r)
        tf_rates = {tf: mean([float(r["signal_rate_pct"]) for r in by_tf.get(tf, [])]) for tf in tfs}
        asset_rates = []
        for symbol in sorted({r["symbol"] for r in rows}):
            vals = [float(r["signal_rate_pct"]) for r in rows if r["symbol"] == symbol]
            asset_rates.append(mean(vals))
        matrix_rows.append({
            "month": args.month,
            "rule": rule,
            "assets": len({r["symbol"] for r in rows}),
            "asset_tf_samples": len(rows),
            "avg_signal_rate_pct": mean([float(r["signal_rate_pct"]) for r in rows]),
            "median_signal_rate_pct": median([float(r["signal_rate_pct"]) for r in rows]),
            "avg_long_pct": mean([float(r["long_pct_of_signals"]) for r in rows]),
            "avg_short_pct": mean([float(r["short_pct_of_signals"]) for r in rows]),
            "avg_directional_bias": mean([float(r["directional_bias"]) for r in rows]),
            "avg_persistence_bars": mean([float(r["avg_persistence_bars"]) for r in rows]),
            "avg_max_persistence_bars": mean([float(r["max_persistence_bars"]) for r in rows]),
            "avg_reversal_rate_pct": mean([float(r["reversal_rate_pct"]) for r in rows]),
            "avg_hold_rate_pct": mean([float(r["hold_rate_pct"]) for r in rows]),
            "tf_5m_signal_rate_pct": tf_rates.get(5, 0.0),
            "tf_15m_signal_rate_pct": tf_rates.get(15, 0.0),
            "tf_30m_signal_rate_pct": tf_rates.get(30, 0.0),
            "tf_60m_signal_rate_pct": tf_rates.get(60, 0.0),
            "asset_signal_rate_spread_pct": max(asset_rates) - min(asset_rates) if asset_rates else 0.0,
        })

    mp = Path(args.matrix_output)
    mp.parent.mkdir(parents=True, exist_ok=True)
    with mp.open("w", newline="", encoding="utf-8") as f:
        fields = list(matrix_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(matrix_rows)

    print(f"CHARACTERIZATION month={args.month} assets={len(symbols)} tfs={','.join(map(str, tfs))} rules=16")
    print(f"SAVED {p} rows={len(detail_rows)}")
    print(f"SAVED {mp} rows={len(matrix_rows)}")
    print("R  ACT% LONG% SHORT% BIAS  PERS REV%  5m  15m  30m  60m")
    for r in matrix_rows:
        print(f"{r['rule']:>3} {float(r['avg_signal_rate_pct']):5.1f} {float(r['avg_long_pct']):5.1f} {float(r['avg_short_pct']):5.1f} {float(r['avg_directional_bias']):+5.2f} {float(r['avg_persistence_bars']):5.2f} {float(r['avg_reversal_rate_pct']):5.1f} {float(r['tf_5m_signal_rate_pct']):5.1f} {float(r['tf_15m_signal_rate_pct']):5.1f} {float(r['tf_30m_signal_rate_pct']):5.1f} {float(r['tf_60m_signal_rate_pct']):5.1f}")


if __name__ == "__main__":
    main()
