from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_pattern_trader.models import Candle
from scripts.july_r_composite_walkforward import components, TESTS, discover_symbols, find_month_file, load_binance, resample

RULES = [f"R{i}" for i in range(1, 17)]
VARIANTS = ["N", "I"]


def raw_sig(c: Candle, rule: str) -> int:
    v = components(c)[rule]
    th = TESTS[rule]
    return 1 if v >= th else -1 if v <= -th else 0


def run_rule(series: dict[str, list[Candle]], rule: str, variant: str, fee_side_pct: float) -> dict:
    initial = 1000.0
    per_symbol = initial / len(series)
    total = 0.0
    trades = wins = 0
    fees_total = 0.0
    max_dd = 0.0

    for symbol, raw in series.items():
        capital = per_symbol
        peak = capital
        c15 = resample(raw, 15)
        c60 = resample(raw, 60)
        # 15m signal closes first; the first eligible 60m candle is entry,
        # next 60m candle is exit. One trade at a time per symbol.
        signal_ptr = 0
        signals = []
        for i, c in enumerate(c15):
            s = raw_sig(c, rule)
            if variant == "I":
                s = -s
            if s:
                signals.append((c.timestamp + 15 * 60, s, i + 1))

        j = 0
        while j + 1 < len(c60):
            ts = c60[j].timestamp
            # Discard stale signals that arrived before the current eligible
            # execution candle. This is the same non-overlap convention used
            # by the validated R16 control test.
            while signal_ptr < len(signals) and signals[signal_ptr][0] < ts:
                signal_ptr += 1
            if signal_ptr >= len(signals):
                j += 1
                continue
            signal_end, direction, signal_id = signals[signal_ptr]
            if ts < signal_end:
                j += 1
                continue

            entry = c60[j].close
            exit_price = c60[j + 1].close
            move = (exit_price / entry - 1.0) * direction if entry > 0 else 0.0
            gross = capital * move
            fee = capital * 2.0 * fee_side_pct / 100.0
            capital = max(0.0, capital + gross - fee)
            fees_total += fee
            trades += 1
            wins += int(gross > 0)
            peak = max(peak, capital)
            max_dd = max(max_dd, 100.0 * (peak - capital) / peak if peak else 0.0)
            signal_ptr += 1
            j += 2

        total += capital

    return {
        "rule": rule,
        "variant": variant,
        "final": total,
        "ret": 100.0 * (total / initial - 1.0),
        "trades": trades,
        "win": 100.0 * wins / trades if trades else 0.0,
        "dd": max_dd,
        "fees": fees_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--fee-side-pct", type=float, default=0.0)
    args = ap.parse_args()

    data_root = Path(args.input_dir)
    symbols = sorted(discover_symbols(data_root, args.test_month)) if args.symbols.upper() == "ALL" else [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    series = {}
    for symbol in symbols:
        p = find_month_file(data_root, symbol, args.test_month)
        if p:
            cs = load_binance(p)
            if len(cs) >= 64:
                series[symbol] = cs
    if not series:
        raise RuntimeError("No test data found")

    print(f"R1_R16_SINGLE_NONOVERLAP architecture=B assets={len(series)} test={args.test_month} fee={args.fee_side_pct:.3f}%/side")
    print("15m: single R signal at candle close -> first eligible 60m entry -> next 60m exit")
    print("Control: one trade per symbol at a time; stale confirmed signals discarded")
    print("RULE VAR FINAL RET T WIN DD FEES")
    print("---- --- ----- ------ - ---- ---- -----")

    results = []
    for rule in RULES:
        for variant in VARIANTS:
            r = run_rule(series, rule, variant, args.fee_side_pct)
            results.append(r)
            print(f"{r['rule']:>3} {r['variant']:>3} {r['final']:>7.2f} {r['ret']:>+6.2f}% {r['trades']:>4} {r['win']:>5.2f}% {r['dd']:>5.2f}% {r['fees']:>6.2f}")

    out = ROOT / "reports" / "july_r1_r16_single_nonoverlap_60m.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rule", "variant", "final", "ret", "trades", "win", "dd", "fees"])
        w.writeheader()
        w.writerows(results)

    ranked = sorted(results, key=lambda x: x["ret"], reverse=True)
    print("--- TOP 10 BY RETURN ---")
    for r in ranked[:10]:
        print(f"{r['rule']:>3} {r['variant']:>3} RET={r['ret']:+.2f}% FINAL={r['final']:.2f} T={r['trades']} WIN={r['win']:.2f}% DD={r['dd']:.2f}%")
    print(f"SAVED {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
