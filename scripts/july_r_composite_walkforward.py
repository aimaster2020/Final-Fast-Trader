from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_rule(candles: list[Candle], rule: str) -> dict[str, float]:
    signals = [signal_for_rule(c, rule, TESTS[rule]) for c in candles]
    nonzero = [s for s in signals if s]
    n = len(signals)
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
        "signal_rate_pct": len(nonzero) / n * 100.0 if n else 0.0,
        "long_pct": long_count / len(nonzero) * 100.0 if nonzero else 0.0,
        "short_pct": short_count / len(nonzero) * 100.0 if nonzero else 0.0,
        "bias": bias,
        "persistence": mean([float(x) for x in runs]),
        "reversal_rate_pct": reversals / transitions * 100.0 if transitions else 0.0,
    }


def build_weights(train: dict[tuple[str, int, str], dict[str, float]], tfs: list[int]) -> tuple[dict[str, float], list[dict]]:
    by_rule: dict[str, list[dict]] = defaultdict(list)
    for (_symbol, _tf, rule), stats in train.items():
        by_rule[rule].append(stats)

    weights: dict[str, float] = {}
    rows: list[dict] = []
    tier_cap = {"CORE": 1.00, "SECONDARY": 0.65, "EVENT": 0.35, "NOISY/RARE": 0.15}

    for rule in RULES:
        rs = by_rule[rule]
        activity = mean([r["signal_rate_pct"] for r in rs])
        persistence = mean([r["persistence"] for r in rs])
        reversal = mean([r["reversal_rate_pct"] for r in rs])
        asset_means = []
        for symbol in sorted({k[0] for k in train if k[2] == rule}):
            asset_means.append(mean([train[(symbol, tf, rule)]["signal_rate_pct"] for tf in tfs if (symbol, tf, rule) in train]))
        tf_means = [mean([train[(symbol, tf, rule)]["signal_rate_pct"] for symbol in sorted({k[0] for k in train if k[2] == rule}) if (symbol, tf, rule) in train]) for tf in tfs]
        asset_spread = max(asset_means) - min(asset_means) if asset_means else 0.0
        tf_spread = max(tf_means) - min(tf_means) if tf_means else 0.0

        activity_score = max(0.0, min(1.0, 1.0 - abs(activity - 35.0) / 35.0))
        persistence_score = max(0.0, min(1.0, (persistence - 1.0) / 2.0))
        reversal_score = max(0.0, min(1.0, 1.0 - reversal / 100.0))
        asset_stability = max(0.0, min(1.0, 1.0 - asset_spread / max(activity, 1.0)))
        tf_stability = max(0.0, min(1.0, 1.0 - tf_spread / max(mean(tf_means), 1.0)))
        direction_score = abs(mean([r["bias"] for r in rs]))
        score = 100.0 * (
            0.25 * activity_score + 0.20 * persistence_score + 0.20 * reversal_score
            + 0.20 * asset_stability + 0.10 * tf_stability + 0.05 * direction_score
        )
        tier = "CORE" if score >= 70 else "SECONDARY" if score >= 50 else "EVENT" if score >= 30 else "NOISY/RARE"
        weight = max(0.0, score / 100.0) * tier_cap[tier]
        weights[rule] = weight
        rows.append({"rule": rule, "behavior_score": score, "tier": tier, "weight": weight})
    return weights, rows


def components(c: Candle) -> dict[str, float]:
    from fast_pattern_trader.ohlc_rule_strategy import (
        evaluate_body_position, evaluate_body_strength, evaluate_close_behavior,
        evaluate_open_behavior, evaluate_range, evaluate_r16, evaluate_rules,
        evaluate_wick_behavior,
    )
    v = evaluate_rules(c); r = evaluate_range(c); ob = evaluate_open_behavior(c)
    cb = evaluate_close_behavior(c); wb = evaluate_wick_behavior(c)
    bp = evaluate_body_position(c); bs = evaluate_body_strength(c); r16 = evaluate_r16(c)
    return {
        "R1": float(v.close_gt_open), "R2": float(v.open_gt_close),
        "R3": float(v.close_eq_high), "R4": float(v.close_eq_low),
        "R5": float(r.directional_vote), "R6": float(ob.open_high_similarity),
        "R7": float(-ob.open_low_similarity), "R8": float(cb.close_high_similarity),
        "R9": float(-cb.close_low_similarity), "R10": float(wb.lower_wick_strength),
        "R11": float(-wb.upper_wick_strength), "R12": float(bp.bullish_position),
        "R13": float(-bp.bearish_position), "R14": float(bs.bullish_strength),
        "R15": float(-bs.bearish_strength), "R16": float(int(r16.direction)),
    }


def composite(c: Candle, weights: dict[str, float], gate: float) -> tuple[int, float, int]:
    vals = components(c)
    total = 0.0
    used = 0.0
    active = 0
    for rule in RULES:
        value = vals[rule]
        if abs(value) < TESTS[rule]:
            continue
        active += 1
        sign = 1 if value > 0 else -1 if value < 0 else 0
        total += weights[rule] * sign
        used += weights[rule]
    score = total / used if used else 0.0
    signal = 1 if score >= gate else -1 if score <= -gate else 0
    return signal, score, active


def evaluate_test(candles: list[Candle], weights: dict[str, float], gate: float) -> dict:
    longs = shorts = holds = correct = evaluated = active_sum = 0
    for i, c in enumerate(candles[:-1]):
        sig, _score, active = composite(c, weights, gate)
        active_sum += active
        if sig == 1: longs += 1
        elif sig == -1: shorts += 1
        else: holds += 1
        nxt = candles[i + 1].close
        realized = 1 if nxt > c.close else -1 if nxt < c.close else 0
        if sig and realized:
            evaluated += 1
            correct += int(sig == realized)
    total = len(candles) - 1
    signals = longs + shorts
    return {
        "candles": len(candles), "signal_rate_pct": signals / total * 100.0 if total else 0.0,
        "long_pct_of_signals": longs / signals * 100.0 if signals else 0.0,
        "directional_accuracy_pct": correct / evaluated * 100.0 if evaluated else 0.0,
        "evaluated_signals": evaluated, "avg_active_rules": active_sum / total if total else 0.0,
        "long_signals": longs, "short_signals": shorts, "hold_candles": holds,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward validation: train structural R weights on prior month, test on July.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--train-month", default="2026-06")
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--timeframes", default="5,15,30,60")
    ap.add_argument("--gate", type=float, default=0.25)
    ap.add_argument("--output", default="reports/july_r_composite_walkforward.csv")
    ap.add_argument("--weights-output", default="reports/july_r_composite_walkforward_weights.csv")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    tfs = [int(x) for x in args.timeframes.split(",") if x.strip()]
    if args.symbols.upper() == "ALL":
        symbols = sorted(set(discover_symbols(input_dir, args.train_month)) | set(discover_symbols(input_dir, args.test_month)))
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]

    train_stats: dict[tuple[str, int, str], dict[str, float]] = {}
    for symbol in symbols:
        path = find_month_file(input_dir, symbol, args.train_month)
        if not path: continue
        raw = load_binance(path)
        for tf in tfs:
            candles = resample(raw, tf)
            for rule in RULES:
                if candles:
                    train_stats[(symbol, tf, rule)] = summarize_rule(candles, rule)

    if not train_stats:
        raise SystemExit(f"No training data found for {args.train_month}")
    weights, weight_rows = build_weights(train_stats, tfs)

    wp = Path(args.weights_output); wp.parent.mkdir(parents=True, exist_ok=True)
    with wp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rule", "behavior_score", "tier", "weight"])
        w.writeheader(); w.writerows(weight_rows)

    rows: list[dict] = []
    for symbol in symbols:
        path = find_month_file(input_dir, symbol, args.test_month)
        if not path: continue
        raw = load_binance(path)
        for tf in tfs:
            candles = resample(raw, tf)
            if len(candles) < 2: continue
            result = evaluate_test(candles, weights, args.gate)
            rows.append({"train_month": args.train_month, "test_month": args.test_month, "symbol": symbol, "timeframe": tf, "gate": args.gate, **result})

    if not rows:
        raise SystemExit(f"No test data found for {args.test_month}")
    p = Path(args.output); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys()); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    all_eval = sum(int(r["evaluated_signals"]) for r in rows)
    all_correct = sum(round(int(r["evaluated_signals"]) * float(r["directional_accuracy_pct"]) / 100.0) for r in rows)
    avg_signal = mean([float(r["signal_rate_pct"]) for r in rows])
    avg_long = mean([float(r["long_pct_of_signals"]) for r in rows if float(r["signal_rate_pct"]) > 0])
    acc = all_correct / all_eval * 100.0 if all_eval else 0.0

    print(f"WF train={args.train_month} test={args.test_month} assets={len(symbols)} tfs={','.join(map(str,tfs))} gate={args.gate}")
    print("ASSET TF  SIGNAL% LONG%  ACC% EVAL  ACTIVE")
    for r in rows:
        print(f"{r['symbol']:<7} {int(r['timeframe']):>3} {float(r['signal_rate_pct']):7.1f} {float(r['long_pct_of_signals']):6.1f} {float(r['directional_accuracy_pct']):6.1f} {int(r['evaluated_signals']):5d} {float(r['avg_active_rules']):6.2f}")
    print(f"WF_AGG signal={avg_signal:.1f}% long={avg_long:.1f}% acc={acc:.1f}% eval={all_eval}")
    print(f"SAVED {p} rows={len(rows)}")
    print(f"SAVED {wp} rows={len(weight_rows)}")


if __name__ == "__main__":
    main()
