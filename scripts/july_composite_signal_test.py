from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fast_pattern_trader.models import Candle

RULES = [f"R{i}" for i in range(1, 17)]
THRESHOLDS = {
    "R1": 1.00, "R2": 0.25, "R3": 1.00, "R4": 1.00,
    "R5": 0.50, "R6": 0.50, "R7": 0.50, "R8": 0.50,
    "R9": 0.50, "R10": 0.50, "R11": 0.50, "R12": 0.50,
    "R13": 0.50, "R14": 0.75, "R15": 0.75, "R16": 1.00,
}


def load_binance(path: Path) -> list[Candle]:
    rows: list[Candle] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.reader(f):
            if len(r) < 5:
                continue
            try:
                ts = int(float(r[0]))
                if ts > 10_000_000_000_000:
                    ts //= 1_000_000
                elif ts > 10_000_000_000:
                    ts //= 1_000
                rows.append(Candle(ts, float(r[1]), float(r[2]), float(r[3]), float(r[4])))
            except (ValueError, TypeError):
                continue
    return sorted(rows, key=lambda x: x.timestamp)


def resample(candles: list[Candle], minutes: int) -> list[Candle]:
    if minutes == 1:
        return candles
    bucket = minutes * 60
    groups: dict[int, list[Candle]] = {}
    for c in candles:
        groups.setdefault((c.timestamp // bucket) * bucket, []).append(c)
    out: list[Candle] = []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda x: x.timestamp)
        ids = sorted({x.timestamp // 60 for x in group})
        if len(ids) != minutes or ids[-1] - ids[0] + 1 != minutes:
            continue
        out.append(Candle(key, group[0].open, max(x.high for x in group), min(x.low for x in group), group[-1].close))
    return out


def find_month_file(input_dir: Path, symbol: str, month: str) -> Path | None:
    exact = [
        input_dir / f"{symbol}-1m-{month}.csv",
        input_dir / f"{symbol}_1m_{month}.csv",
        input_dir / f"{symbol}-{month}.csv",
    ]
    for p in exact:
        if p.exists():
            return p
    found = list(input_dir.rglob(f"*{symbol}*{month}*.csv"))
    return found[0] if found else None


def components(c: Candle) -> dict[str, float]:
    from fast_pattern_trader.ohlc_rule_strategy import (
        evaluate_body_position,
        evaluate_body_strength,
        evaluate_close_behavior,
        evaluate_open_behavior,
        evaluate_range,
        evaluate_r16,
        evaluate_rules,
        evaluate_wick_behavior,
    )
    v = evaluate_rules(c)
    r = evaluate_range(c)
    ob = evaluate_open_behavior(c)
    cb = evaluate_close_behavior(c)
    wb = evaluate_wick_behavior(c)
    bp = evaluate_body_position(c)
    bs = evaluate_body_strength(c)
    r16 = evaluate_r16(c)
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


def fixed_rule_weight(score: float, tier: str) -> float:
    # Score is only a structural reliability magnitude. Tier caps prevent rare/event rules
    # from dominating the composite. No return-based fitting is performed.
    tier_cap = {"CORE": 1.00, "SECONDARY": 0.65, "EVENT": 0.35, "NOISY/RARE": 0.15}
    return max(0.0, score / 100.0) * tier_cap.get(tier, 0.15)


def load_scores(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[r["rule"]] = fixed_rule_weight(float(r["behavior_score"]), r["tier"])
    missing = [r for r in RULES if r not in out]
    if missing:
        raise SystemExit(f"Missing rule scores: {','.join(missing)}")
    return out


def composite_signal(c: Candle, weights: dict[str, float]) -> tuple[int, float, int]:
    vals = components(c)
    total = 0.0
    used = 0.0
    active = 0
    for rule in RULES:
        value = vals[rule]
        threshold = THRESHOLDS[rule]
        if abs(value) < threshold:
            continue
        active += 1
        w = weights[rule]
        # Each component is already signed by the R1-R16 definition.
        # Do not apply a second direction map: doing so reverses R1/R3/R7/R9/R11/R13/R15
        # and creates a structural long bias in the composite.
        signed = 1 if value > 0 else -1 if value < 0 else 0
        total += w * signed
        used += w
    score = total / used if used else 0.0
    # Fixed consensus gate: at least 25% normalized directional agreement.
    signal = 1 if score >= 0.25 else -1 if score <= -0.25 else 0
    return signal, score, active


def main() -> None:
    ap = argparse.ArgumentParser(description="Fixed non-optimized R1-R16 composite July signal test.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--month", default="2026-07")
    ap.add_argument("--timeframes", default="5,15,30,60")
    ap.add_argument("--scores", default="reports/july_r_behavior_scores.csv")
    ap.add_argument("--output", default="reports/july_composite_signal_test.csv")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    scores = load_scores(Path(args.scores))
    if args.symbols.upper() == "ALL":
        symbols = sorted({
            p.name.split("-1m-")[0] for p in input_dir.rglob("*.csv") if "-1m-" + args.month in p.name
        })
        if not symbols:
            symbols = sorted({
                p.name.split("_1m_")[0] for p in input_dir.rglob("*.csv") if "_1m_" + args.month in p.name
            })
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    tfs = [int(x) for x in args.timeframes.split(",") if x.strip()]

    rows: list[dict] = []
    for symbol in symbols:
        src = find_month_file(input_dir, symbol, args.month)
        if not src:
            print(f"SKIP {symbol}: no {args.month} file")
            continue
        base = load_binance(src)
        for tf in tfs:
            candles = resample(base, tf)
            if len(candles) < 2:
                continue
            longs = shorts = holds = correct = evaluated = active_sum = 0
            for i, c in enumerate(candles[:-1]):
                sig, score, active = composite_signal(c, scores)
                active_sum += active
                if sig == 1:
                    longs += 1
                elif sig == -1:
                    shorts += 1
                else:
                    holds += 1
                nxt = candles[i + 1].close
                realized = 1 if nxt > c.close else -1 if nxt < c.close else 0
                if sig and realized:
                    evaluated += 1
                    correct += int(sig == realized)
            total = len(candles) - 1
            signals = longs + shorts
            rows.append({
                "symbol": symbol, "timeframe": tf, "candles": len(candles),
                "signal_candles": signals, "signal_rate_pct": signals / total * 100,
                "long_signals": longs, "short_signals": shorts, "hold_candles": holds,
                "long_pct_of_signals": longs / signals * 100 if signals else 0.0,
                "directional_accuracy_pct": correct / evaluated * 100 if evaluated else 0.0,
                "evaluated_signals": evaluated,
                "avg_active_rules": active_sum / total,
            })

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        fields = list(rows[0].keys()) if rows else ["symbol", "timeframe"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    print(f"COMPOSITE month={args.month} assets={len(symbols)} tfs={','.join(map(str,tfs))} gate=0.25")
    print("ASSET TF  SIGNAL% LONG%  ACC% EVAL  ACTIVE")
    for r in rows:
        print(f"{r['symbol']:<7} {int(r['timeframe']):>3} {r['signal_rate_pct']:7.1f} {r['long_pct_of_signals']:6.1f} {r['directional_accuracy_pct']:6.1f} {int(r['evaluated_signals']):5d} {r['avg_active_rules']:6.2f}")
    print(f"SAVED {p} rows={len(rows)}")


if __name__ == "__main__":
    main()
