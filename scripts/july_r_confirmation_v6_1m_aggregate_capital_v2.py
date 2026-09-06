from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_pattern_trader.models import Candle
from scripts.july_r_composite_walkforward import (
    discover_symbols, find_month_file, load_binance,
)


def pct(c: int, n: int) -> float:
    return 100.0 * c / n if n else 0.0


def raw_sig(c: Candle, rule: str, components, tests) -> int:
    v = components(c)[rule]
    th = tests[rule]
    return 1 if v >= th else -1 if v <= -th else 0


def variant_sig(c: Candle, rule: str, variant: str, components, tests) -> int:
    s = raw_sig(c, rule, components, tests)
    return s if variant == "N" else -s


def chain_signal(cs: list[Candle], i: int, candidate: dict, components, tests) -> tuple[bool, int]:
    trigger = variant_sig(cs[i], candidate["trigger"], candidate["variant"], components, tests)
    if not trigger:
        return False, 0
    for k, x in enumerate(candidate["selected"], 1):
        j = i + k
        if j >= len(cs):
            return False, trigger
        s = variant_sig(cs[j], x["rule"], x["variant"], components, tests)
        if not s:
            return False, trigger
        expected = trigger if x["mode"] == "SAME" else -trigger
        if s != expected:
            return False, trigger
    return True, trigger


def one_min_predictions(cs, candidate, components, tests):
    out = {}
    depth = len(candidate["selected"])
    for i in range(len(cs) - depth - 1):
        ok, direction = chain_signal(cs, i, candidate, components, tests)
        if ok and direction:
            # Prediction becomes known at the close of the last confirmation candle.
            out[i + depth] = direction
    return out


def parse_set(label: str) -> tuple[str, str, list[dict]]:
    parts = [x for x in label.strip().split("+") if x]
    if not parts:
        raise ValueError("empty candidate set")
    first = parts[0].split(":")
    if len(first) < 2:
        raise ValueError(f"invalid candidate: {label}")
    trigger, variant = first[0].strip(), first[1].strip()
    selected = []
    for item in parts[1:]:
        bits = item.split(":")
        if len(bits) < 3:
            raise ValueError(f"invalid confirmation in candidate: {label}")
        mode = "SAME" if bits[2].strip().upper().startswith("S") else "OPPOSITE"
        selected.append({"rule": bits[0].strip(), "variant": bits[1].strip(), "mode": mode})
    return trigger, variant, selected


def load_frozen_candidates(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen V6 candidate file not found: {path}. Run the original V6 characterization first "
            "or pass --candidates with its CSV path."
        )
    out = []
    seen = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = (row.get("set") or row.get("candidate") or "").strip()
            if not label:
                continue
            trigger, variant, selected = parse_set(label)
            key = (trigger, variant, tuple((x["rule"], x["variant"], x["mode"]) for x in selected))
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "trigger": trigger,
                "variant": variant,
                "selected": selected,
                "label": label,
                "source_tf": row.get("timeframe_min", row.get("tf", "")),
                "source_level": row.get("level", ""),
            })
    return out


def block_vote(preds: dict[int, int], start: int, end: int) -> tuple[int, int, int]:
    long_n = short_n = hold_n = 0
    for i in range(start, end):
        d = preds.get(i, 0)
        if d > 0:
            long_n += 1
        elif d < 0:
            short_n += 1
        else:
            hold_n += 1
    return long_n, short_n, hold_n


def aggregate_blocks(preds, block_size: int, strict_majority: float, n_bars: int):
    out = []
    for end in range(block_size, n_bars + 1, block_size):
        start = end - block_size
        l, s, h = block_vote(preds, start, end)
        total = end - start
        if l > s and 100.0 * l / total > strict_majority:
            d = 1
        elif s > l and 100.0 * s / total > strict_majority:
            d = -1
        else:
            d = 0
        out.append((end, d, l, s, h))
    return out


def run_capital(series, candidate, tf: int, initial: float, fee_side_pct: float, vote_majority: float, components, tests):
    per_symbol = initial / max(1, len(series))
    total = 0.0
    trades = wins = losses = 0
    fees = 0.0
    max_dd = 0.0
    for _, cs in series.items():
        local = per_symbol
        preds = one_min_predictions(cs, candidate, components, tests)
        blocks = aggregate_blocks(preds, tf, vote_majority, len(cs))
        peak = local
        for end, direction, _, _, _ in blocks:
            if not direction:
                continue
            entry_i = end - 1
            exit_i = entry_i + tf
            if exit_i >= len(cs):
                break
            entry = cs[entry_i].close
            exit_price = cs[exit_i].close
            if entry <= 0 or exit_price <= 0:
                continue
            move = (exit_price / entry - 1.0) * direction
            gross = local * move
            fee = local * (2.0 * fee_side_pct / 100.0)
            local = max(0.0, local + gross - fee)
            fees += fee
            trades += 1
            wins += int(gross > 0)
            losses += int(gross <= 0)
            peak = max(peak, local)
            if peak:
                max_dd = max(max_dd, 100.0 * (peak - local) / peak)
            if local <= 0:
                break
        total += local
    return {
        "final": total,
        "return_pct": 100.0 * (total / initial - 1.0),
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": pct(wins, trades),
        "fees": fees,
        "max_dd_pct": max_dd,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Freeze previously selected V6 candidates, generate their predictions only on 1m data, aggregate into higher-TF votes, then run capital backtest."
    )
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--timeframes", default="5,15,30,60,120,240")
    ap.add_argument("--vote-majority", type=float, default=50.0)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee-side-pct", type=float, default=1.3)
    ap.add_argument("--candidates", default="reports/july_r_confirmation_all_triggers_wf_v6.csv")
    ap.add_argument("--output", default="reports/july_r_confirmation_v6_1m_aggregate_capital_v3.csv")
    args = ap.parse_args()

    root = Path(args.input_dir)
    tfs = [int(x) for x in args.timeframes.split(",") if x.strip()]
    symbols = (sorted(discover_symbols(root, args.test_month))
               if args.symbols.upper() == "ALL"
               else [x.strip().upper() for x in args.symbols.split(",") if x.strip()])

    from scripts.july_r_composite_walkforward import components, TESTS

    candidates = load_frozen_candidates(root / args.candidates if not Path(args.candidates).is_absolute() else Path(args.candidates))
    test = {}
    for s in symbols:
        p = find_month_file(root, s, args.test_month)
        if not p:
            continue
        cs = load_binance(p)
        if len(cs) > 32:
            test[s] = cs

    if not test:
        raise RuntimeError("No 1m test data found.")

    print(
        f"V6_1M_FROZEN_AGGREGATE_CAPITAL_V3 test={args.test_month} assets={len(test)} "
        f"frozen_candidates={len(candidates)} tfs={','.join(map(str,tfs))} majority>{args.vote_majority:.1f}% initial={args.initial_capital:.2f}"
    )
    print("NO candidate discovery. NO train pass. NO optimization. Candidates are loaded once from the prior V6 CSV.")
    print("Pipeline: frozen V6 candidate -> 1m predictions -> block vote -> capital. Fees are reported at 1.3% per side.")

    rows = []
    for tf in tfs:
        print(f"\n===== TF={tf}m =====")
        for rank, c in enumerate(candidates, 1):
            r = run_capital(test, c, tf, args.initial_capital, args.fee_side_pct, args.vote_majority, components, TESTS)
            print(
                f"{rank:3} {c['label']:<45} SRC={c['source_tf']} L={c['source_level']} "
                f"FINAL={r['final']:.2f} RET={r['return_pct']:+.2f}% T={r['trades']} WIN={r['win_rate_pct']:.2f}% "
                f"DD={r['max_dd_pct']:.2f}% FEES={r['fees']:.2f}"
            )
            rows.append({
                "tf": tf,
                "rank": rank,
                "candidate": c["label"],
                "source_tf": c["source_tf"],
                "source_level": c["source_level"],
                "initial_capital": args.initial_capital,
                "final_capital": r["final"],
                "return_pct": r["return_pct"],
                "trades": r["trades"],
                "wins": r["wins"],
                "losses": r["losses"],
                "win_rate_pct": r["win_rate_pct"],
                "max_dd_pct": r["max_dd_pct"],
                "fees_paid": r["fees"],
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys()) if rows else ["tf"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"SAVED {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
