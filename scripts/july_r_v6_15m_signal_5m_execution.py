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


def raw_sig(c: Candle, rule: str) -> int:
    value = components(c)[rule]
    threshold = TESTS[rule]
    return 1 if value >= threshold else -1 if value <= -threshold else 0


def variant_sig(c: Candle, rule: str, variant: str) -> int:
    s = raw_sig(c, rule)
    return s if variant == "N" else -s


def chain_signal(cs: list[Candle], i: int, candidate: dict) -> tuple[bool, int]:
    trigger = variant_sig(cs[i], candidate["trigger"], candidate["variant"])
    if not trigger:
        return False, 0
    for k, x in enumerate(candidate["selected"], 1):
        j = i + k
        if j >= len(cs):
            return False, trigger
        s = variant_sig(cs[j], x["rule"], x["variant"])
        if not s:
            return False, trigger
        expected = trigger if x["mode"] == "SAME" else -trigger
        if s != expected:
            return False, trigger
    return True, trigger


def parse_set(label: str) -> tuple[str, str, list[dict]]:
    parts = [x for x in label.strip().split("+") if x]
    first = parts[0].split(":")
    if len(first) < 2:
        raise ValueError(f"invalid candidate: {label}")
    trigger, variant = first[0].strip(), first[1].strip()
    selected = []
    for item in parts[1:]:
        bits = item.split(":")
        if len(bits) < 3:
            raise ValueError(f"invalid confirmation: {label}")
        selected.append({
            "rule": bits[0].strip(),
            "variant": bits[1].strip(),
            "mode": "SAME" if bits[2].strip().upper().startswith("S") else "OPPOSITE",
        })
    return trigger, variant, selected


def load_candidates(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Frozen V6 candidates not found: {path}")
    out, seen = [], set()
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
                "label": label,
                "trigger": trigger,
                "variant": variant,
                "selected": selected,
                "source_tf": row.get("timeframe_min", row.get("tf", "")),
                "source_level": row.get("level", ""),
            })
    return out


def signal_times_15m(candles15: list[Candle], candidate: dict) -> list[tuple[int, int]]:
    """Return (15m candle close timestamp, direction). Signal is known only after 15m close."""
    out = []
    depth = len(candidate["selected"])
    for i in range(len(candles15) - depth):
        ok, direction = chain_signal(candles15, i, candidate)
        if ok and direction:
            last = candles15[i + depth]
            out.append((last.timestamp + 15 * 60, direction))
    return out


def run_capital(series: dict[str, list[Candle]], candidate: dict, initial: float, fee_side_pct: float) -> dict:
    per_symbol = initial / max(1, len(series))
    total = 0.0
    trades = wins = losses = 0
    fees = 0.0
    max_dd = 0.0
    for _, raw in series.items():
        c15 = resample(raw, 15)
        c5 = resample(raw, 5)
        signals = signal_times_15m(c15, candidate)
        if len(c5) < 2:
            total += per_symbol
            continue
        local = per_symbol
        peak = local
        j = 0
        for signal_end, direction in signals:
            while j < len(c5) and c5[j].timestamp < signal_end:
                j += 1
            if j >= len(c5) - 1:
                break
            entry_i = j
            exit_i = j + 1
            entry = c5[entry_i].close
            exit_price = c5[exit_i].close
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
            j = exit_i + 1
            if local <= 0:
                break
        total += local
    return {
        "final": total,
        "return_pct": 100.0 * (total / initial - 1.0),
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": 100.0 * wins / trades if trades else 0.0,
        "fees": fees,
        "max_dd_pct": max_dd,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen V6 15m signal -> next 5m candle execution backtest on July 2026.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee-side-pct", type=float, default=1.3)
    ap.add_argument("--candidates", default="reports/july_r_confirmation_all_triggers_wf_v6.csv")
    ap.add_argument("--output", default="reports/july_r_v6_15m_signal_5m_execution.csv")
    args = ap.parse_args()

    data_root = Path(args.input_dir)
    candidate_path = Path(args.candidates)
    if not candidate_path.is_absolute():
        candidate_path = ROOT / candidate_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    if args.symbols.upper() == "ALL":
        symbols = sorted(discover_symbols(data_root, args.test_month))
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]

    candidates = load_candidates(candidate_path)
    series = {}
    for symbol in symbols:
        p = find_month_file(data_root, symbol, args.test_month)
        if not p:
            continue
        raw = load_binance(p)
        if len(raw) >= 32:
            series[symbol] = raw
    if not series:
        raise RuntimeError("No 1m July test data found.")

    print(f"V6_15M_SIGNAL_5M_EXECUTION test={args.test_month} assets={len(series)} frozen_candidates={len(candidates)} initial={args.initial_capital:.2f} fee={args.fee_side_pct:.2f}%/side")
    print(f"CANDIDATES={candidate_path}")
    print("NO candidate discovery. NO train pass. NO optimization. V6 candidates remain frozen.")
    print("Pipeline: 1m raw -> 15m R1-R16 signal/confirmation -> signal known after 15m close -> next 5m candle execution.")
    print("Each 15m signal produces at most one 5m trade; 1x capital, full local capital, 1.3% fee per side.")

    rows = []
    for rank, c in enumerate(candidates, 1):
        r = run_capital(series, c, args.initial_capital, args.fee_side_pct)
        print(f"{rank:3} {c['label']:<45} SRC={c['source_tf']} L={c['source_level']} FINAL={r['final']:.2f} RET={r['return_pct']:+.2f}% T={r['trades']} WIN={r['win_rate_pct']:.2f}% DD={r['max_dd_pct']:.2f}% FEES={r['fees']:.2f}")
        rows.append({"rank": rank, "candidate": c["label"], "source_tf": c["source_tf"], "source_level": c["source_level"], "initial_capital": args.initial_capital, "final_capital": r["final"], "return_pct": r["return_pct"], "trades": r["trades"], "wins": r["wins"], "losses": r["losses"], "win_rate_pct": r["win_rate_pct"], "max_dd_pct": r["max_dd_pct"], "fees_paid": r["fees"]})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"SAVED {output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
