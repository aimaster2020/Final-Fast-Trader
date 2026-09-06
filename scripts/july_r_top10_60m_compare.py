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
    components, TESTS, discover_symbols, find_month_file, load_binance, resample,
)

TOP_AGG = [
    "R16:I",
    "R8:I",
    "R2:N",
    "R6:I+R8:I:S",
    "R6:N",
]

TOP_REVERSE = [
    "R5:N",
    "R15:N",
    "R6:N",
    "R12:N",
    "R16:I+R8:I:O+R6:I:O",
]


def raw_sig(c: Candle, rule: str) -> int:
    v = components(c)[rule]
    th = TESTS[rule]
    return 1 if v >= th else -1 if v <= -th else 0


def variant_sig(c: Candle, rule: str, variant: str) -> int:
    s = raw_sig(c, rule)
    return s if variant == "N" else -s


def parse_set(label: str) -> tuple[str, str, list[dict]]:
    parts = [x for x in label.split("+") if x]
    first = parts[0].split(":")
    trigger, variant = first[0], first[1]
    selected = []
    for item in parts[1:]:
        bits = item.split(":")
        selected.append({
            "rule": bits[0],
            "variant": bits[1],
            "mode": "SAME" if bits[2].upper().startswith("S") else "OPPOSITE",
        })
    return trigger, variant, selected


def chain_signal(cs: list[Candle], i: int, c: dict) -> tuple[bool, int]:
    trigger = variant_sig(cs[i], c["trigger"], c["variant"])
    if not trigger:
        return False, 0
    for k, x in enumerate(c["selected"], 1):
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


def make_candidate(label: str) -> dict:
    trigger, variant, selected = parse_set(label)
    return {"label": label, "trigger": trigger, "variant": variant, "selected": selected}


def one_min_predictions(cs: list[Candle], c: dict) -> dict[int, int]:
    out = {}
    depth = len(c["selected"])
    for i in range(len(cs) - depth - 1):
        ok, d = chain_signal(cs, i, c)
        if ok and d:
            out[i + depth] = d
    return out


def block_vote(preds: dict[int, int], start: int, end: int) -> int:
    long_n = short_n = 0
    for i in range(start, end):
        d = preds.get(i, 0)
        long_n += d > 0
        short_n += d < 0
    total = end - start
    if long_n > short_n and 100.0 * long_n / total > 50.0:
        return 1
    if short_n > long_n and 100.0 * short_n / total > 50.0:
        return -1
    return 0


def run_aggregate(series: dict[str, list[Candle]], c: dict, fee_side_pct: float) -> dict:
    initial = 1000.0
    per_symbol = initial / len(series)
    total = 0.0
    trades = wins = 0
    fees = 0.0
    max_dd = 0.0
    for raw in series.values():
        local = per_symbol
        peak = local
        preds = one_min_predictions(raw, c)
        for end in range(60, len(raw) + 1, 60):
            direction = block_vote(preds, end - 60, end)
            if not direction:
                continue
            entry_i = end - 1
            exit_i = entry_i + 60
            if exit_i >= len(raw):
                break
            move = (raw[exit_i].close / raw[entry_i].close - 1.0) * direction
            gross = local * move
            fee = local * 2.0 * fee_side_pct / 100.0
            local = max(0.0, local + gross - fee)
            fees += fee
            trades += 1
            wins += gross > 0
            peak = max(peak, local)
            if peak:
                max_dd = max(max_dd, 100.0 * (peak - local) / peak)
    total += local
        # one symbol per loop
    return result(total, initial, trades, wins, fees, max_dd)


def result(final: float, initial: float, trades: int, wins: int, fees: float, dd: float) -> dict:
    return {
        "final": final,
        "ret": 100.0 * (final / initial - 1.0),
        "trades": trades,
        "win": 100.0 * wins / trades if trades else 0.0,
        "dd": dd,
        "fees": fees,
    }


def signal_times_15m(c15: list[Candle], c: dict) -> list[tuple[int, int]]:
    out = []
    depth = len(c["selected"])
    for i in range(len(c15) - depth):
        ok, d = chain_signal(c15, i, c)
        if ok and d:
            out.append((c15[i + depth].timestamp + 15 * 60, d))
    return out


def run_reverse(series: dict[str, list[Candle]], c: dict, fee_side_pct: float) -> dict:
    initial = 1000.0
    per_symbol = initial / len(series)
    total = 0.0
    trades = wins = 0
    fees = 0.0
    max_dd = 0.0
    for raw in series.values():
        local = per_symbol
        peak = local
        c15 = resample(raw, 15)
        c60 = resample(raw, 60)
        signals = signal_times_15m(c15, c)
        j = 0
        for signal_end, direction in signals:
            while j < len(c60) and c60[j].timestamp < signal_end:
                j += 1
            if j >= len(c60) - 1:
                break
            entry = c60[j].close
            exit_price = c60[j + 1].close
            move = (exit_price / entry - 1.0) * direction
            gross = local * move
            fee = local * 2.0 * fee_side_pct / 100.0
            local = max(0.0, local + gross - fee)
            fees += fee
            trades += 1
            wins += gross > 0
            peak = max(peak, local)
            if peak:
                max_dd = max(max_dd, 100.0 * (peak - local) / peak)
            j += 2
            if local <= 0:
                break
        total += local
    return result(total, initial, trades, wins, fees, max_dd)


def load_series(data_root: Path, month: str, symbols_arg: str) -> dict[str, list[Candle]]:
    symbols = sorted(discover_symbols(data_root, month)) if symbols_arg.upper() == "ALL" else [x.strip().upper() for x in symbols_arg.split(",") if x.strip()]
    out = {}
    for s in symbols:
        p = find_month_file(data_root, s, month)
        if p:
            cs = load_binance(p)
            if len(cs) >= 64:
                out[s] = cs
    if not out:
        raise RuntimeError("No test data found")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    args = ap.parse_args()

    series = load_series(Path(args.input_dir), args.test_month, args.symbols)
    print(f"TOP10_60M test={args.test_month} assets={len(series)} initial=1000.00")
    print("A=1m predictions -> 60m block vote | B=15m signal -> next 60m execution")
    print("COMM=0 and COMM=1.3%/side")
    print("---")

    rows = []
    for arch, labels, runner in [("A", TOP_AGG, run_aggregate), ("B", TOP_REVERSE, run_reverse)]:
        for label in labels:
            c = make_candidate(label)
            for fee in (0.0, 1.3):
                r = runner(series, c, fee)
                print(f"{arch} {label:<32} FEE={fee:.1f}%/side FINAL={r['final']:.2f} RET={r['ret']:+.2f}% T={r['trades']} WIN={r['win']:.2f}% DD={r['dd']:.2f}% FEES={r['fees']:.2f}")
                rows.append({"arch": arch, "candidate": label, "fee_side_pct": fee, **r})

    out = ROOT / "reports" / "july_r_top10_60m_compare.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"SAVED {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
