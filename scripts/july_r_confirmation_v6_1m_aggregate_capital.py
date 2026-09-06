from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_pattern_trader.models import Candle
from scripts.july_r_confirmation_all_triggers_wf_v6 import (
    RULES,
    choose_variant,
    choose_next,
    chain_ok,
    evaluate_level,
    load_sets,
)
from scripts.july_r_composite_walkforward import (
    components,
    discover_symbols,
    find_month_file,
    load_binance,
    resample,
)


def raw_sig(c: Candle, rule: str) -> int:
    v = components(c)[rule]
    th = __import__("scripts.july_r_composite_walkforward", fromlist=["TESTS"]).TESTS[rule]
    return 1 if v >= th else -1 if v <= -th else 0


def variant_sig(c: Candle, rule: str, variant: str) -> int:
    s = raw_sig(c, rule)
    return s if variant == "N" else -s


def next_direction(cs: list[Candle], i: int) -> int:
    if i + 1 >= len(cs):
        return 0
    return 1 if cs[i + 1].close > cs[i].close else -1 if cs[i + 1].close < cs[i].close else 0


def pct(c: int, n: int) -> float:
    return 100.0 * c / n if n else 0.0


def load_1m(root: Path, symbols: list[str], month: str) -> dict[str, list[Candle]]:
    out: dict[str, list[Candle]] = {}
    for s in symbols:
        p = find_month_file(root, s, month)
        if not p:
            continue
        cs = load_binance(p)
        if len(cs) > 32:
            out[s] = cs
    return out


def build_1m_decisions(
    cs: list[Candle],
    trigger_rule: str,
    trigger_variant: str,
    selected: list[dict],
) -> dict[int, int]:
    """Map each closed 1m bar index to the decision generated from that bar.

    The V6 strategy itself is evaluated on the 1m candle sequence. A decision is
    available only after the complete confirmation chain has closed.
    """
    out: dict[int, int] = {}
    for i in range(len(cs) - len(selected) - 2):
        ok, direction = chain_ok(cs, i, trigger_rule, trigger_variant, selected)
        if not ok or not direction:
            continue
        signal_bar = i + len(selected)
        out[signal_bar] = direction
    return out


def aggregate_prediction(decisions: dict[int, int], start_bar: int, end_bar: int, threshold: float) -> int:
    long_n = short_n = hold_n = 0
    for i in range(start_bar, end_bar):
        d = decisions.get(i, 0)
        if d > 0:
            long_n += 1
        elif d < 0:
            short_n += 1
        else:
            hold_n += 1
    total = end_bar - start_bar
    if total <= 0:
        return 0
    if 100.0 * long_n / total >= threshold and long_n > short_n:
        return 1
    if 100.0 * short_n / total >= threshold and short_n > long_n:
        return -1
    return 0


def aggregate_votes_by_block(decisions: dict[int, int], block_size: int, threshold: float, n_bars: int) -> dict[int, int]:
    """Create a higher-TF prediction from the preceding block of closed 1m predictions.

    For example, 5m uses 5 one-minute predictions, 15m uses 15, etc. The prediction
    for block k is computed from the just-closed block (not the future block), then
    applied to the NEXT higher-TF interval.
    """
    out: dict[int, int] = {}
    for end in range(block_size, n_bars + 1, block_size):
        start = end - block_size
        d = aggregate_prediction(decisions, start, end, threshold)
        out[end] = d
    return out


@dataclass
class Capital:
    value: float
    peak: float
    max_dd: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    fees: float = 0.0

    def trade(self, direction: int, entry: float, exit_price: float, fee_side_pct: float) -> None:
        if entry <= 0 or exit_price <= 0 or not direction:
            return
        move = (exit_price / entry - 1.0) * direction
        gross = self.value * move
        fee = self.value * (2.0 * fee_side_pct / 100.0)
        self.value = max(0.0, self.value + gross - fee)
        self.trades += 1
        self.wins += int(gross > 0)
        self.losses += int(gross <= 0)
        self.fees += fee
        self.peak = max(self.peak, self.value)
        if self.peak > 0:
            self.max_dd = max(self.max_dd, (self.peak - self.value) / self.peak)


def strategy_for_tf(
    train: list[list[Candle]],
    threshold: float,
    min_samples: int,
    max_levels: int,
) -> tuple[tuple[str, str], list[dict]] | None:
    best = None
    for r in RULES:
        tv, tn, nn, ti, ni = choose_variant(train, r, min_samples)
        selected: list[dict] = []
        c, n = 0, 0
        # Use V6 training selection only. Do not inspect July to choose the strategy.
        for i in range(len(train[0]) - len(selected) - 2):
            pass
        # Start with trigger-only and then greedily extend.
        def train_acc(sel: list[dict]) -> tuple[float, int]:
            depth = len(sel)
            correct = count = 0
            for cs in train:
                for i in range(len(cs) - depth - 2):
                    ok, d = chain_ok(cs, i, r, tv, sel)
                    if not ok:
                        continue
                    y = next_direction(cs, i + depth + 1)
                    if not y:
                        continue
                    count += 1; correct += int(d == y)
            return pct(correct, count), count
        a, n = train_acc([])
        if n >= min_samples and a >= threshold:
            cand = ((r, tv), [], a, n)
            if best is None or (a, n) > (best[2], best[3]):
                best = cand
        for _ in range(max_levels):
            ranked = choose_next(train, r, tv, selected, min_samples)
            ranked = [x for x in ranked if x["acc"] >= threshold]
            if not ranked:
                break
            selected.append(ranked[0])
            a, n = train_acc(selected)
            if n >= min_samples and a >= threshold:
                cand = ((r, tv), [dict(x) for x in selected], a, n)
                if best is None or (a, n) > (best[2], best[3]):
                    best = cand
    if best is None:
        return None
    return best[0], best[1]


def run_for_symbol(
    one_min: list[Candle],
    train_1m: list[list[Candle]],
    initial: float,
    fee_side_pct: float,
    agg_tf: int,
    vote_threshold: float,
    min_samples: int,
    max_levels: int,
) -> Capital:
    strategy = strategy_for_tf(train_1m, 55.0, min_samples, max_levels)
    if strategy is None:
        return Capital(initial, initial)
    trigger, selected = strategy
    decisions = build_1m_decisions(one_min, trigger[0], trigger[1], selected)
    n = len(one_min)
    block_predictions = aggregate_votes_by_block(decisions, agg_tf, vote_threshold, n)
    eq = Capital(initial, initial)
    # block end E -> prediction applies to candles E ... E+agg_tf-1, using close-to-close.
    for end, direction in sorted(block_predictions.items()):
        if not direction:
            continue
        if end >= n or end + agg_tf - 1 >= n:
            break
        entry_i = end
        exit_i = end + agg_tf
        if exit_i >= n:
            break
        eq.trade(direction, one_min[entry_i].close, one_min[exit_i].close, fee_side_pct)
    return eq


def main() -> None:
    ap = argparse.ArgumentParser(description="Use 1m V6 predictions and aggregate them into 5/15/30/60m higher-TF decisions, then calculate capital.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--train-month", default="2026-06")
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--timeframes", default="5,15,30,60,120,240")
    ap.add_argument("--vote-threshold", type=float, default=50.0)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--max-levels", type=int, default=2)
    args = ap.parse_args()
    root = Path(args.input_dir)
    tfs = [int(x) for x in args.timeframes.split(",") if x.strip()]
    if args.symbols.upper() == "ALL":
        symbols = sorted(set(discover_symbols(root, args.train_month)) | set(discover_symbols(root, args.test_month)))
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    train_1m_sets = load_1m(root, symbols, args.train_month)
    test_1m = load_1m(root, symbols, args.test_month)
    train = list(train_1m_sets.values())
    print(f"V6_1M_AGGREGATE_CAPITAL train={args.train_month} test={args.test_month} assets={len(symbols)} tfs={','.join(map(str,tfs))} vote={args.vote_threshold:.1f}% initial={args.initial_capital:.2f}")
    print("1m V6 decisions are aggregated; a higher-TF prediction is applied only to the NEXT higher-TF interval. Fees: 0 and 1.3%/side.")
    rows=[]
    for tf in tfs:
        for fee in (0.0, 1.3):
            eq = Capital(args.initial_capital, args.initial_capital)
            used=0
            for s in symbols:
                if s not in test_1m: continue
                local = run_for_symbol(test_1m[s], train, args.initial_capital, fee, tf, args.vote_threshold, args.min_samples, args.max_levels)
                eq.value += local.value - args.initial_capital
                eq.trades += local.trades; eq.wins += local.wins; eq.losses += local.losses; eq.fees += local.fees
                used += 1
            eq.peak = max(args.initial_capital, eq.value)
            print(f"TF={tf}m FEE={fee:.1f}%/side FINAL={eq.value:.2f} RETURN={100*(eq.value/args.initial_capital-1):+.2f}% TRADES={eq.trades} WIN={pct(eq.wins,eq.trades):.2f}% FEES={eq.fees:.2f}")
            rows.append({"tf":tf,"fee_side_pct":fee,"final":eq.value,"return_pct":100*(eq.value/args.initial_capital-1),"trades":eq.trades,"wins":eq.wins,"losses":eq.losses,"win_rate_pct":pct(eq.wins,eq.trades),"fees":eq.fees,"assets":used})
    out=Path("reports/july_r_confirmation_v6_1m_aggregate_capital.csv"); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()) if rows else ["tf"]); w.writeheader(); w.writerows(rows)
    print(f"SAVED {out} rows={len(rows)}")

if __name__ == "__main__":
    main()
