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
from scripts.july_r_composite_walkforward import (
    RULES,
    TESTS,
    components,
    discover_symbols,
    find_month_file,
    load_binance,
    resample,
)


def raw_sig(c: Candle, rule: str) -> int:
    v = components(c)[rule]
    th = TESTS[rule]
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


def load_series(root: Path, symbols: list[str], month: str, tf: int) -> dict[str, list[Candle]]:
    out: dict[str, list[Candle]] = {}
    for symbol in symbols:
        p = find_month_file(root, symbol, month)
        if not p:
            continue
        cs = resample(load_binance(p), tf)
        if len(cs) > 32:
            out[symbol] = cs
    return out


def chain_pass(cs: list[Candle], i: int, trigger: tuple[str, str], selected: list[tuple[str, str, str]]) -> tuple[bool, int]:
    tr, tv = trigger
    if i >= len(cs):
        return False, 0
    s0 = variant_sig(cs[i], tr, tv)
    if not s0:
        return False, 0
    for k, (rule, variant, mode) in enumerate(selected, 1):
        j = i + k
        if j >= len(cs):
            return False, s0
        s = variant_sig(cs[j], rule, variant)
        if not s:
            return False, s0
        expected = s0 if mode == "SAME" else -s0
        if s != expected:
            return False, s0
    return True, s0


def chain_accuracy(train: list[list[Candle]], trigger: tuple[str, str], selected: list[tuple[str, str, str]]) -> tuple[int, int]:
    depth = len(selected)
    correct = n = 0
    for cs in train:
        for i in range(len(cs) - depth - 2):
            ok, direction = chain_pass(cs, i, trigger, selected)
            if not ok:
                continue
            y = next_direction(cs, i + depth + 1)
            if not y:
                continue
            n += 1
            correct += int(direction == y)
    return correct, n


def rank_confirmations(
    train: list[list[Candle]],
    trigger: tuple[str, str],
    selected: list[tuple[str, str, str]],
    min_n: int,
    threshold: float,
) -> list[tuple[str, str, str, float, int]]:
    ranked: list[tuple[str, str, str, float, int]] = []
    used = {x[0] for x in selected} | {trigger[0]}
    for rule in RULES:
        if rule in used:
            continue
        for variant in ("N", "I"):
            for mode in ("SAME", "OPPOSITE"):
                cand = selected + [(rule, variant, mode)]
                correct, n = chain_accuracy(train, trigger, cand)
                acc = pct(correct, n)
                if n >= min_n and acc >= threshold:
                    ranked.append((rule, variant, mode, acc, n))
    ranked.sort(key=lambda x: (x[3], x[4]), reverse=True)
    return ranked


def select_strategy(
    train: list[list[Candle]], threshold: float, min_n: int, max_levels: int
) -> tuple[tuple[str, str], list[tuple[str, str, str]], float, int] | None:
    best = None
    for rule in RULES:
        for variant in ("N", "I"):
            trigger = (rule, variant)
            correct, n = chain_accuracy(train, trigger, [])
            acc = pct(correct, n)
            if n < min_n or acc < threshold:
                continue
            candidate = (trigger, [], acc, n)
            if best is None or (acc, n) > (best[2], best[3]):
                best = candidate

            selected: list[tuple[str, str, str]] = []
            for _ in range(max_levels):
                ranked = rank_confirmations(train, trigger, selected, min_n, threshold)
                if not ranked:
                    break
                selected.append(ranked[0][:3])
                c, nn = chain_accuracy(train, trigger, selected)
                a = pct(c, nn)
                if nn >= min_n and a >= threshold and (best is None or (a, nn) > (best[2], best[3])):
                    best = (trigger, selected.copy(), a, nn)
    return best


@dataclass
class Equity:
    capital: float
    peak: float
    max_dd: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_sum: float = 0.0
    fee_sum: float = 0.0

    def apply(self, direction: int, entry: float, exit_price: float, fee_rate: float) -> None:
        if entry <= 0 or exit_price <= 0 or self.capital <= 0:
            return
        move = (exit_price / entry - 1.0) * direction
        gross_pnl = self.capital * move
        fee = self.capital * fee_rate * 2.0
        self.capital = max(self.capital + gross_pnl - fee, 0.0)
        self.trades += 1
        self.wins += int(gross_pnl > 0)
        self.losses += int(gross_pnl <= 0)
        self.gross_sum += move
        self.fee_sum += fee
        self.peak = max(self.peak, self.capital)
        if self.peak > 0:
            self.max_dd = max(self.max_dd, (self.peak - self.capital) / self.peak)


def evaluate_walkforward(
    series: dict[str, list[Candle]],
    capital: float,
    fee_rate: float,
    train_bars: int,
    retrain_every: int,
    threshold: float,
    min_n: int,
    max_levels: int,
) -> dict:
    eq = Equity(capital, capital)
    items = list(series.items())
    if not items:
        return {"initial": capital, "final": capital, "return_pct": 0.0, "trades": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0, "max_dd_pct": 0.0, "gross_sum_pct": 0.0, "fees_paid": 0.0, "qualified_retrains": 0, "no_trade_retrains": 0}

    total_bars = max(len(cs) for _, cs in items)
    next_retrain = train_bars
    strategy = None
    qualified = no_trade = 0

    for step in range(train_bars, total_bars - 1):
        if strategy is None or step >= next_retrain:
            train_sets: list[list[Candle]] = []
            for _, cs in items:
                end = min(step, len(cs))
                start = max(0, end - train_bars)
                if end - start >= min_n:
                    train_sets.append(cs[start:end])
            strategy = select_strategy(train_sets, threshold, min_n, max_levels) if train_sets else None
            if strategy is None:
                no_trade += 1
            else:
                qualified += 1
            next_retrain = step + retrain_every

        if strategy is None:
            continue
        trigger, selected, train_acc, train_n = strategy
        for _, cs in items:
            if step >= len(cs) - 1:
                continue
            ok, direction = chain_pass(cs, step, trigger, selected)
            if not ok:
                continue
            eq.apply(direction, cs[step].close, cs[step + 1].close, fee_rate)

    return {
        "initial": capital,
        "final": eq.capital,
        "return_pct": 100.0 * (eq.capital / capital - 1.0),
        "trades": eq.trades,
        "wins": eq.wins,
        "losses": eq.losses,
        "win_rate_pct": pct(eq.wins, eq.trades),
        "max_dd_pct": 100.0 * eq.max_dd,
        "gross_sum_pct": 100.0 * eq.gross_sum,
        "fees_paid": eq.fee_sum,
        "qualified_retrains": qualified,
        "no_trade_retrains": no_trade,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Leakage-safe 55%%+ R confirmation walk-forward capital backtest")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--train-month", default="2026-06")
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--timeframes", default="30,60,120,240")
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--threshold", type=float, default=55.0)
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--max-levels", type=int, default=2)
    ap.add_argument("--train-bars", type=int, default=500)
    ap.add_argument("--retrain-every", type=int, default=50)
    ap.add_argument("--output", default="reports/july_r_confirmation_capital_walkforward_v7.csv")
    args = ap.parse_args()

    root = Path(args.input_dir)
    tfs = [int(x.strip()) for x in args.timeframes.split(",") if x.strip()]
    if args.symbols.upper() == "ALL":
        symbols = sorted(set(discover_symbols(root, args.train_month)) | set(discover_symbols(root, args.test_month)))
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]

    rows: list[dict] = []
    print(f"R_CONFIRM_CAPITAL_WF_V7 train={args.train_month} test={args.test_month} threshold={args.threshold:.1f}% min_n={args.min_samples} capital={args.initial_capital}")
    print("No qualified >= threshold strategy => NO TRADE. Walk-forward uses only candles strictly before each retraining point.")

    for tf in tfs:
        test_series = load_series(root, symbols, args.test_month, tf)
        if not test_series:
            continue
        print(f"\nTF={tf}m")
        for fee_pct in (0.0, 1.3):
            res = evaluate_walkforward(
                test_series,
                args.initial_capital,
                fee_pct / 100.0,
                args.train_bars,
                args.retrain_every,
                args.threshold,
                args.min_samples,
                args.max_levels,
            )
            print(
                f"FEE={fee_pct:.1f}%/side FINAL={res['final']:.2f} RETURN={res['return_pct']:.2f}% "
                f"TRADES={res['trades']} WIN={res['win_rate_pct']:.2f}% DD={res['max_dd_pct']:.2f}% "
                f"QUAL={res['qualified_retrains']} NO_TRADE_RETRAIN={res['no_trade_retrains']} FEES={res['fees_paid']:.2f}"
            )
            rows.append({"timeframe_min": tf, "fee_per_side_pct": fee_pct, **res})

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys()) if rows else ["timeframe_min"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"SAVED {p} rows={len(rows)}")


if __name__ == "__main__":
    main()
