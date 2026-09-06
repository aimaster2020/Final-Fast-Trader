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
    components,
    TESTS,
    discover_symbols,
    find_month_file,
    load_binance,
    resample,
)

RULES = [f"R{i}" for i in range(1, 17)]
VARIANTS = ["N", "I"]


def raw_sig(c: Candle, rule: str) -> int:
    v = components(c)[rule]
    th = TESTS[rule]
    return 1 if v >= th else -1 if v <= -th else 0


def run_rule(
    series: dict[str, list[Candle]],
    rule: str,
    variant: str,
    fee_side_pct: float,
    global_trade_id_start: int,
) -> tuple[dict, list[dict]]:
    initial = 1000.0
    per_symbol = initial / len(series)
    total = 0.0
    trades = wins = 0
    fees_total = 0.0
    max_dd = 0.0
    stale_discarded = 0
    signal_count = 0
    rows: list[dict] = []
    global_trade_id = global_trade_id_start

    for symbol, raw in series.items():
        capital = per_symbol
        peak = capital
        c15 = resample(raw, 15)
        c60 = resample(raw, 60)

        signals = []
        for i, c in enumerate(c15):
            raw_signal = raw_sig(c, rule)
            signal = -raw_signal if variant == "I" else raw_signal
            if signal:
                signal_end = c.timestamp + 15 * 60
                signals.append((signal_end, signal, i + 1, c.timestamp, raw_signal))
        signal_count += len(signals)

        signal_ptr = 0
        j = 0
        while j + 1 < len(c60):
            ts = c60[j].timestamp

            while signal_ptr < len(signals) and signals[signal_ptr][0] < ts:
                stale_discarded += 1
                signal_ptr += 1

            if signal_ptr >= len(signals):
                j += 1
                continue

            signal_end, direction, signal_id, signal_ts, trigger_raw = signals[signal_ptr]
            if ts < signal_end:
                j += 1
                continue

            entry = c60[j].close
            exit_price = c60[j + 1].close
            move = (exit_price / entry - 1.0) * direction if entry > 0 else 0.0
            gross = capital * move
            fee = capital * 2.0 * fee_side_pct / 100.0
            before = capital
            after = max(0.0, capital + gross - fee)
            capital = after
            fees_total += fee
            trades += 1
            is_win = gross > 0
            wins += int(is_win)
            peak = max(peak, capital)
            dd = 100.0 * (peak - capital) / peak if peak else 0.0
            max_dd = max(max_dd, dd)
            global_trade_id += 1

            rows.append({
                "trade_id": global_trade_id,
                "rule": rule,
                "variant": variant,
                "symbol": symbol,
                "signal_id": signal_id,
                "signal_ts": signal_ts,
                "signal_end_ts": signal_end,
                "signal_raw": trigger_raw,
                "direction": direction,
                "action": "BUY" if direction > 0 else "SELL",
                "entry_ts": ts,
                "exit_ts": c60[j + 1].timestamp,
                "entry_price": entry,
                "exit_price": exit_price,
                "move_pct": move * 100.0,
                "capital_before": before,
                "gross_pnl": gross,
                "fee": fee,
                "capital_after": after,
                "win": int(is_win),
                "drawdown_pct": dd,
            })

            signal_ptr += 1
            j += 2

        total += capital

    result = {
        "rule": rule,
        "variant": variant,
        "final": total,
        "ret": 100.0 * (total / initial - 1.0),
        "trades": trades,
        "win": 100.0 * wins / trades if trades else 0.0,
        "dd": max_dd,
        "fees": fees_total,
        "signals": signal_count,
        "stale_discarded": stale_discarded,
        "buy": sum(r["action"] == "BUY" for r in rows),
        "sell": sum(r["action"] == "SELL" for r in rows),
    }
    return result, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--fee-side-pct", type=float, default=0.0)
    ap.add_argument("--show-trades", type=int, default=10)
    args = ap.parse_args()

    data_root = Path(args.input_dir)
    symbols = (
        sorted(discover_symbols(data_root, args.test_month))
        if args.symbols.upper() == "ALL"
        else [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    )
    series = {}
    for symbol in symbols:
        p = find_month_file(data_root, symbol, args.test_month)
        if p:
            cs = load_binance(p)
            if len(cs) >= 64:
                series[symbol] = cs
    if not series:
        raise RuntimeError("No test data found")

    print(
        f"R1_R16_SINGLE_NONOVERLAP_DETAIL architecture=B assets={len(series)} "
        f"test={args.test_month} fee={args.fee_side_pct:.3f}%/side"
    )
    print("15m: single R signal at candle close -> first eligible 60m entry -> next 60m exit")
    print("Control: one trade per symbol at a time; stale confirmed signals discarded")
    print("RULE VAR FINAL RET T WIN DD SIGNALS STALE BUY SELL")
    print("---- --- ----- ------ - ---- ---- ------- ----- --- ----")

    results = []
    all_trades = []
    next_trade_id = 0
    for rule in RULES:
        for variant in VARIANTS:
            result, rows = run_rule(series, rule, variant, args.fee_side_pct, next_trade_id)
            next_trade_id += len(rows)
            results.append(result)
            all_trades.extend(rows)
            print(
                f"{rule:>3} {variant:>3} {result['final']:>7.2f} {result['ret']:>+6.2f}% "
                f"{result['trades']:>4} {result['win']:>5.2f}% {result['dd']:>5.2f}% "
                f"{result['signals']:>7} {result['stale_discarded']:>5} "
                f"{result['buy']:>3} {result['sell']:>4}"
            )

    report_dir = ROOT / "reports" / "july_r1_r16_single_nonoverlap_detail"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "r1_r16_single_nonoverlap_detail_summary.csv"
    trades_path = report_dir / "r1_r16_single_nonoverlap_all_trades.csv"
    write_csv(summary_path, results)
    write_csv(trades_path, all_trades)

    ranked = sorted(results, key=lambda x: x["ret"], reverse=True)
    print("--- TOP 10 BY RETURN ---")
    for r in ranked[:10]:
        print(
            f"{r['rule']:>3} {r['variant']:>3} RET={r['ret']:+.2f}% FINAL={r['final']:.2f} "
            f"T={r['trades']} WIN={r['win']:.2f}% DD={r['dd']:.2f}% "
            f"BUY={r['buy']} SELL={r['sell']}"
        )

    print("--- FIRST TRADES BY TOP 10 ---")
    for r in ranked[:10]:
        sample = [
            x for x in all_trades
            if x["rule"] == r["rule"] and x["variant"] == r["variant"]
        ][: args.show_trades]
        print(f"[{r['rule']}{r['variant']}] showing {len(sample)} trades")
        for x in sample:
            print(
                f"  #{x['trade_id']} {x['symbol']} {x['action']} "
                f"entry={x['entry_price']:.8f} exit={x['exit_price']:.8f} "
                f"move={x['move_pct']:+.4f}% cap={x['capital_after']:.4f} "
                f"win={x['win']} signal_id={x['signal_id']}"
            )

    print(f"SAVED {summary_path.relative_to(ROOT)}")
    print(f"SAVED {trades_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
