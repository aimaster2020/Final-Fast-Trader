from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_pattern_trader.models import Candle
from scripts.july_r_composite_walkforward import components, discover_symbols, find_month_file, load_binance, resample
from scripts.ohlc_rule_isolation_monthly import TESTS

RULES = [f"R{i}" for i in range(1, 17)]
MODES = ["CURRENT", "2OF4", "3OF4", "4OF4"]
DEFAULT_FEE_SIDE_PCT = 0.05
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"


def find_requested_month_file(input_dir: Path, symbol: str, month: str) -> Path | None:
    """Find a monthly CSV robustly regardless of the exact filename separator/case."""
    exact = find_month_file(input_dir, symbol, month)
    if exact is not None:
        return exact

    symbol_u = symbol.upper()
    month_u = month.upper()
    candidates: list[Path] = []
    for p in input_dir.rglob("*.csv"):
        name_u = p.name.upper()
        if symbol_u in name_u and month_u in name_u:
            candidates.append(p)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: str(p).upper())[0]


def signal(c: Candle, rule: str, inverted: bool) -> int:
    value = components(c)[rule]
    threshold = TESTS[rule]
    s = 1 if value >= threshold else -1 if value <= -threshold else 0
    return -s if inverted else s


def frame_prediction(candles: list[Candle], i: int, rule: str, inverted: bool, mode: str) -> int:
    if mode == "CURRENT":
        return signal(candles[i], rule, inverted)

    # Four latest completed 1h candles: current candle + previous 3.
    votes = [signal(candles[j], rule, inverted) for j in range(i - 3, i + 1)]
    votes = [x for x in votes if x]
    if not votes:
        return 0
    up = sum(v == 1 for v in votes)
    down = sum(v == -1 for v in votes)
    if mode == "2OF4":
        return 1 if up > down else -1 if down > up else 0
    if mode == "3OF4":
        return 1 if up >= 3 else -1 if down >= 3 else 0
    if mode == "4OF4":
        return 1 if up == 4 else -1 if down == 4 else 0
    raise ValueError(mode)


def evaluate_accuracy(candles: list[Candle], rule: str, inverted: bool, mode: str, horizon: int = 4) -> dict:
    correct = signals = 0
    start_i = 0 if mode == "CURRENT" else 3
    last = len(candles) - horizon
    for i in range(start_i, max(start_i, last)):
        pred = frame_prediction(candles, i, rule, inverted, mode)
        if not pred:
            continue
        entry = candles[i].close
        exit_price = candles[i + horizon].close
        if entry <= 0 or exit_price <= 0:
            continue
        realized = 1 if exit_price > entry else -1 if exit_price < entry else 0
        if realized == 0:
            continue
        signals += 1
        correct += int(pred == realized)
    return {"accuracy_pct": 100.0 * correct / signals if signals else 0.0, "signals": signals, "correct": correct}


def backtest_symbol(candles: list[Candle], rule: str, inverted: bool, mode: str, fee_side_pct: float, horizon: int = 4) -> dict:
    capital = 250.0
    position = 0
    entry_price = 0.0
    exit_i = -1
    trades = wins = 0
    fees = 0.0
    peak = capital
    max_dd = 0.0
    start_i = 3 if mode != "CURRENT" else 0

    for i in range(start_i, len(candles) - horizon):
        if position:
            if i < exit_i:
                continue
            px = candles[exit_i].close
            gross = capital * ((px / entry_price) - 1.0) * position
            open_fee = capital * fee_side_pct / 100.0
            capital = max(0.0, capital + gross - open_fee)
            fees += open_fee
            close_fee = capital * fee_side_pct / 100.0
            capital = max(0.0, capital - close_fee)
            fees += close_fee
            trades += 1
            wins += int(gross > 0)
            position = 0
            entry_price = 0.0
            exit_i = -1
            peak = max(peak, capital)
            max_dd = max(max_dd, 100.0 * (peak - capital) / peak if peak else 0.0)
            if capital <= 0:
                break

        pred = frame_prediction(candles, i, rule, inverted, mode)
        if not pred:
            continue
        entry_price = candles[i].close
        if entry_price <= 0:
            continue
        open_fee = capital * fee_side_pct / 100.0
        capital = max(0.0, capital - open_fee)
        fees += open_fee
        position = pred
        exit_i = i + horizon
        if capital <= 0:
            break

    if position and 0 <= exit_i < len(candles):
        px = candles[exit_i].close
        gross = capital * ((px / entry_price) - 1.0) * position
        close_fee = capital * fee_side_pct / 100.0
        capital = max(0.0, capital + gross - close_fee)
        fees += close_fee
        trades += 1
        wins += int(gross > 0)
        peak = max(peak, capital)
        max_dd = max(max_dd, 100.0 * (peak - capital) / peak if peak else 0.0)

    return {
        "final": capital,
        "return_pct": 100.0 * (capital / 250.0 - 1.0),
        "trades": trades,
        "wins": wins,
        "win_rate_pct": 100.0 * wins / trades if trades else 0.0,
        "fees": fees,
        "max_dd_pct": max_dd,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="R1-R16/inverses: 1h signal, 4h horizon, consensus strength comparison.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee-side-pct", type=float, default=DEFAULT_FEE_SIDE_PCT)
    ap.add_argument("--output", default=None, help="CSV output path; defaults to reports/r1_r16_1h_horizon4_consensus_<month>.csv")
    args = ap.parse_args()

    data_root = Path(args.input_dir)
    if args.symbols.upper() == "ALL":
        symbols = discover_symbols(data_root, args.test_month)
        if not symbols:
            symbols = [x.strip().upper() for x in DEFAULT_SYMBOLS.split(",") if x.strip()]
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]

    series: dict[str, list[Candle]] = {}
    missing: list[str] = []
    loaded_files: dict[str, str] = {}
    for symbol in symbols:
        p = find_requested_month_file(data_root, symbol, args.test_month)
        if not p:
            missing.append(symbol)
            continue
        raw = load_binance(p)
        if len(raw) >= 10:
            series[symbol] = resample(raw, 60)
            loaded_files[symbol] = p.name

    if not series:
        raise RuntimeError(
            f"No 1m data found for test month {args.test_month} and symbols {','.join(symbols)}. "
            "Check that the monthly CSV files exist under --input-dir."
        )

    print(
        f"R1_R16_1H_H4_CONSENSUS test={args.test_month} assets={len(series)} "
        f"initial={args.initial_capital:.2f} fee={args.fee_side_pct:.3f}%/side"
    )
    print("FILES " + " | ".join(f"{s}:{loaded_files[s]}" for s in sorted(loaded_files)))
    if missing:
        print("MISSING " + ",".join(missing))
    print("HORIZON=4h | SIGNAL=1h | CURRENT=1 candle | 2OF4=majority | 3OF4=at least 3 agree | 4OF4=all 4 agree")
    print("4-frame window=current + previous 3 completed 1h candles | NO indicators | NO ML | frozen R1-R16 | non-overlap, one position/symbol")
    print("RULE MODE VAR FINAL RET% ACC% TRADES WIN% DD% FEES")
    print("---- ---- --- ----- ----- ----- ------ ----- ----- -----")

    rows = []
    for rule in RULES:
        for inverted in (False, True):
            variant = "I" if inverted else "N"
            for mode in MODES:
                total_final = 0.0
                acc_num = acc_den = trades = wins = 0
                fees = dd = 0.0
                per_symbol = args.initial_capital / len(series)
                for candles in series.values():
                    a = evaluate_accuracy(candles, rule, inverted, mode)
                    r = backtest_symbol(candles, rule, inverted, mode, args.fee_side_pct)
                    acc_num += a["correct"]
                    acc_den += a["signals"]
                    total_final += r["final"]
                    trades += r["trades"]
                    wins += r["wins"]
                    fees += r["fees"]
                    dd = max(dd, r["max_dd_pct"])
                if abs(args.initial_capital - 1000.0) > 1e-12:
                    scale = per_symbol / 250.0
                    total_final *= scale
                    fees *= scale
                ret = 100.0 * (total_final / args.initial_capital - 1.0)
                acc = 100.0 * acc_num / acc_den if acc_den else 0.0
                win = 100.0 * wins / trades if trades else 0.0
                print(f"{rule:>4} {mode:>4} {variant:>3} {total_final:>7.2f} {ret:>+6.2f} {acc:>5.2f} {trades:>6} {win:>6.2f} {dd:>5.2f} {fees:>7.2f}")
                rows.append({
                    "rule": rule,
                    "mode": mode,
                    "variant": variant,
                    "initial_capital": args.initial_capital,
                    "final_capital": total_final,
                    "return_pct": ret,
                    "directional_accuracy_pct": acc,
                    "trades": trades,
                    "win_rate_pct": win,
                    "max_dd_pct": dd,
                    "fees_paid": fees,
                    "test_month": args.test_month,
                    "fee_side_pct": args.fee_side_pct,
                })

    if args.output:
        out = Path(args.output)
    else:
        out = ROOT / "reports" / f"r1_r16_1h_horizon4_consensus_{args.test_month}.csv"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"SAVED {out.relative_to(ROOT)} rows={len(rows)}")


if __name__ == "__main__":
    main()
