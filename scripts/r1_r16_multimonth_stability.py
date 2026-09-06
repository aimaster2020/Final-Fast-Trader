from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.july_r1_r16_1h_horizon4_consensus import (
    MODES,
    RULES,
    backtest_symbol,
    evaluate_accuracy,
    load_requested_month,
    resample,
)

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
DEFAULT_FEE_SIDE_PCT = 0.13
DEFAULT_MONTHS = "2026-05,2026-06,2026-07,2026-08"


def test_month(
    input_dir: Path,
    month: str,
    symbols: list[str],
    fee_side_pct: float,
    initial_capital: float,
) -> tuple[list[dict], dict[str, list[str]]]:
    series = {}
    loaded_files: dict[str, list[str]] = {}
    for symbol in symbols:
        raw, files = load_requested_month(input_dir, symbol, month)
        if len(raw) >= 10:
            series[symbol] = resample(raw, 60)
            loaded_files[symbol] = files

    if not series:
        raise RuntimeError(f"No usable data found for {month}.")

    rows: list[dict] = []
    for rule in RULES:
        for inverted in (False, True):
            variant = "I" if inverted else "N"
            for mode in MODES:
                total_final = 0.0
                acc_num = acc_den = trades = wins = 0
                fees = dd = 0.0
                per_symbol = initial_capital / len(series)
                for candles in series.values():
                    a = evaluate_accuracy(candles, rule, inverted, mode)
                    r = backtest_symbol(candles, rule, inverted, mode, fee_side_pct)
                    acc_num += a["correct"]
                    acc_den += a["signals"]
                    total_final += r["final"]
                    trades += r["trades"]
                    wins += r["wins"]
                    fees += r["fees"]
                    dd = max(dd, r["max_dd_pct"])
                if abs(initial_capital - 1000.0) > 1e-12:
                    scale = per_symbol / 250.0
                    total_final *= scale
                    fees *= scale
                ret = 100.0 * (total_final / initial_capital - 1.0)
                acc = 100.0 * acc_num / acc_den if acc_den else 0.0
                win = 100.0 * wins / trades if trades else 0.0
                rows.append({
                    "rule": rule,
                    "variant": variant,
                    "mode": mode,
                    "month": month,
                    "final_capital": total_final,
                    "return_pct": ret,
                    "accuracy_pct": acc,
                    "trades": trades,
                    "win_rate_pct": win,
                    "max_dd_pct": dd,
                    "fees_paid": fees,
                })
    return rows, loaded_files


def summarize(monthly_rows: list[dict], months: list[str]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for r in monthly_rows:
        grouped.setdefault((r["rule"], r["variant"], r["mode"]), []).append(r)

    out: list[dict] = []
    n_months = len(months)
    for key, rs in sorted(grouped.items(), key=lambda kv: (int(kv[0][0][1:]), kv[0][1], MODES.index(kv[0][2]))):
        by_month = {r["month"]: r for r in rs}
        returns = [float(by_month[m]["return_pct"]) for m in months if m in by_month]
        accuracies = [float(by_month[m]["accuracy_pct"]) for m in months if m in by_month]
        trades = [int(by_month[m]["trades"]) for m in months if m in by_month]
        dds = [float(by_month[m]["max_dd_pct"]) for m in months if m in by_month]
        positive = sum(x > 0 for x in returns)
        nonnegative = sum(x >= 0 for x in returns)
        mean_ret = sum(returns) / len(returns) if returns else 0.0
        worst = min(returns) if returns else 0.0
        best = max(returns) if returns else 0.0
        volatility = (sum((x - mean_ret) ** 2 for x in returns) / len(returns)) ** 0.5 if returns else 0.0

        # Stability score prioritizes repeated profitability and then penalizes instability.
        # This is a ranking aid, not an optimization target.
        coverage = len(returns) / n_months if n_months else 0.0
        positive_rate = positive / len(returns) if returns else 0.0
        consistency = max(0.0, 1.0 - volatility / 20.0)
        drawdown_quality = max(0.0, 1.0 - (sum(dds) / len(dds)) / 30.0) if dds else 0.0
        stability_score = 100.0 * (
            0.45 * positive_rate
            + 0.20 * coverage
            + 0.20 * consistency
            + 0.15 * drawdown_quality
        )

        out.append({
            "rule": key[0],
            "variant": key[1],
            "mode": key[2],
            "months_tested": len(returns),
            "positive_months": positive,
            "nonnegative_months": nonnegative,
            "positive_rate_pct": 100.0 * positive_rate,
            "mean_return_pct": mean_ret,
            "best_month_pct": best,
            "worst_month_pct": worst,
            "return_std_pct": volatility,
            "mean_accuracy_pct": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "mean_trades": sum(trades) / len(trades) if trades else 0.0,
            "mean_dd_pct": sum(dds) / len(dds) if dds else 0.0,
            "stability_score": stability_score,
            "monthly_returns": ",".join(f"{by_month[m]['return_pct']:+.2f}" if m in by_month else "NA" for m in months),
        })
    return sorted(out, key=lambda r: (-float(r["stability_score"]), -float(r["mean_return_pct"]), -float(r["positive_rate_pct"]), float(r["mean_dd_pct"])))


def main() -> None:
    ap = argparse.ArgumentParser(description="R1-R16 multi-month stability ranking at fixed fee.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--months", default=DEFAULT_MONTHS, help="Comma-separated YYYY-MM months, e.g. 2026-05,2026-06,2026-07,2026-08")
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee-side-pct", type=float, default=DEFAULT_FEE_SIDE_PCT)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    months = [x.strip() for x in args.months.split(",") if x.strip()]
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]

    monthly_rows: list[dict] = []
    print(f"R1_R16_MULTIMONTH_STABILITY months={','.join(months)} fee={args.fee_side_pct:.3f}%/side assets={','.join(symbols)}")

    successful_months: list[str] = []
    for month in months:
        try:
            rows, loaded = test_month(input_dir, month, symbols, args.fee_side_pct, args.initial_capital)
        except RuntimeError as exc:
            print(f"SKIP {month}: {exc}")
            continue
        monthly_rows.extend(rows)
        successful_months.append(month)
        print(f"MONTH {month} OK files=" + " | ".join(f"{s}:{len(loaded.get(s, []))}" for s in symbols))

    if not monthly_rows:
        raise RuntimeError("No monthly data could be tested.")

    summary = summarize(monthly_rows, successful_months)
    print(f"TESTED_MONTHS {','.join(successful_months)}")
    print("RANK RULE MODE POS/ALL MEAN% BEST% WORST% ACC% TRADES DD% SCORE RETURNS")
    print("---- ---- ---- ------- ------ ------ ------ ----- ------ ----- ----- -------")
    for r in summary[:20]:
        print(
            f"{r['rule']:>4} {r['mode']:>4} {r['variant']:>3} "
            f"{r['positive_months']}/{r['months_tested']:<3} "
            f"{float(r['mean_return_pct']):>+6.2f} "
            f"{float(r['best_month_pct']):>+6.2f} "
            f"{float(r['worst_month_pct']):>+6.2f} "
            f"{float(r['mean_accuracy_pct']):>6.2f} "
            f"{float(r['mean_trades']):>6.1f} "
            f"{float(r['mean_dd_pct']):>6.2f} "
            f"{float(r['stability_score']):>6.2f} "
            f"{r['monthly_returns']}"
        )

    out = Path(args.output) if args.output else ROOT / "reports" / f"r1_r16_multimonth_stability_{'-'.join(successful_months)}.csv"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        fields = list(summary[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(summary)

    raw_out = out.with_name(out.stem + "_monthly.csv")
    with raw_out.open("w", newline="", encoding="utf-8") as f:
        fields = list(monthly_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(monthly_rows)

    print(f"SAVED {out.relative_to(ROOT)}")
    print(f"SAVED {raw_out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
