from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def movement_budget(rows: list[dict], i: int, bars: int) -> float:
    if i < bars:
        return float("inf")
    return sum(abs(float(rows[j]["close"]) - float(rows[j - 1]["close"])) for j in range(i - bars, i))


def signal(rows: list[dict], i: int, window: int, move_bars: int, multiplier: float) -> str:
    if i < max(window, move_bars):
        return ""
    previous = rows[i - window:i]
    prev_low = min(float(r["low"]) for r in previous)
    prev_high = max(float(r["high"]) for r in previous)
    budget = movement_budget(rows, i, move_bars) * multiplier
    down = prev_low - float(rows[i]["low"])
    up = float(rows[i]["high"]) - prev_high
    low = float(rows[i]["low"]) < prev_low and down >= budget
    high = float(rows[i]["high"]) > prev_high and up >= budget
    if low and high:
        return ""
    return "LOW" if low else "HIGH" if high else ""


def run(rows: list[dict], window: int, move_bars: int, multiplier: float, horizon: int, allocation: float, fee_rt: float):
    capital = 1000.0
    peak = capital
    max_dd = 0.0
    trades = []
    active_until = -1
    start = max(window, move_bars)
    for i in range(start, len(rows) - horizon - 1):
        if i <= active_until:
            continue
        sig = signal(rows, i, window, move_bars, multiplier)
        if not sig:
            continue
        entry_i = i + 1
        exit_i = i + 1 + horizon
        entry = float(rows[entry_i]["open"])
        exit_price = float(rows[exit_i]["close"])
        side = 1 if sig == "LOW" else -1
        gross_ret = ((exit_price / entry) - 1.0) * side
        notional = capital * allocation
        fee = notional * fee_rt / 100.0
        pnl = notional * gross_ret - fee
        capital += pnl
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak * 100.0)
        trades.append((rows[i]["month"], sig, pnl, fee))
        active_until = exit_i
    wins = sum(p > 0 for _, _, p, _ in trades)
    gross_profit = sum(p for _, _, p, _ in trades if p > 0)
    gross_loss = -sum(p for _, _, p, _ in trades if p < 0)
    pf = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0)
    return capital, max_dd, len(trades), wins / len(trades) * 100 if trades else 0.0, pf, sum(f for _, _, _, f in trades), trades


def main() -> None:
    ap = argparse.ArgumentParser(description="Monthly/symbol validation of the extreme move strategy using the prepared 1H dataset.")
    ap.add_argument("--prepared-file", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--months", default="2026-05,2026-06,2026-07,2026-08")
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--move-bars", type=int, default=5)
    ap.add_argument("--multiplier", type=float, default=2.0)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--allocation", type=float, default=0.30)
    ap.add_argument("--fee-roundtrip-pct", type=float, default=0.26)
    ap.add_argument("--output", default="reports/extreme_move_monthly_validation.csv")
    args = ap.parse_args()

    path = Path(args.prepared_file)
    if not path.is_absolute():
        path = ROOT / path
    symbols = {x.strip().upper() for x in args.symbols.split(",") if x.strip()}
    months = {x.strip() for x in args.months.split(",") if x.strip()}
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["symbol"].upper() in symbols and r["month"] in months:
                grouped[r["symbol"].upper()].append(r)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["timestamp"]))

    print(f"EXTREME_MOVE_MONTHLY_VALIDATION W={args.window} M={args.move_bars} X={args.multiplier:.2f} H={args.horizon} alloc={args.allocation:.0%} fee={args.fee_roundtrip_pct:.2f}%")
    print("SYMBOL MONTH       T    WIN       PNL     FINAL     PF     DD    FEES")
    print("-----------------------------------------------------------------------")
    combined = []
    for symbol in sorted(grouped):
        by_month: dict[str, list[dict]] = defaultdict(list)
        for r in grouped[symbol]:
            by_month[r["month"]].append(r)
        symbol_capital = 1000.0
        for month in sorted(by_month):
            final, dd, n, win, pf, fees, trades = run(by_month[month], args.window, args.move_bars, args.multiplier, args.horizon, args.allocation, args.fee_roundtrip_pct)
            pnl = final - 1000.0
            # Carry monthly P&L onto the symbol's running capital for a separate compact view.
            symbol_capital += pnl
            combined.extend((symbol, month, *x) for x in trades)
            print(f"{symbol:7} {month:10} {n:4d} {win:6.1f}% {pnl:+9.2f} {symbol_capital:9.2f} {pf:6.2f} {dd:6.2f}% {fees:7.2f}")

    print("DONE")
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "month", "signal", "pnl", "fee"])
        for symbol, month, sig, pnl, fee in combined:
            w.writerow([symbol, month, sig, pnl, fee])
    print(f"SAVED {out.relative_to(ROOT)} rows={len(combined)}")


if __name__ == "__main__":
    main()
