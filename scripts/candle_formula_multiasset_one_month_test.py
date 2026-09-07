from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal

TIMEFRAMES = ("5m", "15m", "1h", "4h")
DEFAULT_MONTH = "2026-07"
DEFAULT_COMMISSION = 0.0013
DEFAULT_VALUES = tuple(float(x) for x in range(-200, 201, 25))
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def to_candle(row: dict[str, str]) -> Candle | None:
    try:
        return Candle(
            int(float(row["timestamp"])),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_month(rows: list[dict[str, str]], month: str, symbol: str) -> list[Candle]:
    out: list[Candle] = []
    for row in rows:
        if row.get("symbol") != symbol or row.get("month") != month:
            continue
        candle = to_candle(row)
        if candle is not None:
            out.append(candle)
    return sorted(out, key=lambda c: c.timestamp)


def month_from_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")


def build_4h(rows: list[dict[str, str]], month: str, symbol: str) -> list[Candle]:
    hourly = []
    for row in rows:
        if row.get("symbol") != symbol:
            continue
        candle = to_candle(row)
        if candle is not None:
            hourly.append(candle)
    hourly.sort(key=lambda c: c.timestamp)

    buckets: dict[int, list[Candle]] = {}
    for candle in hourly:
        bucket = candle.timestamp - (candle.timestamp % 14_400)
        buckets.setdefault(bucket, []).append(candle)

    out: list[Candle] = []
    for bucket_ts, group in sorted(buckets.items()):
        group.sort(key=lambda c: c.timestamp)
        expected = [bucket_ts + 3_600 * i for i in range(4)]
        if len(group) != 4 or [c.timestamp for c in group] != expected:
            continue
        if month_from_ts(bucket_ts) != month:
            continue
        out.append(
            Candle(
                bucket_ts,
                group[0].open,
                max(c.high for c in group),
                min(c.low for c in group),
                group[-1].close,
            )
        )
    return out


def signal_side(candle: Candle) -> int:
    signal = decide(candle).signal
    if signal == Signal.BUY:
        return 1
    if signal == Signal.SELL:
        return -1
    return 0


def pnl_pct(side: int, entry: float, exit_price: float) -> float:
    return side * (exit_price - entry) / entry if entry > 0 else 0.0


def run(candles: list[Candle], lower: float, upper: float, commission: float) -> dict[str, float | int]:
    initial = 1000.0
    equity = initial
    position: tuple[int, float, float] | None = None
    trades = wins = 0
    gross_profit = gross_loss = 0.0

    for candle in candles:
        sig = signal_side(candle)
        body = candle.close - candle.open

        if position is None:
            if sig != 0 and not (lower <= body <= upper):
                position = (sig, candle.close, equity)
            continue

        side, entry, capital = position
        opposite = sig != 0 and sig != side
        exit_now = (side > 0 and opposite and body <= upper) or (side < 0 and opposite and body >= lower)
        if not exit_now:
            continue

        gross = capital * pnl_pct(side, entry, candle.close)
        net = gross - capital * commission * 2.0
        equity += net
        trades += 1
        if net > 0:
            wins += 1
            gross_profit += max(gross, 0.0)
        elif net < 0:
            gross_loss += min(gross, 0.0)
        position = None

    if position is not None and candles:
        side, entry, capital = position
        gross = capital * pnl_pct(side, entry, candles[-1].close)
        net = gross - capital * commission * 2.0
        equity += net
        trades += 1
        if net > 0:
            wins += 1
            gross_profit += max(gross, 0.0)
        elif net < 0:
            gross_loss += min(gross, 0.0)

    return {
        "return_pct": (equity / initial - 1.0) * 100.0,
        "trades": trades,
        "win_rate": wins / trades * 100.0 if trades else 0.0,
        "pf": gross_profit / abs(gross_loss) if gross_loss < 0 else (float("inf") if gross_profit else 0.0),
    }


def parse_values(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(x.strip()) for x in raw.split(",") if x.strip()}))
    if len(values) < 2:
        raise SystemExit("Need at least two threshold values")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description="One-month multi-asset candle formula test with independent 4h bias.")
    ap.add_argument("--month", default=DEFAULT_MONTH)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    ap.add_argument("--values", default=",".join(str(v) for v in DEFAULT_VALUES))
    ap.add_argument("--input-5m", default="reports/prepared_price_action_5m.csv")
    ap.add_argument("--input-15m", default="reports/prepared_price_action_15m.csv")
    ap.add_argument("--input-1h", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    month = args.month
    symbols = tuple(x.strip() for x in args.symbols.split(",") if x.strip())
    values = parse_values(args.values)
    paths = {
        "5m": Path(args.input_5m),
        "15m": Path(args.input_15m),
        "1h": Path(args.input_1h),
    }
    source_rows = {tf: read_rows(path) for tf, path in paths.items()}

    data: dict[tuple[str, str], list[Candle]] = {}
    for symbol in symbols:
        for tf in ("5m", "15m", "1h"):
            data[(symbol, tf)] = load_month(source_rows[tf], month, symbol)
        data[(symbol, "4h")] = build_4h(source_rows["1h"], month, symbol)

    fast_pairs = [(lo, hi) for lo in values for hi in values if lo < hi]

    print(
        f"MULTI_ASSET_1M | month={month} | commission={args.commission * 100:.2f}% per side | "
        f"roundtrip={args.commission * 200:.2f}% | 4h=RESAMPLED_FROM_1H"
    )

    overall: list[tuple] = []
    for symbol in symbols:
        missing = [tf for tf in TIMEFRAMES if not data[(symbol, tf)]]
        if missing:
            print(f"{symbol} | SKIP missing={','.join(missing)}")
            continue

        results = []
        for flo, fhi in fast_pairs:
            fast = {
                tf: run(data[(symbol, tf)], flo, fhi, args.commission)
                for tf in ("5m", "15m", "1h")
            }
            for hlo, hhi in fast_pairs:
                h4 = run(data[(symbol, "4h")], hlo, hhi, args.commission)
                combined = [
                    float(fast["5m"]["return_pct"]),
                    float(fast["15m"]["return_pct"]),
                    float(fast["1h"]["return_pct"]),
                    float(h4["return_pct"]),
                ]
                avg_tf = sum(combined) / 4.0
                worst_tf = min(combined)
                total_trades = sum(int(fast[tf]["trades"]) for tf in ("5m", "15m", "1h")) + int(h4["trades"])
                results.append((avg_tf, worst_tf, total_trades, flo, fhi, hlo, hhi, fast, h4))

        # First priority is positive average across all four TFs, then worst TF, then activity.
        results.sort(key=lambda x: (x[0] > 0, x[0], x[1], x[2]), reverse=True)
        best = results[: max(1, args.top)]

        print(f"\n===== {symbol} =====")
        print("FAST_BIAS | 4H_BIAS | AVG_4TF | WORST_TF | TRADES | 5m_RET/T-D | 15m_RET/T-D | 1h_RET/T-D | 4h_RET/T-D")
        for avg_tf, worst_tf, total_trades, flo, fhi, hlo, hhi, fast, h4 in best:
            days_5m = max(1, len(data[(symbol, "5m")]) / 288.0)
            days_15m = max(1, len(data[(symbol, "15m")]) / 96.0)
            days_1h = max(1, len(data[(symbol, "1h")]) / 24.0)
            days_4h = max(1, len(data[(symbol, "4h")]) / 6.0)
            print(
                f"[{flo:.0f},{fhi:.0f}] | [{hlo:.0f},{hhi:.0f}] | {avg_tf:+6.2f}% | {worst_tf:+7.2f}% | {total_trades:6d} | "
                f"{float(fast['5m']['return_pct']):+6.2f}%/{int(fast['5m']['trades'])/days_5m:.2f} | "
                f"{float(fast['15m']['return_pct']):+7.2f}%/{int(fast['15m']['trades'])/days_15m:.2f} | "
                f"{float(fast['1h']['return_pct']):+6.2f}%/{int(fast['1h']['trades'])/days_1h:.2f} | "
                f"{float(h4['return_pct']):+6.2f}%/{int(h4['trades'])/days_4h:.2f}"
            )
        top = best[0]
        overall.append((top[0], top[1], symbol, top))

    overall.sort(reverse=True, key=lambda x: (x[0] > 0, x[0], x[1]))
    print("\nASSET_RANK")
    for avg, worst, symbol, top in overall:
        print(f"{symbol} | avg_4tf={avg:+.2f}% | worst_tf={worst:+.2f}% | fast=[{top[3]:.0f},{top[4]:.0f}] | 4h=[{top[5]:.0f},{top[6]:.0f}] | trades={top[2]}")


if __name__ == "__main__":
    main()
