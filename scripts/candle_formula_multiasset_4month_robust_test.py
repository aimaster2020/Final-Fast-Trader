from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal

TIMEFRAMES = ("5m", "15m", "1h", "4h")
DEFAULT_MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
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
    hourly: list[Candle] = []
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


def compound(returns: list[float]) -> float:
    equity = 1.0
    for ret in returns:
        equity *= 1.0 + ret / 100.0
    return (equity - 1.0) * 100.0


def avg4(returns: dict[str, float]) -> float:
    return sum(returns.values()) / 4.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Four-month multi-asset robust candle formula test with fixed bias pairs.")
    ap.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    ap.add_argument("--values", default=",".join(str(v) for v in DEFAULT_VALUES))
    ap.add_argument("--input-5m", default="reports/prepared_price_action_5m.csv")
    ap.add_argument("--input-15m", default="reports/prepared_price_action_15m.csv")
    ap.add_argument("--input-1h", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    months = tuple(x.strip() for x in args.months.split(",") if x.strip())
    symbols = tuple(x.strip() for x in args.symbols.split(",") if x.strip())
    values = parse_values(args.values)
    fast_pairs = [(lo, hi) for lo in values for hi in values if lo < hi]

    paths = {
        "5m": Path(args.input_5m),
        "15m": Path(args.input_15m),
        "1h": Path(args.input_1h),
    }
    source_rows = {tf: read_rows(path) for tf, path in paths.items()}

    data: dict[tuple[str, str, str], list[Candle]] = {}
    for symbol in symbols:
        for month in months:
            for tf in ("5m", "15m", "1h"):
                data[(symbol, month, tf)] = load_month(source_rows[tf], month, symbol)
            data[(symbol, month, "4h")] = build_4h(source_rows["1h"], month, symbol)

    print(
        f"MULTI_ASSET_4M | months={','.join(months)} | symbols={','.join(symbols)} | "
        f"commission={args.commission * 100:.2f}% per side | roundtrip={args.commission * 200:.2f}% | "
        f"4h=RESAMPLED_FROM_1H"
    )

    for symbol in symbols:
        missing = [f"{month}:{tf}" for month in months for tf in TIMEFRAMES if not data[(symbol, month, tf)]]
        if missing:
            print(f"\n===== {symbol} =====\nSKIP missing={','.join(missing)}")
            continue

        results = []
        for flo, fhi in fast_pairs:
            for hlo, hhi in fast_pairs:
                month_tf: dict[str, dict[str, dict[str, float | int]]] = {}
                for month in months:
                    month_tf[month] = {
                        "5m": run(data[(symbol, month, "5m")], flo, fhi, args.commission),
                        "15m": run(data[(symbol, month, "15m")], flo, fhi, args.commission),
                        "1h": run(data[(symbol, month, "1h")], flo, fhi, args.commission),
                        "4h": run(data[(symbol, month, "4h")], hlo, hhi, args.commission),
                    }

                tf_compounded = {
                    tf: compound([float(month_tf[m][tf]["return_pct"]) for m in months])
                    for tf in TIMEFRAMES
                }
                tf_positive_months = {
                    tf: sum(1 for m in months if float(month_tf[m][tf]["return_pct"]) > 0)
                    for tf in TIMEFRAMES
                }
                worst_month = min(
                    float(month_tf[m][tf]["return_pct"])
                    for m in months for tf in TIMEFRAMES
                )
                positive_cells = sum(tf_positive_months.values())
                total_trades = sum(
                    int(month_tf[m][tf]["trades"])
                    for m in months for tf in TIMEFRAMES
                )
                avg_4tf = sum(tf_compounded.values()) / 4.0
                worst_tf = min(tf_compounded.values())
                results.append(
                    (
                        avg_4tf,
                        worst_tf,
                        positive_cells,
                        total_trades,
                        flo,
                        fhi,
                        hlo,
                        hhi,
                        tf_compounded,
                        tf_positive_months,
                        worst_month,
                    )
                )

        results.sort(key=lambda x: (x[0] > 0, x[2], x[1], x[0], x[3]), reverse=True)
        best = results[: max(1, args.top)]

        print(f"\n===== {symbol} =====")
        print("FAST_BIAS | 4H_BIAS | AVG_4TF_COMPOUND | WORST_TF | POS_CELLS | TRADES | 5m | 15m | 1h | 4h")
        for row in best:
            avg_4tf, worst_tf, positive_cells, total_trades, flo, fhi, hlo, hhi, comp, posm, worst_month = row
            print(
                f"[{flo:.0f},{fhi:.0f}] | [{hlo:.0f},{hhi:.0f}] | {avg_4tf:+8.2f}% | {worst_tf:+7.2f}% | "
                f"{positive_cells:2d}/{len(months)*4} | {total_trades:5d} | "
                f"{comp['5m']:+7.2f}%({posm['5m']}/{len(months)}) | "
                f"{comp['15m']:+7.2f}%({posm['15m']}/{len(months)}) | "
                f"{comp['1h']:+7.2f}%({posm['1h']}/{len(months)}) | "
                f"{comp['4h']:+7.2f}%({posm['4h']}/{len(months)})"
            )

        top = best[0]
        print(
            f"ROBUST_BEST | {symbol} | avg_4tf={top[0]:+.2f}% | worst_tf={top[1]:+.2f}% | "
            f"positive_cells={top[2]}/{len(months)*4} | trades={top[3]} | "
            f"fast=[{top[4]:.0f},{top[5]:.0f}] | 4h=[{top[6]:.0f},{top[7]:.0f}] | "
            f"worst_single_month_tf={top[10]:+.2f}%"
        )


if __name__ == "__main__":
    main()
