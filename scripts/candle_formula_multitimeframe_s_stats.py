from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fast_pattern_trader.models import Candle

MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h", "2h", "4h")
S_VALUES = (0, 1, 2, 3)


def excel_s(candle: Candle) -> int:
    j = candle.close - candle.open
    k = candle.high - candle.close
    l = candle.low - candle.close
    m = candle.high - candle.open
    p = int(k > j)
    q = int(k > m)
    r = int(l > j)
    return p + q + r


def excel_actual(j_current: float, j_next: float) -> int:
    if j_next > j_current:
        return 1
    if j_next < j_current:
        return -1
    return 0


def predicted_side(s: int) -> int:
    return 1 if s in (2, 3) else -1


def parse_candle(row: dict[str, str]) -> Candle | None:
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


def discover_files(root: Path, timeframe: str, symbol: str) -> list[Path]:
    names = {
        f"{symbol}_{timeframe}.csv",
        f"{symbol}.csv",
        f"prepared_price_action_{timeframe}.csv",
    }
    candidates: list[Path] = []

    preferred = (
        root / timeframe / f"{symbol}_{timeframe}.csv",
        root / timeframe / f"{symbol}.csv",
        root / f"prepared_price_action_{timeframe}.csv",
    )
    for path in preferred:
        if path.is_file():
            candidates.append(path)

    try:
        for path in root.rglob("*.csv"):
            if path in candidates:
                continue
            if path.name in names:
                candidates.append(path)
    except OSError:
        pass

    return candidates


def load_rows(paths: list[Path], symbol: str, months: set[str]) -> list[tuple[str, Candle]]:
    rows: list[tuple[str, Candle]] = []
    seen: set[tuple[str, int]] = set()
    for path in paths:
        filename_has_symbol = symbol in path.name
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_symbol = row.get("symbol", "")
                    if row_symbol and row_symbol != symbol:
                        continue
                    if not row_symbol and not filename_has_symbol:
                        continue
                    month = row.get("month", "")
                    if month and month not in months:
                        continue
                    candle = parse_candle(row)
                    if candle is None:
                        continue
                    key = (month, candle.timestamp)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append((month, candle))
        except (OSError, UnicodeError):
            continue

    return sorted(rows, key=lambda x: x[1].timestamp)


def compute_stats(rows: list[tuple[str, Candle]]) -> dict[int, list[int]]:
    stats = {s: [0, 0] for s in S_VALUES}  # count, correct
    if len(rows) < 2:
        return stats

    for i in range(len(rows) - 1):
        _, candle = rows[i]
        _, next_candle = rows[i + 1]
        j_current = candle.close - candle.open
        j_next = next_candle.close - next_candle.open
        s = excel_s(candle)
        prediction = predicted_side(s)
        actual = excel_actual(j_current, j_next)

        stats[s][0] += 1
        stats[s][1] += int(prediction == actual)

    return stats


def pct(stat: list[int]) -> float:
    return stat[1] / stat[0] * 100.0 if stat[0] else 0.0


def fmt(stats: dict[int, list[int]]) -> str:
    return " | ".join(
        f"S{s}={stats[s][0]}/{stats[s][1]}/{pct(stats[s]):.1f}%"
        for s in S_VALUES
    )


def add_stats(dst: dict[int, list[int]], src: dict[int, list[int]]) -> None:
    for s in S_VALUES:
        dst[s][0] += src[s][0]
        dst[s][1] += src[s][1]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Exact Excel S0/S1/S2/S3 direction-accuracy test across "
            "1m, 5m, 15m, 30m, 1h, 2h and 4h."
        )
    )
    ap.add_argument("--root", default="reports")
    ap.add_argument("--months", default=",".join(MONTHS))
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    ap.add_argument("--timeframes", default=",".join(TIMEFRAMES))
    args = ap.parse_args()

    root = Path(args.root)
    months = {x.strip() for x in args.months.split(",") if x.strip()}
    symbols = tuple(x.strip() for x in args.symbols.split(",") if x.strip())
    timeframes = tuple(x.strip() for x in args.timeframes.split(",") if x.strip())

    print(
        "EXCEL_MULTI_TF_S_TEST | exact I(row)=sign(J_next-J_current) | "
        "J=C-O | S=P+Q+R | S0/S1=SELL | S2/S3=BUY"
    )
    print(f"months={','.join(sorted(months))} | symbols={','.join(symbols)}")
    print("format: TF SYMBOL | S0=count/correct/acc | S1=... | S2=... | S3=...")

    for timeframe in timeframes:
        tf_total = {s: [0, 0] for s in S_VALUES}
        found_any = False

        for symbol in symbols:
            paths = discover_files(root, timeframe, symbol)
            if not paths:
                print(f"{timeframe:>3} {symbol} | NO_DATA")
                continue

            rows = load_rows(paths, symbol, months)
            if len(rows) < 2:
                print(f"{timeframe:>3} {symbol} | NO_USABLE_DATA")
                continue

            found_any = True
            stats = compute_stats(rows)
            add_stats(tf_total, stats)
            print(f"{timeframe:>3} {symbol} | {fmt(stats)}")

        if found_any:
            print(f"{timeframe:>3} ALL      | {fmt(tf_total)}")
        else:
            print(f"{timeframe:>3} ALL      | NO_DATA")


if __name__ == "__main__":
    main()
