from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from fast_pattern_trader.models import Candle

MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def load_all(path: Path, symbol: str) -> list[tuple[str, Candle]]:
    rows: list[tuple[str, Candle]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("symbol") != symbol:
                continue
            try:
                rows.append(
                    (
                        row["month"],
                        Candle(
                            int(float(row["timestamp"])),
                            float(row["open"]),
                            float(row["high"]),
                            float(row["low"]),
                            float(row["close"]),
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(rows, key=lambda x: x[1].timestamp)


def excel_s(c: Candle) -> int:
    j = c.close - c.open
    k = c.high - c.close
    m = c.high - c.open
    l = c.low - c.close
    return int(k > j) + int(k > m) + int(l > j)


def signal_from_s(s: int) -> int:
    return 1 if s in (2, 3) else -1


def stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    return statistics.mean(values), statistics.median(values), statistics.mean(abs(x) for x in values)


def run(rows: list[tuple[str, Candle]], month: str) -> dict[int, list[float]]:
    result: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: []}
    indices = [i for i, (m, _) in enumerate(rows) if m == month]
    for idx in indices:
        if idx + 1 >= len(rows):
            continue
        if rows[idx + 1][0] != month:
            continue
        current = rows[idx][1]
        nxt = rows[idx + 1][1]
        s = excel_s(current)
        # Magnitude target is the next close minus the current close.
        move = nxt.close - current.close
        result[s].append(move)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure next close movement for the exact Excel S0-S3 candle formula.")
    ap.add_argument("--input", default="reports/prepared_price_action_1h.csv")
    args = ap.parse_args()
    path = Path(args.input)

    print("EXCEL_MAGNITUDE_TEST | tf=1h | fee=0 | exact S0-S3 formula | target=next_close-current_close | same_month_only")
    print("format: N / mean_move / median_move / mean_abs_move | dollars")

    for symbol in SYMBOLS:
        rows = load_all(path, symbol)
        print(f"\n{symbol}")
        all_stats: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: []}
        for month in MONTHS:
            by_s = run(rows, month)
            parts = []
            for s in range(4):
                values = by_s[s]
                all_stats[s].extend(values)
                mean, median, abs_mean = stats(values)
                parts.append(f"S{s}={len(values)}/{mean:+.2f}/{median:+.2f}/{abs_mean:.2f}")
            print(f"{month} | " + " ".join(parts))

        print("4M | " + " ".join(
            f"S{s}={len(all_stats[s])}/{stats(all_stats[s])[0]:+.2f}/{stats(all_stats[s])[1]:+.2f}/{stats(all_stats[s])[2]:.2f}"
            for s in range(4)
        ))

        # Direction-conditioned magnitude: positive distance for UP predictions,
        # positive distance for DOWN predictions. This shows whether the signal's
        # direction is associated with a useful move size.
        print("4M_DIR_MAG | " + " ".join(
            _direction_magnitude(rows, s) for s in range(4)
        ))


def _direction_magnitude(rows: list[tuple[str, Candle]], s_value: int) -> str:
    values: list[float] = []
    for idx in range(len(rows) - 1):
        month, current = rows[idx]
        next_month, nxt = rows[idx + 1]
        if month not in MONTHS or next_month != month:
            continue
        if excel_s(current) != s_value:
            continue
        signal = signal_from_s(s_value)
        signed_move = nxt.close - current.close
        values.append(signal * signed_move)
    mean, median, abs_mean = stats(values)
    return f"S{s_value}={len(values)}/{mean:+.2f}/{median:+.2f}/{abs_mean:.2f}"


if __name__ == "__main__":
    main()
