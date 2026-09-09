from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def load(path: Path, symbol: str):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return rows
        if len(header) < 11:
            raise ValueError(f"CSV must contain at least 11 physical columns; found {len(header)}")
        for r in reader:
            if len(r) < 11:
                continue
            try:
                if r[0].strip() != symbol:
                    continue
                # A-G are OHLCV layout: A=symbol B=month C=timestamp D=open E=high F=low G=close H=volume.
                # I/J/K are the three spreadsheet columns requested by the user.
                rows.append({
                    "timestamp": int(float(r[2])),
                    "close": float(r[6]),
                    "I": float(r[8]),
                    "J": float(r[9]),
                    "K": float(r[10]),
                })
            except (ValueError, TypeError, IndexError):
                continue
    return sorted(rows, key=lambda x: x["timestamp"])


def err_metrics(values):
    n = len(values)
    if not n:
        return 0, math.inf, math.inf, 0.0
    mae = sum(abs(p - a) for p, a in values) / n
    rmse = math.sqrt(sum((p - a) ** 2 for p, a in values) / n)
    correct = sum((p > 0 and a > 0) or (p < 0 and a < 0) or (p == 0 and a == 0) for p, a in values)
    return n, mae, rmse, 100.0 * correct / n


def main():
    ap = argparse.ArgumentParser(description="Audit spreadsheet columns I/J/K against next-close change.")
    ap.add_argument("--input", type=Path, required=True, help="CSV containing the spreadsheet columns A..K or more.")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    args = ap.parse_args()
    symbols = tuple(x.strip() for x in args.symbols.split(",") if x.strip())

    print("IJK_NEXT_CLOSE_AUDIT | tf=1h | fee=0 | physical columns")
    print("A-G assumed = symbol/month/timestamp/open/high/low/close; H=volume; I/J/K are read as positions 9/10/11.")
    print("Target = Close_next - Close_current.")
    print("For I/J/K, both RAW value and VALUE-current-close are tested.")

    for sym in symbols:
        rows = load(args.input, sym)
        print(sym)
        if len(rows) < 2:
            print("  NO_DATA")
            continue

        actuals = [rows[i + 1]["close"] - rows[i]["close"] for i in range(len(rows) - 1)]
        zero = [(0.0, a) for a in actuals]
        _, zmae, zrmse, zacc = err_metrics(zero)
        print(f"  ZERO_BASELINE       MAE={zmae:.6g} RMSE={zrmse:.6g} dir_acc={zacc:.2f}%")

        candidates = [
            ("I_RAW", lambda r: r["I"]),
            ("J_RAW", lambda r: r["J"]),
            ("K_RAW", lambda r: r["K"]),
            ("I-CLOSE", lambda r: r["I"] - r["close"]),
            ("J-CLOSE", lambda r: r["J"] - r["close"]),
            ("K-CLOSE", lambda r: r["K"] - r["close"]),
            ("K-J", lambda r: r["K"] - r["J"]),
            ("J-I", lambda r: r["J"] - r["I"]),
            ("K-I", lambda r: r["K"] - r["I"]),
            ("AVG_IJ-CLOSE", lambda r: (r["I"] + r["J"]) / 2.0 - r["close"]),
            ("AVG_IJK-CLOSE", lambda r: (r["I"] + r["J"] + r["K"]) / 3.0 - r["close"]),
        ]

        for name, fn in candidates:
            pairs = []
            for i in range(len(rows) - 1):
                p = fn(rows[i])
                if math.isfinite(p):
                    pairs.append((p, actuals[i]))
            _, mae, rmse, acc = err_metrics(pairs)
            print(f"  {name:<18} MAE={mae:.6g} RMSE={rmse:.6g} dir_acc={acc:.2f}%")

        # Oracle check: determine whether K on current row equals the next candle close.
        matches = 0
        for i in range(len(rows) - 1):
            k = rows[i]["K"]
            nxt = rows[i + 1]["close"]
            tol = max(1e-9, abs(nxt) * 1e-10)
            matches += int(abs(k - nxt) <= tol)
        pct = 100.0 * matches / (len(rows) - 1)
        print(f"  K_EQUALS_NEXT_CLOSE {pct:.2f}%")
        if pct >= 95.0:
            print("  WARNING: K is effectively the next Close; K-based results are LOOKAHEAD/ORACLE, not a valid predictor.")
        print()


if __name__ == "__main__":
    main()
