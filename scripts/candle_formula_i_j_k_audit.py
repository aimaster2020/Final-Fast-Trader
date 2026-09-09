from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def load(path: Path, symbol: str):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return rows
        for r in reader:
            if len(r) < 11:
                continue
            try:
                if r[0] != symbol:
                    continue
                rows.append((int(float(r[2])), float(r[7]), float(r[8]), float(r[9]), float(r[10])))
            except (ValueError, TypeError, IndexError):
                continue
    return sorted(rows, key=lambda x: x[0])


def metrics(rows, predictor):
    abs_sum = sq_sum = 0.0
    correct = 0
    n = 0
    for i in range(len(rows) - 1):
        cur = rows[i]
        nxt = rows[i + 1]
        actual = nxt[1] - cur[1]
        pred = predictor(cur, nxt)
        if pred is None or not math.isfinite(pred):
            continue
        err = pred - actual
        abs_sum += abs(err)
        sq_sum += err * err
        if (pred > 0 and actual > 0) or (pred < 0 and actual < 0) or (pred == 0 and actual == 0):
            correct += 1
        n += 1
    if n == 0:
        return 0, math.inf, math.inf, 0.0
    return n, abs_sum / n, math.sqrt(sq_sum / n), 100.0 * correct / n


def close_delta(cur, nxt):
    return nxt[1] - cur[1]


def main():
    ap = argparse.ArgumentParser(description="Audit spreadsheet columns I/J/K as next-close-delta predictors.")
    ap.add_argument("--input", type=Path, default=Path("reports/prepared_price_action_1h_with_formula.csv"))
    args = ap.parse_args()

    print("IJK_NEXT_CLOSE_AUDIT | tf=1h | fee=0")
    print("Rows: I/J/K are read by physical CSV positions 9/10/11 (1-based).")
    print("Target = Close_next - Close_current. I/J/K are tested without fitting.")
    print("NOTE: if K equals Close_next, it is an oracle/lookahead column and is reported separately.")

    for sym in SYMBOLS:
        rows = load(args.input, sym)
        print(sym)
        if len(rows) < 2:
            print("  NO_DATA")
            continue

        k_match = sum(abs(r[4] - rows[i + 1][1]) <= max(1e-9, abs(rows[i + 1][1]) * 1e-10) for i, r in enumerate(rows[:-1]))
        k_match_pct = 100.0 * k_match / (len(rows) - 1)
        print(f"  K_EQUALS_NEXT_CLOSE={k_match_pct:.2f}%")

        candidates = [
            ("I_RAW", lambda c, n: c[2]),
            ("J_RAW", lambda c, n: c[3]),
            ("K_RAW", lambda c, n: c[4]),
            ("I_AS_DELTA", lambda c, n: c[2]),
            ("J_MINUS_CLOSE", lambda c, n: c[3] - c[1]),
            ("K_MINUS_CLOSE", lambda c, n: c[4] - c[1]),
            ("K_MINUS_J", lambda c, n: c[4] - c[3]),
            ("J_MINUS_I", lambda c, n: c[3] - c[2]),
            ("K_MINUS_I", lambda c, n: c[4] - c[2]),
            ("AVG_IJ", lambda c, n: (c[2] + c[3]) / 2.0),
            ("AVG_IJK", lambda c, n: (c[2] + c[3] + c[4]) / 3.0),
            ("MID_IK_MINUS_CLOSE", lambda c, n: (c[2] + c[4]) / 2.0 - c[1]),
        ]

        results = []
        for name, fn in candidates:
            n, mae, rmse, acc = metrics(rows, fn)
            results.append((mae, rmse, -acc, name, n))

        # True zero baseline on exactly the same rows.
        n, zmae, zrmse, zacc = metrics(rows, lambda c, n: 0.0)
        print(f"  ZERO_BASELINE     MAE={zmae:.6g} RMSE={zrmse:.6g} dir_acc={zacc:.2f}%")
        for mae, rmse, negacc, name, n in sorted(results):
            flag = "  [ORACLE]" if name == "K_RAW" and k_match_pct >= 95.0 else ""
            print(f"  {name:<18} MAE={mae:.6g} RMSE={rmse:.6g} dir_acc={-negacc:.2f}%{flag}")
        print()


if __name__ == "__main__":
    main()
