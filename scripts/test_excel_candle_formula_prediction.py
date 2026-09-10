from __future__ import annotations

import argparse
import csv
from pathlib import Path


def sign(v: float) -> int:
    return 1 if v > 0 else -1 if v < 0 else 0


def excel_values(o: float, h: float, l: float, c: float) -> tuple[float, float, float, float, int, float, float, float]:
    # Excel columns from the supplied workbook:
    # N = C-O, O = H-C, P = L-C, Q = H-O, R = L-O.
    n = c - o
    hc = h - c
    lc = l - c
    ho = h - o
    lo = l - o

    # AA, AB, AC and AD exactly as supplied.
    r1 = int(hc > n)
    r2 = int(hc > ho)
    r3 = int(lc > n)
    ad = r1 + r2 + r3

    # I = IF(AD>=3,P/O,IF(AD<1,ABS(P/O),0))
    if ad >= 3:
        i = lc / hc if hc != 0 else 0.0
    elif ad < 1:
        i = abs(lc / hc) if hc != 0 else 0.0
    else:
        i = 0.0

    # J = (R*I)+G, where R = L-O and G = Close.
    predicted_close = (lo * i) + c
    return n, hc, lc, ho, ad, lo, i, predicted_close


def load_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "timestamp": row.get("timestamp", ""),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def run(path: Path) -> None:
    rows = load_rows(path)
    if len(rows) < 2:
        raise ValueError(f"Not enough rows: {path}")

    total = 0
    scored = 0
    direction_correct = 0
    abs_errors: list[float] = []
    rel_errors: list[float] = []
    ad_stats = {0: [0, 0, 0.0], 1: [0, 0, 0.0], 2: [0, 0, 0.0], 3: [0, 0, 0.0]}

    for idx in range(len(rows) - 1):
        cur = rows[idx]
        nxt = rows[idx + 1]
        n, hc, lc, ho, ad, lo, i, pred = excel_values(
            float(cur["open"]), float(cur["high"]), float(cur["low"]), float(cur["close"])
        )
        actual = float(nxt["close"])
        actual_delta = actual - float(cur["close"])
        pred_delta = pred - float(cur["close"])

        total += 1
        error = abs(pred - actual)
        abs_errors.append(error)
        rel_errors.append(error / abs(actual_delta) if abs(actual_delta) > 1e-12 else 0.0)

        # Direction of the predicted move versus actual close-to-close move.
        if pred_delta != 0 and actual_delta != 0:
            scored += 1
            direction_correct += int(sign(pred_delta) == sign(actual_delta))

        st = ad_stats[ad]
        st[0] += 1
        # Correct magnitude direction relative to the actual delta.
        if pred_delta != 0 and actual_delta != 0 and sign(pred_delta) == sign(actual_delta):
            st[1] += 1
        st[2] += error

    print(f"FILE={path}")
    print(f"ROWS={len(rows)} TESTS={total}")
    print(f"MAE={sum(abs_errors)/len(abs_errors):.6f}")
    print(f"DIR_ACC={direction_correct/scored*100:.2f}% N={scored}" if scored else "DIR_ACC=0.00% N=0")
    print("AD  N     DIR_ACC    MAE")
    for ad in range(4):
        n, wins, err_sum = ad_stats[ad]
        acc = wins / n * 100 if n else 0.0
        mae = err_sum / n if n else 0.0
        print(f"{ad}   {n:<5} {acc:>7.2f}%   {mae:>10.6f}")



def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce the supplied Excel candle prediction formulas.")
    ap.add_argument("--input", required=True)
    args = ap.parse_args()
    run(Path(args.input))


if __name__ == "__main__":
    main()
