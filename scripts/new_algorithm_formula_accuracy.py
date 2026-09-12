from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_THRESHOLD_PCT = 0.26


def norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def read_ohlc(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = {norm(x): x for x in (reader.fieldnames or [])}
        required = ["open", "high", "low", "close"]
        missing = [x for x in required if x not in fields]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        rows: list[dict[str, float]] = []
        for raw in reader:
            try:
                rows.append({x: float(raw[fields[x]]) for x in required})
            except (TypeError, ValueError):
                continue

    if len(rows) < 4:
        raise ValueError("Need at least 4 valid OHLC rows")
    return rows


def b(value: bool) -> int:
    return 1 if value else 0


def calculate(rows: list[dict[str, float]], threshold_pct: float) -> list[dict[str, float | int | None]]:
    """Reproduce the supplied Excel logic, with the final prediction taken from column F."""
    # First calculate the per-candle U values from the supplied N:AD rules.
    base: list[dict[str, float | int | None]] = []

    for i, row in enumerate(rows):
        open_ = row["open"]
        high = row["high"]
        low = row["low"]
        close = row["close"]
        next_close = rows[i + 1]["close"] if i + 1 < len(rows) else None

        co = close - open_
        hc = high - close
        lc = close - low
        ho = high - open_

        # Excel N: T exactly as supplied.
        n = b(hc > co)
        o = 0
        p = b(hc > ho)
        r = b(hc < co)
        s = 0
        t = b(hc < ho)

        # Exact AD formula as supplied. The misplaced closing parenthesis
        # means the bearish branch is selected whenever SUM(R:T)=2.
        if (n + o + p == 2) and ((high - open_) > 50):
            ad = 1
        elif r + s + t == 2:
            ad = -1
        else:
            ad = 0

        # Excel U formula.
        if ad == 1:
            u = high - open_ + close
        elif ad == -1:
            u = low - open_ + close
        else:
            u = open_

        base.append({
            **row,
            "CO": co,
            "HC": hc,
            "LC": lc,
            "HO": ho,
            "N": n,
            "O": o,
            "P": p,
            "R": r,
            "S": s,
            "T": t,
            "U": u,
            "AD": ad,
            "NEXT_CLOSE": next_close,
        })

    out: list[dict[str, float | int | None]] = []

    # Excel F formula supplied for row 25:
    # =IF(E25-B25>0,AVERAGE(U23:U25),IF(E25-B25<0,AVERAGE(D23:D25),E25))
    # Therefore F uses the current candle plus the prior two candles.
    for i, row in enumerate(base):
        close = float(row["close"])
        open_ = float(row["open"])
        next_close = row["NEXT_CLOSE"]

        if i >= 2:
            if close - open_ > 0:
                predicted_close = sum(float(base[j]["U"]) for j in range(i - 2, i + 1)) / 3.0
            elif close - open_ < 0:
                predicted_close = sum(float(base[j]["low"]) for j in range(i - 2, i + 1)) / 3.0
            else:
                predicted_close = close
        else:
            predicted_close = None

        v = w = x = y = z = aa = ab = ac = ae = None
        if predicted_close is not None and next_close is not None:
            predicted_move = predicted_close - close
            actual_move = float(next_close) - close
            predicted_move_pct = abs(predicted_move) / abs(close) * 100 if close != 0 else None
            actual_move_pct = abs(actual_move) / abs(close) * 100 if close != 0 else None

            w = predicted_move
            x = predicted_move_pct
            y = actual_move
            z = actual_move_pct

            if x is not None:
                v = b(x >= threshold_pct)
            if x is not None and z is not None:
                aa = abs(x - z)
                ab = b((x >= threshold_pct) == (z >= threshold_pct))

            ad = int(row["AD"])
            ac = 1 if predicted_move > 0 else (-1 if predicted_move < 0 else 0)
            actual_direction = 1 if actual_move > 0 else (-1 if actual_move < 0 else 0)
            ae = 0 if actual_direction == 0 or ac == 0 else b(ac == actual_direction)

        out.append({
            **row,
            "F": predicted_close,
            "V": v,
            "W": w,
            "X": x,
            "Y": y,
            "Z": z,
            "AA": aa,
            "AB": ab,
            "AC": ac,
            "AE": ae,
        })

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Test the supplied Excel prediction formula using column F")
    ap.add_argument("--input", required=True)
    ap.add_argument("--threshold-pct", type=float, default=DEFAULT_THRESHOLD_PCT)
    args = ap.parse_args()

    data = calculate(read_ohlc(Path(args.input)), args.threshold_pct)

    test_rows = [r for r in data if r["F"] is not None and r["Y"] is not None]
    direction_rows = [r for r in test_rows if r["AC"] != 0 and r["AE"] is not None]
    magnitude_rows = [r for r in test_rows if r["AB"] is not None]
    threshold_rows = [r for r in test_rows if r["V"] == 1]

    direction_correct = sum(int(r["AE"] == 1) for r in direction_rows)
    magnitude_correct = sum(int(r["AB"] == 1) for r in magnitude_rows)

    direction_accuracy = direction_correct / len(direction_rows) if direction_rows else 0.0
    magnitude_accuracy = magnitude_correct / len(magnitude_rows) if magnitude_rows else 0.0
    threshold_coverage = len(threshold_rows) / len(test_rows) if test_rows else 0.0
    threshold_precision = (
        sum(int(r["AB"] == 1) for r in threshold_rows) / len(threshold_rows)
        if threshold_rows else 0.0
    )

    predicted_pct = [float(r["X"]) for r in threshold_rows if r["X"] is not None]
    actual_pct = [float(r["Z"]) for r in threshold_rows if r["Z"] is not None]
    avg_predicted = sum(predicted_pct) / len(predicted_pct) if predicted_pct else 0.0
    avg_actual = sum(actual_pct) / len(actual_pct) if actual_pct else 0.0

    print("=" * 72)
    print("NEW ALGORITHM FORMULA ACCURACY - COLUMN F")
    print("=" * 72)
    print(f"input               : {args.input}")
    print(f"OHLC rows           : {len(data)}")
    print(f"test samples        : {len(test_rows)}")
    print(f"threshold %         : {args.threshold_pct:.4f}")
    print(f"direction N         : {len(direction_rows)}")
    print(f"direction correct   : {direction_correct}")
    print(f"direction accuracy  : {direction_accuracy * 100:.2f}%")
    print(f"magnitude N         : {len(magnitude_rows)}")
    print(f"magnitude correct   : {magnitude_correct}")
    print(f"magnitude accuracy  : {magnitude_accuracy * 100:.2f}%")
    print(f"threshold signal N  : {len(threshold_rows)}")
    print(f"threshold coverage  : {threshold_coverage * 100:.2f}%")
    print(f"threshold precision : {threshold_precision * 100:.2f}%")
    print(f"avg predicted move% : {avg_predicted:.6f}%")
    print(f"avg actual move%    : {avg_actual:.6f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
