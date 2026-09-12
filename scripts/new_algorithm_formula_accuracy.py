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

    if len(rows) < 2:
        raise ValueError("Need at least 2 valid OHLC rows")
    return rows


def b(value: bool) -> int:
    return 1 if value else 0


def calculate(rows: list[dict[str, float]], threshold_pct: float) -> list[dict[str, float | int | None]]:
    """Reproduce the supplied Excel N:AE logic as written."""
    out: list[dict[str, float | int | None]] = []

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

        # Excel:
        # N = --(H>G)
        # O = 0
        # P = --(H>J)
        # R = --(H<G)
        # S = 0
        # T = --(H<J)
        n = b(hc > co)
        o = 0
        p = b(hc > ho)
        r = b(hc < co)
        s = 0
        t = b(hc < ho)

        # Exact supplied Excel AD formula:
        # =IF(OR(N="",O="",P="",R="",S="",T=""),"",IF(AND(SUM(N:P)=2,(C-B)>50),1,IF(AND(SUM(R:T)=2,(D-B))<50,-1,0)))
        # The bearish comparison is outside AND(). Excel therefore compares
        # the TRUE/FALSE result of AND(...) with 50; both 1 and 0 are < 50.
        # Consequently, the bearish branch is selected whenever SUM(R:T)=2.
        if (n + o + p == 2) and ((high - open_) > 50):
            ad = 1
        elif (r + s + t == 2):
            ad = -1
        else:
            ad = 0

        # Excel U:
        # =IF(OR(AD="",B="",C="",D="",E=""),"",IF(AD=1,C-B+E,IF(AD=-1,D-B+E,IF(AD=0,B,""))))
        if ad == 1:
            predicted_close = high - open_ + close
        elif ad == -1:
            predicted_close = low - open_ + close
        else:
            predicted_close = open_

        # Excel Q: valid flag.
        q = 1 if next_close is not None else None

        v = w = x = y = z = aa = ab = ac = ae = None
        if q == 1:
            predicted_move = predicted_close - close
            predicted_move_pct = abs(predicted_move) / abs(close) * 100 if close != 0 else None
            actual_move = next_close - close
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

            ac = 1 if ad > 0 else (-1 if ad < 0 else 0)
            actual_direction = 1 if actual_move > 0 else (-1 if actual_move < 0 else 0)
            ae = 0 if ad == 0 or actual_direction == 0 else b(ad == actual_direction)

        out.append(
            {
                **row,
                "CO": co,
                "HC": hc,
                "LC": lc,
                "HO": ho,
                "N": n,
                "O": o,
                "P": p,
                "Q": q,
                "R": r,
                "S": s,
                "T": t,
                "U": predicted_close,
                "V": v,
                "W": w,
                "X": x,
                "Y": y,
                "Z": z,
                "AA": aa,
                "AB": ab,
                "AC": ac,
                "AD": ad,
                "AE": ae,
            }
        )

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Test the supplied Excel candle direction + move formula")
    ap.add_argument("--input", required=True)
    ap.add_argument("--threshold-pct", type=float, default=DEFAULT_THRESHOLD_PCT)
    args = ap.parse_args()

    data = calculate(read_ohlc(Path(args.input)), args.threshold_pct)

    test_rows = [r for r in data if r["AD"] is not None]
    direction_rows = [r for r in test_rows if r["AD"] != 0 and r["AE"] is not None]
    magnitude_rows = [r for r in test_rows if r["AB"] is not None]
    threshold_rows = [r for r in test_rows if r["V"] == 1]

    direction_correct = sum(int(r["AE"] == 1) for r in direction_rows)
    magnitude_correct = sum(int(r["AB"] == 1) for r in magnitude_rows)

    direction_accuracy = direction_correct / len(direction_rows) if direction_rows else 0.0
    magnitude_accuracy = magnitude_correct / len(magnitude_rows) if magnitude_rows else 0.0
    threshold_coverage = len(threshold_rows) / len(test_rows) if test_rows else 0.0

    precision_numerator = sum(int(r["AB"] == 1) for r in threshold_rows)
    threshold_precision = precision_numerator / len(threshold_rows) if threshold_rows else 0.0

    predicted_pct = [float(r["X"]) for r in threshold_rows if r["X"] is not None]
    actual_pct = [float(r["Z"]) for r in threshold_rows if r["Z"] is not None]
    avg_predicted = sum(predicted_pct) / len(predicted_pct) if predicted_pct else 0.0
    avg_actual = sum(actual_pct) / len(actual_pct) if actual_pct else 0.0

    print("=" * 72)
    print("NEW ALGORITHM FORMULA ACCURACY - EXCEL EXACT")
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
