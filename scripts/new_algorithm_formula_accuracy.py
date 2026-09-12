from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


THRESHOLD_PCT = 0.26


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


def excel_bool(value: bool) -> int:
    return 1 if value else 0


def calculate_columns(rows: list[dict[str, float]]) -> list[dict[str, float | int | None]]:
    """
    Reproduce the supplied Excel columns N:AE exactly.

    N = --(HC>CO)
    O = 0
    P = --(HC>HO)
    Q = 1 when the row has data and has a next candle
    R = --(HC<CO)
    S = 0
    T = --(HC<HO)

    AD is the direction formula exactly as supplied, including the
    parenthesis placement in the bearish condition.
    U is predicted close from AD.
    V..AE are the supplied threshold, move, magnitude and direction metrics.
    """
    out: list[dict[str, float | int | None]] = []

    for i, row in enumerate(rows):
        b = row["open"]
        c = row["high"]
        d = row["low"]
        e = row["close"]
        next_close = rows[i + 1]["close"] if i + 1 < len(rows) else None

        g = e - b
        h = c - e
        j = c - b

        n = excel_bool(h > g)
        o = 0
        p = excel_bool(h > j)
        q = 1 if next_close is not None else None
        r = excel_bool(h < g)
        s = 0
        t = excel_bool(h < j)

        # Exact Excel syntax supplied by the user:
        # IF(AND(SUM(R:T)=2,(D-B))<50,-1,0)
        # Because Excel compares the final AND() result (TRUE/FALSE)
        # with 50, this outer comparison is always TRUE. Therefore,
        # whenever SUM(R:T)=2, the bearish branch returns -1.
        bullish = (n + o + p == 2) and ((c - b) > 50)
        bearish_exact = (r + s + t == 2)

        if bullish:
            ad = 1
        elif bearish_exact:
            ad = -1
        else:
            ad = 0

        if ad == 1:
            u = c - b + e
        elif ad == -1:
            u = d - b + e
        else:
            u = b

        v: int | None = None
        w: float | None = None
        x: float | None = None
        y: float | None = None
        z: float | None = None
        aa: float | None = None
        ab: int | None = None
        ac: int | None = None
        ae: int | None = None

        if q is not None:
            v = excel_bool(x := (abs(u - e) / abs(e) * 100.0) >= THRESHOLD_PCT) if e != 0 else 0
            w = u - e
            x = abs(w) / abs(e) * 100.0 if e != 0 else None
            y = next_close - e
            z = abs(y) / abs(e) * 100.0 if e != 0 else None

            if x is not None and z is not None:
                aa = abs(x - z)
                ab = excel_bool((x >= THRESHOLD_PCT) == (z >= THRESHOLD_PCT))

            ac = 1 if ad > 0 else (-1 if ad < 0 else 0)
            actual_direction = 1 if y > 0 else (-1 if y < 0 else 0)
            ae = 0 if ad == 0 or actual_direction == 0 else excel_bool(ad == actual_direction)

        out.append(
            {
                **row,
                "G": g,
                "H": h,
                "J": j,
                "N": n,
                "O": o,
                "P": p,
                "Q": q,
                "R": r,
                "S": s,
                "T": t,
                "U": u,
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
    ap = argparse.ArgumentParser(description="Test the supplied Excel candle direction + move formula exactly")
    ap.add_argument("--input", required=True)
    ap.add_argument("--threshold-pct", type=float, default=THRESHOLD_PCT)
    args = ap.parse_args()

    global THRESHOLD_PCT
    THRESHOLD_PCT = args.threshold_pct

    rows = read_ohlc(Path(args.input))
    data = calculate_columns(rows)

    tested = [r for r in data if r["AD"] is not None and r["Q"] == 1]
    direction_rows = [r for r in tested if r["AD"] != 0 and r["AE"] is not None]
    magnitude_rows = [r for r in tested if r["AB"] is not None]
    threshold_rows = [r for r in tested if r["V"] == 1]

    direction_correct = sum(int(r["AE"] == 1) for r in direction_rows)
    magnitude_correct = sum(int(r["AB"] == 1) for r in magnitude_rows)

    abs_errors = [float(r["W"] - r["Y"]) for r in tested if r["W"] is not None and r["Y"] is not None]
    signed_errors = abs_errors

    mae = sum(abs(e) for e in abs_errors) / len(abs_errors) if abs_errors else 0.0
    rmse = math.sqrt(sum(e * e for e in signed_errors) / len(signed_errors)) if signed_errors else 0.0
    mean_error = sum(signed_errors) / len(signed_errors) if signed_errors else 0.0

    direction_accuracy = direction_correct / len(direction_rows) * 100.0 if direction_rows else 0.0
    magnitude_accuracy = magnitude_correct / len(magnitude_rows) * 100.0 if magnitude_rows else 0.0

    print("=" * 66)
    print("NEW ALGORITHM FORMULA ACCURACY - EXCEL EXACT")
    print("=" * 66)
    print(f"input               : {args.input}")
    print(f"OHLC rows           : {len(rows)}")
    print(f"test samples        : {len(tested)}")
    print(f"threshold %         : {THRESHOLD_PCT:.4f}")
    print(f"threshold signal N  : {len(threshold_rows)}")
    print(f"direction N         : {len(direction_rows)}")
    print(f"direction correct   : {direction_correct}")
    print(f"direction accuracy  : {direction_accuracy:.2f}%")
    print(f"magnitude N         : {len(magnitude_rows)}")
    print(f"magnitude correct   : {magnitude_correct}")
    print(f"magnitude accuracy  : {magnitude_accuracy:.2f}%")
    print(f"MAE (move)          : {mae:.4f}")
    print(f"RMSE (move)         : {rmse:.4f}")
    print(f"mean signed error   : {mean_error:.4f}")
    print("=" * 66)


if __name__ == "__main__":
    main()
