from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


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
        rows = []
        for raw in reader:
            try:
                row = {x: float(raw[fields[x]]) for x in required}
            except (TypeError, ValueError):
                continue
            rows.append(row)
    if len(rows) < 6:
        raise ValueError("Need at least 6 valid OHLC rows")
    return rows


def calc_ad(row: dict[str, float]) -> int:
    # Reproduces the spreadsheet logic in AC/AD exactly, including the 50 threshold.
    co = row["close"] - row["open"]
    hc = row["high"] - row["close"]
    ho = row["high"] - row["open"]
    lc = row["close"] - row["low"]

    n = int(hc > co)
    o = int(lc > co)  # spreadsheet column O is explicitly 0 in the supplied data
    p = int(hc > ho)
    r = int(hc < co)
    s = int(lc < co)  # spreadsheet column S is explicitly 0 in the supplied data
    t = int(hc < ho)

    if n + o + p == 2 and (row["high"] - row["open"]) > 50:
        ac = 1
    elif r + s + t == 2 and (row["low"] - row["open"]) < 50:
        ac = -1
    else:
        ac = 0
    return ac if ac in (-1, 0, 1) else 0


def calc_u(row: dict[str, float], ad: int) -> float:
    if ad == 1:
        return row["high"] - row["open"] + row["close"]
    if ad == -1:
        return row["low"] - row["open"] + row["close"]
    return row["open"]


def sign(v: float, eps: float = 1e-12) -> int:
    if v > eps:
        return 1
    if v < -eps:
        return -1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Test the exact Excel candle prediction formula")
    ap.add_argument("--input", required=True, help="CSV with Timestamp/Open/High/Low/Close (or lowercase equivalents)")
    args = ap.parse_args()

    rows = read_ohlc(Path(args.input))
    ads = [calc_ad(r) for r in rows]
    us = [calc_u(r, ads[i]) for i, r in enumerate(rows)]

    tested = 0
    direction_correct = 0
    abs_errors = []
    signed_errors = []

    # Spreadsheet starts prediction at row 25 using U/L from the current row and prior 2 rows,
    # then compares Predicted Close with Close_next (column K).
    for i in range(4, len(rows) - 1):
        row = rows[i]
        if row["close"] > row["open"]:
            predicted_close = sum(us[i - 2 : i + 1]) / 3.0
        elif row["close"] < row["open"]:
            predicted_close = sum(rows[j]["low"] for j in range(i - 2, i + 1)) / 3.0
        else:
            predicted_close = row["close"]

        actual_close = rows[i + 1]["close"]
        predicted_move = predicted_close - row["close"]
        actual_move = actual_close - row["close"]
        ps = sign(predicted_move)
        a_s = sign(actual_move)

        tested += 1
        if ps == a_s:
            direction_correct += 1
        err = predicted_close - actual_close
        signed_errors.append(err)
        abs_errors.append(abs(err))

    if not tested:
        raise ValueError("No testable rows")

    accuracy = direction_correct / tested * 100.0
    mae = sum(abs_errors) / tested
    mean_error = sum(signed_errors) / tested
    rmse = math.sqrt(sum(e * e for e in signed_errors) / tested)

    print("=" * 58)
    print("NEW ALGORITHM FORMULA ACCURACY")
    print("=" * 58)
    print(f"input              : {args.input}")
    print(f"OHLC rows          : {len(rows)}")
    print(f"tested predictions : {tested}")
    print(f"direction correct  : {direction_correct}")
    print(f"direction wrong    : {tested - direction_correct}")
    print(f"direction accuracy : {accuracy:.2f}%")
    print(f"MAE (price)        : {mae:.4f}")
    print(f"RMSE (price)       : {rmse:.4f}")
    print(f"mean signed error  : {mean_error:.4f}")
    print("=" * 58)


if __name__ == "__main__":
    main()
