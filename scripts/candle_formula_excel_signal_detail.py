from __future__ import annotations

import argparse
import csv
from pathlib import Path

MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def load(path: Path, month: str, symbol: str) -> list[dict[str, float | int]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("month") != month or row.get("symbol") != symbol:
                continue
            try:
                rows.append({
                    "timestamp": int(float(row["timestamp"])),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(rows, key=lambda r: int(r["timestamp"]))


def body(row: dict[str, float | int]) -> float:
    return float(row["close"]) - float(row["open"])


def direction_excel(current_body: float, previous_body: float | None) -> int:
    if previous_body is None:
        return 0
    if current_body > previous_body:
        return 1
    if current_body < previous_body:
        return -1
    return 0


def rules(row: dict[str, float | int]) -> dict[str, int]:
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])
    o = float(row["open"])
    hc = h - c
    co = c - o
    lc = l - c
    ho = h - o
    # Exact formulas used for S = R1 + R2 + R3.
    return {
        "R1": int(hc > co),
        "R2": int(ho < hc),
        "R3": int(lc > co),
        "R4": int(hc < co),
        "R5": int(ho > hc),
        "R6": int(lc < co),
    }


def signal_from_s(s: int) -> int:
    if s == 3:
        return 1
    if s == 0:
        return -1
    return 0


def fmt(v: int) -> str:
    return "BUY" if v > 0 else "SELL" if v < 0 else "HOLD"


def main() -> None:
    ap = argparse.ArgumentParser(description="Row-by-row exact Excel signal and direction detail. No commission.")
    ap.add_argument("--input", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--output", default="reports/candle_formula_excel_signal_detail.csv")
    args = ap.parse_args()

    path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        for month in MONTHS:
            candles = load(path, month, symbol)
            previous_body: float | None = None
            previous_signal = 0
            for i, row in enumerate(candles):
                b = body(row)
                r = rules(row)
                s = r["R1"] + r["R2"] + r["R3"]
                signal = signal_from_s(s)
                actual_direction = direction_excel(b, previous_body)
                prediction_correct = int(previous_signal != 0 and previous_signal == actual_direction)
                rows_out.append({
                    "symbol": symbol,
                    "month": month,
                    "row": i + 1,
                    "timestamp": row["timestamp"],
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "L_body_close_minus_open": b,
                    "previous_L": previous_body if previous_body is not None else "",
                    "actual_direction_L_vs_previous_L": fmt(actual_direction),
                    "actual_direction_value": actual_direction,
                    "R1": r["R1"],
                    "R2": r["R2"],
                    "R3": r["R3"],
                    "R4": r["R4"],
                    "R5": r["R5"],
                    "R6": r["R6"],
                    "S_R1_plus_R2_plus_R3": s,
                    "signal": fmt(signal),
                    "signal_value": signal,
                    "previous_signal": fmt(previous_signal),
                    "previous_signal_value": previous_signal,
                    "previous_prediction_correct": prediction_correct,
                })
                previous_body = b
                previous_signal = signal

    fields = list(rows_out[0].keys()) if rows_out else []
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"EXCEL_SIGNAL_DETAIL_CSV={out_path}")
    print(f"ROWS={len(rows_out)}")
    print("FEE=0")
    print("SIGNAL=S3_BUY/S0_SELL/S1,S2_HOLD")
    print("DIRECTION=IF(L_current>L_previous,1,IF(L_current<L_previous,-1,0))")


if __name__ == "__main__":
    main()
