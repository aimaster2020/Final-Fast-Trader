from __future__ import annotations

import argparse
import csv
from pathlib import Path

MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def load_all(path: Path, symbol: str) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("symbol") != symbol:
                continue
            try:
                rows.append(
                    {
                        "timestamp": int(float(row["timestamp"])),
                        "month": row["month"],
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row.get("volume", 0) or 0),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(rows, key=lambda r: r["timestamp"])


def excel_direction(current_j: float, previous_j: float | None) -> int | None:
    """Exact Excel I formula: =IF(Jcurrent>Jprevious,1,IF(Jcurrent<Jprevious,-1,0))."""
    if previous_j is None:
        return None
    if current_j > previous_j:
        return 1
    if current_j < previous_j:
        return -1
    return 0


def excel_rules(open_: float, high: float, low: float, close: float) -> tuple[int, ...]:
    """Exact Excel J:O and P:R logic from the user's formulas."""
    j = close - open_  # J = C-O
    k = high - close  # K = H-C
    l = low - close  # L = L-C
    m = high - open_  # M = H-O
    n = low - open_  # N = L-O
    o = high - low  # O = H-L
    _ = n, o  # retained because N/O are explicit Excel columns

    # P = IF(K>J,1,0)
    p = int(k > j)
    # Q = IF(K>M,1,0)
    q = int(k > m)
    # R = IF(L>J,1,0)
    r = int(l > j)
    return j, k, l, m, n, o, p, q, r


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact user-provided Excel candle formulas; no commission.")
    ap.add_argument("--input", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--output", default="reports/candle_formula_excel_signal_detail.csv")
    args = ap.parse_args()

    out = []
    for symbol in SYMBOLS:
        all_data = load_all(Path(args.input), symbol)
        previous_prediction = ""
        position = 0
        entry_price = None
        trade_id = 0

        for i, row in enumerate(all_data):
            if row["month"] not in MONTHS:
                continue

            ohlc = row["open"], row["high"], row["low"], row["close"]
            j, k, l, m, n, o, p, q, r = excel_rules(*ohlc)
            s = p + q + r

            previous_j = None
            if i > 0:
                previous_row = all_data[i - 1]
                previous_j = previous_row["close"] - previous_row["open"]
            actual_direction = excel_direction(j, previous_j)

            # T = IF(AND(S=3,I=1),TRUE,"")
            t = bool(s == 3 and actual_direction == 1)
            # U = IF(AND(S=0,I=-1),TRUE,"")
            u = bool(s == 0 and actual_direction == -1)

            # Prediction generated only from S, exactly as implied by T/U:
            # S=3 -> BUY, S=0 -> SELL, otherwise HOLD.
            signal = 1 if s == 3 else -1 if s == 0 else 0

            previous_prediction_correct = ""
            if previous_prediction != "" and actual_direction is not None:
                previous_prediction_correct = int(
                    (previous_prediction == "BUY" and actual_direction == 1)
                    or (previous_prediction == "SELL" and actual_direction == -1)
                )

            in_bias = int(-args.width <= j <= args.width)
            entry = 0
            exit_ = 0
            trade_pnl_pct = ""

            # Existing trading layer retained; formula layer above is exact Excel.
            if position == 0 and signal != 0 and not in_bias and previous_prediction_correct == 1:
                position = signal
                entry_price = row["close"]
                trade_id += 1
                entry = 1
            elif position != 0 and in_bias and entry_price is not None:
                trade_pnl_pct = position * (row["close"] - entry_price) / entry_price * 100.0
                position = 0
                entry_price = None
                exit_ = 1

            out.append(
                {
                    "symbol": symbol,
                    "month": row["month"],
                    "row": i + 1,
                    "timestamp": row["timestamp"],
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                    "J_C-O": j,
                    "K_H-C": k,
                    "L_L-C": l,
                    "M_H-O": m,
                    "N_L-O": n,
                    "O_H-L": o,
                    "I_direction": "" if actual_direction is None else actual_direction,
                    "P_HC_gt_CO": p,
                    "Q_HC_gt_HO": q,
                    "R_LC_gt_CO": r,
                    "S": s,
                    "T_S3_and_I1": t,
                    "U_S0_and_Iminus1": u,
                    "signal": "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD",
                    "previous_prediction": previous_prediction,
                    "previous_prediction_correct": previous_prediction_correct,
                    "bias_width": args.width,
                    "in_bias": in_bias,
                    "position_after": "BUY" if position == 1 else "SELL" if position == -1 else "FLAT",
                    "entry": entry,
                    "exit": exit_,
                    "trade_id": trade_id if (position or entry or exit_) else "",
                    "entry_price": "" if entry_price is None else entry_price,
                    "trade_pnl_pct": trade_pnl_pct,
                }
            )

            previous_prediction = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"

            # Trading layer closes at month end; Excel formula calculations do not reset.
            if i + 1 < len(all_data) and all_data[i + 1]["month"] != row["month"] and position != 0 and entry_price is not None:
                position = 0
                entry_price = None

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)

    print(f"DETAIL_CSV={output}")
    print(f"ROWS={len(out)}")
    print("FORMULAS=J:C-O; K:H-C; L:Low-C; M:H-O; N:Low-O; O:H-L")
    print("DIRECTION=IF(J_current>J_previous,1,IF(J_current<J_previous,-1,0))")
    print("SIGNAL=S3_BUY/S0_SELL/S1,S2_HOLD")
    print("T=AND(S=3,I=1)")
    print("U=AND(S=0,I=-1)")
    print("FEE=0")
    print(f"WIDTH={args.width:g}")


if __name__ == "__main__":
    main()
