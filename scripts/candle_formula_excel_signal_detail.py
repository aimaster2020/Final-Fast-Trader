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


def excel_rules(open_: float, high: float, low: float, close: float) -> dict[str, float | int]:
    # Exact Excel J:O formulas from the user.
    j = close - open_      # J = G-D = C-O
    k = high - close       # K = E-G = H-C
    l = low - close        # L = F-G = Low-C
    m = high - open_       # M = E-D = H-O
    n = low - open_        # N = F-D = Low-O
    o = high - low         # O = E-F = H-L

    # Exact Excel P:R formulas.
    p = int(k > j)         # P = IF(K>J,1,0)
    q = int(k > m)         # Q = IF(K>M,1,0)
    r = int(l > j)         # R = IF(L>J,1,0)
    s = p + q + r          # S = SUM(P:R)

    return {"J": j, "K": k, "L": l, "M": m, "N": n, "O": o, "P": p, "Q": q, "R": r, "S": s}


def excel_i_for_row(current: dict, next_row: dict | None) -> int | None:
    """Exact Excel placement of I.

    The user's Excel formula in row 2 is:
        =IF((J3)>J2,1,IF((J3)<J2,-1,0))

    So the I value on Excel row N compares J(N+1) against J(N).
    """
    if next_row is None:
        return None
    current_j = current["close"] - current["open"]
    next_j = next_row["close"] - next_row["open"]
    if next_j > current_j:
        return 1
    if next_j < current_j:
        return -1
    return 0


def direction_name(code: int | None) -> str:
    if code == 1:
        return "BUY"
    if code == -1:
        return "SELL"
    if code == 0:
        return "FLAT"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact user-provided Excel candle formulas with exact row alignment; no commission.")
    ap.add_argument("--input", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--output", default="reports/candle_formula_excel_signal_detail.csv")
    args = ap.parse_args()

    out = []

    for symbol in SYMBOLS:
        all_data = load_all(Path(args.input), symbol)

        for i, row in enumerate(all_data):
            if row["month"] not in MONTHS:
                continue

            f = excel_rules(row["open"], row["high"], row["low"], row["close"])
            next_row = all_data[i + 1] if i + 1 < len(all_data) else None

            # EXACT Excel I formula on this row:
            # I(row) = IF(J(next_row)>J(row),1,IF(J(next_row)<J(row),-1,0))
            i_direction = excel_i_for_row(row, next_row)

            # EXACT Excel T/U formulas on the SAME row.
            t = bool(f["S"] == 3 and i_direction == 1)
            u = bool(f["S"] == 0 and i_direction == -1)

            # The raw prediction classification from S only.
            # This is NOT I, T, or U.
            signal = 1 if f["S"] == 3 else -1 if f["S"] == 0 else 0
            in_bias = int(-args.width <= f["J"] <= args.width)

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
                    "J_C-O": f["J"],
                    "K_H-C": f["K"],
                    "L_L-C": f["L"],
                    "M_H-O": f["M"],
                    "N_L-O": f["N"],
                    "O_H-L": f["O"],
                    "I_direction": "" if i_direction is None else i_direction,
                    "I_direction_name": direction_name(i_direction),
                    "P_HC_gt_CO": f["P"],
                    "Q_HC_gt_HO": f["Q"],
                    "R_LC_gt_CO": f["R"],
                    "S": f["S"],
                    "T_S3_and_I1": t,
                    "U_S0_and_Iminus1": u,
                    "signal": "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD",
                    "next_J": "" if next_row is None else next_row["close"] - next_row["open"],
                    "bias_width": args.width,
                    "in_bias": in_bias,
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)

    print(f"DETAIL_CSV={output}")
    print(f"ROWS={len(out)}")
    print("I_EXCEL_ROW_2=IF(J3>J2,1,IF(J3<J2,-1,0))")
    print("I_ALIGNMENT=ROW_N_USES_J(N+1)_VS_J(N)")
    print("SIGNAL=S3_BUY/S0_SELL/S1,S2_HOLD")
    print("T=AND(S=3,I=1)")
    print("U=AND(S=0,I=-1)")
    print("FEE=0")
    print(f"WIDTH={args.width:g}")


if __name__ == "__main__":
    main()
