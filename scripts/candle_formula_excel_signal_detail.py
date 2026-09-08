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
                rows.append({
                    "timestamp": int(float(row["timestamp"])),
                    "month": row["month"],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(rows, key=lambda r: r["timestamp"])


def direction(cur_l: float, prev_l: float | None) -> int:
    if prev_l is None:
        return 0
    return 1 if cur_l > prev_l else -1 if cur_l < prev_l else 0


def signal(o: float, h: float, low: float, c: float) -> tuple[int, tuple[int, int, int, int, int, int], int]:
    body = c - o
    hc = h - c
    lc = low - c
    ho = h - o
    r1 = int(hc > body)
    r2 = int(hc > ho)
    r3 = int(lc > body)
    r4 = int(hc < body)
    r5 = int(ho > hc)
    r6 = int(lc < body)
    s = r1 + r2 + r3
    sig = 1 if s == 3 else -1 if s == 0 else 0
    return sig, (r1, r2, r3, r4, r5, r6), s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--output", default="reports/candle_formula_excel_signal_detail.csv")
    args = ap.parse_args()

    out = []
    for symbol in SYMBOLS:
        all_data = load_all(Path(args.input), symbol)
        prev_pred = 0
        pos = 0
        entry_price = None
        trade_id = 0
        for i, r in enumerate(all_data):
            if r["month"] not in MONTHS:
                if r["month"] < MONTHS[0]:
                    prev_pred, pos, entry_price = 0, 0, None
                continue
            o, h, low, c = r["open"], r["high"], r["low"], r["close"]
            lc = c - o
            prev = all_data[i - 1] if i > 0 and all_data[i - 1]["month"] <= r["month"] else None
            pl = None if prev is None else prev["close"] - prev["open"]
            actual = direction(lc, pl)
            sig, rules, s = signal(o, h, low, c)
            R1, R2, R3, R4, R5, R6 = rules
            correct = int(prev_pred != 0 and prev_pred == actual)
            bias = int(-args.width <= lc <= args.width)
            entry = exit_ = 0
            pnl = ""
            if pos == 0 and sig and not bias and correct:
                pos = sig
                entry_price = c
                trade_id += 1
                entry = 1
            elif pos and bias and entry_price is not None:
                pnl = pos * (c - entry_price) / entry_price * 100.0
                pos = 0
                entry_price = None
                exit_ = 1
            out.append({
                "symbol": symbol, "month": r["month"], "row": i + 1, "timestamp": r["timestamp"],
                "open": o, "high": h, "low": low, "close": c,
                "L_current": lc, "L_previous": "" if pl is None else pl,
                "actual_direction_code": actual,
                "R1": R1, "R2": R2, "R3": R3, "R4": R4, "R5": R5, "R6": R6,
                "S": s, "signal_code": sig, "previous_prediction_code": prev_pred,
                "previous_prediction_correct": correct, "bias_width": args.width, "in_bias": bias,
                "position_after": pos, "entry": entry, "exit": exit_,
                "trade_id": trade_id if (pos or entry or exit_) else "",
                "entry_price": "" if entry_price is None else entry_price,
                "trade_pnl_pct": pnl,
            })
            prev_pred = sig

            if r["month"] != MONTHS[-1] and i + 1 < len(all_data) and all_data[i + 1]["month"] != r["month"]:
                prev_pred = 0
                if pos and entry_price is not None:
                    pos = 0
                    entry_price = None

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"DETAIL_CSV={p}")
    print(f"ROWS={len(out)}")
    print("SIGNAL=S3_BUY/S0_SELL/S1,S2_HOLD")
    print("DIRECTION=IF(L_current>L_previous,1,IF(L_current<L_previous,-1,0))")
    print("FEE=0")
    print(f"WIDTH={args.width:g}")


if __name__ == "__main__":
    main()
