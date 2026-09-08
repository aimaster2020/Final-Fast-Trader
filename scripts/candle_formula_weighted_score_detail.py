from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")

WEIGHTS = {"R1": 2, "R2": 4, "R3": 5, "R4": 6, "R5": 3, "R6": 1}


@dataclass(frozen=True)
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float


def load(path: Path, month: str, symbol: str) -> list[Candle]:
    rows: list[Candle] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("month") != month or row.get("symbol") != symbol:
                continue
            try:
                rows.append(
                    Candle(
                        int(float(row["timestamp"])),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(rows, key=lambda c: c.timestamp)


def active_rules(c: Candle) -> dict[str, bool]:
    hc = c.high - c.close
    co = c.close - c.open
    lc = c.low - c.close
    ho = c.high - c.open
    return {
        "R1": hc > co,
        "R2": ho < hc,
        "R3": lc > co,
        "R4": hc < co,
        "R5": ho > hc,
        "R6": lc < co,
    }


def weighted_signal(r: dict[str, bool]) -> tuple[int, int, int]:
    up = sum(WEIGHTS[k] for k in ("R1", "R2", "R3") if r[k])
    down = sum(WEIGHTS[k] for k in ("R4", "R5", "R6") if r[k])
    if up > down:
        return 1, up, down
    if down > up:
        return -1, up, down
    return 0, up, down


def body_direction(c: Candle) -> int:
    body = c.close - c.open
    if body > 0:
        return 1
    if body < 0:
        return -1
    return 0


def fmt_direction(value: int) -> str:
    return "BUY" if value > 0 else "SELL" if value < 0 else "FLAT"


def run_detail(candles: list[Candle], symbol: str, month: str, width: float) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    position_side = 0
    entry_price: float | None = None
    previous_prediction = 0
    trade_id = 0

    for i, c in enumerate(candles):
        r = active_rules(c)
        current_signal, up_score, down_score = weighted_signal(r)
        body = c.close - c.open
        body_dir = body_direction(c)
        in_bias = -width <= body <= width
        previous_correct = previous_prediction != 0 and previous_prediction == body_dir

        entry = False
        exit_ = False
        trade_pnl_pct: float | None = None

        if position_side == 0:
            if current_signal != 0 and not in_bias and previous_correct:
                position_side = current_signal
                entry_price = c.close
                trade_id += 1
                entry = True
        elif in_bias and entry_price is not None:
            gross_return = position_side * (c.close - entry_price) / entry_price * 100.0
            trade_pnl_pct = gross_return
            exit_ = True
            position_side = 0
            entry_price = None

        out.append(
            {
                "symbol": symbol,
                "month": month,
                "row": i + 1,
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "body": body,
                "body_direction": fmt_direction(body_dir),
                "R1": int(r["R1"]),
                "R2": int(r["R2"]),
                "R3": int(r["R3"]),
                "R4": int(r["R4"]),
                "R5": int(r["R5"]),
                "R6": int(r["R6"]),
                "R1_weighted": WEIGHTS["R1"] if r["R1"] else 0,
                "R2_weighted": WEIGHTS["R2"] if r["R2"] else 0,
                "R3_weighted": WEIGHTS["R3"] if r["R3"] else 0,
                "R4_weighted": WEIGHTS["R4"] if r["R4"] else 0,
                "R5_weighted": WEIGHTS["R5"] if r["R5"] else 0,
                "R6_weighted": WEIGHTS["R6"] if r["R6"] else 0,
                "up_score": up_score,
                "down_score": down_score,
                "signal": fmt_direction(current_signal),
                "previous_prediction": fmt_direction(previous_prediction),
                "previous_prediction_correct": int(previous_correct),
                "bias_width": width,
                "in_bias": int(in_bias),
                "position_before": fmt_direction(position_side),
                "entry": int(entry),
                "exit": int(exit_),
                "trade_id": trade_id if position_side != 0 or entry or exit_ else "",
                "entry_price": entry_price if entry_price is not None else "",
                "trade_pnl_pct": trade_pnl_pct if trade_pnl_pct is not None else "",
            }
        )
        previous_prediction = current_signal

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Write row-by-row weighted R1..R6 backtest details. No commission.")
    ap.add_argument("--input", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--output", default="reports/candle_formula_weighted_score_detail.csv")
    args = ap.parse_args()

    path = Path(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        for month in MONTHS:
            candles = load(path, month, symbol)
            all_rows.extend(run_detail(candles, symbol, month, args.width))

    if not all_rows:
        raise SystemExit("No matching rows found.")

    fields = list(all_rows[0].keys())
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"DETAIL_CSV={output}")
    print(f"ROWS={len(all_rows)}")
    print(f"WEIGHTS={WEIGHTS}")
    print("FEE=0")
    print(f"WIDTH={args.width:g}")


if __name__ == "__main__":
    main()
