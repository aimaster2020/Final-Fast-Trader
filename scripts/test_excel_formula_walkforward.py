from __future__ import annotations

import argparse
import csv
from pathlib import Path

RAW_FIELDS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
OUTPUT_FIELDS = [
    "Timestamp",
    "Open",
    "High",
    "Low",
    "Close",
    "predict",
    "ct",
    "pt",
    "pt=ct",
    "(-)pt=ct",
    "Long_Return",
    "Short_Return",
]


def num(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sign_asymmetric(value: float, up_threshold: float, down_threshold: float) -> int:
    if value > up_threshold:
        return 1
    if value < -down_threshold:
        return -1
    return 0


def looks_like_header(row: list[str]) -> bool:
    normalized = [str(x).strip().lower() for x in row]
    return {"timestamp", "open", "high", "low", "close"}.issubset(normalized)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return []

        if looks_like_header(first):
            fields = [str(x).strip() for x in first]
            return [
                {k: raw[idx] if idx < len(raw) else "" for idx, k in enumerate(fields)}
                for raw in reader
            ]

        if len(first) < 5:
            raise ValueError(
                "Expected headerless order: Timestamp,Open,High,Low,Close[,Volume]."
            )

        rows = [first] + list(reader)
        return [
            {k: raw[idx] if idx < len(raw) else "" for idx, k in enumerate(RAW_FIELDS)}
            for raw in rows
        ]


def build_rows(
    rows: list[dict[str, str]],
    ct_up_threshold: float,
    ct_down_threshold: float,
) -> list[dict[str, object]]:
    """Match the supplied Excel formulas row-for-row.

    Predict formula for row r (once six candles exist):
      IF(Close-Open>0, AVERAGE(last 6 closes),
         IF(Close-Open<0, AVERAGE(last 6 highs), Close))

    ct formula:
      +1 when next close - current close > ct_up_threshold
      -1 when next close - current close < -ct_down_threshold

    pt formula:
      +1 when predict - current close > ct_up_threshold
      -1 when predict - current close < -ct_down_threshold

    pt=ct and (-)pt=ct are exact Excel-style agreement flags.

    Final return formulas from the supplied sheet:
      Long_Return  = IF(OR(pt_current=1, pt_previous=1),
                         (next_close-current_close)/current_close, 0)
      Short_Return = IFERROR(IF(OR(pt_current=-1, pt_previous=-1),
                         (current_close-next_close)/next_close, 0), 0)
    """
    output: list[dict[str, object]] = []

    # The Excel sample starts calculations after six candles are available.
    for i in range(len(rows) - 1):
        current = rows[i]
        nxt = rows[i + 1]

        o = num(current.get("Open"))
        h = num(current.get("High"))
        c = num(current.get("Close"))
        next_c = num(nxt.get("Close"))
        if None in (o, h, c, next_c):
            continue

        predict: float | None = None
        ct = ""
        pt = ""
        pt_eq_ct = ""
        neg_pt_eq_ct = ""
        long_return = 0.0
        short_return = 0.0

        # Excel has blank formulas until the six-row moving window exists.
        if i >= 5:
            window = rows[i - 5 : i + 1]
            closes = [num(x.get("Close")) for x in window]
            highs = [num(x.get("High")) for x in window]
            if all(v is not None for v in closes + highs):
                body = c - o
                if body > 0:
                    predict = sum(closes) / 6.0
                elif body < 0:
                    predict = sum(highs) / 6.0
                else:
                    predict = c

                actual_delta = next_c - c
                predicted_delta = predict - c
                ct_value = sign_asymmetric(
                    actual_delta, ct_up_threshold, ct_down_threshold
                )
                pt_value = sign_asymmetric(
                    predicted_delta, ct_up_threshold, ct_down_threshold
                )
                ct = ct_value
                pt = pt_value
                pt_eq_ct = int(pt_value == ct_value and pt_value == 1)
                neg_pt_eq_ct = int(pt_value == ct_value and pt_value == -1)

                previous_pt = None
                if i - 1 >= 5:
                    previous_predict = output[i - 1].get("predict")
                    previous_pt = output[i - 1].get("pt")
                    # previous_pt is already the exact Excel pt value for row i-1.
                    _ = previous_predict

                if pt_value == 1 or previous_pt == 1:
                    long_return = (next_c - c) / c if c else 0.0

                if pt_value == -1 or previous_pt == -1:
                    short_return = (c - next_c) / next_c if next_c else 0.0

        output.append(
            {
                "Timestamp": current.get("Timestamp", ""),
                "Open": o,
                "High": h,
                "Low": num(current.get("Low")),
                "Close": c,
                "predict": predict if predict is not None else "",
                "ct": ct,
                "pt": pt,
                "pt=ct": pt_eq_ct,
                "(-)pt=ct": neg_pt_eq_ct,
                "Long_Return": long_return,
                "Short_Return": short_return,
            }
        )

    return output


def summarize(result: list[dict[str, object]], commission_rate: float) -> dict[str, float | int]:
    valid = [r for r in result if r["pt"] != ""]
    up = [r for r in valid if r["pt"] == 1]
    down = [r for r in valid if r["pt"] == -1]
    hold = [r for r in valid if r["pt"] == 0]
    long_returns = [float(r["Long_Return"]) for r in result if float(r["Long_Return"]) != 0.0]
    short_returns = [float(r["Short_Return"]) for r in result if float(r["Short_Return"]) != 0.0]

    round_trip_pct = commission_rate * 100.0
    long_net = [x - commission_rate for x in long_returns]
    short_net = [x - commission_rate for x in short_returns]

    return {
        "all_pt": len(valid),
        "pt_up": len(up),
        "pt_down": len(down),
        "pt_hold": len(hold),
        "pt_up_accuracy": sum(1 for r in up if r.get("ct") == 1) / len(up) if up else 0.0,
        "pt_down_accuracy": sum(1 for r in down if r.get("ct") == -1) / len(down) if down else 0.0,
        "pt_agreement_up": sum(int(r["pt=ct"] == 1) for r in valid),
        "pt_agreement_down": sum(int(r["(-)pt=ct"] == 1) for r in valid),
        "long_signals": len(long_returns),
        "short_signals": len(short_returns),
        "long_total_return": sum(long_returns),
        "short_total_return": sum(short_returns),
        "long_avg_return": sum(long_returns) / len(long_returns) if long_returns else 0.0,
        "short_avg_return": sum(short_returns) / len(short_returns) if short_returns else 0.0,
        "long_total_net": sum(long_net),
        "short_total_net": sum(short_net),
        "long_avg_net": sum(long_net) / len(long_net) if long_net else 0.0,
        "short_avg_net": sum(short_net) / len(short_net) if short_net else 0.0,
        "commission_round_trip_pct": round_trip_pct,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Exact implementation of the supplied Excel sample/formula logic"
    )
    p.add_argument("--input", required=True)
    p.add_argument("--ct-up", type=float, default=600.0,
                   help="Excel F3-style upward threshold; default 600")
    p.add_argument("--ct-down", type=float, default=100.0,
                   help="Excel F4-style downward threshold; default 100")
    p.add_argument("--commission", type=float, default=0.0,
                   help="Round-trip commission as decimal; default 0")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if args.ct_up < 0 or args.ct_down < 0 or args.commission < 0:
        raise ValueError("thresholds and commission must be >= 0")

    result = build_rows(load_rows(Path(args.input)), args.ct_up, args.ct_down)
    write_csv(Path(args.output), result)
    stats = summarize(result, args.commission)

    print("=" * 100)
    print("EXACT EXCEL SAMPLE / FORMULA TEST")
    print("=" * 100)
    print(f"CT_UP={args.ct_up:g}")
    print(f"CT_DOWN={args.ct_down:g}")
    print(f"COMMISSION_ROUND_TRIP={args.commission * 100:.4f}%")
    print(f"output={args.output}")
    print()
    for key, value in stats.items():
        if isinstance(value, float):
            if "accuracy" in key or "agreement" in key:
                print(f"{key}={value * 100:.6f}%")
            else:
                print(f"{key}={value:.10f}")
        else:
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
