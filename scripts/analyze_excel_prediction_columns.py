from __future__ import annotations

import argparse
import csv
from pathlib import Path

RAW_FIELDS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]


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
            raise ValueError("Expected headerless order: Timestamp,Open,High,Low,Close[,Volume].")
        rows = [first] + list(reader)
        return [
            {k: raw[idx] if idx < len(raw) else "" for idx, k in enumerate(RAW_FIELDS)}
            for raw in rows
        ]


def calculate(rows, ct_up: float, ct_down: float) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    previous_pt: int | None = None

    for i in range(len(rows) - 1):
        current = rows[i]
        nxt = rows[i + 1]
        o = num(current.get("Open"))
        h = num(current.get("High"))
        low = num(current.get("Low"))
        c = num(current.get("Close"))
        next_c = num(nxt.get("Close"))

        if None in (o, h, low, c, next_c):
            result.append({"pt": "", "ct": "", "long_return": 0.0, "short_return": 0.0})
            continue

        predict: float | None = None
        ct: int | None = None
        pt: int | None = None
        pt_eq_ct = 0
        neg_pt_eq_ct = 0
        long_return = 0.0
        short_return = 0.0

        # Exact Excel predict formula: six-candle window ending at current row.
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

                ct = sign_asymmetric(next_c - c, ct_up, ct_down)
                pt = sign_asymmetric(predict - c, ct_up, ct_down)
                pt_eq_ct = int(pt == ct and pt == 1)
                neg_pt_eq_ct = int(pt == ct and pt == -1)

                if pt == 1 or previous_pt == 1:
                    long_return = (next_c - c) / c if c else 0.0
                if pt == -1 or previous_pt == -1:
                    short_return = (c - next_c) / next_c if next_c else 0.0

                previous_pt = pt

        result.append({
            "Timestamp": current.get("Timestamp", ""),
            "Open": o,
            "High": h,
            "Low": low,
            "Close": c,
            "predict": predict if predict is not None else "",
            "ct": ct if ct is not None else "",
            "pt": pt if pt is not None else "",
            "pt=ct": pt_eq_ct,
            "(-)pt=ct": neg_pt_eq_ct,
            "Long_Return": long_return,
            "Short_Return": short_return,
        })

    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Timestamp", "Open", "High", "Low", "Close",
        "predict", "ct", "pt", "pt=ct", "(-)pt=ct",
        "Long_Return", "Short_Return",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], commission: float) -> None:
    valid = [r for r in rows if r.get("pt") != ""]
    up = [r for r in valid if r["pt"] == 1]
    down = [r for r in valid if r["pt"] == -1]
    hold = [r for r in valid if r["pt"] == 0]

    up_correct = sum(1 for r in up if r["ct"] == 1)
    down_correct = sum(1 for r in down if r["ct"] == -1)
    long_returns = [float(r["Long_Return"]) for r in rows if float(r["Long_Return"]) != 0.0]
    short_returns = [float(r["Short_Return"]) for r in rows if float(r["Short_Return"]) != 0.0]

    long_net = [x - commission for x in long_returns]
    short_net = [x - commission for x in short_returns]

    print("=" * 100)
    print("EXACT EXCEL SAMPLE / FORMULA TEST")
    print("=" * 100)
    print(f"frames={len(valid)}")
    print(f"CT_UP={ARGS.ct_up:g}")
    print(f"CT_DOWN={ARGS.ct_down:g}")
    print(f"COMMISSION_ROUND_TRIP={commission * 100:.4f}%")
    print()
    print("pt summary")
    print(f"all={len(valid)}")
    print(f"up={len(up)} accuracy_vs_ct={up_correct / len(up) * 100:.3f}%" if up else "up=0")
    print(f"down={len(down)} accuracy_vs_ct={down_correct / len(down) * 100:.3f}%" if down else "down=0")
    print(f"hold={len(hold)}")
    print()
    print("final return formulas")
    print(f"long_signals={len(long_returns)} avg_return={sum(long_returns) / len(long_returns) if long_returns else 0.0:.10f} total_return={sum(long_returns):.10f}")
    print(f"short_signals={len(short_returns)} avg_return={sum(short_returns) / len(short_returns) if short_returns else 0.0:.10f} total_return={sum(short_returns):.10f}")
    print(f"long_avg_net={sum(long_net) / len(long_net) if long_net else 0.0:.10f}")
    print(f"short_avg_net={sum(short_net) / len(short_net) if short_net else 0.0:.10f}")


ARGS = None


def main() -> None:
    global ARGS
    p = argparse.ArgumentParser(description="Exact implementation of the supplied Excel sample/formula logic")
    p.add_argument("--input", required=True)
    p.add_argument("--ct-up", type=float, default=600.0)
    p.add_argument("--ct-down", type=float, default=100.0)
    p.add_argument("--commission", type=float, default=0.0)
    p.add_argument("--output", required=True)
    ARGS = p.parse_args()

    if ARGS.ct_up < 0 or ARGS.ct_down < 0 or ARGS.commission < 0:
        raise ValueError("thresholds and commission must be >= 0")

    rows = calculate(load_rows(Path(ARGS.input)), ARGS.ct_up, ARGS.ct_down)
    write_csv(Path(ARGS.output), rows)
    print(f"output={ARGS.output}")
    summarize(rows, ARGS.commission)


if __name__ == "__main__":
    main()
