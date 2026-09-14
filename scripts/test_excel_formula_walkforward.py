from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


BASE_FIELDS = ["Timestamp", "Open", "High", "Low", "Close"]
RAW_FIELDS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
FINAL_FIELDS = [
    "Timestamp",
    "Open",
    "High",
    "Low",
    "Close",
    "predict",
    "ct",
    "pt",
    "pt=ct",
    "positive_pt=ct",
    "negative_pt=ct",
    "bullish_evidence",
    "bearish_evidence",
    "positive_direction_match",
    "positive_move",
    "negative_direction_match",
    "negative_move",
    "CO",
    "HC",
    "LC",
    "HO",
    "Close_next",
    "V",
    "Next_Return",
    "HC>CO",
    "LC>CO",
    "HC>HO",
    "Fold/Valid",
    "HC<CO",
    "LC<CO",
    "HC<HO",
    "Predicted Close",
    "Threshold",
    "Predicted Move",
    "Predicted Move %",
    "Actual Move",
    "Actual Move %",
    "Abs Error (pp)",
    "Magnitude Correct",
    "Direction Score",
    "Direction",
    "Direction Correct",
]


def num(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sign(value: float, threshold: float) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def looks_like_header(row: list[str]) -> bool:
    normalized = [str(x).strip().lower() for x in row]
    required = {"timestamp", "open", "high", "low", "close"}
    return required.issubset(set(normalized))


def load_rows(path: Path):
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
                "Input CSV has neither a recognized header nor at least 5 OHLC columns. "
                "Expected headerless order: Timestamp,Open,High,Low,Close[,Volume]."
            )

        rows = [first] + list(reader)
        return [
            {k: raw[idx] if idx < len(raw) else "" for idx, k in enumerate(RAW_FIELDS)}
            for raw in rows
        ]


def evaluate(
    rows,
    g_threshold: float,
    h_threshold: float,
    magnitude_threshold: float,
    am_mode: str,
):
    out = []

    for i in range(5, len(rows) - 1):
        current, nxt = rows[i], rows[i + 1]
        o, h, l, c = (
            num(current.get(k)) for k in ("Open", "High", "Low", "Close")
        )
        next_c = num(nxt.get("Close"))
        if None in (o, h, l, c, next_c):
            continue

        window = rows[i - 5 : i + 1]
        closes = [num(x.get("Close")) for x in window]
        highs = [num(x.get("High")) for x in window]
        if any(value is None for value in closes + highs):
            continue

        body = c - o
        predict = (
            sum(closes) / 6.0
            if body > 0
            else (sum(highs) / 6.0 if body < 0 else c)
        )

        ct = sign(next_c - c, g_threshold)
        pt = sign(predict - c, h_threshold)

        pt_eq_ct = int(pt == ct and pt != 0)
        positive_pt_eq_ct = int(pt == ct and pt == 1)
        negative_pt_eq_ct = int(pt == ct and pt == -1)

        q = c - o
        r = h - c
        s = l - c
        t = h - o

        hc_gt_co = int(r > q)
        lc_gt_co = int(s > q)
        hc_gt_ho = int(r > t)
        hc_lt_co = int(r < q)
        lc_lt_co = int(s < q)
        hc_lt_ho = int(r < t)

        bullish_evidence = int(hc_gt_co + lc_gt_co + hc_gt_ho == 3)
        bearish_evidence = -1 if hc_lt_co + lc_lt_co + hc_lt_ho == 3 else 0

        if am_mode == "literal":
            bullish = int(hc_gt_co + lc_gt_co + hc_gt_ho == 2 and q > 50)
            bearish = int(hc_lt_co + lc_lt_co + hc_lt_ho == 2)
        elif am_mode == "corrected":
            bullish = int(hc_gt_co + lc_gt_co + hc_gt_ho == 2 and (h - o) > 50)
            bearish = int(hc_lt_co + lc_lt_co + hc_lt_ho == 2 and (l - o) < 50)
        else:
            raise ValueError(f"Unsupported am_mode: {am_mode}")

        direction_score = 1 if bullish else (-1 if bearish else 0)
        an = direction_score

        positive_direction_match = int(pt == 1 and bullish_evidence == 1)
        positive_move = (next_c - c) if positive_direction_match else 0.0
        negative_direction_match = int(pt == -1 and bearish_evidence == -1)
        negative_move = (c - next_c) if negative_direction_match else 0.0

        fold_valid = 1
        predicted_close = (
            h - o + c
            if an == 1
            else (l - o + c if an == -1 else o)
        )
        predicted_move = predicted_close - c
        predicted_move_pct = abs(predicted_move) / abs(c) * 100 if c else 0.0
        threshold = int(predicted_move_pct >= magnitude_threshold)
        actual_move = next_c - c
        actual_move_pct = abs(actual_move) / abs(c) * 100 if c else 0.0
        abs_error_pp = abs(predicted_move_pct - actual_move_pct)
        magnitude_correct = int(
            (predicted_move_pct >= magnitude_threshold)
            == (actual_move_pct >= magnitude_threshold)
        )
        actual_direction = 1 if actual_move > 0 else (-1 if actual_move < 0 else 0)
        direction_correct = int(
            an != 0
            and actual_direction != 0
            and an == actual_direction
        )

        out.append(
            {
                "Timestamp": current.get("Timestamp", ""),
                "Open": o,
                "High": h,
                "Low": l,
                "Close": c,
                "predict": predict,
                "ct": ct,
                "pt": pt,
                "pt=ct": pt_eq_ct,
                "positive_pt=ct": positive_pt_eq_ct,
                "negative_pt=ct": negative_pt_eq_ct,
                "bullish_evidence": bullish_evidence,
                "bearish_evidence": bearish_evidence,
                "positive_direction_match": positive_direction_match,
                "positive_move": positive_move,
                "negative_direction_match": negative_direction_match,
                "negative_move": negative_move,
                "CO": q,
                "HC": r,
                "LC": s,
                "HO": t,
                "Close_next": next_c,
                "V": actual_move,
                "Next_Return": actual_move / c if c else 0.0,
                "HC>CO": hc_gt_co,
                "LC>CO": lc_gt_co,
                "HC>HO": hc_gt_ho,
                "Fold/Valid": fold_valid,
                "HC<CO": hc_lt_co,
                "LC<CO": lc_lt_co,
                "HC<HO": hc_lt_ho,
                "Predicted Close": predicted_close,
                "Threshold": threshold,
                "Predicted Move": predicted_move,
                "Predicted Move %": predicted_move_pct,
                "Actual Move": actual_move,
                "Actual Move %": actual_move_pct,
                "Abs Error (pp)": abs_error_pp,
                "Magnitude Correct": magnitude_correct,
                "Direction Score": direction_score,
                "Direction": an,
                "Direction Correct": direction_correct,
            }
        )

    signals = [r for r in out if r["Direction"] != 0]
    correct = sum(int(r["Direction Correct"]) for r in signals)
    magnitude_correct = sum(int(r["Magnitude Correct"]) for r in out)
    stats = {
        "frames": len(out),
        "signals": len(signals),
        "signal_pct": len(signals) / len(out) * 100 if out else 0.0,
        "direction_correct": correct,
        "direction_accuracy_pct": correct / len(signals) * 100 if signals else 0.0,
        "pt_ct": sum(int(r["pt=ct"]) for r in out),
        "positive_pt_ct": sum(int(r["positive_pt=ct"]) for r in out),
        "negative_pt_ct": sum(int(r["negative_pt=ct"]) for r in out),
        "magnitude_correct": magnitude_correct,
        "magnitude_accuracy_pct": magnitude_correct / len(out) * 100 if out else 0.0,
    }
    return out, stats


def write_csv(path: Path, rows: Iterable[dict]):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="Final output equivalent of the supplied Excel formulas")
    p.add_argument("--input", required=True)
    p.add_argument("--g", type=float, default=0.0)
    p.add_argument("--h", type=float, default=0.0)
    p.add_argument("--magnitude-threshold", type=float, default=0.0)
    p.add_argument("--am-mode", choices=("literal", "corrected"), default="literal")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    result, stats = evaluate(
        load_rows(Path(args.input)),
        args.g,
        args.h,
        args.magnitude_threshold,
        args.am_mode,
    )
    write_csv(Path(args.output), result)

    print(f"G={args.g:g} H={args.h:g} MAGNITUDE_THRESHOLD={args.magnitude_threshold:g} AM_MODE={args.am_mode}")
    print(f"output={args.output}")
    for key, value in stats.items():
        print(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}")


if __name__ == "__main__":
    main()
