from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


RAW_FIELDS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
FINAL_FIELDS = [
    "Timestamp", "Open", "High", "Low", "Close",
    "predict", "ct", "pt", "pt=ct", "positive_pt=ct", "negative_pt=ct",
    "HC>CO", "LC>CO", "HC>HO", "M_long_signal", "N_Long_Move",
    "HC<CO", "LC<CO", "HC<HO", "O_short_signal", "P_Short_Move",
    "CO", "HC", "LC", "HO", "Close_next", "V", "Next_Return",
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
    return {"timestamp", "open", "high", "low", "close"}.issubset(normalized)


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


def evaluate(rows, g_threshold: float, h_threshold: float):
    out = []

    for i in range(5, len(rows) - 1):
        current, nxt = rows[i], rows[i + 1]
        o, h, l, c = (num(current.get(k)) for k in ("Open", "High", "Low", "Close"))
        next_c = num(nxt.get("Close"))
        if None in (o, h, l, c, next_c):
            continue

        window = rows[i - 5:i + 1]
        closes = [num(x.get("Close")) for x in window]
        highs = [num(x.get("High")) for x in window]
        if any(value is None for value in closes + highs):
            continue

        # F: IF(E-B>0,AVERAGE(E_window),IF(E-B<0,AVERAGE(C_window),E))
        body = c - o
        predict = sum(closes) / 6.0 if body > 0 else (sum(highs) / 6.0 if body < 0 else c)

        # G / H
        ct = sign(next_c - c, g_threshold)
        pt = sign(predict - c, h_threshold)
        pt_eq_ct = int(pt == ct and pt != 0)
        positive_pt_eq_ct = int(pt == ct and pt == 1)
        negative_pt_eq_ct = int(pt == ct and pt == -1)

        # Q:T
        q = c - o       # CO
        r = h - c       # HC
        s = l - c       # LC
        t = h - o       # HO

        # X:Y:Z
        hc_gt_co = int(r > q)
        lc_gt_co = int(s > q)
        hc_gt_ho = int(r > t)

        # AB:AC:AD
        hc_lt_co = int(r < q)
        lc_lt_co = int(s < q)
        hc_lt_ho = int(r < t)

        # J = SUM(X:Z); L = SUM(AB:AD)
        j_bullish_confirmation = int(hc_gt_co + lc_gt_co + hc_gt_ho == 3)
        l_bearish_confirmation = int(hc_lt_co + lc_lt_co + hc_lt_ho == 0)

        # FINAL EXCEL OUTPUTS:
        # M26 = IF(AND(H26=1,J26=1),1,0)
        # N26 = IF(M26=1,(E27-E26),0)
        # O26 = IF(AND(H26=-1,L26=-1),1,0)
        # P26 = IF(O26,E26-E27,0)
        m_long_signal = int(pt == 1 and j_bullish_confirmation == 1)
        n_long_move = (next_c - c) if m_long_signal == 1 else 0.0
        o_short_signal = int(pt == -1 and l_bearish_confirmation == 1)
        p_short_move = (c - next_c) if o_short_signal == 1 else 0.0

        actual_move = next_c - c

        out.append({
            "Timestamp": current.get("Timestamp", ""),
            "Open": o, "High": h, "Low": l, "Close": c,
            "predict": predict, "ct": ct, "pt": pt,
            "pt=ct": pt_eq_ct,
            "positive_pt=ct": positive_pt_eq_ct,
            "negative_pt=ct": negative_pt_eq_ct,
            "HC>CO": hc_gt_co, "LC>CO": lc_gt_co, "HC>HO": hc_gt_ho,
            "M_long_signal": m_long_signal, "N_Long_Move": n_long_move,
            "HC<CO": hc_lt_co, "LC<CO": lc_lt_co, "HC<HO": hc_lt_ho,
            "O_short_signal": o_short_signal, "P_Short_Move": p_short_move,
            "CO": q, "HC": r, "LC": s, "HO": t,
            "Close_next": next_c, "V": actual_move,
            "Next_Return": actual_move / c if c else 0.0,
        })

    long_signals = [r for r in out if r["M_long_signal"] == 1]
    short_signals = [r for r in out if r["O_short_signal"] == 1]
    long_moves = [float(r["N_Long_Move"]) for r in long_signals]
    short_moves = [float(r["P_Short_Move"]) for r in short_signals]
    all_moves = long_moves + short_moves

    stats = {
        "frames": len(out),
        "long_signals": len(long_signals),
        "short_signals": len(short_signals),
        "total_signals": len(all_moves),
        "long_avg_move": sum(long_moves) / len(long_moves) if long_moves else 0.0,
        "short_avg_move": sum(short_moves) / len(short_moves) if short_moves else 0.0,
        "combined_avg_move": sum(all_moves) / len(all_moves) if all_moves else 0.0,
        "long_total_move": sum(long_moves),
        "short_total_move": sum(short_moves),
        "combined_total_move": sum(all_moves),
    }
    return out, stats


def write_csv(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="Final output matching the supplied Excel final move formulas")
    p.add_argument("--input", required=True)
    p.add_argument("--g", type=float, default=0.0)
    p.add_argument("--h", type=float, default=0.0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    result, stats = evaluate(load_rows(Path(args.input)), args.g, args.h)
    write_csv(Path(args.output), result)

    print(f"G={args.g:g} H={args.h:g}")
    print(f"output={args.output}")
    for key, value in stats.items():
        print(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}")


if __name__ == "__main__":
    main()
