from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((str(r.get("timestamp", "")), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def body(row):
    return row[4] - row[1]


def ad(row):
    b = body(row)
    hc = row[2] - row[4]
    ho = row[2] - row[1]
    lc = row[3] - row[4]
    return int(hc > b) + int(ho < hc) + int(lc > b)


def pct(x, n):
    return x / n * 100.0 if n else 0.0


def run(rows):
    out = {}
    for score in (0, 3):
        n = correct = 0
        ret_long = ret_short = 0.0
        long_n = short_n = 0
        for i in range(len(rows) - 1):
            if ad(rows[i]) != score:
                continue
            b = body(rows[i + 1])
            if abs(b) < 1e-12:
                continue
            n += 1
            predicted_side = "LONG" if score == 0 else "SHORT"
            actual_side = "LONG" if b > 0 else "SHORT"
            correct += int(predicted_side == actual_side)
            o, c = rows[i + 1][1], rows[i + 1][4]
            r = c / o - 1.0 if predicted_side == "LONG" else 1.0 - c / o
            if predicted_side == "LONG":
                ret_long += r
                long_n += 1
            else:
                ret_short += r
                short_n += 1
        out[score] = (n, correct, pct(correct, n), ret_long + ret_short, long_n + short_n)

    n = correct = 0
    signed_move_correct = 0
    for i in range(1, len(rows) - 1):
        score = ad(rows[i])
        if score not in (0, 3):
            continue
        cur = body(rows[i])
        nxt = body(rows[i + 1])
        if abs(cur) < 1e-12 or abs(nxt) < 1e-12:
            continue
        n += 1
        m = 1 if nxt > cur else -1 if nxt < cur else 0
        pred_m = -1 if score == 0 else 1
        signed_move_correct += int(m == pred_m)

    return out, (n, signed_move_correct, pct(signed_move_correct, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    args = ap.parse_args()

    for symbol in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        rows = load(Path(args.input_dir) / f"{symbol}_1h.csv")
        result, move = run(rows)
        print(f"=== {symbol} rows={len(rows)} ===")
        for score in (3, 0):
            n, correct, acc, gross_sum, trades = result[score]
            side = "SHORT" if score == 3 else "LONG"
            print(f"AD={score} -> {side}: N={n} BODY_DIR={acc:.2f}% NEXT_OPEN_CLOSE_SUM={gross_sum*100:.2f}%")
        print(f"82pct_reproduction_signed_body_move: N={move[0]} ACC={move[2]:.2f}%")


if __name__ == "__main__":
    main()
