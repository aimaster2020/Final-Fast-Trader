from __future__ import annotations

import argparse
import csv
from pathlib import Path

EPS = 1e-12
BINS = (0.0, 0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, float("inf"))


def load(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append(
                    (
                        str(r.get("timestamp", "")),
                        float(r["open"]),
                        float(r["high"]),
                        float(r["low"]),
                        float(r["close"]),
                    )
                )
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


def gap_pct(cur, nxt):
    if abs(cur[4]) <= EPS:
        return 0.0
    return (nxt[1] - cur[4]) / cur[4]


def bucket(x):
    a = abs(x)
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        if lo <= a < hi:
            return f"{lo*100:.3f}-{hi*100:.3f}%" if hi != float("inf") else f">={lo*100:.3f}%"
    return "?"


def run(rows):
    out = {}
    for score in (3, 0):
        for sign_name in ("UP", "DOWN"):
            for bname in [bucket(BINS[i]) for i in range(len(BINS)-1)]:
                out[(score, sign_name, bname)] = [0, 0, 0.0]

        for i in range(len(rows) - 1):
            if ad(rows[i]) != score:
                continue
            nxt_body = body(rows[i + 1])
            if abs(nxt_body) <= EPS:
                continue
            g = gap_pct(rows[i], rows[i + 1])
            sign_name = "UP" if g >= 0 else "DOWN"
            bname = bucket(g)
            n, correct, ret_sum = out[(score, sign_name, bname)]
            predicted_side = "SHORT" if score == 3 else "LONG"
            actual_side = "LONG" if nxt_body > 0 else "SHORT"
            correct += int(predicted_side == actual_side)
            ret = nxt_body / rows[i + 1][1]
            ret_sum += ret if predicted_side == "LONG" else -ret
            out[(score, sign_name, bname)] = [n + 1, correct, ret_sum]
    return out


def main():
    ap = argparse.ArgumentParser(description="Condition candle-formula body direction on the observed next-open gap")
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--min-samples", type=int, default=30)
    args = ap.parse_args()

    for symbol in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        rows = load(Path(args.input_dir) / f"{symbol}_1h.csv")
        data = run(rows)
        print(f"=== {symbol} rows={len(rows)} ===")
        for score, side in ((3, "SHORT"), (0, "LONG")):
            print(f"AD={score} base={side}")
            for sign_name in ("UP", "DOWN"):
                for i in range(len(BINS) - 1):
                    bname = bucket(BINS[i])
                    n, correct, ret_sum = data[(score, sign_name, bname)]
                    if n < args.min_samples:
                        continue
                    acc = correct / n * 100.0
                    avg = ret_sum / n * 100.0
                    print(f"  GAP_{sign_name} {bname}: N={n} BODY_DIR={acc:.1f}% AVG_BODY_RET={avg:.3f}%")


if __name__ == "__main__":
    main()
