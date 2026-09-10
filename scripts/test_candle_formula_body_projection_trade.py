from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median

EPS = 1e-12
LOOKBACK = 50
MIN_HISTORY = 15
CAPITAL = 1000.0


def load(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((str(r.get("timestamp", "")), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def body(r):
    return r[4] - r[1]


def ad(r):
    b = body(r)
    hc = r[2] - r[4]
    ho = r[2] - r[1]
    lc = r[3] - r[4]
    return int(hc > b) + int(ho < hc) + int(lc > b)


def signed_move_ratio(cur, nxt):
    if abs(cur) <= EPS:
        return None
    return (nxt - cur) / abs(cur)


def run(rows, threshold, score):
    capital = CAPITAL
    trades = 0
    wins = 0

    for i in range(1, len(rows) - 1):
        if ad(rows[i]) != score:
            continue
        cur = body(rows[i])
        if abs(cur) <= EPS:
            continue

        vals = []
        for j in range(max(1, i - LOOKBACK), i):
            if ad(rows[j]) != score:
                continue
            r = signed_move_ratio(body(rows[j]), body(rows[j + 1]))
            if r is not None:
                vals.append(r)
        if len(vals) < MIN_HISTORY:
            continue

        expected_move = median(vals)
        predicted_body = cur + abs(cur) * expected_move
        entry = rows[i + 1][1]
        exit_price = rows[i + 1][4]
        if entry <= EPS:
            continue

        pred_pct = abs(predicted_body) / entry
        if pred_pct < threshold:
            continue

        side = 1 if predicted_body > EPS else -1 if predicted_body < -EPS else 0
        if side == 0:
            continue

        gross = exit_price / entry - 1.0 if side > 0 else 1.0 - exit_price / entry
        capital *= 1.0 + gross
        trades += 1
        wins += int(gross > 0)

    return trades, wins, capital


def main():
    ap = argparse.ArgumentParser(description="Candle-formula body projection traded next open -> next close")
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--thresholds", default="0,0.0001,0.0002,0.0003,0.0005,0.00075,0.001")
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",")]
    for symbol in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        rows = load(Path(args.input_dir) / f"{symbol}_1h.csv")
        print(f"=== {symbol} rows={len(rows)} ===")
        for score, label in ((3, "AD3"), (0, "AD0")):
            best = None
            for t in thresholds:
                tr, w, final = run(rows, t, score)
                wr = w / tr * 100.0 if tr else 0.0
                ret = (final / CAPITAL - 1.0) * 100.0
                print(f"{label} T={t*100:.3f}% TR={tr} WR={wr:.1f}% RETURN={ret:.2f}% FINAL={final:.2f}")
                if best is None or final > best[0]:
                    best = (final, t, tr, wr, ret)
            if best:
                print(f"{label} BEST_ZERO_FEE T={best[1]*100:.3f}% TR={best[2]} WR={best[3]:.1f}% RETURN={best[4]:.2f}% FINAL={best[0]:.2f}")


if __name__ == "__main__":
    main()
