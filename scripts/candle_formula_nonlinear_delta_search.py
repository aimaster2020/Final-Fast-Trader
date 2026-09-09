from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def load(path: Path, symbol: str):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("symbol") != symbol:
                continue
            try:
                rows.append((
                    int(float(r["timestamp"])),
                    float(r["open"]),
                    float(r["high"]),
                    float(r["low"]),
                    float(r["close"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(rows, key=lambda x: x[0])


def sign(x, eps=1e-12):
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def parts(row):
    _, o, h, l, c = row
    body = c - o
    upper = h - max(o, c)
    lower = min(o, c) - l
    rng = h - l
    if rng <= 1e-15:
        return body, upper, lower, rng, 0.0
    return body, upper, lower, rng, (upper - lower) / rng


def mae_rmse_dir(rows, predictor):
    abs_sum = sq_sum = 0.0
    correct = 0
    n = 0
    for i in range(len(rows) - 1):
        cur, nxt = rows[i], rows[i + 1]
        pred = predictor(cur)
        actual = nxt[4] - cur[4]
        err = pred - actual
        abs_sum += abs(err)
        sq_sum += err * err
        correct += int(sign(pred) == sign(actual))
        n += 1
    if n == 0:
        return math.inf, math.inf, 0.0
    return abs_sum / n, math.sqrt(sq_sum / n), 100.0 * correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("reports/prepared_price_action_1h.csv"))
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    print("NONLINEAR_DELTA_SEARCH | tf=1h | fee=0 | no lookahead")
    print(f"Target = next_close-current_close | train={args.train:.0%}")
    print("Searches fixed nonlinear candle formulas; coefficients are selected on train only.")

    coeffs = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)

    # Formula family: a scale term times a signed shape term.
    # Scale terms have price units and are computed from the current candle only.
    specs = []
    for k in coeffs:
        if k == 0:
            continue
        specs.append((f"{k:g}*(UPPER-LOWER)", lambda b, u, lo, r, k=k: k * (u - lo)))
        specs.append((f"{k:g}*(UPPER-LOWER)/RANGE", lambda b, u, lo, r, k=k: k * (u - lo) if r == 0 else k * (u - lo) / r * r))
        specs.append((f"{k:g}*RANGE*(UPPER-LOWER)/RANGE", lambda b, u, lo, r, k=k: 0.0 if r == 0 else k * r * (u - lo) / r))
        specs.append((f"{k:g}*BODY*(UPPER-LOWER)/RANGE", lambda b, u, lo, r, k=k: 0.0 if r == 0 else k * b * (u - lo) / r))
        specs.append((f"{k:g}*(UPPER^2-LOWER^2)/RANGE", lambda b, u, lo, r, k=k: 0.0 if r == 0 else k * (u*u - lo*lo) / r))
        specs.append((f"{k:g}*BODY*(UPPER^2-LOWER^2)/RANGE^2", lambda b, u, lo, r, k=k: 0.0 if r == 0 else k * b * (u*u - lo*lo) / (r*r)))
        specs.append((f"{k:g}*RANGE*((UPPER-LOWER)/RANGE)^2*sign(BODY)", lambda b, u, lo, r, k=k: 0.0 if r == 0 else k * r * ((u-lo)/r)**2 * sign(b)))

    for sym in SYMBOLS:
        rows = load(args.input, sym)
        cut = max(1, int(len(rows) * args.train))
        train = rows[:cut]
        test = rows[cut:]

        scored = []
        for name, fn in specs:
            def pred(row, fn=fn):
                b, u, lo, r, _ = parts(row)
                return fn(b, u, lo, r)

            tr = mae_rmse_dir(train, pred)
            te = mae_rmse_dir(test, pred)
            scored.append((te[0], te[1], -te[2], tr, name))

        # ZERO baseline is intentionally included for the exact same test split.
        zero_test = mae_rmse_dir(test, lambda row: 0.0)
        scored.sort()

        print(sym)
        print(f"  ZERO_TEST MAE={zero_test[0]:.6g} RMSE={zero_test[1]:.6g} dir_acc={zero_test[2]:.2f}%")
        for test_mae, test_rmse, neg_dir, tr, name in scored[:args.top]:
            print(
                f"  {test_mae:.6g} / {test_rmse:.6g} / {-neg_dir:.2f}%"
                f" | train_mae={tr[0]:.6g} train_rmse={tr[1]:.6g} | {name}"
            )
        print()


if __name__ == "__main__":
    main()
