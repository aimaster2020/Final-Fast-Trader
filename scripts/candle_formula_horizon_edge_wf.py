from __future__ import annotations

import argparse
import csv
from pathlib import Path

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
HORIZONS = (1, 3, 6, 12, 24)
FEE_ROUNDTRIP_PCT = 0.26


def load(path: Path, symbol: str):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("symbol") != symbol:
                continue
            try:
                rows.append(
                    (
                        r["month"],
                        int(float(r["timestamp"])),
                        float(r["open"]),
                        float(r["high"]),
                        float(r["low"]),
                        float(r["close"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(rows, key=lambda x: x[1])


def score(row):
    _, _, o, h, l, c = row
    j = c - o
    k = h - c
    m = h - o
    return int(k > j) + int(k > m) + int(l > j)


def sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def evaluate(rows, horizon: int, rule: str):
    correct = 0
    n = 0
    gross_sum = 0.0
    net_sum = 0.0
    positive_net = 0
    by_s = {1: [0, 0, 0.0], 2: [0, 0, 0.0], 3: [0, 0, 0.0]}

    for i in range(len(rows) - horizon):
        cur = rows[i]
        nxt = rows[i + horizon]
        st = score(cur)
        if st not in by_s:
            continue

        if rule == "S13":
            pred = -1 if st == 1 else 1
        elif rule == "S1_SHORT_S3_LONG":
            pred = -1 if st == 1 else 1
        elif rule == "S1_LONG_S3_SHORT":
            pred = 1 if st == 1 else -1
        else:
            raise ValueError(rule)

        if cur[5] <= 0:
            continue
        ret = pred * (nxt[5] - cur[5]) / cur[5] * 100.0
        net = ret - FEE_ROUNDTRIP_PCT
        actual = sign(nxt[5] - cur[5])
        correct += int(pred == actual)
        n += 1
        gross_sum += ret
        net_sum += net
        positive_net += int(net > 0)
        z = by_s[st]
        z[0] += 1
        z[1] += int(pred == actual)
        z[2] += net

    return n, correct, gross_sum, net_sum, positive_net, by_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("reports/prepared_price_action_1h.csv"),
    )
    args = ap.parse_args()

    print("HORIZON_EDGE_WF | tf=1h | fee=0.13%/side | gross/net price edge | past_only")
    print("Tests S1/S3 at multiple holding horizons; S0 absent in current dataset. No optimization; fixed horizons only.")
    print("S13: S1=SHORT, S3=LONG. Reverse is the opposite mapping.")

    for symbol in SYMBOLS:
        rows = load(args.input, symbol)
        print(f"\n{symbol}")
        for horizon in HORIZONS:
            for rule in ("S13", "S1_LONG_S3_SHORT"):
                n, correct, gross, net, pos, by_s = evaluate(rows, horizon, rule)
                acc = 100.0 * correct / n if n else 0.0
                avg_gross = gross / n if n else 0.0
                avg_net = net / n if n else 0.0
                net_win = 100.0 * pos / n if n else 0.0
                label = "S13" if rule == "S13" else "REV"
                print(
                    f"  H={horizon:>2} {label:<3} n={n} acc={acc:.1f}% "
                    f"avg_gross={avg_gross:+.4f}% avg_net={avg_net:+.4f}% "
                    f"net_win={net_win:.1f}% sum_net={net:+.2f}%"
                )
            print()


if __name__ == "__main__":
    main()
