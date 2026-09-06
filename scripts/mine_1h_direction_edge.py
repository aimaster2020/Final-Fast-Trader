from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def f(v: str | None) -> float:
    try:
        return float(v or 0)
    except ValueError:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure 1H direction-pattern edge versus return thresholds, symbols and months.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--min-count", type=int, default=30)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--fee-roundtrip-pct", type=float, default=0.26)
    args = ap.parse_args()

    symbols = {x.strip().upper() for x in args.symbols.split(",") if x.strip()}
    thresholds = [args.fee_roundtrip_pct, 0.50, 1.00]
    data = []
    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("symbol", "").upper() not in symbols:
                continue
            code = (r.get("four_direction_past_code") or "").strip()
            if len(code) != 5 or any(c not in "UDF" for c in code):
                continue
            month = (r.get("timestamp_utc") or "")[:7]
            ret4 = f(r.get("future_4_return_pct"))
            data.append((r.get("symbol", "").upper(), month, code, ret4))

    groups = defaultdict(list)
    for sym, month, code, ret4 in data:
        groups[(sym, month, code)].append(ret4)
    combined = defaultdict(list)
    for (sym, month, code), vals in groups.items():
        combined[code].extend(vals)

    print(f"EDGE_MINING rows={len(data)} symbols={','.join(sorted(symbols))} min_count={args.min_count} roundtrip_fee={args.fee_roundtrip_pct:.2f}%")
    print("thresholds=%s" % ",".join(f"{x:.2f}%" for x in thresholds))
    print("CODE N EDGE@FEE EDGE@0.50 EDGE@1.00 AVG4H MEDIAN4H")
    print("---- --- -------- --------- -------- ------ ---------")

    ranked = []
    for code, vals in combined.items():
        if len(vals) < args.min_count:
            continue
        absvals = sorted(abs(x) for x in vals)
        med = absvals[len(absvals) // 2]
        avg = sum(vals) / len(vals)
        edges = []
        for th in thresholds:
            long_n = sum(x >= th for x in vals)
            short_n = sum(x <= -th for x in vals)
            best_side_n = max(long_n, short_n)
            edge = 100.0 * best_side_n / len(vals)
            edges.append(edge)
        score = edges[0] * max(abs(avg), 0.001)
        ranked.append((score, code, len(vals), *edges, avg, med))

    for row in sorted(ranked, reverse=True)[: args.top]:
        score, code, n, e0, e05, e1, avg, med = row
        print(f"{code:>4} {n:>3} {e0:>8.1f} {e05:>9.1f} {e1:>8.1f} {avg:>+6.3f} {med:>+8.3f}")

    print("MONTH_STABILITY_TOP")
    candidates = sorted(ranked, reverse=True)[: min(args.top, len(ranked))]
    for _, code, total, *rest in candidates:
        month_stats = []
        for month in sorted({m for _, m, c, _ in data if c == code}):
            vals = [r for _, m, c, r in data if m == month and c == code]
            if len(vals) < 10:
                continue
            long_n = sum(x >= args.fee_roundtrip_pct for x in vals)
            short_n = sum(x <= -args.fee_roundtrip_pct for x in vals)
            edge = max(long_n, short_n) / len(vals) * 100.0
            avg = sum(vals) / len(vals)
            month_stats.append(f"{month}:{len(vals)}/{edge:.0f}%/{avg:+.2f}")
        print(f"{code} total={total} " + " ".join(month_stats))


if __name__ == "__main__":
    main()
