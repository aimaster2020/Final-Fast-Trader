from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def f(v: str | None) -> float:
    try:
        return float(v or 0)
    except (ValueError, TypeError):
        return 0.0


def pick(r: dict, *names: str) -> str:
    for name in names:
        value = r.get(name)
        if value not in (None, ""):
            return str(value).strip()
    return ""


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
    total_rows = symbol_rows = valid_code_rows = return_rows = 0

    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        print("INPUT_COLUMNS " + ",".join(fields))
        for r in reader:
            total_rows += 1
            sym = pick(r, "symbol").upper()
            if sym not in symbols:
                continue
            symbol_rows += 1

            code = pick(r, "four_direction_past_code", "seq_5_past_code", "past_code")
            if len(code) != 5 or any(c not in "UDF" for c in code):
                continue
            valid_code_rows += 1

            timestamp = pick(r, "timestamp_utc", "time_utc")
            month = timestamp[:7]
            ret4_raw = pick(r, "future_4_return_pct", "future_4h_return_pct")
            try:
                ret4 = float(ret4_raw)
            except (ValueError, TypeError):
                continue
            return_rows += 1
            data.append((sym, month, code, ret4))

    groups = defaultdict(list)
    for sym, month, code, ret4 in data:
        groups[(sym, month, code)].append(ret4)
    combined = defaultdict(list)
    for (_sym, _month, code), vals in groups.items():
        combined[code].extend(vals)

    print(
        f"EDGE_MINING rows={len(data)} raw_rows={total_rows} symbol_rows={symbol_rows} "
        f"valid_codes={valid_code_rows} valid_returns={return_rows} "
        f"symbols={','.join(sorted(symbols))} min_count={args.min_count} "
        f"roundtrip_fee={args.fee_roundtrip_pct:.2f}%"
    )
    print("thresholds=%s" % ",".join(f"{x:.2f}%" for x in thresholds))
    print("CODE N EDGE@FEE EDGE@0.50 EDGE@1.00 AVG4H MEDIAN4H")
    print("---- --- -------- --------- -------- ------ ---------")

    ranked = []
    for code, vals in combined.items():
        if len(vals) < args.min_count:
            continue
        sorted_abs = sorted(abs(x) for x in vals)
        med = sorted_abs[len(sorted_abs) // 2]
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

    print("TOP_DIRECTIONAL_PATTERNS")
    for _, code, total, e0, e05, e1, avg, med in sorted(ranked, reverse=True)[: args.top]:
        long_n = sum(x >= args.fee_roundtrip_pct for x in combined[code])
        short_n = sum(x <= -args.fee_roundtrip_pct for x in combined[code])
        side = "LONG" if long_n > short_n else "SHORT" if short_n > long_n else "TIE"
        print(
            f"{code} -> {side:5} n={total:4d} edge_fee={max(long_n, short_n)/total*100:5.1f}% "
            f"edge_0.50={e05:5.1f}% edge_1.00={e1:5.1f}% avg4h={avg:+.3f}%"
        )

    print("MONTH_STABILITY_TOP")
    candidates = sorted(ranked, reverse=True)[: min(args.top, len(ranked))]
    months_all = sorted({m for _, m, _, _ in data if m})
    for _, code, total, *_rest in candidates:
        month_stats = []
        for month in months_all:
            vals = [ret for _, m, c, ret in data if m == month and c == code]
            if len(vals) < 10:
                continue
            long_n = sum(x >= args.fee_roundtrip_pct for x in vals)
            short_n = sum(x <= -args.fee_roundtrip_pct for x in vals)
            edge = max(long_n, short_n) / len(vals) * 100.0
            avg = sum(vals) / len(vals)
            side = "L" if long_n > short_n else "S" if short_n > long_n else "T"
            month_stats.append(f"{month}:{len(vals)}/{side}/{edge:.0f}%/{avg:+.2f}")
        print(f"{code} total={total} " + " ".join(month_stats))


if __name__ == "__main__":
    main()
