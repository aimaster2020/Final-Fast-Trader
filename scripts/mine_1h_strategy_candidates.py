from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median


def num(v: str | None) -> float:
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def stats(values: list[float], side: str, fee: float) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "fav": 0, "fav_pct": 0.0, "avg": 0.0, "med": 0.0, "net_avg": 0.0}
    if side == "LONG":
        fav = sum(x >= fee for x in values)
        avg = sum(values) / n
        med = median(values)
        net_avg = avg - fee
    else:
        fav = sum(x <= -fee for x in values)
        avg = -sum(values) / n
        med = median([-x for x in values])
        net_avg = avg - fee
    return {"n": n, "fav": fav, "fav_pct": 100.0 * fav / n, "avg": avg, "med": med, "net_avg": net_avg}


def main() -> None:
    ap = argparse.ArgumentParser(description="Find directional 1H pattern candidates with fee-aware and month-stable edge.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--min-count", type=int, default=30)
    ap.add_argument("--min-month-count", type=int, default=10)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--fee-roundtrip-pct", type=float, default=0.26)
    args = ap.parse_args()

    symbols = {x.strip().upper() for x in args.symbols.split(",") if x.strip()}
    rows: list[tuple[str, str, str, float]] = []
    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            sym = (r.get("symbol") or "").upper()
            if sym not in symbols:
                continue
            code = (r.get("seq_5_past_code") or r.get("four_direction_past_code") or "").strip()
            ret = num(r.get("future_4h_return_pct") or r.get("future_4_return_pct"))
            month = (r.get("time_utc") or r.get("timestamp_utc") or "")[:7]
            if len(code) == 5 and all(c in "UDF" for c in code) and month:
                rows.append((sym, month, code, ret))

    by_code: defaultdict[str, list[float]] = defaultdict(list)
    by_code_month: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for sym, month, code, ret in rows:
        by_code[code].append(ret)
        by_code_month[(code, month)].append(ret)

    print(f"STRATEGY_CANDIDATES rows={len(rows)} symbols={','.join(sorted(symbols))} fee_roundtrip={args.fee_roundtrip_pct:.2f}%")
    print("Direction is chosen from the sign of mean 4H return; fee is charged against that directional return.")
    print("CODE SIDE N AVG4H MED4H NET_AFTER_FEE FAV@FEE FAV@0.50 FAV@1.00 POS_MONTHS MONTHS")
    print("---- ---- --- ------ ------ ------------ ------- --------- --------- ----------- ------")

    ranked = []
    for code, vals in by_code.items():
        if len(vals) < args.min_count:
            continue
        mean_raw = sum(vals) / len(vals)
        side = "LONG" if mean_raw > 0 else "SHORT" if mean_raw < 0 else "FLAT"
        if side == "FLAT":
            continue
        s = stats(vals, side, args.fee_roundtrip_pct)
        f05 = stats(vals, side, 0.50)["fav_pct"]
        f10 = stats(vals, side, 1.00)["fav_pct"]
        months = sorted({m for _, m, c, _ in rows if c == code})
        good_months = 0
        month_edges: list[str] = []
        stable_net = 0
        for month in months:
            mv = by_code_month[(code, month)]
            if len(mv) < args.min_month_count:
                continue
            ms = stats(mv, side, args.fee_roundtrip_pct)
            good_months += int(ms["net_avg"] > 0)
            stable_net += int(ms["net_avg"] > 0)
            month_edges.append(f"{month}:{len(mv)}/{ms['fav_pct']:.0f}%/{ms['net_avg']:+.2f}")
        score = s["net_avg"] * max(s["fav_pct"], 0.0) * (1.0 + 0.25 * stable_net)
        ranked.append((score, code, side, s, f05, f10, good_months, len(month_edges), month_edges))

    ranked.sort(key=lambda x: (-x[0], -x[3]["net_avg"], -x[3]["fav_pct"], x[1]))
    for score, code, side, s, f05, f10, good_months, month_n, month_edges in ranked[:args.top]:
        print(f"{code:>4} {side:>4} {s['n']:>3} {s['avg']:>+6.3f} {s['med']:>+6.3f} {s['net_avg']:>+12.3f} {s['fav_pct']:>7.1f}% {f05:>8.1f}% {f10:>8.1f}% {good_months:>5}/{month_n:<5} {' '.join(month_edges)}")

    print("TOP_ACTIONABLE")
    actionable = [r for r in ranked if r[3]["net_avg"] > 0 and r[6] >= 2]
    for score, code, side, s, f05, f10, good_months, month_n, _ in actionable[:args.top]:
        print(f"{code} -> {side} n={s['n']} avg={s['avg']:+.3f}% net={s['net_avg']:+.3f}% fav_fee={s['fav_pct']:.1f}% pos_months={good_months}/{month_n}")
    if not actionable:
        print("NONE")


if __name__ == "__main__":
    main()
