from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from statistics import median


def num(v: str | None) -> float:
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def stats(values: list[float], side: str, fee: float) -> dict:
    n = len(values)
    if not n or side == "FLAT":
        return {"n": n, "avg": 0.0, "med": 0.0, "net": 0.0, "fav": 0.0}
    signed = values if side == "LONG" else [-x for x in values]
    return {
        "n": n,
        "avg": sum(signed) / n,
        "med": median(signed),
        "net": sum(signed) / n - fee,
        "fav": 100.0 * sum(x >= fee for x in signed) / n,
    }


def bucket(value: float, cuts: list[float]) -> str:
    for c in sorted(cuts, reverse=True):
        if value >= c:
            label = str(c).replace(".", "p")
            return f">={label}"
    label = str(min(cuts)).replace(".", "p")
    return f"<{label}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine 1H direction patterns conditioned on current candle strength.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--min-month-count", type=int, default=8)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--fee-roundtrip-pct", type=float, default=0.26)
    args = ap.parse_args()

    symbols = {x.strip().upper() for x in args.symbols.split(",") if x.strip()}
    rows: list[tuple[str, str, str, float, float, float, float]] = []
    with open(args.input, "r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            sym = (r.get("symbol") or "").upper()
            if sym not in symbols:
                continue
            code = (r.get("seq_5_past_code") or r.get("four_direction_past_code") or "").strip()
            ret4 = num(r.get("future_4h_return_pct") or r.get("future_4_return_pct"))
            month = (r.get("time_utc") or r.get("timestamp_utc") or "")[:7]
            o, h, l, c = num(r.get("open")), num(r.get("high")), num(r.get("low")), num(r.get("close"))
            if not code or len(code) != 5 or any(x not in "UDF" for x in code) or not month or o <= 0:
                continue
            body_pct = abs(c - o) / o * 100.0
            range_pct = max(0.0, h - l) / o * 100.0
            close_pos = (c - l) / (h - l) if h > l else 0.5
            rows.append((sym, month, code, ret4, body_pct, range_pct, close_pos))

    groups = defaultdict(list)
    for sym, month, code, ret, body, rng, pos in rows:
        body_b = bucket(body, [0.10, 0.20, 0.50])
        range_b = bucket(rng, [0.20, 0.50, 1.00])
        pos_b = "LOW" if pos < 0.35 else "MID" if pos < 0.65 else "HIGH"
        groups[(code, body_b, range_b, pos_b)].append((sym, month, ret))

    print(f"PATTERN_STRENGTH rows={len(rows)} symbols={','.join(sorted(symbols))} fee={args.fee_roundtrip_pct:.2f}%")
    print("condition = 5-direction code + current candle body/range/close-position")
    print("PATTERN BODY RANGE POS SIDE N AVG4H MED4H NET FAV% POS_MONTHS")
    print("------- ---- ----- --- ---- --- ----- ----- ---- ----- ---------")

    ranked = []
    months_all = sorted({x[1] for x in rows})
    for (code, body_b, range_b, pos_b), vals in groups.items():
        if len(vals) < args.min_count:
            continue
        raw = [x[2] for x in vals]
        mean_raw = sum(raw) / len(raw)
        side = "LONG" if mean_raw > 0 else "SHORT" if mean_raw < 0 else "FLAT"
        if side == "FLAT":
            continue
        s = stats(raw, side, args.fee_roundtrip_pct)
        pos_months = 0
        month_parts = []
        for m in months_all:
            mv = [x[2] for x in vals if x[1] == m]
            if len(mv) < args.min_month_count:
                continue
            ms = stats(mv, side, args.fee_roundtrip_pct)
            pos_months += int(ms["net"] > 0)
            month_parts.append(f"{m}:{len(mv)}/{ms['net']:+.2f}")
        if pos_months < 2 or s["net"] <= 0:
            continue
        score = s["net"] * (s["fav"] / 100.0) * (1 + 0.25 * pos_months)
        ranked.append((score, code, body_b, range_b, pos_b, side, s, pos_months, month_parts))

    ranked.sort(key=lambda x: (-x[0], -x[6]["net"], -x[6]["fav"], x[1]))
    for score, code, body_b, range_b, pos_b, side, s, pos_months, month_parts in ranked[: args.top]:
        print(f"{code:>7} {body_b:>5} {range_b:>5} {pos_b:>3} {side:>5} {s['n']:>3} {s['avg']:>+5.3f} {s['med']:>+5.3f} {s['net']:>+5.3f} {s['fav']:>5.1f}% {pos_months:>3}/{len(month_parts):<3} {' '.join(month_parts)}")

    print("TOP_ACTIONABLE")
    if not ranked:
        print("NONE")
    else:
        for _, code, body_b, range_b, pos_b, side, s, pos_months, _ in ranked[:20]:
            print(f"{code}|BODY={body_b}|RANGE={range_b}|POS={pos_b} -> {side} n={s['n']} avg={s['avg']:+.3f}% net={s['net']:+.3f}% fav={s['fav']:.1f}% positive_months={pos_months}")


if __name__ == "__main__":
    main()
