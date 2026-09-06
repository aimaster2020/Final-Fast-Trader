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


def body_bucket(v: float) -> str:
    if v >= 0.50:
        return ">=0p5"
    if v >= 0.20:
        return ">=0p2"
    if v >= 0.10:
        return ">=0p1"
    return "<0p1"


def range_bucket(v: float) -> str:
    if v >= 1.00:
        return ">=1p0"
    if v >= 0.50:
        return ">=0p5"
    if v >= 0.20:
        return ">=0p2"
    return "<0p2"


def condition(row: dict) -> tuple[str, str, str, str]:
    o, h, l, c = (num(row.get(k)) for k in ("open", "high", "low", "close"))
    body = abs(c - o) / o * 100.0 if o > 0 else 0.0
    rng = max(0.0, h - l) / o * 100.0 if o > 0 else 0.0
    pos = (c - l) / (h - l) if h > l else 0.5
    pos_b = "LOW" if pos < 0.35 else "MID" if pos < 0.65 else "HIGH"
    return (
        (row.get("seq_5_past_code") or row.get("four_direction_past_code") or "").strip(),
        body_bucket(body),
        range_bucket(rng),
        pos_b,
    )


def signed_stats(vals: list[float], side: str, fee: float) -> tuple[float, float, float, float]:
    signed = vals if side == "LONG" else [-x for x in vals]
    avg = sum(signed) / len(signed)
    med = median(signed)
    net = avg - fee
    fav = 100.0 * sum(x >= fee for x in signed) / len(signed)
    return avg, med, net, fav


def load(path: str, symbols: set[str]) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if (r.get("symbol") or "").upper() not in symbols:
                continue
            r["month"] = (r.get("time_utc") or r.get("timestamp_utc") or "")[:7]
            code = (r.get("seq_5_past_code") or r.get("four_direction_past_code") or "").strip()
            if len(code) != 5 or any(x not in "UDF" for x in code):
                continue
            ret = r.get("future_4h_return_pct") or r.get("future_4_return_pct")
            if not r["month"] or ret is None:
                continue
            r["ret4"] = num(ret)
            r["cond"] = condition(r)
            out.append(r)
    return out


def discover(train_rows: list[dict], min_count: int, min_month_count: int, fee: float, min_pos_months: int) -> list[dict]:
    groups: defaultdict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for r in train_rows:
        groups[r["cond"]].append(r)
    months = sorted({r["month"] for r in train_rows})
    candidates = []
    for cond, rs in groups.items():
        if len(rs) < min_count:
            continue
        raw = [r["ret4"] for r in rs]
        mean_raw = sum(raw) / len(raw)
        side = "LONG" if mean_raw > 0 else "SHORT" if mean_raw < 0 else "FLAT"
        if side == "FLAT":
            continue
        avg, med, net, fav = signed_stats(raw, side, fee)
        if net <= 0:
            continue
        pos_months = 0
        for m in months:
            vals = [r["ret4"] for r in rs if r["month"] == m]
            if len(vals) < min_month_count:
                continue
            mnet = signed_stats(vals, side, fee)[2]
            pos_months += int(mnet > 0)
        if pos_months < min_pos_months:
            continue
        candidates.append({
            "cond": cond,
            "side": side,
            "train_n": len(rs),
            "train_avg": avg,
            "train_med": med,
            "train_net": net,
            "train_fav": fav,
            "train_pos_months": pos_months,
        })
    candidates.sort(key=lambda x: (-x["train_net"], -x["train_fav"], -x["train_n"]))
    return candidates


def evaluate(test_rows: list[dict], candidate: dict, fee: float) -> dict:
    rs = [r for r in test_rows if r["cond"] == candidate["cond"]]
    side = candidate["side"]
    vals = [r["ret4"] for r in rs]
    if not vals:
        return {"n": 0, "avg": 0.0, "net": 0.0, "fav": 0.0, "win": 0.0}
    avg, med, net, fav = signed_stats(vals, side, fee)
    win = 100.0 * sum((x >= fee if side == "LONG" else x <= -fee) for x in vals) / len(vals)
    return {"n": len(vals), "avg": avg, "net": net, "fav": fav, "win": win}


def main() -> None:
    ap = argparse.ArgumentParser(description="Leave-one-month-out validation for 1H direction+strength patterns.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--months", default="2026-05,2026-06,2026-07,2026-08")
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--min-month-count", type=int, default=8)
    ap.add_argument("--min-pos-months", type=int, default=2)
    ap.add_argument("--fee-roundtrip-pct", type=float, default=0.26)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    symbols = {x.strip().upper() for x in args.symbols.split(",") if x.strip()}
    months = [x.strip() for x in args.months.split(",") if x.strip()]
    rows = load(args.input, symbols)
    print(f"OOS_PATTERN_STRENGTH rows={len(rows)} fee={args.fee_roundtrip_pct:.2f}% symbols={','.join(sorted(symbols))}")
    print("RULE TEST_MONTH TRAIN_N TRAIN_NET TEST_N TEST_AVG TEST_NET TEST_WIN%")
    print("---- ---------- -------- --------- ------ -------- -------- --------")

    all_oos = []
    for target in months:
        train = [r for r in rows if r["month"] != target]
        test = [r for r in rows if r["month"] == target]
        candidates = discover(train, args.min_count, args.min_month_count, args.fee_roundtrip_pct, args.min_pos_months)
        tested = []
        for c in candidates:
            ev = evaluate(test, c, args.fee_roundtrip_pct)
            if ev["n"] == 0:
                continue
            row = {**c, "test_month": target, **ev}
            tested.append(row)
            all_oos.append(row)
        tested.sort(key=lambda x: (-x["net"], -x["win"], -x["n"]))
        print(f"TARGET {target} candidates={len(candidates)} matched={len(tested)}")
        for r in tested[: args.top]:
            print(f"{r['cond'][0]}|BODY={r['cond'][1]}|RANGE={r['cond'][2]}|POS={r['cond'][3]} {r['side']:>5} {target} {r['train_n']:>8} {r['train_net']:>+9.3f} {r['n']:>6} {r['avg']:>+8.3f} {r['net']:>+8.3f} {r['win']:>8.1f}")

    print("OOS_SUMMARY")
    durable = defaultdict(list)
    for r in all_oos:
        if r["n"] >= 5:
            durable[(r["cond"], r["side"])].append(r)
    ranked = []
    for key, rs in durable.items():
        pos = sum(r["net"] > 0 for r in rs)
        mean = sum(r["net"] for r in rs) / len(rs)
        total_n = sum(r["n"] for r in rs)
        ranked.append((pos, mean, total_n, key, rs))
    for pos, mean, total_n, key, rs in sorted(ranked, key=lambda x: (-x[0], -x[1], -x[2]))[: args.top]:
        print(f"{key[0]}|BODY={key[1]}|RANGE={key[2]}|POS={key[3]} side={rs[0]['side']} positive_targets={pos}/{len(rs)} mean_oos_net={mean:+.3f}% total_test_n={total_n}")


if __name__ == "__main__":
    main()
