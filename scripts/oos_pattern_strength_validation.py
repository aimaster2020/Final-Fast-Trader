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
    code = (row.get("seq_5_past_code") or row.get("four_direction_past_code") or "").strip()
    return code, body_bucket(body), range_bucket(rng), pos_b


def signed_values(values: list[float], side: str) -> list[float]:
    return values if side == "LONG" else [-x for x in values]


def stats(values: list[float], side: str, fee: float) -> dict:
    signed = signed_values(values, side)
    n = len(signed)
    avg = sum(signed) / n if n else 0.0
    med = median(signed) if n else 0.0
    net = avg - fee
    wins = sum(x >= fee for x in signed)
    return {"n": n, "avg": avg, "med": med, "net": net, "wins": wins, "win_pct": 100.0 * wins / n if n else 0.0}


def load(path: str, symbols: set[str]) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if (r.get("symbol") or "").upper() not in symbols:
                continue
            month = (r.get("time_utc") or r.get("timestamp_utc") or "")[:7]
            code = (r.get("seq_5_past_code") or r.get("four_direction_past_code") or "").strip()
            ret = r.get("future_4h_return_pct") or r.get("future_4_return_pct")
            if len(code) != 5 or any(c not in "UDF" for c in code) or not month or ret is None:
                continue
            r["month"] = month
            r["ret4"] = num(ret)
            r["cond"] = condition(r)
            out.append(r)
    return out


def discover(train_rows: list[dict], min_count: int, min_month_count: int, fee: float, min_pos_months: int) -> list[dict]:
    groups: defaultdict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for r in train_rows:
        groups[r["cond"]].append(r)
    months = sorted({r["month"] for r in train_rows})
    candidates: list[dict] = []
    for cond, rs in groups.items():
        if len(rs) < min_count:
            continue
        raw = [r["ret4"] for r in rs]
        mean_raw = sum(raw) / len(raw)
        if mean_raw == 0:
            continue
        side = "LONG" if mean_raw > 0 else "SHORT"
        s = stats(raw, side, fee)
        if s["net"] <= 0:
            continue
        positive_months = 0
        for month in months:
            vals = [r["ret4"] for r in rs if r["month"] == month]
            if len(vals) < min_month_count:
                continue
            positive_months += int(stats(vals, side, fee)["net"] > 0)
        if positive_months < min_pos_months:
            continue
        candidates.append({
            "cond": cond,
            "side": side,
            "train_n": len(rs),
            "train_raw_avg": mean_raw,
            "train_avg": s["avg"],
            "train_net": s["net"],
            "train_win_pct": s["win_pct"],
            "train_positive_months": positive_months,
        })
    candidates.sort(key=lambda x: (-x["train_net"], -x["train_win_pct"], -x["train_n"]))
    return candidates


def evaluate(test_rows: list[dict], candidate: dict, fee: float) -> dict:
    matched = [r for r in test_rows if r["cond"] == candidate["cond"]]
    raw = [r["ret4"] for r in matched]
    s = stats(raw, candidate["side"], fee)
    return {"n": s["n"], "raw_avg": sum(raw) / len(raw) if raw else 0.0, "signed_avg": s["avg"], "net": s["net"], "win": s["win_pct"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict leave-one-month-out validation for 1H direction+strength patterns.")
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
    print("SIDE is selected from TRAIN mean 4H return. TEST_SIGNED_AVG is profit/loss for that side; TEST_NET=SIGNED_AVG-fee.")
    print("RULE TEST_MONTH SIDE TRAIN_N TRAIN_RAW TEST_N TEST_RAW TEST_SIGNED_AVG TEST_NET TEST_WIN%")
    print("---- ---------- ---- -------- --------- ------- -------- --------- -------- --------")

    all_oos: list[dict] = []
    for target in months:
        train = [r for r in rows if r["month"] != target]
        test = [r for r in rows if r["month"] == target]
        candidates = discover(train, args.min_count, args.min_month_count, args.fee_roundtrip_pct, args.min_pos_months)
        tested: list[dict] = []
        for c in candidates:
            ev = evaluate(test, c, args.fee_roundtrip_pct)
            if ev["n"] == 0:
                continue
            row = {**c, "test_month": target, **ev}
            tested.append(row)
            all_oos.append(row)
        tested.sort(key=lambda x: (-x["net"], -x["win"], -x["n"]))
        print(f"TARGET {target} candidates={len(candidates)} matched={len(tested)}")
        for r in tested[:args.top]:
            c = r["cond"]
            print(
                f"{c[0]}|BODY={c[1]}|RANGE={c[2]}|POS={c[3]} {r['side']:>5} {target} "
                f"{r['train_n']:>7} {r['train_raw_avg']:>+9.3f} {r['n']:>7} {r['raw_avg']:>+9.3f} "
                f"{r['signed_avg']:>+15.3f} {r['net']:>+9.3f} {r['win']:>9.1f}"
            )

    print("OOS_SUMMARY")
    grouped: defaultdict[tuple, list[dict]] = defaultdict(list)
    for r in all_oos:
        if r["n"] >= 5:
            grouped[(r["cond"], r["side"])].append(r)

    ranked = []
    for key, rs in grouped.items():
        positive_targets = sum(r["net"] > 0 for r in rs)
        total_n = sum(r["n"] for r in rs)
        weighted_signed_avg = sum(r["signed_avg"] * r["n"] for r in rs) / total_n if total_n else 0.0
        mean_oos_net = weighted_signed_avg - args.fee_roundtrip_pct
        ranked.append((positive_targets, mean_oos_net, total_n, key, len(rs)))

    for positive_targets, mean_oos_net, total_n, key, target_n in sorted(ranked, key=lambda x: (-x[0], -x[1], -x[2]))[:args.top]:
        cond, side = key
        print(
            f"{cond[0]}|BODY={cond[1]}|RANGE={cond[2]}|POS={cond[3]} side={side} "
            f"positive_targets={positive_targets}/{target_n} mean_oos_net={mean_oos_net:+.3f}% total_test_n={total_n}"
        )


if __name__ == "__main__":
    main()
