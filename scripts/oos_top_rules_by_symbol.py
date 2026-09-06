from __future__ import annotations

import argparse
import csv
from collections import defaultdict


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


def signed_avg(vals: list[float], side: str) -> float:
    if not vals:
        return 0.0
    if side == "SHORT":
        return -sum(vals) / len(vals)
    return sum(vals) / len(vals)


def load(path: str, symbols: set[str]) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            symbol = (r.get("symbol") or "").upper()
            if symbol not in symbols:
                continue
            month = (r.get("time_utc") or r.get("timestamp_utc") or "")[:7]
            ret = r.get("future_4h_return_pct") or r.get("future_4_return_pct")
            code = (r.get("seq_5_past_code") or r.get("four_direction_past_code") or "").strip()
            if not month or ret is None or len(code) != 5 or any(c not in "UDF" for c in code):
                continue
            r["symbol"] = symbol
            r["month"] = month
            r["ret4"] = num(ret)
            r["cond"] = condition(r)
            out.append(r)
    return out


def discover(train_rows: list[dict], min_count: int, min_month_count: int, fee: float, min_pos_months: int) -> list[dict]:
    groups: defaultdict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for r in train_rows:
        groups[r["cond"]].append(r)

    candidates: list[dict] = []
    for cond, rs in groups.items():
        if len(rs) < min_count:
            continue
        raw = [r["ret4"] for r in rs]
        mean_raw = sum(raw) / len(raw)
        side = "LONG" if mean_raw > 0 else "SHORT" if mean_raw < 0 else "FLAT"
        if side == "FLAT":
            continue
        train_signed = signed_avg(raw, side)
        train_net = train_signed - fee
        if train_net <= 0:
            continue

        positive_months = 0
        months = sorted({r["month"] for r in rs})
        for m in months:
            vals = [r["ret4"] for r in rs if r["month"] == m]
            if len(vals) < min_month_count:
                continue
            positive_months += int(signed_avg(vals, side) - fee > 0)
        if positive_months < min_pos_months:
            continue

        candidates.append({
            "cond": cond,
            "side": side,
            "train_n": len(rs),
            "train_raw": mean_raw,
            "train_signed": train_signed,
            "train_net": train_net,
            "positive_months": positive_months,
        })

    candidates.sort(key=lambda x: (-x["train_net"], -x["train_n"]))
    return candidates


def evaluate(test_rows: list[dict], candidate: dict, fee: float) -> dict:
    vals = [r["ret4"] for r in test_rows if r["cond"] == candidate["cond"]]
    if not vals:
        return {"n": 0, "raw": 0.0, "signed": 0.0, "net": 0.0, "win": 0.0}
    side = candidate["side"]
    signed_vals = vals if side == "LONG" else [-x for x in vals]
    signed = sum(signed_vals) / len(signed_vals)
    net = signed - fee
    win = 100.0 * sum(x >= fee for x in signed_vals) / len(signed_vals)
    return {
        "n": len(vals),
        "raw": sum(vals) / len(vals),
        "signed": signed,
        "net": net,
        "win": win,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict LOMO OOS test of the two leading pattern-strength rules, separately per symbol.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--months", default="2026-05,2026-06,2026-07,2026-08")
    ap.add_argument("--min-count", type=int, default=8)
    ap.add_argument("--min-month-count", type=int, default=4)
    ap.add_argument("--min-pos-months", type=int, default=2)
    ap.add_argument("--fee-roundtrip-pct", type=float, default=0.26)
    args = ap.parse_args()

    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    months = [x.strip() for x in args.months.split(",") if x.strip()]
    rows = load(args.input, set(symbols))

    targets = {
        ("UUUUU", ">=0p5", ">=1p0", "HIGH"): "UUUUU_HIGH_SHORT",
        ("DUDDD", ">=0p5", ">=1p0", "LOW"): "DUDDD_LOW_LONG",
    }

    print(f"SYMBOL_OOS_TOP_RULES rows={len(rows)} fee={args.fee_roundtrip_pct:.2f}%")
    print("TESTS are selected from the other 3 months for the same symbol only.")
    print("RULE SYMBOL MONTH SIDE TRAIN_N TRAIN_NET TEST_N TEST_RAW TEST_SIGNED TEST_NET WIN%")
    print("---- ------ ----- ---- -------- -------- ------- --------- ----------- -------- -----")

    all_rows: list[dict] = []
    for symbol in symbols:
        symbol_rows = [r for r in rows if r["symbol"] == symbol]
        for target_month in months:
            train = [r for r in symbol_rows if r["month"] != target_month]
            test = [r for r in symbol_rows if r["month"] == target_month]
            discovered = {c["cond"]: c for c in discover(train, args.min_count, args.min_month_count, args.fee_roundtrip_pct, args.min_pos_months)}

            for cond, label in targets.items():
                candidate = discovered.get(cond)
                if candidate is None:
                    continue
                ev = evaluate(test, candidate, args.fee_roundtrip_pct)
                if ev["n"] == 0:
                    continue
                row = {**candidate, **ev, "symbol": symbol, "month": target_month, "label": label}
                all_rows.append(row)
                print(
                    f"{label:18} {symbol:7} {target_month} {candidate['side']:>5} "
                    f"{candidate['train_n']:>7} {candidate['train_net']:>+8.3f} "
                    f"{ev['n']:>7} {ev['raw']:>+9.3f} {ev['signed']:>+11.3f} "
                    f"{ev['net']:>+8.3f} {ev['win']:>5.1f}"
                )

    print("SYMBOL_SUMMARY")
    grouped: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_rows:
        grouped[(r["label"], r["symbol"])].append(r)

    for (label, symbol), rs in grouped.items():
        positive = sum(r["net"] > 0 for r in rs)
        n = sum(r["n"] for r in rs)
        weighted_signed = sum(r["signed"] * r["n"] for r in rs) / n if n else 0.0
        weighted_net = weighted_signed - args.fee_roundtrip_pct
        print(f"{label:18} {symbol:7} positive_months={positive}/{len(rs)} mean_oos_net={weighted_net:+.3f}% total_test_n={n}")

    print("RULE_SUMMARY")
    for label in targets.values():
        rs = [r for r in all_rows if r["label"] == label]
        positive_symbols = 0
        for symbol in symbols:
            srs = [r for r in rs if r["symbol"] == symbol]
            if srs and sum(r["net"] > 0 for r in srs) >= 2:
                positive_symbols += 1
        n = sum(r["n"] for r in rs)
        weighted_signed = sum(r["signed"] * r["n"] for r in rs) / n if n else 0.0
        weighted_net = weighted_signed - args.fee_roundtrip_pct
        print(f"{label:18} symbols_ge2_positive_months={positive_symbols}/{len(symbols)} mean_oos_net={weighted_net:+.3f}% total_test_n={n}")


if __name__ == "__main__":
    main()
