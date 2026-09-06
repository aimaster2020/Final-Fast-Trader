from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


RULES = [
    ("UUUUD", ">=0p1", ">=0p2", "MID", "SHORT"),
    ("DDUUD", ">=0p1", ">=0p2", "MID", "SHORT"),
    ("DUDDD", "<0p5", ">=0p2", "MID", "LONG"),
    ("UUDDD", "<0p5", ">=0p2", "LOW", "LONG"),
    ("UDUDU", ">=0p1", ">=0p2", "MID", "LONG"),
]


def num(v: str | None) -> float:
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def body_bucket(body: float) -> str:
    if body >= 0.50:
        return ">=0p5"
    if body >= 0.20:
        return ">=0p2"
    if body >= 0.10:
        return ">=0p1"
    return "<0p1"


def range_bucket(rng: float) -> str:
    if rng >= 1.00:
        return ">=1p0"
    if rng >= 0.50:
        return ">=0p5"
    if rng >= 0.20:
        return ">=0p2"
    return "<0p2"


def matches(r: dict, rule) -> bool:
    code, body_b, range_b, pos_b, side = rule
    return (
        (r.get("seq_5_past_code") or "") == code
        and (r.get("body_bucket") or "") == body_b
        and (r.get("range_bucket") or "") == range_b
        and (r.get("pos_bucket") or "") == pos_b
        and side in {"LONG", "SHORT"}
    )


def backtest(rows: list[dict], rule, fee: float, capital: float, horizon: int) -> dict:
    code, body_b, range_b, pos_b, side = rule
    by_symbol: defaultdict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_symbol[(r.get("symbol") or "").upper()].append(r)

    start_cap = capital
    trades = wins = 0
    fees = 0.0
    per_month: defaultdict[str, list[float]] = defaultdict(list)

    for symbol, rs in by_symbol.items():
        rs.sort(key=lambda x: x.get("timestamp_ms") or x.get("time_utc") or "")
        pos_until = -1
        for i, r in enumerate(rs):
            if i <= pos_until:
                continue
            if not matches(r, rule):
                continue
            j = i + horizon
            if j >= len(rs):
                continue
            entry = num(r.get("close"))
            exit_px = num(rs[j].get("close"))
            if entry <= 0 or exit_px <= 0:
                continue
            gross_ret = (exit_px / entry - 1.0) if side == "LONG" else (entry / exit_px - 1.0)
            gross_pct = gross_ret * 100.0
            fee_amt = capital * fee / 100.0
            capital = max(0.0, capital - fee_amt)
            fees += fee_amt
            pnl = capital * gross_ret
            capital = max(0.0, capital + pnl)
            fee_amt = capital * fee / 100.0
            capital = max(0.0, capital - fee_amt)
            fees += fee_amt
            net_pct_on_start = gross_pct - fee
            trades += 1
            wins += int(gross_pct > fee)
            month = (r.get("time_utc") or r.get("timestamp_utc") or "")[:7]
            per_month[month].append(net_pct_on_start)
            pos_until = j
            if capital <= 0:
                break

    monthly = {}
    for m, vals in sorted(per_month.items()):
        monthly[m] = {
            "trades": len(vals),
            "net_sum_pct": sum(vals),
            "avg_net_pct": sum(vals) / len(vals) if vals else 0.0,
            "positive": sum(v > 0 for v in vals),
        }
    return {
        "final": capital,
        "return_pct": (capital / start_cap - 1.0) * 100.0,
        "trades": trades,
        "wins": wins,
        "win_rate": wins / trades * 100.0 if trades else 0.0,
        "fees": fees,
        "monthly": monthly,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--fee-roundtrip-pct", type=float, default=0.26)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--horizon", type=int, default=4)
    args = ap.parse_args()

    rows = []
    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            o, h, l, c = num(r.get("open")), num(r.get("high")), num(r.get("low")), num(r.get("close"))
            if o <= 0:
                continue
            rng = max(0.0, h - l) / o * 100.0
            body = abs(c - o) / o * 100.0
            pos = (c - l) / (h - l) if h > l else 0.5
            r["body_bucket"] = body_bucket(body)
            r["range_bucket"] = range_bucket(rng)
            r["pos_bucket"] = "LOW" if pos < 0.35 else "MID" if pos < 0.65 else "HIGH"
            rows.append(r)

    print(f"PATTERN_STRENGTH_BACKTEST rows={len(rows)} fee_roundtrip={args.fee_roundtrip_pct:.2f}% horizon={args.horizon}h")
    print("RULE SIDE TRADES WIN% RETURN% FINAL FEES POS_MONTHS MONTHS")
    for rule in RULES:
        r = backtest(rows, rule, args.fee_roundtrip_pct, args.initial_capital, args.horizon)
        pos_months = sum(1 for x in r["monthly"].values() if x["net_sum_pct"] > 0)
        month_txt = " ".join(f"{m}:{v['trades']}/{v['net_sum_pct']:+.2f}" for m, v in r["monthly"].items())
        print(f"{rule[0]} {rule[4]:>5} {r['trades']:>6} {r['win_rate']:>5.1f} {r['return_pct']:>+7.2f} {r['final']:>8.2f} {r['fees']:>7.2f} {pos_months:>3} {month_txt}")


if __name__ == "__main__":
    main()
