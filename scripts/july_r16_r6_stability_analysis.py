from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES = ROOT / "reports" / "july_r1_r16_single_nonoverlap_detail" / "r1_r16_single_nonoverlap_all_trades.csv"
RULES = [("R16", "N"), ("R6", "N")]
FEES = [0.0, 0.05]


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def week_label(ts: str) -> str:
    """Return ISO week for either epoch seconds/ms or ISO timestamp."""
    s = str(ts).strip()
    try:
        value = float(s)
        if value > 10_000_000_000:
            value /= 1000.0
        dt = datetime.fromtimestamp(value, tz=timezone.utc)
    except ValueError:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return f"W{dt.isocalendar().week:02d}"


def aggregate(rows: list[dict], fee_side: float) -> dict:
    if not rows:
        return {"t": 0, "wins": 0, "move": 0.0, "ret": 0.0, "final": 250.0}
    capital = 250.0
    wins = 0
    move_sum = 0.0
    for r in sorted(rows, key=lambda x: float(x["entry_ts"])):
        move = float(r["move_pct"]) / 100.0
        move_sum += move * 100.0
        wins += int(r["win"])
        capital = max(0.0, capital * (1.0 + move - 2.0 * fee_side / 100.0))
    return {
        "t": len(rows),
        "wins": wins,
        "move": move_sum,
        "ret": (capital / 250.0 - 1.0) * 100.0,
        "final": capital,
    }


def print_symbol(rows: list[dict]) -> None:
    print("--- SYMBOL STABILITY: fee=0 ---")
    print("RULE SYMBOL T WIN MOVE_SUM RET")
    print("---- ------ - --- -------- ---")
    for rule, variant in RULES:
        symbols = sorted({r["symbol"] for r in rows if r["rule"] == rule and r["variant"] == variant})
        for symbol in symbols:
            x = [r for r in rows if r["rule"] == rule and r["variant"] == variant and r["symbol"] == symbol]
            a = aggregate(x, 0.0)
            print(f"{rule:>3} {symbol:>6} {a['t']:>3} {100*a['wins']/a['t']:>5.2f}% {a['move']:>+8.2f} {a['ret']:>+6.2f}%")


def print_week(rows: list[dict]) -> None:
    print("--- WEEKLY STABILITY: fee=0 ---")
    print("RULE WEEK T WIN MOVE_SUM RET")
    print("---- --- - --- -------- ---")
    for rule, variant in RULES:
        groups = defaultdict(list)
        for r in rows:
            if r["rule"] == rule and r["variant"] == variant:
                groups[week_label(r["entry_ts"])].append(r)
        for week in sorted(groups):
            a = aggregate(groups[week], 0.0)
            print(f"{rule:>3} {week:>3} {a['t']:>3} {100*a['wins']/a['t']:>5.2f}% {a['move']:>+8.2f} {a['ret']:>+6.2f}%")


def print_fee(rows: list[dict]) -> None:
    print("--- FEE SENSITIVITY: R16N / R6N ---")
    print("RULE FEE T RET FINAL")
    print("---- --- - --- -----")
    for rule, variant in RULES:
        x = [r for r in rows if r["rule"] == rule and r["variant"] == variant]
        for fee in FEES:
            a = aggregate(x, fee)
            print(f"{rule:>3} {fee:>3.2f}% {a['t']:>4} {a['ret']:>+7.2f}% {a['final']:>8.2f}")


def print_week_fee(rows: list[dict]) -> None:
    print("--- WEEKLY FEE CHECK: 0.05%/side ---")
    print("RULE WEEK T RET")
    print("---- --- - ---")
    for rule, variant in RULES:
        groups = defaultdict(list)
        for r in rows:
            if r["rule"] == rule and r["variant"] == variant:
                groups[week_label(r["entry_ts"])].append(r)
        for week in sorted(groups):
            a = aggregate(groups[week], 0.05)
            print(f"{rule:>3} {week:>3} {a['t']:>3} {a['ret']:>+7.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default=str(TRADES))
    args = ap.parse_args()
    rows = load_rows(Path(args.trades))
    rows = [r for r in rows if (r["rule"], r["variant"]) in RULES]
    if not rows:
        raise RuntimeError("No R16N/R6N trades found")
    print("R16_R6_STABILITY_ANALYSIS")
    print(f"TRADES={len(rows)} SOURCE={Path(args.trades).as_posix()}")
    print("Architecture inherited from frozen non-overlap report: 15m signal -> 60m entry -> next 60m exit")
    print_symbol(rows)
    print_week(rows)
    print_fee(rows)
    print_week_fee(rows)


if __name__ == "__main__":
    main()
