from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def pct(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine 1H direction sequences and their next-direction/return behavior.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--future-step", type=int, default=1, choices=(1, 2, 3, 4))
    args = ap.parse_args()

    symbols = {x.strip().upper() for x in args.symbols.split(",") if x.strip()}
    groups = defaultdict(lambda: {"n": 0, "up": 0, "down": 0, "ret_sum": 0.0})

    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    future_dir_col = f"next_d{args.future_step}"
    future_ret_col = f"future_{args.future_step}h_return_pct"

    for r in rows:
        if r.get("symbol", "").upper() not in symbols:
            continue
        code = (r.get("seq_5_past_code") or "").strip()
        if len(code) != 5 or any(ch not in "UDF" for ch in code):
            continue
        future_dir = (r.get(future_dir_col) or "").strip().upper()
        ret = pct(r.get(future_ret_col))
        if future_dir not in {"UP", "DOWN", "FLAT"}:
            continue
        g = groups[code]
        g["n"] += 1
        g["ret_sum"] += ret
        if future_dir == "UP":
            g["up"] += 1
        elif future_dir == "DOWN":
            g["down"] += 1

    print(f"PATTERN_MINING rows={len(rows)} symbols={','.join(sorted(symbols))} horizon={args.future_step}h min_count={args.min_count}")
    print("CODE = previous 4 directions + current | U=UP D=DOWN F=FLAT")
    print("CODE N DOMINANT WIN% AVG_RET% UP DOWN")
    print("---- --- -------- ----- -------- ------ ----")

    ranked = []
    for code, g in groups.items():
        if g["n"] < args.min_count:
            continue
        direction = "UP" if g["up"] > g["down"] else "DOWN" if g["down"] > g["up"] else "FLAT"
        wins = max(g["up"], g["down"])
        win_rate = 100.0 * wins / g["n"] if g["n"] else 0.0
        avg_ret = g["ret_sum"] / g["n"] if g["n"] else 0.0
        ranked.append((win_rate, abs(avg_ret), avg_ret, code, direction, g["n"], g["up"], g["down"]))

    ranked.sort(key=lambda x: (-x[0], -x[1], -x[5], x[3]))
    for win_rate, _, avg_ret, code, direction, n, up, down in ranked[:args.top]:
        print(f"{code:>4} {n:>3} {direction:>8} {win_rate:>5.1f} {avg_ret:>+8.3f} {up:>6} {down:>4}")

    print("TOP_DIRECTIONAL_CODES")
    for win_rate, _, avg_ret, code, direction, n, up, down in ranked[:args.top]:
        print(f"{code:>5} -> {direction:>5} n={n:>4} win={win_rate:>5.1f}% avg_ret={avg_ret:>+8.3f}% up={up:>4} down={down:>4}")


if __name__ == "__main__":
    main()
