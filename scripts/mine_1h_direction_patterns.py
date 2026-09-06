from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sign(v: str) -> int:
    return 1 if v == "U" else -1 if v == "D" else 0


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
    groups = defaultdict(lambda: {"n": 0, "up": 0, "down": 0, "ret_sum": 0.0, "ret_pos": 0, "ret_neg": 0})

    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        if r.get("symbol", "").upper() not in symbols:
            continue
        code = (r.get("four_direction_past_code") or "").strip()
        if len(code) != 5 or any(ch not in "UDF" for ch in code):
            continue
        future_dir = (r.get("next_dir_%d" % args.future_step) or "").strip()
        ret = pct(r.get("future_%dh_return_pct" % args.future_step))
        if future_dir not in {"UP", "DOWN", "FLAT"}:
            continue
        g = groups[(code, future_dir)]
        g["n"] += 1
        g["ret_sum"] += ret
        if future_dir == "UP":
            g["up"] += 1
        elif future_dir == "DOWN":
            g["down"] += 1
        if ret > 0:
            g["ret_pos"] += 1
        elif ret < 0:
            g["ret_neg"] += 1

    print(f"PATTERN_MINING rows={len(rows)} symbols={','.join(sorted(symbols))} horizon={args.future_step}h min_count={args.min_count}")
    print("CODE = previous 4 directions + current | U=UP D=DOWN F=FLAT")
    print("CODE FUTURE N WIN% AVG_RET%")
    print("---- ------ --- ----- --------")

    ranked = []
    by_code = defaultdict(lambda: {"total": 0, "up": 0, "down": 0, "ret_sum": 0.0})
    for (code, future_dir), g in groups.items():
        b = by_code[code]
        b["total"] += g["n"]
        b["up"] += g["up"]
        b["down"] += g["down"]
        b["ret_sum"] += g["ret_sum"]
        if g["n"] >= args.min_count:
            wins = g["up"] if future_dir == "UP" else g["down"] if future_dir == "DOWN" else 0
            win_rate = 100.0 * wins / g["n"] if g["n"] else 0.0
            avg_ret = g["ret_sum"] / g["n"] if g["n"] else 0.0
            ranked.append((win_rate * (abs(avg_ret) + 1e-9), win_rate, avg_ret, code, future_dir, g["n"]))

    for _, win_rate, avg_ret, code, future_dir, n in sorted(ranked, reverse=True)[: args.top * 3]:
        print(f"{code:>4} {future_dir:>6} {n:>3} {win_rate:>5.1f} {avg_ret:>+8.3f}")

    print("TOP_DIRECTIONAL_CODES")
    code_ranked = []
    for code, b in by_code.items():
        if b["total"] < args.min_count:
            continue
        direction = "UP" if b["up"] > b["down"] else "DOWN" if b["down"] > b["up"] else "FLAT"
        wins = max(b["up"], b["down"])
        win_rate = 100.0 * wins / b["total"] if b["total"] else 0.0
        avg_ret = b["ret_sum"] / b["total"] if b["total"] else 0.0
        code_ranked.append((win_rate, abs(avg_ret), avg_ret, code, direction, b["total"], b["up"], b["down"]))

    for win_rate, abs_avg, avg_ret, code, direction, n, up, down in sorted(code_ranked, reverse=True)[: args.top]:
        print(f"{code:>4} -> {direction:>5} n={n:>4} win={win_rate:>5.1f}% avg_ret={avg_ret:>+8.3f}% up={up:>4} down={down:>4}")


if __name__ == "__main__":
    main()
