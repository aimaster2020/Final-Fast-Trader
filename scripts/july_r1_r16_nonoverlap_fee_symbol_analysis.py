from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="reports/july_r1_r16_single_nonoverlap_detail/r1_r16_single_nonoverlap_all_trades.csv",
    )
    ap.add_argument("--fees", default="0,0.05,0.1,0.2,0.3")
    args = ap.parse_args()

    rows = load_rows(ROOT / args.input)
    fees = [float(x.strip()) for x in args.fees.split(",") if x.strip()]

    print("R1_R16_NONOVERLAP_SYMBOL_FEE_ANALYSIS")
    print(f"TRADES={len(rows)} FEES_SIDE_PCT={','.join(f'{x:g}' for x in fees)}")
    print("--- PER SYMBOL / RULE (fee=0) ---")
    print("RULE VAR SYMBOL T WIN MOVE_SUM RET_ON_250")
    print("---- --- ------ - --- --------- ---------")

    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        key = (r["rule"], r["variant"], r["symbol"])
        grouped.setdefault(key, []).append(r)

    symbol_rows = []
    for (rule, variant, symbol), g in sorted(grouped.items()):
        wins = sum(int(r["win"]) for r in g)
        move_sum = sum(float(r["move_pct"]) for r in g)
        capital = 250.0
        for r in g:
            capital *= 1.0 + float(r["move_pct"]) / 100.0
        ret = 100.0 * (capital / 250.0 - 1.0)
        symbol_rows.append((rule, variant, symbol, len(g), 100.0 * wins / len(g), move_sum, ret))
        print(f"{rule:>3} {variant:>3} {symbol:<7} {len(g):>4} {100.0*wins/len(g):>6.2f}% {move_sum:>10.4f} {ret:>9.2f}%")

    print("--- FEE SENSITIVITY: ALL 32 CANDIDATES ---")
    print("RULE VAR " + " ".join(f"F{x:g}" for x in fees))
    print("---- --- " + " ".join("--------" for _ in fees))

    for rule in [f"R{i}" for i in range(1, 17)]:
        for variant in ["N", "I"]:
            g = [r for r in rows if r["rule"] == rule and r["variant"] == variant]
            # Rebuild each symbol's capital independently, matching the main test.
            out = []
            for fee_side in fees:
                total = 0.0
                by_symbol = {}
                for symbol in sorted({r["symbol"] for r in g}):
                    capital = 250.0
                    for r in g:
                        if r["symbol"] != symbol:
                            continue
                        move = float(r["move_pct"]) / 100.0
                        fee = 2.0 * fee_side / 100.0
                        capital = max(0.0, capital * (1.0 + move - fee))
                    by_symbol[symbol] = capital
                    total += capital
                ret = 100.0 * (total / 1000.0 - 1.0)
                out.append(f"{ret:+7.2f}%")
            print(f"{rule:>3} {variant:>3} " + " ".join(out))

    print("--- BEST CANDIDATES BY FEE ---")
    for fee_side in fees:
        ranked = []
        for rule in [f"R{i}" for i in range(1, 17)]:
            for variant in ["N", "I"]:
                g = [r for r in rows if r["rule"] == rule and r["variant"] == variant]
                total = 0.0
                for symbol in sorted({r["symbol"] for r in g}):
                    capital = 250.0
                    for r in g:
                        if r["symbol"] == symbol:
                            move = float(r["move_pct"]) / 100.0
                            fee = 2.0 * fee_side / 100.0
                            capital = max(0.0, capital * (1.0 + move - fee))
                    total += capital
                ranked.append((100.0 * (total / 1000.0 - 1.0), rule, variant, total))
        print(f"FEE={fee_side:g}%/side")
        for ret, rule, variant, final in sorted(ranked, reverse=True)[:5]:
            print(f"  {rule}{variant}: RET={ret:+.2f}% FINAL={final:.2f}")


if __name__ == "__main__":
    main()
