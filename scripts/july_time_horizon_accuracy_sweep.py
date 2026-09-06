from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.july_time_horizon_oracle_fee_hold import load_series

HORIZONS = [("30m", 30), ("1h", 60), ("2h", 120), ("4h", 240)]
ACCURACIES = list(range(50, 101, 5))
DEFAULT_FEE_SIDE_PCT = 0.13
DEFAULT_SEEDS = 20


def build_frames(raw: list, minutes: int) -> list[tuple[int, float, float, int]]:
    """Canonical non-overlapping intervals: (start_i, entry, exit, oracle_dir)."""
    frames = []
    i = 0
    while i < len(raw):
        entry = raw[i].close
        target_ts = raw[i].timestamp + minutes * 60
        j = i + 1
        while j < len(raw) and raw[j].timestamp < target_ts:
            j += 1
        if j >= len(raw) or entry <= 0:
            break
        exit_price = raw[j].close
        if exit_price <= 0:
            break
        move = exit_price / entry - 1.0
        direction = 1 if move > 0 else -1 if move < 0 else 0
        frames.append((i, entry, exit_price, direction))
        i = j
    return frames


def make_predictions(directions: list[int], target_accuracy: int, seed: int) -> list[int]:
    """Keep flats unchanged and flip exactly the required number of non-flat labels."""
    preds = directions[:]
    directional = [i for i, d in enumerate(directions) if d != 0]
    wrong = round(len(directional) * (100 - target_accuracy) / 100.0)
    rng = random.Random(seed)
    flip_indices = rng.sample(directional, wrong) if wrong else []
    for i in flip_indices:
        preds[i] = -preds[i]
    return preds


def charge_fee(capital: float, fee_side_pct: float) -> tuple[float, float]:
    fee = capital * fee_side_pct / 100.0
    return capital - fee, fee


def simulate(
    series: dict[str, list],
    minutes: int,
    target_accuracy: int,
    seed: int,
    fee_side_pct: float,
) -> dict:
    initial = 1000.0
    per_symbol = initial / len(series)
    total = 0.0
    directional = correct = intervals = entries = reversals = 0
    fees = 0.0

    for symbol, raw in series.items():
        frames = build_frames(raw, minutes)
        dirs = [x[3] for x in frames]
        preds = make_predictions(dirs, target_accuracy, seed * 1009 + sum(map(ord, symbol)))
        capital = per_symbol
        position = 0

        for (_, entry, exit_price, actual_dir), pred in zip(frames, preds):
            intervals += 1
            if actual_dir != 0:
                directional += 1
                correct += int(pred == actual_dir)

            if pred == 0:
                if position != 0:
                    capital, f = charge_fee(capital, fee_side_pct)
                    fees += f
                    position = 0
            elif position == 0:
                capital, f = charge_fee(capital, fee_side_pct)
                fees += f
                entries += 1
                position = pred
            elif pred != position:
                capital, f = charge_fee(capital, fee_side_pct)
                fees += f
                capital, f = charge_fee(capital, fee_side_pct)
                fees += f
                reversals += 1
                entries += 1
                position = pred

            if position == 1:
                capital *= exit_price / entry
            elif position == -1:
                capital *= entry / exit_price

        if position != 0:
            capital, f = charge_fee(capital, fee_side_pct)
            fees += f

        total += capital

    return {
        "final": total,
        "ret_pct": 100.0 * (total / initial - 1.0),
        "accuracy_pct": 100.0 * correct / directional if directional else 0.0,
        "intervals": intervals,
        "entries": entries,
        "reversals": reversals,
        "fees": fees,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--fee-side-pct", type=float, default=DEFAULT_FEE_SIDE_PCT)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    args = ap.parse_args()

    series = load_series(Path(args.input_dir), args.test_month, args.symbols)
    print(f"JULY_TIME_HORIZON_ACCURACY_SWEEP test={args.test_month} assets={len(series)}")
    print("CONTROLLED SYNTHETIC PREDICTION | ORACLE LABELS + RANDOM FLIPS")
    print(f"fee={args.fee_side_pct:.3f}%/side | SAME PREDICTION=HOLD | FLIP=REVERSE | seeds={args.seeds}")
    print("HORIZON ACC TARGET_ACC AVG_RET MIN_RET MAX_RET AVG_FINAL AVG_ENTRIES")
    print("------- --- ---------- ------- ------- ------- --------- -----------")

    rows = []
    for horizon, minutes in HORIZONS:
        for target in ACCURACIES:
            results = [simulate(series, minutes, target, seed, args.fee_side_pct) for seed in range(args.seeds)]
            rets = [r["ret_pct"] for r in results]
            finals = [r["final"] for r in results]
            entries = [r["entries"] for r in results]
            actual_acc = sum(r["accuracy_pct"] for r in results) / len(results)
            avg_ret = sum(rets) / len(rets)
            min_ret = min(rets)
            max_ret = max(rets)
            avg_final = sum(finals) / len(finals)
            avg_entries = sum(entries) / len(entries)
            print(
                f"{horizon:>7} {target:>3}% {actual_acc:>10.2f}% "
                f"{avg_ret:>+7.2f}% {min_ret:>+7.2f}% {max_ret:>+7.2f}% "
                f"{avg_final:>9.2f} {avg_entries:>11.0f}"
            )
            rows.append({
                "horizon": horizon,
                "target_accuracy_pct": target,
                "actual_accuracy_pct": actual_acc,
                "avg_ret_pct": avg_ret,
                "min_ret_pct": min_ret,
                "max_ret_pct": max_ret,
                "avg_final": avg_final,
                "avg_entries": avg_entries,
                "seeds": args.seeds,
                "fee_side_pct": args.fee_side_pct,
            })

    out = ROOT / "reports" / "july_time_horizon_accuracy_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"SAVED {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
