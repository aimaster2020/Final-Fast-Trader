from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path
from statistics import median


def load(path: Path) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((
                    float(r["open"]),
                    float(r["high"]),
                    float(r["low"]),
                    float(r["close"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def ad_score(o: float, h: float, l: float, c: float) -> int:
    body = c - o
    hc = h - c
    ho = h - o
    lc = l - c
    return int(hc > body) + int(ho < hc) + int(lc > body)


def evaluate(rows: list[tuple[float, float, float, float]], lookback: int, min_history: int) -> dict[str, float | int]:
    # History stores the OOS-observable ratio:
    # next_abs_body / current_abs_body, keyed by the current AD score.
    history = {0: deque(maxlen=lookback), 3: deque(maxlen=lookback)}

    tests = 0
    size_mae = 0.0
    pct_errors: list[float] = []
    growth_correct = 0
    predicted_ratios: list[float] = []
    actual_ratios: list[float] = []

    for i in range(len(rows) - 1):
        o, h, l, c = rows[i]
        no, nh, nl, nc = rows[i + 1]
        ad = ad_score(o, h, l, c)

        current_size = abs(c - o)
        next_size = abs(nc - no)

        if ad in history and current_size > 1e-12 and len(history[ad]) >= min_history:
            coeff = median(history[ad])
            pred_size = current_size * coeff
            err = abs(pred_size - next_size)
            size_mae += err
            actual_ratio = next_size / current_size if current_size > 1e-12 else 0.0
            predicted_ratios.append(coeff)
            actual_ratios.append(actual_ratio)
            pct_errors.append(abs(pred_size - next_size) / next_size if next_size > 1e-12 else 0.0)

            actual_grow = next_size > current_size
            pred_grow = pred_size > current_size
            growth_correct += int(actual_grow == pred_grow)
            tests += 1

        if ad in history and current_size > 1e-12:
            history[ad].append(next_size / current_size)

    if not tests:
        return {"tests": 0, "mae": 0.0, "mape": 0.0, "growth_acc": 0.0, "median_pred_ratio": 0.0, "median_actual_ratio": 0.0}

    return {
        "tests": tests,
        "mae": size_mae / tests,
        "mape": median(pct_errors) * 100.0,
        "growth_acc": growth_correct / tests * 100.0,
        "median_pred_ratio": median(predicted_ratios),
        "median_actual_ratio": median(actual_ratios),
    }


def evaluate_by_ad(rows: list[tuple[float, float, float, float]], lookback: int, min_history: int) -> None:
    for score in (3, 0):
        filtered = []
        # Keep all rows: evaluate() needs temporal continuity to build causal history.
        # score filtering is applied by temporarily using only the requested history.
        history = deque(maxlen=lookback)
        tests = 0
        mae = 0.0
        mape_values: list[float] = []
        grow_correct = 0

        for i in range(len(rows) - 1):
            o, h, l, c = rows[i]
            no, nh, nl, nc = rows[i + 1]
            ad = ad_score(o, h, l, c)
            current_size = abs(c - o)
            next_size = abs(nc - no)

            if ad == score and current_size > 1e-12 and len(history) >= min_history:
                coeff = median(history)
                pred_size = current_size * coeff
                mae += abs(pred_size - next_size)
                if next_size > 1e-12:
                    mape_values.append(abs(pred_size - next_size) / next_size)
                grow_correct += int((pred_size > current_size) == (next_size > current_size))
                tests += 1

            if ad == score and current_size > 1e-12:
                history.append(next_size / current_size)

        if tests:
            print(
                f"AD={score} TESTS={tests} "
                f"MAE={mae/tests:.6f} "
                f"MED_ABS_PCT_ERR={median(mape_values)*100:.2f}% "
                f"GROW_SHRINK_ACC={grow_correct/tests*100:.2f}%"
            )
        else:
            print(f"AD={score} TESTS=0")


def main() -> None:
    ap = argparse.ArgumentParser(description="Causal adaptive candle-body magnitude prediction from prior same-AD observations.")
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--lookback", type=int, default=50)
    ap.add_argument("--min-history", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.input_dir)
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]

    print(f"LOOKBACK={args.lookback} MIN_HISTORY={args.min_history}")
    for symbol in symbols:
        path = root / f"{symbol}_1h.csv"
        rows = load(path)
        print(f"{symbol} rows={len(rows)}")
        evaluate_by_ad(rows, args.lookback, args.min_history)


if __name__ == "__main__":
    main()
