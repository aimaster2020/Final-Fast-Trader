from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median

DEFAULT_LOOKBACK = 50
MIN_HISTORY = 15
EPS = 1e-12


def load(path: Path) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def ad_score(row: tuple[float, float, float, float]) -> int:
    o, h, l, c = row
    body = c - o
    hc = h - c
    ho = h - o
    lc = l - c
    return int(hc > body) + int(ho < hc) + int(lc > body)


def parts(row: tuple[float, float, float, float]) -> tuple[float, float, float, float, float]:
    o, h, l, c = row
    size = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    rng = h - l
    wick = upper + lower
    return size, upper, lower, rng, wick


def predict(formula: str, cur: tuple[float, float, float, float], prev: tuple[float, float, float, float]) -> float | None:
    cs, cu, cl, cr, cw = parts(cur)
    ps, pu, pl, pr, pw = parts(prev)

    if formula == "PERSIST":
        return cs
    if formula == "DOUBLE_MINUS_PREV":
        return max(0.0, 2.0 * cs - ps)
    if formula == "AVG_BODY":
        return 0.5 * (cs + ps)
    if formula == "GEOMEAN_BODY":
        return math.sqrt(cs * ps)
    if formula == "HARMONIC_BODY":
        return 2.0 * cs * ps / (cs + ps) if cs + ps > EPS else None
    if formula == "RANGE_PREV_BODY_SHARE":
        if pr <= EPS:
            return None
        return cr * (ps / pr)
    if formula == "BODY_CUR_RANGE_PREV_RATIO":
        if pr <= EPS:
            return None
        return cs * (cr / pr)
    if formula == "SQRT_RANGE_RATIO":
        if pr <= EPS:
            return None
        return cs * math.sqrt(max(cr / pr, 0.0))
    if formula == "SQRT_BODY_RATIO":
        if ps <= EPS:
            return None
        return cs * math.sqrt(max(cs / ps, 0.0))
    if formula == "WICK_SHARE_TRANSFER":
        if cr <= EPS or pr <= EPS:
            return None
        cur_share = cw / cr
        prev_share = pw / pr
        if prev_share <= EPS:
            return None
        return cr * max(0.0, 1.0 - cur_share) * prev_share
    if formula == "BODY_SHARE_COMPLEMENT":
        return cr * max(0.0, 1.0 - (cw / cr)) if cr > EPS else None
    return None


def score_predictions(pairs: list[tuple[float, float]]) -> tuple[float, float, float]:
    if not pairs:
        return math.inf, math.inf, 0.0
    abs_err = [abs(p - y) for p, y in pairs]
    rel_err = [abs(p - y) / y for p, y in pairs if y > EPS]
    grow_ok = sum((p > 0) == (y > 0) for p, y in pairs)
    # The growth score is filled by caller; here return MAE, median relative error, unused.
    return sum(abs_err) / len(abs_err), median(rel_err) if rel_err else math.inf, grow_ok / len(pairs) * 100.0


def evaluate(rows: list[tuple[float, float, float, float]], score: int, lookback: int) -> dict[str, float | int | str]:
    formulas = [
        "PERSIST",
        "DOUBLE_MINUS_PREV",
        "AVG_BODY",
        "GEOMEAN_BODY",
        "HARMONIC_BODY",
        "RANGE_PREV_BODY_SHARE",
        "BODY_CUR_RANGE_PREV_RATIO",
        "SQRT_RANGE_RATIO",
        "SQRT_BODY_RATIO",
        "BODY_SHARE_COMPLEMENT",
    ]

    errors: list[float] = []
    rel_errors: list[float] = []
    grow_correct = 0
    tests = 0
    selected_counts = {f: 0 for f in formulas}

    for i in range(1, len(rows) - 1):
        if ad_score(rows[i]) != score:
            continue

        history_start = max(1, i - lookback)
        history: dict[str, list[tuple[float, float]]] = {f: [] for f in formulas}
        for j in range(history_start, i):
            hist_cur = rows[j]
            hist_prev = rows[j - 1]
            actual = parts(rows[j + 1])[0]
            for formula in formulas:
                pred = predict(formula, hist_cur, hist_prev)
                if pred is not None and math.isfinite(pred):
                    history[formula].append((pred, actual))

        ranked: list[tuple[float, str]] = []
        for formula in formulas:
            pairs = history[formula]
            if len(pairs) < MIN_HISTORY:
                continue
            rel = [abs(p - y) / y for p, y in pairs if y > EPS]
            if not rel:
                continue
            ranked.append((median(rel), formula))

        if not ranked:
            continue
        _, best = min(ranked)
        pred = predict(best, rows[i], rows[i - 1])
        actual = parts(rows[i + 1])[0]
        current = parts(rows[i])[0]
        if pred is None or not math.isfinite(pred):
            continue

        selected_counts[best] += 1
        errors.append(abs(pred - actual))
        if actual > EPS:
            rel_errors.append(abs(pred - actual) / actual)
        grow_correct += int((actual > current) == (pred > current))
        tests += 1

    mae = sum(errors) / len(errors) if errors else 0.0
    med_rel = median(rel_errors) if rel_errors else 0.0
    acc = grow_correct / tests * 100.0 if tests else 0.0
    top_formula = max(selected_counts, key=selected_counts.get) if tests else "NONE"
    return {"tests": tests, "mae": mae, "med_rel": med_rel, "acc": acc, "top_formula": top_formula, **{f"sel_{k}": v for k, v in selected_counts.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Causal rolling selector over symbolic body-size formulas.")
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    args = ap.parse_args()

    lookback = max(10, args.lookback)
    root = Path(args.input_dir)
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    print(f"LOOKBACK={lookback} MIN_HISTORY={MIN_HISTORY}")
    print("MODEL=causal rolling symbolic formula selector; formulas use current+previous candle only")
    for symbol in symbols:
        rows = load(root / f"{symbol}_1h.csv")
        print(f"{symbol} rows={len(rows)}")
        for score in (3, 0):
            r = evaluate(rows, score, lookback)
            print(
                f"AD={score} TESTS={int(r['tests'])} MAE={r['mae']:.6f} "
                f"MED_REL_ERR={r['med_rel']*100:.2f}% GROW_SHRINK_ACC={r['acc']:.2f}% "
                f"TOP={r['top_formula']}"
            )


if __name__ == "__main__":
    main()
