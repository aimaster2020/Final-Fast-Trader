from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MIN_CANDLES = 48


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    cols = ["timestamp", "open", "high", "low", "close"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    body = df["close"] - df["open"]
    hc = df["high"] - df["close"]
    ho = df["high"] - df["open"]
    lc = df["low"] - df["close"]
    score = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)

    formula_dir = np.where(score <= 1, -1, np.where(score >= 2, 1, 0)).astype(int)
    upper = (df["high"] - df["open"]).abs()
    lower = (df["open"] - df["low"]).abs()
    side_formula = np.where(lower > upper, 1, np.where(upper > lower, -1, 0)).astype(int)

    trade_direction = formula_dir.copy()
    trade_direction[(score == 1) & (side_formula == 1)] = 0
    trade_direction[(score == 2) & (side_formula == -1)] = 0

    next_body = body.shift(-1)
    actual_dir = np.sign(next_body).astype(float)

    J = body
    K = df["high"] - df["close"]
    L = df["close"] - df["low"]
    U = J.shift(-1) - J

    out = df.copy()
    out["J"] = J
    out["K"] = K
    out["L"] = L
    out["U"] = U
    out["pred_direction"] = trade_direction
    out["score_direction"] = formula_dir
    out["actual_direction"] = actual_dir
    return out.dropna(subset=["U"]).reset_index(drop=True)


def direction_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    m = (df["pred_direction"] != 0) & (df["actual_direction"] != 0)
    y = df.loc[m, "actual_direction"].to_numpy(float)
    p = df.loc[m, "pred_direction"].to_numpy(float)
    correct = int(np.sum(y == p))
    n = len(y)
    long_n = int(np.sum(p == 1))
    short_n = int(np.sum(p == -1))
    return {
        "dir_n": n,
        "dir_correct": correct,
        "dir_accuracy": correct / n if n else 0.0,
        "dir_long_accuracy": float(np.mean(y[p == 1] == 1)) if long_n else 0.0,
        "dir_short_accuracy": float(np.mean(y[p == -1] == -1)) if short_n else 0.0,
        "dir_coverage": n / len(df) if len(df) else 0.0,
    }


def _empty_mag() -> dict[str, float | int]:
    return {
        "mag_n": 0,
        "mag_dir_accuracy": 0.0,
        "mag_mae": np.nan,
        "mag_nmae": np.nan,
        "mag_rmse": np.nan,
        "mag_corr": np.nan,
        "mag_r2": np.nan,
        "mag_within_25pct": np.nan,
        "mag_within_50pct": np.nan,
        "mag_within_100pct": np.nan,
        "mag_mean_abs_actual": np.nan,
        "mag_mean_abs_pred": np.nan,
    }


def magnitude_walkforward(df: pd.DataFrame, folds: int = 5, train_ratio: float = 0.5) -> dict[str, float | int]:
    features = ["J", "K", "L"]
    n = len(df)
    if n < max(100, folds * 20):
        return _empty_mag()

    initial_train = max(20, int(n * train_ratio))
    remaining = n - initial_train
    if remaining < folds * 10:
        folds = max(2, remaining // 10)
    test_size = remaining // folds
    if test_size <= 0:
        return _empty_mag()

    ys = []
    ps = []

    for i in range(folds):
        start = initial_train + i * test_size
        end = initial_train + (i + 1) * test_size if i < folds - 1 else n
        train = df.iloc[:start]
        test = df.iloc[start:end]
        if len(test) == 0:
            continue

        x_train = np.column_stack([np.ones(len(train)), train[features].to_numpy(float)])
        y_train = train["U"].to_numpy(float)
        coef = np.linalg.lstsq(x_train, y_train, rcond=None)[0]

        x_test = np.column_stack([np.ones(len(test)), test[features].to_numpy(float)])
        y = test["U"].to_numpy(float)
        p = x_test @ coef
        ys.append(y)
        ps.append(p)

    if not ys:
        return _empty_mag()

    y = np.concatenate(ys)
    p = np.concatenate(ps)
    err = p - y
    abs_y = np.abs(y)
    abs_p = np.abs(p)

    mask = (np.sign(y) != 0) & (np.sign(p) != 0)
    acc = float(np.mean(np.sign(y[mask]) == np.sign(p[mask]))) if mask.any() else 0.0

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mean_abs_y = float(np.mean(abs_y))
    nmae = mae / mean_abs_y if mean_abs_y > 0 else np.nan

    corr = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0 else 0.0
    sst = float(np.sum((y - np.mean(y)) ** 2))
    sse = float(np.sum((y - p) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    # Relative-error buckets are reported only where the realized |U| is non-zero.
    nz = abs_y > 0
    rel_err = np.full_like(abs_y, np.nan, dtype=float)
    rel_err[nz] = np.abs(err[nz]) / abs_y[nz]

    return {
        "mag_n": len(y),
        "mag_dir_accuracy": acc,
        "mag_mae": mae,
        "mag_nmae": nmae,
        "mag_rmse": rmse,
        "mag_corr": corr,
        "mag_r2": r2,
        "mag_within_25pct": float(np.mean(rel_err[nz] <= 0.25)) if nz.any() else np.nan,
        "mag_within_50pct": float(np.mean(rel_err[nz] <= 0.50)) if nz.any() else np.nan,
        "mag_within_100pct": float(np.mean(rel_err[nz] <= 1.00)) if nz.any() else np.nan,
        "mag_mean_abs_actual": mean_abs_y,
        "mag_mean_abs_pred": float(np.mean(abs_p)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-market formula direction and direct magnitude accuracy on all non-IRT Nobitex markets.")
    ap.add_argument("--data-dir", default="reports/nobitex_1h")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.5)
    ap.add_argument("--min-candles", type=int, default=MIN_CANDLES)
    ap.add_argument("--output", default="reports/nobitex_1h/nonirt_formula_accuracy_by_market.csv")
    args = ap.parse_args()

    root = Path(args.data_dir)
    files = sorted(p for p in root.glob("*_1h.csv") if not p.stem[:-3].upper().endswith("IRT"))
    results = []
    skipped = 0

    for path in files:
        symbol = path.stem[:-3] if path.stem.endswith("_1h") else path.stem
        try:
            df = load(path)
            if len(df) < args.min_candles:
                skipped += 1
                continue
            d = direction_metrics(df)
            m = magnitude_walkforward(df, args.folds, args.train_ratio)
            results.append({"symbol": symbol, "candles": len(df), **d, **m})
        except Exception as exc:
            skipped += 1
            print(f"SKIP {symbol}: {exc}")

    out = pd.DataFrame(results)
    if out.empty:
        raise SystemExit("No valid non-IRT markets found.")
    out["dir_accuracy_pct"] = out["dir_accuracy"] * 100
    out["mag_dir_accuracy_pct"] = out["mag_dir_accuracy"] * 100
    out["mag_within_25pct_pct"] = out["mag_within_25pct"] * 100
    out["mag_within_50pct_pct"] = out["mag_within_50pct"] * 100
    out["mag_within_100pct_pct"] = out["mag_within_100pct"] * 100
    out = out.sort_values(["dir_accuracy", "mag_dir_accuracy"], ascending=False).reset_index(drop=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    print("=" * 150)
    print("NOBITEX NON-IRT | PER-MARKET FORMULA + DIRECT MAGNITUDE ACCURACY")
    print("=" * 150)
    print(f"Markets found       : {len(files)}")
    print(f"Markets tested      : {len(out)}")
    print(f"Skipped             : {skipped}")
    print("Direction formula   : score 0/1 SHORT, score 2/3 LONG + ambiguity filter")
    print("Direction target    : next candle body sign (Close-Open)")
    print("Magnitude target    : U = J_next - J_current")
    print("Magnitude features  : J,K,L | walk-forward OOS linear regression")
    print(f"Walk-forward        : folds={args.folds} train_ratio={args.train_ratio:.2f}")
    print("nMAE                : MAE / mean(|actual U|); lower is better")
    print("Within X%           : |prediction-actual| / |actual| <= X%, excluding exact-zero U")
    print()
    print("AGGREGATE")
    print(f"Direction mean accuracy       : {out.dir_accuracy.mean()*100:.2f}%")
    print(f"Direction median accuracy     : {out.dir_accuracy.median()*100:.2f}%")
    print(f"Direction markets >= 55%      : {(out.dir_accuracy >= 0.55).sum()} / {len(out)}")
    print(f"Direction markets >= 60%      : {(out.dir_accuracy >= 0.60).sum()} / {len(out)}")
    print(f"Magnitude mean corr           : {out.mag_corr.mean():+.3f}")
    print(f"Magnitude median corr         : {out.mag_corr.median():+.3f}")
    print(f"Magnitude mean OOS dir acc    : {out.mag_dir_accuracy.mean()*100:.2f}%")
    print(f"Magnitude median OOS dir acc  : {out.mag_dir_accuracy.median()*100:.2f}%")
    print(f"Magnitude mean nMAE           : {out.mag_nmae.mean()*100:.2f}%")
    print(f"Magnitude median nMAE         : {out.mag_nmae.median()*100:.2f}%")
    print(f"Magnitude mean within 25%     : {out.mag_within_25pct.mean()*100:.2f}%")
    print(f"Magnitude median within 25%   : {out.mag_within_25pct.median()*100:.2f}%")
    print(f"Magnitude mean within 50%     : {out.mag_within_50pct.mean()*100:.2f}%")
    print(f"Magnitude median within 50%   : {out.mag_within_50pct.median()*100:.2f}%")
    print()
    print("ALL MARKETS — SORTED BY DIRECTION ACCURACY")
    for _, r in out.iterrows():
        print(
            f"{r.symbol:20s} candles={int(r.candles):3d} "
            f"dir={r.dir_accuracy_pct:6.2f}% n={int(r.dir_n):3d} cov={r.dir_coverage*100:5.1f}% "
            f"| long={r.dir_long_accuracy*100:6.2f}% short={r.dir_short_accuracy*100:6.2f}% "
            f"| mag_dir={r.mag_dir_accuracy_pct:6.2f}% corr={r.mag_corr:+.3f} "
            f"r2={r.mag_r2:+.3f} nMAE={r.mag_nmae*100:6.2f}% "
            f"W25={r.mag_within_25pct_pct:6.2f}% W50={r.mag_within_50pct_pct:6.2f}% "
            f"MAE={r.mag_mae:.6f}"
        )
    print()
    print(f"Detailed CSV: {path}")
    print("=" * 150)


if __name__ == "__main__":
    main()
