from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MIN_CANDLES = 48


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
    df = df.reset_index(drop=True)

    body = df["close"] - df["open"]
    K = df["high"] - df["close"]
    L = df["close"] - df["low"]
    U = body.shift(-1) - body

    out = pd.DataFrame({"J": body, "K": K, "L": L, "U": U})
    return out.dropna().reset_index(drop=True)


def walkforward_predictions(
    df: pd.DataFrame, folds: int = 5, train_ratio: float = 0.5
) -> pd.DataFrame:
    n = len(df)
    if n < max(100, folds * 20):
        return pd.DataFrame()

    initial_train = max(20, int(n * train_ratio))
    remaining = n - initial_train
    if remaining < folds * 10:
        folds = max(2, remaining // 10)
    test_size = remaining // folds
    rows: list[pd.DataFrame] = []
    features = ["J", "K", "L"]

    for i in range(folds):
        start = initial_train + i * test_size
        end = initial_train + (i + 1) * test_size if i < folds - 1 else n
        train = df.iloc[:start]
        test = df.iloc[start:end].copy()
        if test.empty:
            continue

        x_train = np.column_stack([np.ones(len(train)), train[features].to_numpy(float)])
        y_train = train["U"].to_numpy(float)
        coef = np.linalg.lstsq(x_train, y_train, rcond=None)[0]

        train_scale = float(np.mean(np.abs(y_train)))
        if not np.isfinite(train_scale) or train_scale <= 0:
            continue

        x_test = np.column_stack([np.ones(len(test)), test[features].to_numpy(float)])
        test["pred_U"] = x_test @ coef
        test["actual_abs_U"] = test["U"].abs()
        test["pred_abs_U"] = test["pred_U"].abs()
        test["actual_abs_norm"] = test["actual_abs_U"] / train_scale
        test["pred_abs_norm"] = test["pred_abs_U"] / train_scale
        rows.append(test[["U", "pred_U", "actual_abs_U", "pred_abs_U", "actual_abs_norm", "pred_abs_norm"]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def market_metrics(preds: pd.DataFrame) -> dict[str, float | int]:
    if preds.empty:
        return {
            "n": 0,
            "rank_corr_abs": np.nan,
            "top20_mean_actual": np.nan,
            "bottom20_mean_actual": np.nan,
            "top_bottom_ratio": np.nan,
            "top20_sign_acc": np.nan,
            "bottom20_sign_acc": np.nan,
        }

    p = preds.copy()
    p = p.replace([np.inf, -np.inf], np.nan).dropna(subset=["pred_abs_norm", "actual_abs_norm", "U", "pred_U"])
    if len(p) < 20:
        return {
            "n": len(p),
            "rank_corr_abs": np.nan,
            "top20_mean_actual": np.nan,
            "bottom20_mean_actual": np.nan,
            "top_bottom_ratio": np.nan,
            "top20_sign_acc": np.nan,
            "bottom20_sign_acc": np.nan,
        }

    rank_corr = p["pred_abs_norm"].rank(method="average").corr(
        p["actual_abs_norm"].rank(method="average")
    )

    q20 = p["pred_abs_norm"].quantile(0.20)
    q80 = p["pred_abs_norm"].quantile(0.80)
    bottom = p[p["pred_abs_norm"] <= q20]
    top = p[p["pred_abs_norm"] >= q80]

    top_mean = float(top["actual_abs_norm"].mean())
    bottom_mean = float(bottom["actual_abs_norm"].mean())
    ratio = top_mean / bottom_mean if bottom_mean > 0 else np.nan

    top_sign = float(np.mean(np.sign(top["U"]) == np.sign(top["pred_U"])))
    bottom_sign = float(np.mean(np.sign(bottom["U"]) == np.sign(bottom["pred_U"])))

    return {
        "n": len(p),
        "rank_corr_abs": float(rank_corr) if pd.notna(rank_corr) else np.nan,
        "top20_mean_actual": top_mean,
        "bottom20_mean_actual": bottom_mean,
        "top_bottom_ratio": ratio,
        "top20_sign_acc": top_sign,
        "bottom20_sign_acc": bottom_sign,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Test whether the U=J_next-J magnitude model can rank future move size across Nobitex non-IRT markets."
    )
    ap.add_argument("--data-dir", default="reports/nobitex_1h")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.5)
    ap.add_argument("--min-candles", type=int, default=MIN_CANDLES)
    ap.add_argument("--output", default="reports/nobitex_1h/nonirt_magnitude_ranking_by_market.csv")
    args = ap.parse_args()

    root = Path(args.data_dir)
    files = sorted(p for p in root.glob("*_1h.csv") if not p.stem[:-3].upper().endswith("IRT"))
    results: list[dict[str, float | int | str]] = []
    skipped = 0
    all_preds: list[pd.DataFrame] = []

    for path in files:
        symbol = path.stem[:-3] if path.stem.endswith("_1h") else path.stem
        try:
            df = load(path)
            if len(df) < args.min_candles:
                skipped += 1
                continue
            preds = walkforward_predictions(df, args.folds, args.train_ratio)
            m = market_metrics(preds)
            results.append({"symbol": symbol, **m})
            if not preds.empty:
                all_preds.append(preds.assign(symbol=symbol))
        except Exception as exc:
            skipped += 1
            print(f"SKIP {symbol}: {exc}")

    out = pd.DataFrame(results)
    if out.empty:
        raise SystemExit("No valid non-IRT markets found.")

    out["rank_corr_abs_pct"] = out["rank_corr_abs"] * 100
    out["top20_sign_acc_pct"] = out["top20_sign_acc"] * 100
    out["bottom20_sign_acc_pct"] = out["bottom20_sign_acc"] * 100
    out = out.sort_values("top_bottom_ratio", ascending=False, na_position="last").reset_index(drop=True)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    all_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    print("=" * 120)
    print("NOBITEX NON-IRT | MAGNITUDE RANKING TEST")
    print("=" * 120)
    print(f"Markets found       : {len(files)}")
    print(f"Markets tested      : {len(out)}")
    print(f"Skipped             : {skipped}")
    print("Target              : U = J_next - J_current")
    print("Model               : 5-fold walk-forward linear regression using J,K,L")
    print("Ranking test        : compare top 20% vs bottom 20% by predicted |U|")
    print("Scale               : each test fold normalized by mean |U| in its training set")
    print()

    print("AGGREGATE")
    print(f"Mean absolute rank corr           : {out.rank_corr_abs.mean():+.3f}")
    print(f"Median absolute rank corr         : {out.rank_corr_abs.median():+.3f}")
    print(f"Markets rank corr >= +0.30        : {(out.rank_corr_abs >= 0.30).sum()} / {len(out)}")
    print(f"Markets rank corr >= +0.50        : {(out.rank_corr_abs >= 0.50).sum()} / {len(out)}")
    print(f"Mean top/bottom actual magnitude   : {out.top_bottom_ratio.mean():.2f}x")
    print(f"Median top/bottom actual magnitude : {out.top_bottom_ratio.median():.2f}x")
    print(f"Markets top/bottom >= 1.25x        : {(out.top_bottom_ratio >= 1.25).sum()} / {len(out)}")
    print(f"Markets top/bottom >= 1.50x        : {(out.top_bottom_ratio >= 1.50).sum()} / {len(out)}")
    if not all_df.empty:
        print(f"Pooled top20 actual magnitude      : {all_df.loc[all_df.pred_abs_norm >= all_df.pred_abs_norm.quantile(0.80), 'actual_abs_norm'].mean():.3f}x")
        print(f"Pooled bottom20 actual magnitude   : {all_df.loc[all_df.pred_abs_norm <= all_df.pred_abs_norm.quantile(0.20), 'actual_abs_norm'].mean():.3f}x")

    print()
    print("ALL MARKETS — SORTED BY TOP/BOTTOM MAGNITUDE RATIO")
    for _, r in out.iterrows():
        print(
            f"{r.symbol:20s} n={int(r.n):3d} rank_corr={r.rank_corr_abs_pct:6.2f}% "
            f"top/bottom={r.top_bottom_ratio:5.2f}x "
            f"top20={r.top20_mean_actual:6.3f}x bottom20={r.bottom20_mean_actual:6.3f}x "
            f"sign_top={r.top20_sign_acc_pct:6.2f}% sign_bottom={r.bottom20_sign_acc_pct:6.2f}%"
        )

    print()
    print(f"Detailed CSV: {path}")
    print("=" * 120)


if __name__ == "__main__":
    main()
