from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TIMEFRAMES = ("5m", "15m", "30m", "1h")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
ROOT = Path("reports/nobitex_intraday_native")
MIN_TRAIN = 50
FOLDS = 5

# Commission reference: 0.13% entry + 0.13% exit = 0.26% round trip.
# This test keeps only predictions whose absolute expected move covers 50%
# of the round-trip commission: 0.13%.
ROUND_TRIP_FEE_PCT = 0.26
HALF_ROUND_TRIP_FEE_PCT = ROUND_TRIP_FEE_PCT / 2.0
ROUND_TRIP_FEE = ROUND_TRIP_FEE_PCT / 100.0
HALF_ROUND_TRIP_FEE = HALF_ROUND_TRIP_FEE_PCT / 100.0


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    body = df["close"] - df["open"]
    hc = df["high"] - df["close"]
    ho = df["high"] - df["open"]
    lc = df["low"] - df["close"]
    score = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)

    df["formula_score"] = score
    df["formula_direction"] = np.where(score <= 1, -1, 1)

    df["J"] = body
    df["K"] = df["high"] - df["close"]
    df["L"] = df["close"] - df["low"]

    # Original body-change target, kept for diagnostics only.
    df["J_next"] = df["J"].shift(-1)
    df["U"] = df["J_next"] - df["J"]

    # True trading target: current close -> next close.
    df["close_next"] = df["close"].shift(-1)
    df["V"] = df["close_next"] - df["close"]
    df["next_close_return"] = df["V"] / df["close"]

    return df.dropna(subset=["J_next", "U", "close_next", "V", "next_close_return"]).reset_index(drop=True)


def fit_model(train: pd.DataFrame) -> np.ndarray:
    x = np.column_stack([np.ones(len(train)), train[["J", "K", "L"]].to_numpy(float)])
    y = train["V"].to_numpy(float)
    return np.linalg.lstsq(x, y, rcond=None)[0]


def predict(model: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    x = np.column_stack([np.ones(len(df)), df[["J", "K", "L"]].to_numpy(float)])
    return x @ model


def walk_forward_predictions(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    initial_train = max(MIN_TRAIN, n // 2)
    remaining = n - initial_train
    if remaining < 20:
        return pd.DataFrame()

    fold_size = max(1, remaining // FOLDS)
    parts = []

    for fold in range(FOLDS):
        test_start = initial_train + fold * fold_size
        test_end = n if fold == FOLDS - 1 else min(n, initial_train + (fold + 1) * fold_size)
        if test_start >= n:
            continue

        train = df.iloc[:test_start]
        test = df.iloc[test_start:test_end].copy()
        if len(train) < MIN_TRAIN or test.empty:
            continue

        model = fit_model(train)
        test["pred_V"] = predict(model, test)
        test["magnitude_direction"] = np.where(test["pred_V"] >= 0, 1, -1)
        test["pred_abs_move_pct"] = np.abs(test["pred_V"]) / np.abs(test["close"]) * 100.0
        test["magnitude_threshold"] = test["pred_abs_move_pct"] >= HALF_ROUND_TRIP_FEE_PCT
        parts.append(test)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def evaluate(symbol: str, timeframe: str, pred: pd.DataFrame) -> dict:
    if pred.empty:
        return {"timeframe": timeframe, "symbol": symbol, "samples": 0}

    actual_direction = np.where(pred["V"] >= 0, 1, -1)
    magnitude_direction = pred["magnitude_direction"].to_numpy(int)
    actual_move = pred["V"].to_numpy(float)
    predicted_move = pred["pred_V"].to_numpy(float)
    actual_return = pred["next_close_return"].to_numpy(float)
    magnitude_accuracy = (magnitude_direction == actual_direction).mean()

    magnitude_returns = magnitude_direction * actual_return

    if np.std(predicted_move) > 0 and np.std(actual_move) > 0:
        corr = float(np.corrcoef(predicted_move, actual_move)[0, 1])
    else:
        corr = np.nan

    current_close = pred["close"].to_numpy(float)
    valid = (
        np.isfinite(predicted_move)
        & np.isfinite(actual_move)
        & np.isfinite(current_close)
        & (current_close != 0)
    )
    error_pct = np.abs(predicted_move[valid] - actual_move[valid]) / np.abs(current_close[valid])
    actual_abs_pct = np.abs(actual_move[valid]) / np.abs(current_close[valid])

    selected = pred["magnitude_threshold"].to_numpy(bool)
    selected_n = int(selected.sum())
    selected_coverage_pct = 100.0 * selected_n / len(pred)

    if selected_n:
        selected_actual_direction = actual_direction[selected]
        selected_direction = magnitude_direction[selected]
        selected_returns = magnitude_returns[selected]
        selected_accuracy = float((selected_direction == selected_actual_direction).mean())
        selected_gross_return = float(np.prod(1.0 + selected_returns) - 1.0)
        selected_fee_adjusted_returns = selected_returns - ROUND_TRIP_FEE
        selected_fee_adjusted_return = float(np.prod(1.0 + selected_fee_adjusted_returns) - 1.0)
        selected_abs_move_pct = float(np.abs(actual_return[selected]).mean() * 100.0)
        selected_mean_predicted_move_pct = float(np.abs(predicted_move[selected] / current_close[selected]).mean() * 100.0)
    else:
        selected_accuracy = np.nan
        selected_gross_return = np.nan
        selected_fee_adjusted_return = np.nan
        selected_abs_move_pct = np.nan
        selected_mean_predicted_move_pct = np.nan

    return {
        "timeframe": timeframe,
        "symbol": symbol,
        "samples": len(pred),
        "magnitude_direction_accuracy_pct": magnitude_accuracy * 100.0,
        "magnitude_gross_return_pct": (np.prod(1.0 + magnitude_returns) - 1.0) * 100.0,
        "magnitude_corr": corr,
        "magnitude_mae_pct": error_pct.mean() * 100.0,
        "actual_mean_abs_move_pct": actual_abs_pct.mean() * 100.0,
        "threshold_n": selected_n,
        "threshold_coverage_pct": selected_coverage_pct,
        "threshold_accuracy_pct": selected_accuracy * 100.0 if np.isfinite(selected_accuracy) else np.nan,
        "threshold_gross_return_pct": selected_gross_return * 100.0 if np.isfinite(selected_gross_return) else np.nan,
        "threshold_fee_adjusted_return_pct": selected_fee_adjusted_return * 100.0 if np.isfinite(selected_fee_adjusted_return) else np.nan,
        "threshold_actual_mean_abs_move_pct": selected_abs_move_pct,
        "threshold_mean_predicted_abs_move_pct": selected_mean_predicted_move_pct,
    }


def aggregate_metrics(out: pd.DataFrame) -> dict[str, float]:
    total_samples = int(out["samples"].sum())
    magnitude_correct = (out["magnitude_direction_accuracy_pct"] / 100.0 * out["samples"]).sum()
    threshold_n = int(out["threshold_n"].sum())
    threshold_correct = (
        out["threshold_accuracy_pct"].fillna(0.0) / 100.0 * out["threshold_n"]
    ).sum()

    base_acc = 100.0 * magnitude_correct / total_samples
    threshold_acc = 100.0 * threshold_correct / threshold_n if threshold_n else np.nan

    return {
        "magnitude_pooled_acc": base_acc,
        "threshold_pooled_acc": threshold_acc,
        "delta_pp": threshold_acc - base_acc if np.isfinite(threshold_acc) else np.nan,
        "threshold_n": threshold_n,
        "threshold_coverage": 100.0 * threshold_n / total_samples if total_samples else np.nan,
        "threshold_mean_return": out.loc[out["threshold_n"] > 0, "threshold_gross_return_pct"].mean(),
        "threshold_mean_fee_adjusted_return": out.loc[out["threshold_n"] > 0, "threshold_fee_adjusted_return_pct"].mean(),
        "threshold_mean_abs_move": out.loc[out["threshold_n"] > 0, "threshold_actual_mean_abs_move_pct"].mean(),
        "threshold_mean_predicted_abs_move": out.loc[out["threshold_n"] > 0, "threshold_mean_predicted_abs_move_pct"].mean(),
    }


def main() -> None:
    all_rows = []

    print("=" * 110)
    print("NOBITEX | MAJOR ASSETS | MAGNITUDE THRESHOLD TEST")
    print("=" * 110)
    print("Target: current Close -> next Close | Model: V = Close_next - Close_current | X = J,K,L")
    print("Threshold: |predicted move| >= 0.13% = 50% of 0.26% round-trip commission")
    print("Walk-forward: 5 folds | MIN_TRAIN=50 | Baseline has no threshold")
    print()

    for tf in TIMEFRAMES:
        rows = []

        for symbol in SYMBOLS:
            path = ROOT / tf / f"{symbol}_{tf}.csv"
            if not path.exists():
                print(f"MISSING {tf} {symbol}: {path}")
                continue

            try:
                df = load(path)
                if len(df) < MIN_TRAIN + 20:
                    print(f"SKIP {tf} {symbol}: only {len(df)} rows")
                    continue
                pred = walk_forward_predictions(df)
                result = evaluate(symbol, tf, pred)
                if result["samples"]:
                    rows.append(result)
                    all_rows.append(result)
            except Exception as exc:
                print(f"SKIP {tf} {symbol}: {exc}")

        if not rows:
            continue

        out = pd.DataFrame(rows).sort_values("symbol")
        print(f"TIMEFRAME = {tf}")
        print("SYMBOL      N   M_ACC  TH_N  COV    TH_ACC  TH_RET  TH_NET")
        print("-" * 72)
        for _, r in out.iterrows():
            print(
                f"{r['symbol']:8s} {int(r['samples']):4d} "
                f"{r['magnitude_direction_accuracy_pct']:7.2f}% "
                f"{int(r['threshold_n']):5d} {r['threshold_coverage_pct']:6.2f}% "
                f"{r['threshold_accuracy_pct']:7.2f}% "
                f"{r['threshold_gross_return_pct']:+7.2f}% "
                f"{r['threshold_fee_adjusted_return_pct']:+7.2f}%"
            )

        agg = aggregate_metrics(out)
        print(
            f"AGG {tf}: BASE_M_ACC={agg['magnitude_pooled_acc']:.2f}% "
            f"TH_M_ACC={agg['threshold_pooled_acc']:.2f}% "
            f"DELTA={agg['delta_pp']:+.2f}pp "
            f"COV={agg['threshold_coverage']:.2f}% "
            f"TH_RET={agg['threshold_mean_return']:+.2f}% "
            f"TH_NET={agg['threshold_mean_fee_adjusted_return']:+.2f}%"
        )
        print()

    result_df = pd.DataFrame(all_rows)
    output_dir = ROOT / "multi_tf"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "magnitude_half_commission_threshold_major_assets.csv"
    result_df.to_csv(output, index=False)

    print("=" * 110)
    print(f"CSV: {output}")
    print("=" * 110)


if __name__ == "__main__":
    main()
