from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TIMEFRAMES = ("5m", "15m", "30m", "1h")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
ROOT = Path("reports/nobitex_intraday_native")
MIN_TRAIN = 50
FOLDS = 5


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
        parts.append(test)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def evaluate(symbol: str, timeframe: str, pred: pd.DataFrame) -> dict:
    if pred.empty:
        return {"timeframe": timeframe, "symbol": symbol, "samples": 0}

    actual_direction = np.where(pred["V"] >= 0, 1, -1)
    formula_direction = pred["formula_direction"].to_numpy(int)
    magnitude_direction = pred["magnitude_direction"].to_numpy(int)

    actual_move = pred["V"].to_numpy(float)
    predicted_move = pred["pred_V"].to_numpy(float)
    actual_return = pred["next_close_return"].to_numpy(float)

    formula_accuracy = (formula_direction == actual_direction).mean()
    magnitude_accuracy = (magnitude_direction == actual_direction).mean()

    formula_returns = formula_direction * actual_return
    magnitude_returns = magnitude_direction * actual_return

    formula_return = np.prod(1.0 + formula_returns) - 1.0
    magnitude_return = np.prod(1.0 + magnitude_returns) - 1.0

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

    return {
        "timeframe": timeframe,
        "symbol": symbol,
        "samples": len(pred),
        "formula_accuracy_pct": formula_accuracy * 100.0,
        "formula_gross_return_pct": formula_return * 100.0,
        "formula_mean_trade_pct": formula_returns.mean() * 100.0,
        "magnitude_direction_accuracy_pct": magnitude_accuracy * 100.0,
        "magnitude_gross_return_pct": magnitude_return * 100.0,
        "magnitude_mean_trade_pct": magnitude_returns.mean() * 100.0,
        "magnitude_corr": corr,
        "magnitude_mae_pct": error_pct.mean() * 100.0,
        "actual_mean_abs_move_pct": actual_abs_pct.mean() * 100.0,
    }


def aggregate_metrics(out: pd.DataFrame) -> dict[str, float]:
    total_samples = int(out["samples"].sum())
    formula_correct = (out["formula_accuracy_pct"] / 100.0 * out["samples"]).sum()
    magnitude_correct = (out["magnitude_direction_accuracy_pct"] / 100.0 * out["samples"]).sum()

    return {
        "formula_pooled_acc": 100.0 * formula_correct / total_samples,
        "formula_mean_acc": out["formula_accuracy_pct"].mean(),
        "formula_mean_return": out["formula_gross_return_pct"].mean(),
        "magnitude_pooled_acc": 100.0 * magnitude_correct / total_samples,
        "magnitude_mean_acc": out["magnitude_direction_accuracy_pct"].mean(),
        "magnitude_mean_return": out["magnitude_gross_return_pct"].mean(),
        "magnitude_mean_corr": out["magnitude_corr"].mean(),
        "magnitude_mean_mae": out["magnitude_mae_pct"].mean(),
        "actual_mean_abs_move": out["actual_mean_abs_move_pct"].mean(),
    }


def main() -> None:
    all_rows = []

    print("=" * 110)
    print("NOBITEX | MAJOR ASSETS | DIRECTION VS MAGNITUDE | NO COMMISSION")
    print("=" * 110)
    print("Target: current Close -> next Close | Model: V = Close_next - Close_current | X = J,K,L")
    print("Formula: score <=1 SHORT | score >=2 LONG | Walk-forward: 5 folds | MIN_TRAIN=50")
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
        print("SYMBOL      N    F_ACC  F_RET   M_ACC  M_RET   CORR    MAE")
        print("-" * 72)
        for _, r in out.iterrows():
            print(
                f"{r['symbol']:8s} {int(r['samples']):4d} "
                f"{r['formula_accuracy_pct']:7.2f}% {r['formula_gross_return_pct']:+7.2f}% "
                f"{r['magnitude_direction_accuracy_pct']:7.2f}% {r['magnitude_gross_return_pct']:+7.2f}% "
                f"{r['magnitude_corr']:+6.3f} {r['magnitude_mae_pct']:6.3f}%"
            )

        agg = aggregate_metrics(out)
        print(
            f"AGG {tf}: F_ACC={agg['formula_pooled_acc']:.2f}% "
            f"M_ACC={agg['magnitude_pooled_acc']:.2f}% "
            f"F_RET={agg['formula_mean_return']:+.2f}% "
            f"M_RET={agg['magnitude_mean_return']:+.2f}% "
            f"CORR={agg['magnitude_mean_corr']:+.3f} "
            f"MAE={agg['magnitude_mean_mae']:.3f}% "
            f"MOVE={agg['actual_mean_abs_move']:.3f}%"
        )
        print()

    result_df = pd.DataFrame(all_rows)
    output_dir = ROOT / "multi_tf"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "direction_vs_magnitude_major_assets.csv"
    result_df.to_csv(output, index=False)

    print("=" * 110)
    print(f"CSV: {output}")
    print("=" * 110)


if __name__ == "__main__":
    main()
