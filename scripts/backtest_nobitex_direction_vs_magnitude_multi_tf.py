from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TIMEFRAMES = ("5m", "15m", "30m", "1h")
ROOT = Path("reports/nobitex_intraday_native")
NO_SIGNAL = {"DOSUSDT", "GRVTUSDT", "SLVONUSDT"}
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
    df["J_next"] = df["J"].shift(-1)
    df["U"] = df["J_next"] - df["J"]
    df["next_body_return"] = (
        (df["close"].shift(-1) - df["open"].shift(-1))
        / df["open"].shift(-1)
    )

    return df.dropna(subset=["J_next", "U", "next_body_return"]).reset_index(drop=True)


def fit_model(train: pd.DataFrame) -> np.ndarray:
    x = np.column_stack([np.ones(len(train)), train[["J", "K", "L"]].to_numpy(float)])
    y = train["U"].to_numpy(float)
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
        test["pred_U"] = predict(model, test)
        test["pred_J_next"] = test["J"] + test["pred_U"]
        test["magnitude_direction"] = np.where(test["pred_J_next"] >= 0, 1, -1)
        parts.append(test)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def evaluate(symbol: str, timeframe: str, pred: pd.DataFrame) -> dict:
    if pred.empty:
        return {"timeframe": timeframe, "symbol": symbol, "samples": 0}

    actual_direction = np.where(pred["next_body_return"] >= 0, 1, -1)
    formula_direction = pred["formula_direction"].to_numpy(int)
    magnitude_direction = pred["magnitude_direction"].to_numpy(int)

    actual_move = pred["J_next"].to_numpy(float)
    predicted_move = pred["pred_J_next"].to_numpy(float)
    actual_return = pred["next_body_return"].to_numpy(float)

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

    actual_open = pred["open"].shift(-1).to_numpy(float)
    valid = np.isfinite(predicted_move) & np.isfinite(actual_move) & np.isfinite(actual_open) & (actual_open != 0)
    error_pct = np.abs(predicted_move[valid] - actual_move[valid]) / actual_open[valid]
    actual_abs_pct = np.abs(actual_move[valid]) / actual_open[valid]

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


def main() -> None:
    all_rows = []

    print("=" * 180)
    print("NOBITEX NON-IRT | DIRECTION VS MAGNITUDE | NO COMMISSION | NEXT-CANDLE TEST")
    print("=" * 180)
    print("Formula direction : score <=1 SHORT | score >=2 LONG")
    print("Magnitude model    : U = J_next - J | X = J,K,L")
    print("Magnitude direction: sign(predicted J_next)")
    print("Evaluation         : next candle Open -> Close")
    print("Training           : walk-forward, no future leakage")
    print()

    for tf in TIMEFRAMES:
        data_dir = ROOT / tf
        files = sorted(data_dir.glob(f"*_{tf}.csv"))
        loaded = {}

        for path in files:
            symbol = path.stem[:-len(f"_{tf}")].upper()
            if symbol.endswith("IRT") or symbol in NO_SIGNAL:
                continue
            try:
                df = load(path)
                if len(df) >= MIN_TRAIN + 20:
                    loaded[symbol] = df
            except Exception as exc:
                print(f"SKIP {symbol} {tf}: {exc}")

        print(f"\nTIMEFRAME = {tf} | MARKETS = {len(loaded)}")
        rows = []

        for symbol in sorted(loaded):
            pred = walk_forward_predictions(loaded[symbol])
            if pred.empty:
                continue
            result = evaluate(symbol, tf, pred)
            rows.append(result)
            all_rows.append(result)

        if not rows:
            continue

        out = pd.DataFrame(rows)
        print("SYMBOL               N     F_ACC   F_RET     M_ACC   M_RET   CORR   MAE")
        print("-" * 90)
        for _, r in out.iterrows():
            print(
                f"{r['symbol']:20s} {int(r['samples']):6d} "
                f"{r['formula_accuracy_pct']:7.2f}% {r['formula_gross_return_pct']:+8.2f}% "
                f"{r['magnitude_direction_accuracy_pct']:7.2f}% {r['magnitude_gross_return_pct']:+8.2f}% "
                f"{r['magnitude_corr']:+6.3f} {r['magnitude_mae_pct']:6.3f}%"
            )

        print(f"\n{tf} AGGREGATE:")
        print(f"Formula direction accuracy : {out['formula_accuracy_pct'].mean():.2f}%")
        print(f"Formula gross return mean   : {out['formula_gross_return_pct'].mean():+.2f}%")
        print(f"Magnitude direction accuracy: {out['magnitude_direction_accuracy_pct'].mean():.2f}%")
        print(f"Magnitude gross return mean : {out['magnitude_gross_return_pct'].mean():+.2f}%")
        print(f"Magnitude correlation mean  : {out['magnitude_corr'].mean():+.3f}")
        print(f"Magnitude MAE mean          : {out['magnitude_mae_pct'].mean():.3f}%")

    result_df = pd.DataFrame(all_rows)
    output_dir = ROOT / "multi_tf"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "direction_vs_magnitude_by_asset.csv"
    result_df.to_csv(output, index=False)

    print()
    print("=" * 180)
    print(f"CSV: {output}")
    print("=" * 180)


if __name__ == "__main__":
    main()
