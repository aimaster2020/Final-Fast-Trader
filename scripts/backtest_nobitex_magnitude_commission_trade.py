from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TIMEFRAMES = ("5m", "15m", "30m", "1h")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
ROOT = Path("reports/nobitex_intraday_native")
MIN_TRAIN = 50
FOLDS = 5

# Real trading cost: 0.13% entry + 0.13% exit.
ENTRY_FEE_PCT = 0.13
EXIT_FEE_PCT = 0.13
ROUND_TRIP_FEE_PCT = ENTRY_FEE_PCT + EXIT_FEE_PCT
ENTRY_FEE = ENTRY_FEE_PCT / 100.0
EXIT_FEE = EXIT_FEE_PCT / 100.0

# Only enter when the predicted absolute move covers the full round-trip fee.
ENTRY_THRESHOLD_PCT = ROUND_TRIP_FEE_PCT


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    body = df["close"] - df["open"]
    df["J"] = body
    df["K"] = df["high"] - df["close"]
    df["L"] = df["close"] - df["low"]

    df["close_next"] = df["close"].shift(-1)
    df["V"] = df["close_next"] - df["close"]
    df["next_close_return"] = df["V"] / df["close"]

    return df.dropna(subset=["close_next", "V", "next_close_return"]).reset_index(drop=True)


def fit_model(train: pd.DataFrame) -> np.ndarray:
    x = np.column_stack(
        [np.ones(len(train)), train[["J", "K", "L"]].to_numpy(float)]
    )
    y = train["V"].to_numpy(float)
    return np.linalg.lstsq(x, y, rcond=None)[0]


def predict(model: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    x = np.column_stack(
        [np.ones(len(df)), df[["J", "K", "L"]].to_numpy(float)]
    )
    return x @ model


def walk_forward_predictions(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    initial_train = max(MIN_TRAIN, n // 2)
    remaining = n - initial_train
    if remaining < 20:
        return pd.DataFrame()

    fold_size = max(1, remaining // FOLDS)
    parts: list[pd.DataFrame] = []

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
        test["pred_move_pct"] = test["pred_V"] / test["close"] * 100.0
        test["pred_abs_move_pct"] = np.abs(test["pred_move_pct"])
        test["pred_direction"] = np.where(test["pred_V"] >= 0, 1, -1)
        test["actual_direction"] = np.where(test["V"] >= 0, 1, -1)
        test["take_trade"] = test["pred_abs_move_pct"] >= ENTRY_THRESHOLD_PCT
        parts.append(test)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def evaluate(symbol: str, timeframe: str, pred: pd.DataFrame) -> dict:
    if pred.empty:
        return {"timeframe": timeframe, "symbol": symbol, "samples": 0}

    selected = pred["take_trade"].to_numpy(bool)
    selected_n = int(selected.sum())
    samples = len(pred)

    baseline_accuracy = float(
        (pred["pred_direction"].to_numpy(int) == pred["actual_direction"].to_numpy(int)).mean()
    )

    if selected_n == 0:
        return {
            "timeframe": timeframe,
            "symbol": symbol,
            "samples": samples,
            "baseline_accuracy_pct": baseline_accuracy * 100.0,
            "trade_n": 0,
            "coverage_pct": 0.0,
            "trade_accuracy_pct": np.nan,
            "gross_return_pct": np.nan,
            "net_return_pct": np.nan,
            "avg_gross_trade_pct": np.nan,
            "avg_net_trade_pct": np.nan,
            "avg_predicted_move_pct": np.nan,
            "avg_actual_move_pct": np.nan,
        }

    s = pred.loc[selected].copy()
    direction = s["pred_direction"].to_numpy(float)
    actual_return = s["next_close_return"].to_numpy(float)
    gross_trade_returns = direction * actual_return

    # Apply fees multiplicatively, once on entry and once on exit.
    net_trade_returns = (1.0 + gross_trade_returns) * (1.0 - ENTRY_FEE) * (1.0 - EXIT_FEE) - 1.0

    trade_accuracy = float(
        (s["pred_direction"].to_numpy(int) == s["actual_direction"].to_numpy(int)).mean()
    )

    return {
        "timeframe": timeframe,
        "symbol": symbol,
        "samples": samples,
        "baseline_accuracy_pct": baseline_accuracy * 100.0,
        "trade_n": selected_n,
        "coverage_pct": 100.0 * selected_n / samples,
        "trade_accuracy_pct": trade_accuracy * 100.0,
        "gross_return_pct": (np.prod(1.0 + gross_trade_returns) - 1.0) * 100.0,
        "net_return_pct": (np.prod(1.0 + net_trade_returns) - 1.0) * 100.0,
        "avg_gross_trade_pct": gross_trade_returns.mean() * 100.0,
        "avg_net_trade_pct": net_trade_returns.mean() * 100.0,
        "avg_predicted_move_pct": s["pred_abs_move_pct"].mean(),
        "avg_actual_move_pct": np.abs(actual_return).mean() * 100.0,
    }


def aggregate(rows: pd.DataFrame) -> dict[str, float]:
    total_samples = int(rows["samples"].sum())
    total_trades = int(rows["trade_n"].sum())

    baseline_correct = (
        rows["baseline_accuracy_pct"] / 100.0 * rows["samples"]
    ).sum()
    trade_correct = (
        rows["trade_accuracy_pct"].fillna(0.0) / 100.0 * rows["trade_n"]
    ).sum()

    pooled_baseline_accuracy = 100.0 * baseline_correct / total_samples
    pooled_trade_accuracy = 100.0 * trade_correct / total_trades if total_trades else np.nan

    trade_returns = []
    net_returns = []
    for _, r in rows.iterrows():
        n = int(r["trade_n"])
        if n <= 0:
            continue
        # Rebuild a pooled sequence approximately from per-trade mean is impossible;
        # aggregate headline uses the mean across symbol-level compounded returns.
        trade_returns.append(r["gross_return_pct"])
        net_returns.append(r["net_return_pct"])

    return {
        "baseline_accuracy_pct": pooled_baseline_accuracy,
        "trade_accuracy_pct": pooled_trade_accuracy,
        "trade_n": total_trades,
        "coverage_pct": 100.0 * total_trades / total_samples if total_samples else np.nan,
        "mean_symbol_gross_return_pct": float(np.mean(trade_returns)) if trade_returns else np.nan,
        "mean_symbol_net_return_pct": float(np.mean(net_returns)) if net_returns else np.nan,
    }


def main() -> None:
    all_rows: list[dict] = []

    print("=" * 112)
    print("NOBITEX | MAJOR ASSETS | NEW MAGNITUDE + REAL COMMISSION TEST")
    print("=" * 112)
    print("Target: current Close -> next Close | V = Close_next - Close_current | X = J,K,L")
    print(f"Fees: {ENTRY_FEE_PCT:.2f}% entry + {EXIT_FEE_PCT:.2f}% exit = {ROUND_TRIP_FEE_PCT:.2f}% round trip")
    print(f"Entry rule: abs(predicted move) >= {ENTRY_THRESHOLD_PCT:.2f}%")
    print("Walk-forward: 5 folds | MIN_TRAIN=50 | no future leakage")
    print()

    for tf in TIMEFRAMES:
        rows: list[dict] = []

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
        print("SYMBOL      N   BASE_ACC  TRADES  COV    TR_ACC  GROSS    NET      AVG_NET")
        print("-" * 82)
        for _, r in out.iterrows():
            print(
                f"{r['symbol']:8s} {int(r['samples']):4d} "
                f"{r['baseline_accuracy_pct']:8.2f}% "
                f"{int(r['trade_n']):6d} {r['coverage_pct']:6.2f}% "
                f"{r['trade_accuracy_pct']:7.2f}% "
                f"{r['gross_return_pct']:+7.2f}% "
                f"{r['net_return_pct']:+7.2f}% "
                f"{r['avg_net_trade_pct']:+7.3f}%"
            )

        agg = aggregate(out)
        print(
            f"AGG {tf}: BASE_ACC={agg['baseline_accuracy_pct']:.2f}% "
            f"TR_ACC={agg['trade_accuracy_pct']:.2f}% "
            f"TRADES={agg['trade_n']} "
            f"COV={agg['coverage_pct']:.2f}% "
            f"MEAN_GROSS={agg['mean_symbol_gross_return_pct']:+.2f}% "
            f"MEAN_NET={agg['mean_symbol_net_return_pct']:+.2f}%"
        )
        print()

    result_df = pd.DataFrame(all_rows)
    output_dir = ROOT / "multi_tf"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "magnitude_commission_trade_major_assets.csv"
    result_df.to_csv(output, index=False)

    print("=" * 112)
    print(f"CSV: {output}")
    print("=" * 112)


if __name__ == "__main__":
    main()
