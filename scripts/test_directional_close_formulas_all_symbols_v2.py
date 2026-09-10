from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close"]
THRESHOLDS = (0.001, 0.0025, 0.005, 0.01)


def load_data(path: str, limit: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in REQUIRED)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED).copy()
    if limit > 0:
        df = df.tail(limit + 1).copy()
    df["J"] = df["close"] - df["open"]
    df["K"] = df["high"] - df["close"]
    df["L_checkpoint"] = df["close"] - df["low"]
    df["U"] = df["J"].shift(-1) - df["J"]
    df["next_close"] = df["close"].shift(-1)
    df["high_open_mag"] = np.abs(df["high"] - df["open"])
    df["low_open_mag"] = np.abs(df["low"] - df["open"])
    return df.dropna(subset=["U", "next_close"]).reset_index(drop=True)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = ["J", "K", "L_checkpoint"]
    x_train = np.column_stack([np.ones(len(train)), train[features].to_numpy(float)])
    y_train = train["U"].to_numpy(float)
    coef = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
    x_test = np.column_stack([np.ones(len(test)), test[features].to_numpy(float)])
    return y_train, x_test @ coef


def stats(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if len(actual) == 0:
        return {"n": 0, "mae": 0.0, "mape": 0.0, **{f"within_{int(t*10000)}bp": 0.0 for t in THRESHOLDS}}
    rel = np.abs(predicted - actual) / np.maximum(np.abs(actual), 1e-12)
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "mape": float(np.mean(rel)),
        "within_10bp": float(np.mean(rel <= THRESHOLDS[0])),
        "within_25bp": float(np.mean(rel <= THRESHOLDS[1])),
        "within_50bp": float(np.mean(rel <= THRESHOLDS[2])),
        "within_100bp": float(np.mean(rel <= THRESHOLDS[3])),
        "bias": float(np.mean(predicted - actual)),
    }


def run_symbol(path: Path, symbol: str, folds: int, train_ratio: float, detail_rows: list[dict]) -> list[dict]:
    df = load_data(str(path))
    n = len(df)
    initial_train = int(n * train_ratio)
    remaining = n - initial_train
    test_size = remaining // folds
    rows = []
    print("=" * 132)
    print(f"{symbol} — DIRECTION CHECKPOINT + DIRECTIONAL CLOSE FORMULAS V2")
    print("=" * 132)
    print(f"candles={n} train0={initial_train} folds={folds} test/fold={test_size}")
    print("checkpoint: U=J_next-J_current, features=J,K,L_checkpoint, expanding walk-forward")
    print("LONG = C + |HIGH-OPEN|   SHORT = C - |LOW-OPEN|")
    print("groups selected only from OOS checkpoint prediction; no threshold tuning")

    agg = {"LONG": [], "SHORT": []}
    direction_hits = 0
    direction_total = 0

    for fold in range(folds):
        start = initial_train + fold * test_size
        end = initial_train + (fold + 1) * test_size if fold < folds - 1 else n
        train = df.iloc[:start]
        test = df.iloc[start:end].copy()
        _, pred_u = fit_predict(train, test)
        predicted = np.sign(pred_u)
        actual_u = test["U"].to_numpy(float)
        actual = np.sign(actual_u)
        valid = np.isfinite(pred_u) & np.isfinite(actual_u)
        predicted = predicted[valid]
        actual = actual[valid]
        t = test.iloc[np.flatnonzero(valid)]

        dm = (predicted != 0) & (actual != 0)
        direction_hits += int(np.sum(predicted[dm] == actual[dm]))
        direction_total += int(np.sum(dm))

        close = t["close"].to_numpy(float)
        next_close = t["next_close"].to_numpy(float)
        long_pred = close + t["high_open_mag"].to_numpy(float)
        short_pred = close - t["low_open_mag"].to_numpy(float)

        mask_long = predicted > 0
        mask_short = predicted < 0
        long_stats = stats(next_close[mask_long], long_pred[mask_long])
        short_stats = stats(next_close[mask_short], short_pred[mask_short])
        agg["LONG"].append(long_stats)
        agg["SHORT"].append(short_stats)
        fold_acc = float(np.mean(predicted[dm] == actual[dm])) if dm.any() else 0.0
        print(f"fold {fold+1}: dir={fold_acc*100:.2f}% LONG_N={long_stats['n']} LONG_MAPE={long_stats['mape']*100:.3f}% LONG<=.25={long_stats['within_25bp']*100:.2f}% LONG<=.50={long_stats['within_50bp']*100:.2f}% SHORT_N={short_stats['n']} SHORT_MAPE={short_stats['mape']*100:.3f}% SHORT<=.25={short_stats['within_25bp']*100:.2f}% SHORT<=.50={short_stats['within_50bp']*100:.2f}%")

        for i in range(len(t)):
            direction_label = "LONG" if predicted[i] > 0 else "SHORT" if predicted[i] < 0 else "FLAT"
            actual_label = "UP" if actual[i] > 0 else "DOWN" if actual[i] < 0 else "FLAT"
            pclose = long_pred[i] if direction_label == "LONG" else short_pred[i] if direction_label == "SHORT" else np.nan
            err = abs(pclose - next_close[i]) / max(abs(next_close[i]), 1e-12) if np.isfinite(pclose) else np.nan
            detail_rows.append({
                "symbol": symbol,
                "fold": fold + 1,
                "row_index": int(start + np.flatnonzero(valid)[i]),
                "predicted_direction": direction_label,
                "actual_U_direction": actual_label,
                "open": float(t["open"].iloc[i]),
                "high": float(t["high"].iloc[i]),
                "low": float(t["low"].iloc[i]),
                "close": float(close[i]),
                "high_minus_open_abs": float(t["high_open_mag"].iloc[i]),
                "low_minus_open_abs": float(t["low_open_mag"].iloc[i]),
                "predicted_next_close": float(pclose) if np.isfinite(pclose) else np.nan,
                "actual_next_close": float(next_close[i]),
                "absolute_error": float(abs(pclose - next_close[i])) if np.isfinite(pclose) else np.nan,
                "relative_error": float(err) if np.isfinite(err) else np.nan,
                "within_0.10pct": bool(err <= 0.001) if np.isfinite(err) else False,
                "within_0.25pct": bool(err <= 0.0025) if np.isfinite(err) else False,
                "within_0.50pct": bool(err <= 0.005) if np.isfinite(err) else False,
                "within_1.00pct": bool(err <= 0.01) if np.isfinite(err) else False,
            })

    out = []
    for direction in ("LONG", "SHORT"):
        all_mask = []
        # Re-read detail rows for this symbol/direction for exact aggregate metrics.
        selected = [r for r in detail_rows if r["symbol"] == symbol and r["predicted_direction"] == direction]
        actual = np.array([r["actual_next_close"] for r in selected], dtype=float)
        pred = np.array([r["predicted_next_close"] for r in selected], dtype=float)
        s = stats(actual, pred)
        out.append({"symbol": symbol, "direction": direction, "direction_accuracy_pct": direction_hits / direction_total * 100 if direction_total else 0.0, **s})
        print(f"{direction} N={s['n']:4d} MAE={s['mae']:10.4f} MAPE={s['mape']*100:7.3f}% <=0.10%={s['within_10bp']*100:6.2f}% <=0.25%={s['within_25bp']*100:6.2f}% <=0.50%={s['within_50bp']*100:6.2f}% <=1.00%={s['within_100bp']*100:6.2f}% bias={s['bias']:+.4f}")
    print(f"DIRECTION_ACCURACY={direction_hits / direction_total * 100:.2f}% ({direction_hits}/{direction_total})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.50)
    ap.add_argument("--output-dir", default="reports/formula_direction_test")
    args = ap.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary: list[dict] = []
    details: list[dict] = []
    for symbol in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        path = Path(args.data_dir) / f"{symbol}_1h.csv"
        if not path.exists():
            print(f"SKIP {symbol}: missing {path}")
            continue
        all_summary.extend(run_symbol(path, symbol, args.folds, args.train_ratio, details))

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(details).to_csv(output_dir / "details.csv", index=False)

    print("=" * 132)
    print("ALL SYMBOL SUMMARY")
    print("=" * 132)
    cols = ["symbol", "direction", "n", "direction_accuracy_pct", "mape", "mae", "within_25bp", "within_50bp", "within_100bp", "bias"]
    for _, r in summary_df.iterrows():
        print(f"{r['symbol']:8} {r['direction']:5} N={int(r['n']):4d} dirAcc={r['direction_accuracy_pct']:6.2f}% MAPE={r['mape']*100:6.3f}% MAE={r['mae']:10.4f} <=.25={r['within_25bp']*100:6.2f}% <=.50={r['within_50bp']*100:6.2f}% <=1={r['within_100bp']*100:6.2f}% bias={r['bias']:+.4f}")
    print(f"summary_csv={output_dir / 'summary.csv'}")
    print(f"details_csv={output_dir / 'details.csv'}")


if __name__ == "__main__":
    main()
