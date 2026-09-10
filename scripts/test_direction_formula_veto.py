from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS_DEFAULT = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=lambda c: str(c).strip().lower() in {"open", "high", "low", "close"},
    )
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    df["J"] = df.close - df.open
    df["K"] = df.high - df.close
    df["L_checkpoint"] = df.close - df.low
    df["U"] = df.J.shift(-1) - df.J
    return df.dropna(subset=["U"]).reset_index(drop=True)


def checkpoint_predictions(df: pd.DataFrame, folds: int, train_ratio: float):
    n = len(df)
    train0 = int(n * train_ratio)
    rem = n - train0
    size = rem // folds
    pred_all = []
    actual_all = []
    formula_all = []
    fold_all = []

    for fold in range(folds):
        start = train0 + fold * size
        end = train0 + (fold + 1) * size if fold < folds - 1 else n
        train = df.iloc[:start]
        test = df.iloc[start:end]
        X = np.column_stack(
            [np.ones(len(train)), train[["J", "K", "L_checkpoint"]].to_numpy(float)]
        )
        y = train.U.to_numpy(float)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        Xt = np.column_stack(
            [np.ones(len(test)), test[["J", "K", "L_checkpoint"]].to_numpy(float)]
        )
        pred_u = Xt @ coef
        pred_dir = np.sign(pred_u)
        actual_dir = np.sign(test.U.to_numpy(float))
        formula_dir = np.where(
            test.high.to_numpy(float) - test.open.to_numpy(float)
            > test.open.to_numpy(float) - test.low.to_numpy(float),
            1.0,
            -1.0,
        )
        pred_all.append(pred_dir)
        actual_all.append(actual_dir)
        formula_all.append(formula_dir)
        fold_all.append(np.full(len(test), fold + 1, dtype=int))

    return (
        np.concatenate(pred_all),
        np.concatenate(actual_all),
        np.concatenate(formula_all),
        np.concatenate(fold_all),
    )


def report(name: str, mask: np.ndarray, pred: np.ndarray, actual: np.ndarray):
    valid = mask & (pred != 0) & (actual != 0)
    n = int(valid.sum())
    if n == 0:
        print(f"{name:30} N=0 coverage=0.00% accuracy=0.00%")
        return {"rule": name, "n": 0, "coverage": 0.0, "accuracy": 0.0}
    acc = float(np.mean(pred[valid] == actual[valid]))
    coverage = n / len(pred)
    print(f"{name:30} N={n:4d} coverage={coverage*100:6.2f}% accuracy={acc*100:6.2f}%")
    return {"rule": name, "n": n, "coverage": coverage, "accuracy": acc}


def main() -> None:
    ap = argparse.ArgumentParser(description="Test formula direction as a veto/filter on the original checkpoint")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=SYMBOLS_DEFAULT)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.50)
    ap.add_argument("--output-dir", default="reports/direction_formula_veto")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    print("=" * 116)
    print("FORMULA AS A VETO / ANTI-AGREEMENT FILTER — OOS WALK-FORWARD")
    print("=" * 116)
    print("checkpoint = original U=J_next-J_current, features=J,K,L_checkpoint")
    print("formula_dir = LONG when HIGH-OPEN > OPEN-LOW, else SHORT")
    print("key hypothesis: checkpoint + formula AGREEMENT is weak, while DISAGREEMENT may identify stronger checkpoint cases")
    print("no future information, no threshold tuning")

    for symbol in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        df = load(str(Path(args.data_dir) / f"{symbol}_1h.csv"))
        pred, actual, formula, fold = checkpoint_predictions(df, args.folds, args.train_ratio)
        agree = pred == formula
        disagree = pred != formula
        print(f"\n{symbol}")
        print("-" * 116)
        for rule, mask in (
            ("checkpoint_all", np.ones(len(pred), dtype=bool)),
            ("checkpoint_when_formula_agrees", agree),
            ("checkpoint_when_formula_disagrees", disagree),
            ("formula_only", np.ones(len(pred), dtype=bool)),
        ):
            p = pred if rule != "formula_only" else formula
            row = report(rule, mask, p, actual)
            row["symbol"] = symbol
            rows.append(row)

        # Per-fold stability for the potentially useful veto rule.
        for f in range(1, args.folds + 1):
            m = (fold == f) & disagree & (pred != 0) & (actual != 0)
            if m.any():
                print(
                    f"  fold {f}: disagreement N={int(m.sum()):4d} "
                    f"accuracy={np.mean(pred[m] == actual[m])*100:6.2f}%"
                )

    result = pd.DataFrame(rows)
    result.to_csv(out / "direction_formula_veto_summary.csv", index=False)
    print("=" * 116)
    print(f"summary_csv={out / 'direction_formula_veto_summary.csv'}")


if __name__ == "__main__":
    main()
