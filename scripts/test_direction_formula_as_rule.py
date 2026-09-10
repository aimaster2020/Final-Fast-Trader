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


def accuracy(pred: np.ndarray, actual: np.ndarray) -> float:
    mask = (pred != 0) & (actual != 0)
    return float(np.mean(pred[mask] == actual[mask])) if mask.any() else 0.0


def eval_rule(pred: np.ndarray, actual: np.ndarray) -> tuple[float, int]:
    mask = (pred != 0) & (actual != 0)
    return (accuracy(pred[mask], actual[mask]), int(mask.sum())) if mask.any() else (0.0, 0)


def evaluate_symbol(path: Path, folds: int, train_ratio: float, output_dir: Path) -> pd.DataFrame:
    df = load(str(path))
    n = len(df)
    train0 = int(n * train_ratio)
    remaining = n - train0
    size = remaining // folds
    rows: list[dict[str, object]] = []
    details: list[pd.DataFrame] = []

    for fold in range(folds):
        start = train0 + fold * size
        end = train0 + (fold + 1) * size if fold < folds - 1 else n
        train = df.iloc[:start]
        test = df.iloc[start:end].copy()

        X = np.column_stack([np.ones(len(train)), train[["J", "K", "L_checkpoint"]].to_numpy(float)])
        y = train.U.to_numpy(float)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        Xt = np.column_stack([np.ones(len(test)), test[["J", "K", "L_checkpoint"]].to_numpy(float)])
        pred_u = Xt @ coef
        checkpoint = np.sign(pred_u)
        actual = np.sign(test.U.to_numpy(float))

        up_range = test.high.to_numpy(float) - test.open.to_numpy(float)
        down_range = test.open.to_numpy(float) - test.low.to_numpy(float)

        # Formula-derived direction: choose the side whose directional excursion is larger.
        formula_dir = np.where(up_range > down_range, 1.0, np.where(down_range > up_range, -1.0, 0.0))
        agreement = np.where(checkpoint == formula_dir, checkpoint, 0.0)
        # Formula can be used as an override only when its directional excursion is dominant.
        formula_override = formula_dir.copy()

        rules = {
            "checkpoint": checkpoint,
            "formula_only": formula_dir,
            "agreement_only": agreement,
            "checkpoint_then_formula": np.where(agreement != 0, agreement, 0.0),
            "formula_override": formula_override,
        }

        actual_mask = actual != 0
        for name, pred in rules.items():
            m = actual_mask & (pred != 0)
            acc = float(np.mean(pred[m] == actual[m])) if m.any() else 0.0
            rows.append(
                {
                    "symbol": path.stem.replace("_1h", ""),
                    "fold": fold + 1,
                    "rule": name,
                    "n": int(m.sum()),
                    "coverage_pct": float(m.sum() / actual_mask.sum() * 100.0) if actual_mask.any() else 0.0,
                    "accuracy_pct": acc * 100.0,
                    "checkpoint_accuracy_pct": accuracy(checkpoint[actual_mask], actual[actual_mask]) * 100.0,
                }
            )

        details.append(
            pd.DataFrame(
                {
                    "fold": fold + 1,
                    "row_index": test.index.to_numpy(),
                    "open": test.open.to_numpy(float),
                    "high": test.high.to_numpy(float),
                    "low": test.low.to_numpy(float),
                    "close": test.close.to_numpy(float),
                    "high_minus_open": up_range,
                    "open_minus_low": down_range,
                    "range_ratio_up_down": np.divide(up_range, np.maximum(down_range, 1e-12)),
                    "checkpoint_dir": checkpoint,
                    "formula_dir": formula_dir,
                    "agreement_dir": agreement,
                    "actual_U_dir": actual,
                    "agreement": (checkpoint == formula_dir) & (formula_dir != 0),
                }
            )
        )

    summary = pd.DataFrame(rows)
    details_df = pd.concat(details, ignore_index=True)
    symbol = path.stem.replace("_1h", "")
    output_dir.mkdir(parents=True, exist_ok=True)
    details_df.to_csv(output_dir / f"{symbol}_direction_rule_details.csv", index=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Test H-O / O-L as an additional direction rule")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=SYMBOLS_DEFAULT)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.50)
    ap.add_argument("--output-dir", default="reports/direction_formula_rule")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    all_summary = []
    print("=" * 132)
    print("DIRECTION FORMULA AS AN ADDITIONAL RULE — OOS WALK-FORWARD")
    print("=" * 132)
    print("checkpoint = original U=J_next-J_current, features=J,K,L_checkpoint")
    print("formula direction = LONG when HIGH-OPEN > OPEN-LOW; SHORT when OPEN-LOW > HIGH-OPEN")
    print("agreement_only = keep checkpoint direction only when formula direction agrees")
    print("no future information, no threshold tuning")
    print("symbol rule N coverage accuracy checkpoint_acc")

    for symbol in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        path = Path(args.data_dir) / f"{symbol}_1h.csv"
        summary = evaluate_symbol(path, args.folds, args.train_ratio, output_dir)
        all_summary.append(summary)

    result = pd.concat(all_summary, ignore_index=True)
    result.to_csv(output_dir / "direction_rule_summary.csv", index=False)
    print("-" * 132)

    for symbol in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        s = result[result.symbol == symbol]
        for rule in ["checkpoint", "formula_only", "agreement_only", "checkpoint_then_formula", "formula_override"]:
            r = s[s.rule == rule]
            n = int(r.n.sum())
            correct = float(np.sum(r.n.to_numpy(float) * r.accuracy_pct.to_numpy(float) / 100.0))
            acc = correct / n * 100.0 if n else 0.0
            coverage = n / max(int(0.5 * (len(load(str(Path(args.data_dir) / f"{symbol}_1h.csv"))))), 1) * 100.0
            print(f"{symbol:8} {rule:22} N={n:4d} accuracy={acc:6.2f}%")

    print("summary_csv=" + str(output_dir / "direction_rule_summary.csv"))
    print("details_csv=" + str(output_dir / "<SYMBOL>_direction_rule_details.csv"))


if __name__ == "__main__":
    main()
