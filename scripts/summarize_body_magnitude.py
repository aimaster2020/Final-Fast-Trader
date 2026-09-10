from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


def corr(a: pd.Series, b: pd.Series) -> float:
    data = pd.concat([a, b], axis=1).apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 2:
        return float("nan")
    return float(data.iloc[:, 0].corr(data.iloc[:, 1]))


def regression(df: pd.DataFrame, features: list[str], target: str = "U") -> dict[str, float]:
    data = df[features + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < len(features) + 2:
        return {"n": float(len(data)), "r2": float("nan"), "mae": float("nan"), "rmse": float("nan"), "corr": float("nan"), "intercept": float("nan"), **{f"coef_{x}": float("nan") for x in features}}

    X = np.column_stack([np.ones(len(data)), data[features].to_numpy(float)])
    y = data[target].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    err = y - pred
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum(err ** 2) / ss_tot if ss_tot > 0 else float("nan")

    out = {
        "n": float(len(data)),
        "r2": float(r2),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "corr": float(np.corrcoef(pred, y)[0, 1]),
        "intercept": float(beta[0]),
    }
    for i, name in enumerate(features, 1):
        out[f"coef_{name}"] = float(beta[i])
    return out


def fmt(x: float, digits: int = 6) -> str:
    return "NA" if not np.isfinite(x) else f"{x:.{digits}f}"


def summarize(df: pd.DataFrame, label: str) -> list[str]:
    lines = [f"[{label}]", f"n={len(df)}"]
    if df.empty:
        return lines + [""]

    for col in ["J", "K", "L", "U", "body_pct", "U_pct", "abs_U_pct"]:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if not s.empty:
            lines.append(f"{col}: mean={fmt(s.mean())} median={fmt(s.median())} std={fmt(s.std(ddof=1))} min={fmt(s.min())} max={fmt(s.max())}")

    lines += ["", "CORRELATIONS"]
    lines.append(f"corr(U,J)={fmt(corr(df['U'], df['J']))}")
    lines.append(f"corr(U,K)={fmt(corr(df['U'], df['K']))}")
    lines.append(f"corr(U,L)={fmt(corr(df['U'], df['L']))}")
    lines.append(f"corr(U,J+L)={fmt(corr(df['U'], df['J'] + df['L']))}")
    lines.append(f"corr(U,J+K)={fmt(corr(df['U'], df['J'] + df['K']))}")
    lines.append(f"corr(U,K+L)={fmt(corr(df['U'], df['K'] + df['L']))}")
    lines.append(f"corr(J_next,J)={fmt(corr(df['J_next'], df['J']))}")

    lines += ["", "REGRESSION_U"]
    for features in [["J"], ["L"], ["J", "L"], ["J", "K", "L"]]:
        r = regression(df, features)
        parts = [fmt(r["intercept"])]
        for feature in features:
            c = r[f"coef_{feature}"]
            parts.append(f"{'+' if c >= 0 else '-'} {fmt(abs(c))}*{feature}")
        lines.append(f"features={','.join(features)} n={int(r['n'])} R2={fmt(r['r2'])} MAE={fmt(r['mae'])} RMSE={fmt(r['rmse'])} corr(pred,U)={fmt(r['corr'])}")
        lines.append("formula: U = " + " ".join(parts))

    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument("--input", default=None, help="Raw CSV path")
    parser.add_argument("--output", default=None, help="TXT output path")
    args = parser.parse_args()

    input_path = Path(args.input or f"reports/body_magnitude_raw_{args.month}.csv")
    output_path = Path(args.output or f"reports/body_magnitude_summary_{args.month}.txt")

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    required = ["symbol", "J", "K", "L", "J_next", "U"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    lines = [
        "BODY MAGNITUDE SUMMARY",
        f"month={args.month}",
        f"rows={len(df)}",
        f"symbols={','.join(sorted(df['symbol'].dropna().unique()))}",
        "",
    ]
    lines += summarize(df, "ALL")

    for symbol in SYMBOLS:
        part = df[df["symbol"] == symbol].copy()
        lines += summarize(part, symbol)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("output=" + str(output_path))
    print("-----")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
