from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
FEATURE_SETS = {
    "J": ["J"],
    "J_L": ["J", "L"],
    "J_K_L": ["J", "K", "L"],
}


def corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fit_model(train: pd.DataFrame, features: list[str]) -> np.ndarray:
    data = train[features + ["U"]].apply(pd.to_numeric, errors="coerce").dropna()
    X = np.column_stack([np.ones(len(data)), data[features].to_numpy(float)])
    y = data["U"].to_numpy(float)
    return np.linalg.lstsq(X, y, rcond=None)[0]


def predict(row: pd.Series, beta: np.ndarray, features: list[str]) -> float:
    x = np.array([1.0] + [float(row[f]) for f in features])
    return float(x @ beta)


def metric_block(actual_u: np.ndarray, pred_u: np.ndarray, j: np.ndarray, j_next: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(actual_u) & np.isfinite(pred_u) & np.isfinite(j) & np.isfinite(j_next)
    u = actual_u[mask]
    p = pred_u[mask]
    j0 = j[mask]
    jn = j_next[mask]
    if len(u) == 0:
        return {"n": 0, "mae_u": np.nan, "rmse_u": np.nan, "corr_u": np.nan, "r2_u": np.nan, "u_sign": np.nan, "jnext_mae": np.nan, "abs_jnext_mae": np.nan, "abs_corr": np.nan, "jnext_sign": np.nan}

    err = u - p
    ss_tot = np.sum((u - u.mean()) ** 2)
    r2 = 1.0 - np.sum(err**2) / ss_tot if ss_tot > 0 else np.nan
    jn_hat = j0 + p
    sign_mask = jn != 0

    return {
        "n": int(len(u)),
        "mae_u": float(np.mean(np.abs(err))),
        "rmse_u": float(np.sqrt(np.mean(err**2))),
        "corr_u": corr(u, p),
        "r2_u": float(r2),
        "u_sign": float(np.mean(np.sign(u) == np.sign(p))),
        "jnext_mae": float(np.mean(np.abs(jn - jn_hat))),
        "abs_jnext_mae": float(np.mean(np.abs(np.abs(jn) - np.abs(jn_hat)))),
        "abs_corr": corr(np.abs(jn), np.abs(jn_hat)),
        "jnext_sign": float(np.mean(np.sign(jn[sign_mask]) == np.sign(jn_hat[sign_mask]))) if sign_mask.any() else np.nan,
    }


def load_raw(path: Path, month: str, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol].copy()
    elif "timestamp" in df.columns:
        pass
    else:
        raise ValueError(f"{path}: missing symbol/timestamp columns")

    required = ["J", "K", "L", "J_next", "U"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "timestamp" in df.columns:
        try:
            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            if ts.notna().any():
                df = df.assign(_timestamp=ts).sort_values("_timestamp")
        except Exception:
            pass

    if month and "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.loc[ts.dt.strftime("%Y-%m") == month].copy()

    return df.dropna(subset=required).reset_index(drop=True)


def run_symbol(df: pd.DataFrame, min_train: int) -> dict[str, dict[str, float]]:
    if len(df) <= min_train:
        raise ValueError(f"Need more than min_train={min_train} rows; got {len(df)}")

    out: dict[str, dict[str, float]] = {}
    actual = {name: [] for name in ["BASELINE_-J", *FEATURE_SETS.keys()]}
    preds = {name: [] for name in ["BASELINE_-J", *FEATURE_SETS.keys()]}
    js = []
    jnexts = []

    for i in range(min_train, len(df)):
        train = df.iloc[:i]
        row = df.iloc[i]
        j = float(row["J"])
        u = float(row["U"])
        jn = float(row["J_next"])

        actual["BASELINE_-J"].append(u)
        preds["BASELINE_-J"].append(-j)
        for name, features in FEATURE_SETS.items():
            beta = fit_model(train, features)
            actual[name].append(u)
            preds[name].append(predict(row, beta, features))
        js.append(j)
        jnexts.append(jn)

    for name in actual:
        out[name] = metric_block(
            np.asarray(actual[name], dtype=float),
            np.asarray(preds[name], dtype=float),
            np.asarray(js, dtype=float),
            np.asarray(jnexts, dtype=float),
        )
    return out


def fmt(x: float, digits: int = 5) -> str:
    return "NA" if not np.isfinite(x) else f"{x:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanding walk-forward OOS test for next-candle body magnitude")
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument("--input", default=None, help="Raw body magnitude CSV")
    parser.add_argument("--min-train", type=int, default=200)
    parser.add_argument("--output", default=None, help="TXT output path")
    args = parser.parse_args()

    input_path = Path(args.input or f"reports/body_magnitude_raw_{args.month}.csv")
    output_path = Path(args.output or f"reports/body_magnitude_walkforward_{args.month}.txt")
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    lines = [
        "BODY MAGNITUDE WALK-FORWARD",
        f"month={args.month}",
        f"input={input_path}",
        f"min_train={args.min_train}",
        "training=expanding_history_only",
        "target=U=J_next-J",
        "",
    ]

    pooled = {name: {"u": [], "p": [], "j": [], "jn": []} for name in ["BASELINE_-J", *FEATURE_SETS.keys()]}

    for symbol in SYMBOLS:
        part = load_raw(input_path, args.month, symbol)
        results = run_symbol(part, args.min_train)
        lines.append(f"[{symbol}] n={len(part)} oos={len(part) - args.min_train}")
        for name, m in results.items():
            lines.append(
                f"{name}: n={m['n']} MAE_U={fmt(m['mae_u'])} RMSE_U={fmt(m['rmse_u'])} "
                f"corr_U={fmt(m['corr_u'])} R2_U={fmt(m['r2_u'])} "
                f"U_sign={fmt(m['u_sign'] * 100, 2)}% Jnext_MAE={fmt(m['jnext_mae'])} "
                f"absJnext_MAE={fmt(m['abs_jnext_mae'])} abs_corr={fmt(m['abs_corr'])} "
                f"Jnext_sign={fmt(m['jnext_sign'] * 100, 2)}%"
            )
        lines.append("")

        # Rebuild the OOS arrays once for pooled metrics.
        for i in range(args.min_train, len(part)):
            train = part.iloc[:i]
            row = part.iloc[i]
            j = float(row["J"])
            u = float(row["U"])
            jn = float(row["J_next"])
            pooled["BASELINE_-J"]["u"].append(u)
            pooled["BASELINE_-J"]["p"].append(-j)
            pooled["BASELINE_-J"]["j"].append(j)
            pooled["BASELINE_-J"]["jn"].append(jn)
            for name, features in FEATURE_SETS.items():
                beta = fit_model(train, features)
                pooled[name]["u"].append(u)
                pooled[name]["p"].append(predict(row, beta, features))
                pooled[name]["j"].append(j)
                pooled[name]["jn"].append(jn)

    lines.append("[ALL_POOLED]")
    for name, data in pooled.items():
        m = metric_block(
            np.asarray(data["u"], dtype=float),
            np.asarray(data["p"], dtype=float),
            np.asarray(data["j"], dtype=float),
            np.asarray(data["jn"], dtype=float),
        )
        lines.append(
            f"{name}: n={m['n']} MAE_U={fmt(m['mae_u'])} RMSE_U={fmt(m['rmse_u'])} "
            f"corr_U={fmt(m['corr_u'])} R2_U={fmt(m['r2_u'])} "
            f"U_sign={fmt(m['u_sign'] * 100, 2)}% Jnext_MAE={fmt(m['jnext_mae'])} "
            f"absJnext_MAE={fmt(m['abs_jnext_mae'])} abs_corr={fmt(m['abs_corr'])} "
            f"Jnext_sign={fmt(m['jnext_sign'] * 100, 2)}%"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("output=" + str(output_path))
    print("-----")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
