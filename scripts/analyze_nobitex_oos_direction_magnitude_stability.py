from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MIN_CANDLES = 100
QUANTILES = [0.0, 0.50, 0.70, 0.80, 0.90]


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
    hc = df["high"] - df["close"]
    ho = df["high"] - df["open"]
    lc = df["low"] - df["close"]
    score = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)
    formula_dir = np.where(score <= 1, -1, np.where(score >= 2, 1, 0)).astype(int)

    upper = (df["high"] - df["open"]).abs()
    lower = (df["open"] - df["low"]).abs()
    side_formula = np.where(lower > upper, 1, np.where(upper > lower, -1, 0)).astype(int)
    signal_dir = formula_dir.copy()
    signal_dir[(score == 1) & (side_formula == 1)] = 0
    signal_dir[(score == 2) & (side_formula == -1)] = 0

    j = body
    k = df["high"] - df["close"]
    l = df["close"] - df["low"]
    u = j.shift(-1) - j
    next_body = j.shift(-1)

    out = df.copy()
    out["J"] = j
    out["K"] = k
    out["L"] = l
    out["U"] = u
    out["signal_dir"] = signal_dir
    out["actual_dir"] = np.sign(next_body)
    out["next_body_pct"] = next_body / df["close"].replace(0, np.nan)
    return out.dropna(subset=["U", "next_body_pct"]).reset_index(drop=True)


def fold_predictions(df: pd.DataFrame, folds: int, train_ratio: float) -> pd.DataFrame:
    n = len(df)
    initial_train = max(50, int(n * train_ratio))
    remaining = n - initial_train
    if remaining < max(20, folds * 10):
        return pd.DataFrame()
    test_size = remaining // folds
    parts = []
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
        x_test = np.column_stack([np.ones(len(test)), test[features].to_numpy(float)])
        test["pred_u"] = x_test @ coef
        test["pred_abs_u"] = test["pred_u"].abs()
        parts.append(test)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def stats(frame: pd.DataFrame, half_fee: float, round_fee: float) -> dict[str, float | int]:
    if frame.empty:
        return {"n": 0, "win": np.nan, "mean_signed": np.nan, "half": np.nan, "round": np.nan}
    signed = frame["signal_dir"] * frame["next_body_pct"] * 100.0
    return {
        "n": len(frame),
        "win": float((signed > 0).mean() * 100.0),
        "mean_signed": float(signed.mean()),
        "half": float((signed >= half_fee).mean() * 100.0),
        "round": float((signed >= round_fee).mean() * 100.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="OOS direction stability and magnitude ranking by Nobitex market.")
    ap.add_argument("--data-dir", default="reports/nobitex_1h")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.5)
    ap.add_argument("--half-fee", type=float, default=0.13)
    ap.add_argument("--round-fee", type=float, default=0.26)
    ap.add_argument("--output", default="reports/nobitex_1h/nonirt_oos_direction_magnitude_stability.csv")
    args = ap.parse_args()

    root = Path(args.data_dir)
    files = sorted(p for p in root.glob("*_1h.csv") if not p.stem[:-3].upper().endswith("IRT"))
    rows = []

    for path in files:
        symbol = path.stem[:-3]
        try:
            df = load(path)
            if len(df) < MIN_CANDLES:
                continue
            pred = fold_predictions(df, args.folds, args.train_ratio)
            if pred.empty:
                continue
            sig = pred[(pred["signal_dir"] != 0) & (pred["actual_dir"] != 0)].copy()
            if sig.empty:
                continue
            sig["mag_rank"] = sig["pred_abs_u"].rank(method="first", pct=True)
            base = stats(sig, args.half_fee, args.round_fee)
            row = {
                "symbol": symbol,
                "oos_n": base["n"],
                "base_win": base["win"],
                "base_mean_signed": base["mean_signed"],
                "base_half": base["half"],
                "base_round": base["round"],
            }
            for q in QUANTILES[1:]:
                sel = sig[sig["mag_rank"] >= q]
                s = stats(sel, args.half_fee, args.round_fee)
                label = f"q{int(q*100):02d}"
                row[f"{label}_n"] = s["n"]
                row[f"{label}_win"] = s["win"]
                row[f"{label}_mean_signed"] = s["mean_signed"]
                row[f"{label}_half"] = s["half"]
                row[f"{label}_round"] = s["round"]
            rows.append(row)
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No valid markets.")
    out.to_csv(args.output, index=False)

    print("=" * 120)
    print("NOBITEX NON-IRT | OOS DIRECTION STABILITY + MAGNITUDE")
    print("=" * 120)
    print(f"Markets found/tested : {len(files)} / {len(out)}")
    print(f"Walk-forward         : folds={args.folds} train_ratio={args.train_ratio:.2f}")
    print()
    print("OOS DIRECTION BASELINE")
    print(f"Mean market win      : {out.base_win.mean():.2f}%")
    print(f"Median market win    : {out.base_win.median():.2f}%")
    print(f">=50% markets        : {(out.base_win >= 50).sum()} / {len(out)}")
    print(f">=55% markets        : {(out.base_win >= 55).sum()} / {len(out)}")
    print(f">=60% markets        : {(out.base_win >= 60).sum()} / {len(out)}")
    print(f"<40% markets         : {(out.base_win < 40).sum()} / {len(out)}")
    print()
    for q in [50, 70, 80, 90]:
        label = f"q{q:02d}"
        print(f"{label}: mean_win={out[f'{label}_win'].mean():.2f}% median_win={out[f'{label}_win'].median():.2f}% "
              f"markets>=50={(out[f'{label}_win']>=50).sum()}/{len(out)} "
              f"mean_move={out[f'{label}_mean_signed'].mean():+.3f}% "
              f"half>=50={(out[f'{label}_half']>=50).sum()}/{len(out)} "
              f"round>=50={(out[f'{label}_round']>=50).sum()}/{len(out)}")

    print()
    print("SELECTED MAJOR MARKETS")
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]:
        r = out[out.symbol == symbol]
        if r.empty:
            continue
        r = r.iloc[0]
        print(f"{symbol:8s} base={r.base_win:6.2f}% q50={r.q50_win:6.2f}% q70={r.q70_win:6.2f}% q80={r.q80_win:6.2f}% q90={r.q90_win:6.2f}% "
              f"q80move={r.q80_mean_signed:+.3f}% q80round={r.q80_round:6.2f}%")

    print()
    print("ALL MARKETS — BASELINE OOS WIN DESC")
    for _, r in out.sort_values("base_win", ascending=False).iterrows():
        print(f"{r.symbol:20s} n={int(r.oos_n):4d} base={r.base_win:6.2f}% q80={r.q80_win:6.2f}% "
              f"q80move={r.q80_mean_signed:+.3f}% q80half={r.q80_half:6.2f}% q80round={r.q80_round:6.2f}%")

    print()
    print(f"Detailed CSV: {args.output}")
    print("=" * 120)


if __name__ == "__main__":
    main()
