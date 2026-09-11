from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MIN_CANDLES = 100


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
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
    out["next_body"] = next_body
    out["next_body_pct"] = next_body / df["close"].replace(0, np.nan)
    out["signed_move_pct"] = signal_dir * out["next_body_pct"]
    return out.dropna(subset=["U", "next_body_pct"]).reset_index(drop=True)


def fold_predictions(df: pd.DataFrame, folds: int, train_ratio: float) -> pd.DataFrame:
    n = len(df)
    initial_train = max(50, int(n * train_ratio))
    remaining = n - initial_train
    if remaining < max(20, folds * 10):
        return pd.DataFrame()
    test_size = remaining // folds
    parts: list[pd.DataFrame] = []
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
        return {"n": 0, "mean_signed": np.nan, "win": np.nan, "half": np.nan, "round": np.nan, "mean_abs": np.nan}
    signed = frame["signed_move_pct"] * 100.0
    return {
        "n": len(frame),
        "mean_signed": float(signed.mean()),
        "win": float((signed > 0).mean() * 100.0),
        "half": float((signed >= half_fee).mean() * 100.0),
        "round": float((signed >= round_fee).mean() * 100.0),
        "mean_abs": float(frame["next_body_pct"].abs().mean() * 100.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="OOS audit of formula direction versus magnitude-ranked entries.")
    ap.add_argument("--data-dir", default="reports/nobitex_1h")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.5)
    ap.add_argument("--half-fee", type=float, default=0.13)
    ap.add_argument("--round-fee", type=float, default=0.26)
    ap.add_argument("--output", default="reports/nobitex_1h/nonirt_oos_magnitude_entry_audit.csv")
    args = ap.parse_args()

    root = Path(args.data_dir)
    files = sorted(p for p in root.glob("*_1h.csv") if not p.stem[:-3].upper().endswith("IRT"))
    rows = []
    pooled = {q: [] for q in [0.0, 0.50, 0.70, 0.80, 0.90]}

    for path in files:
        symbol = path.stem[:-3]
        try:
            df = load(path)
            if len(df) < MIN_CANDLES:
                continue
            pred = fold_predictions(df, args.folds, args.train_ratio)
            if pred.empty:
                continue
            sig = pred[pred["signal_dir"] != 0].copy()
            if sig.empty:
                continue
            sig["mag_percentile"] = sig["pred_abs_u"].rank(method="first", pct=True)
            base = stats(sig, args.half_fee, args.round_fee)
            row = {"symbol": symbol, "base_n": base["n"], "base_mean_signed": base["mean_signed"], "base_win": base["win"], "base_half": base["half"], "base_round": base["round"]}
            for q in pooled:
                label = f"q{int(q*100):02d}"
                selected = sig[sig["mag_percentile"] >= q]
                s = stats(selected, args.half_fee, args.round_fee)
                row[f"{label}_n"] = s["n"]
                row[f"{label}_mean_signed"] = s["mean_signed"]
                row[f"{label}_win"] = s["win"]
                row[f"{label}_half"] = s["half"]
                row[f"{label}_round"] = s["round"]
                pooled[q].append(selected)
            rows.append(row)
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No valid markets.")
    out.to_csv(args.output, index=False)

    print("=" * 120)
    print("NOBITEX NON-IRT | OOS MAGNITUDE ENTRY AUDIT")
    print("=" * 120)
    print(f"Markets found : {len(files)}")
    print(f"Markets tested: {len(out)}")
    print("All metrics use the SAME OOS prediction rows used by the magnitude model.")
    print()
    print("POOLED OOS BASELINE vs MAGNITUDE RANK")
    for q in [0.0, 0.50, 0.70, 0.80, 0.90]:
        label = f"q{int(q*100):02d}"
        frames = pooled[q]
        selected = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        s = stats(selected, args.half_fee, args.round_fee)
        print(f"{label:4s} n={s['n']:5d} mean_signed={s['mean_signed']:+.3f}% win={s['win']:.2f}% >=half={s['half']:.2f}% >=round={s['round']:.2f}%")

    base_win = out["q00_win"].mean()
    q80_win = out["q80_win"].mean()
    print()
    print(f"Mean market OOS win rate baseline : {base_win:.2f}%")
    print(f"Mean market OOS win rate q80      : {q80_win:.2f}%")
    print(f"Markets q80 win >= 50%            : {(out.q80_win >= 50).sum()} / {len(out)}")
    print(f"Markets q80 half-fee >= 50%       : {(out.q80_half >= 50).sum()} / {len(out)}")
    print(f"Markets q80 round-fee >= 50%       : {(out.q80_round >= 50).sum()} / {len(out)}")
    print()
    print("BTC / ETH / SOL / XRP")
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]:
        r = out[out.symbol == symbol]
        if r.empty:
            continue
        r = r.iloc[0]
        print(f"{symbol:8s} base_win={r.base_win:6.2f}% q80_win={r.q80_win:6.2f}% q80_mean={r.q80_mean_signed:+.3f}% q80_half={r.q80_half:6.2f}% q80_round={r.q80_round:6.2f}%")

    print()
    print(f"Detailed CSV: {args.output}")
    print("=" * 120)


if __name__ == "__main__":
    main()
