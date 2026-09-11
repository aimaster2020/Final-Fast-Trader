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
    out["pred_direction"] = signal_dir
    out["next_body"] = next_body
    out["next_body_pct"] = next_body / df["close"].replace(0, np.nan)
    out["signed_next_body_pct"] = signal_dir * out["next_body_pct"]
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


def summarize_market(symbol: str, pred: pd.DataFrame, quantiles: list[float], half_fee_pct: float, round_fee_pct: float) -> dict[str, float | int | str]:
    sig = pred[pred["pred_direction"] != 0].copy()
    if sig.empty:
        return {"symbol": symbol, "signals": 0}

    sig["mag_percentile"] = sig["pred_abs_u"].rank(method="first", pct=True)
    rows: dict[str, float | int | str] = {"symbol": symbol, "signals": len(sig)}

    for q in quantiles:
        label = f"q{int(q * 100):02d}"
        sel = sig[sig["mag_percentile"] >= q]
        if sel.empty:
            for suffix in ["n", "mean_signed", "mean_abs", "win", "half_fee", "round_fee"]:
                rows[f"{label}_{suffix}"] = np.nan
            continue
        signed = sel["signed_next_body_pct"] * 100.0
        absolute = sel["next_body_pct"].abs() * 100.0
        rows[f"{label}_n"] = len(sel)
        rows[f"{label}_mean_signed"] = float(signed.mean())
        rows[f"{label}_mean_abs"] = float(absolute.mean())
        rows[f"{label}_win"] = float((signed > 0).mean() * 100.0)
        rows[f"{label}_half_fee"] = float((signed >= half_fee_pct).mean() * 100.0)
        rows[f"{label}_round_fee"] = float((signed >= round_fee_pct).mean() * 100.0)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Magnitude-ranked entry filter test across all non-IRT Nobitex 1H markets.")
    ap.add_argument("--data-dir", default="reports/nobitex_1h")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=0.5)
    ap.add_argument("--half-fee", type=float, default=0.13, help="Half round-trip fee in percent.")
    ap.add_argument("--round-fee", type=float, default=0.26, help="Round-trip fee in percent.")
    ap.add_argument("--output", default="reports/nobitex_1h/nonirt_magnitude_entry_filter_by_market.csv")
    args = ap.parse_args()

    root = Path(args.data_dir)
    files = sorted(p for p in root.glob("*_1h.csv") if not p.stem[:-3].upper().endswith("IRT"))
    quantiles = [0.0, 0.50, 0.70, 0.80, 0.90]
    results = []
    pooled = {q: [] for q in quantiles}
    tested = 0
    skipped = 0

    for path in files:
        symbol = path.stem[:-3] if path.stem.endswith("_1h") else path.stem
        try:
            df = load(path)
            if len(df) < MIN_CANDLES:
                skipped += 1
                continue
            pred = fold_predictions(df, args.folds, args.train_ratio)
            if pred.empty:
                skipped += 1
                continue
            row = summarize_market(symbol, pred, quantiles, args.half_fee, args.round_fee)
            results.append(row)
            tested += 1
            sig = pred[pred["pred_direction"] != 0].copy()
            sig["mag_percentile"] = sig["pred_abs_u"].rank(method="first", pct=True)
            for q in quantiles:
                pooled[q].append(sig[sig["mag_percentile"] >= q])
        except Exception as exc:
            skipped += 1
            print(f"SKIP {symbol}: {exc}")

    out = pd.DataFrame(results)
    out.to_csv(args.output, index=False)

    print("=" * 120)
    print("NOBITEX NON-IRT | MAGNITUDE-RANKED ENTRY FILTER")
    print("=" * 120)
    print(f"Markets found      : {len(files)}")
    print(f"Markets tested     : {tested}")
    print(f"Skipped            : {skipped}")
    print("Signal             : existing formula direction + ambiguity filter")
    print("Magnitude ranking  : walk-forward prediction of |U|, ranked within each market")
    print(f"Half fee threshold : {args.half_fee:.2f}%")
    print(f"Round fee threshold: {args.round_fee:.2f}%")
    print()
    print("POOLED RESULTS — ONLY FORMULA SIGNALS")
    for q in quantiles:
        label = f"q{int(q * 100):02d}"
        frames = pooled[q]
        all_sig = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if all_sig.empty:
            print(f"{label}: n=0")
            continue
        signed = all_sig["signed_next_body_pct"] * 100.0
        absolute = all_sig["next_body_pct"].abs() * 100.0
        print(
            f"{label:4s} n={len(all_sig):5d} "
            f"mean_signed={signed.mean():+7.3f}% "
            f"mean_abs={absolute.mean():7.3f}% "
            f"win={((signed > 0).mean() * 100):6.2f}% "
            f">=half_fee={((signed >= args.half_fee).mean() * 100):6.2f}% "
            f">=round_fee={((signed >= args.round_fee).mean() * 100):6.2f}%"
        )

    print()
    print("MARKET COUNTS — TOP 20% MAGNITUDE")
    q = "q80"
    print(f"markets with positive mean signed move : {(out[f'{q}_mean_signed'] > 0).sum()} / {len(out)}")
    print(f"markets with >= half-fee success >=50% : {(out[f'{q}_half_fee'] >= 50).sum()} / {len(out)}")
    print(f"markets with >= round-fee success >=50% : {(out[f'{q}_round_fee'] >= 50).sum()} / {len(out)}")
    print()
    print("ALL MARKETS — SORTED BY TOP-20% MEAN SIGNED MOVE")
    cols = ["symbol", "signals", "q80_n", "q80_mean_signed", "q80_mean_abs", "q80_win", "q80_half_fee", "q80_round_fee"]
    view = out[cols].sort_values("q80_mean_signed", ascending=False)
    for _, r in view.iterrows():
        print(
            f"{r.symbol:20s} sig={int(r.signals):4d} top20={int(r.q80_n):4d} "
            f"mean_signed={r.q80_mean_signed:+7.3f}% mean_abs={r.q80_mean_abs:7.3f}% "
            f"win={r.q80_win:6.2f}% half={r.q80_half_fee:6.2f}% round={r.q80_round_fee:6.2f}%"
        )

    print()
    print(f"Detailed CSV: {args.output}")
    print("=" * 120)


if __name__ == "__main__":
    main()
