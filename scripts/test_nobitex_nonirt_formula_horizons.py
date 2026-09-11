from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MIN_CANDLES = 100
DEFAULT_HORIZONS = [1, 2, 3, 4, 5]

# The established active non-IRT universe used in the prior Nobitex analysis
# contains 233 markets. These three files had no usable formula signals in the
# earlier direct formula audit, so they are excluded from this horizon test.
EXCLUDED_NO_SIGNAL = {"DOSUSDT", "GRVTUSDT", "SLVONUSDT"}
TARGET_MARKETS = 233


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

    df["signal_dir"] = signal_dir
    return df


def horizon_stats(df: pd.DataFrame, horizon: int, half_fee: float, round_fee: float) -> dict[str, float | int]:
    frame = df.copy()
    future_close = frame["close"].shift(-horizon)
    move_pct = (future_close / frame["close"] - 1.0) * 100.0
    signed = frame["signal_dir"] * move_pct
    valid = frame["signal_dir"].ne(0) & move_pct.notna() & move_pct.ne(0)
    signed = signed[valid]
    if signed.empty:
        return {"n": 0, "win": np.nan, "mean": np.nan, "median": np.nan, "half": np.nan, "round": np.nan}
    return {
        "n": len(signed),
        "win": float((signed > 0).mean() * 100.0),
        "mean": float(signed.mean()),
        "median": float(signed.median()),
        "half": float((signed >= half_fee).mean() * 100.0),
        "round": float((signed >= round_fee).mean() * 100.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Test the fixed Nobitex formula across multiple holding horizons.")
    ap.add_argument("--data-dir", default="reports/nobitex_1h")
    ap.add_argument("--horizons", default="1,2,3,4,5")
    ap.add_argument("--half-fee", type=float, default=0.13)
    ap.add_argument("--round-fee", type=float, default=0.26)
    ap.add_argument("--output", default="reports/nobitex_1h/nonirt_formula_horizons.csv")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    root = Path(args.data_dir)
    files = sorted(
        p
        for p in root.glob("*_1h.csv")
        if not p.stem[:-3].upper().endswith("IRT")
        and p.stem.upper() not in EXCLUDED_NO_SIGNAL
    )

    rows: list[dict[str, float | int | str]] = []
    tested_symbols: set[str] = set()

    for path in files:
        symbol = path.stem[:-3]
        try:
            df = load(path)
            if len(df) < MIN_CANDLES:
                continue
            symbol_had_signal = False
            for horizon in horizons:
                s = horizon_stats(df, horizon, args.half_fee, args.round_fee)
                if not s["n"]:
                    continue
                symbol_had_signal = True
                rows.append({"symbol": symbol, "horizon": horizon, **s})
            if symbol_had_signal:
                tested_symbols.add(symbol)
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No valid markets.")

    out.to_csv(args.output, index=False)

    print("=" * 110)
    print("NOBITEX NON-IRT | FORMULA HORIZON TEST")
    print("=" * 110)
    print(f"Markets found       : {len(files) + len(EXCLUDED_NO_SIGNAL)}")
    print(f"Established universe: {TARGET_MARKETS}")
    print(f"Markets tested      : {len(tested_symbols)}")
    print("Entry               : current candle close")
    print("Exit                : close of the horizon candle")
    print("Signal              : fixed formula + ambiguity filter")
    print()

    summary = []
    for h in horizons:
        x = out[out.horizon == h]
        summary.append({
            "horizon": h,
            "markets": x.symbol.nunique(),
            "mean_win": x.win.mean(),
            "median_win": x.win.median(),
            "mkt_ge55": int((x.win >= 55).sum()),
            "mean_move": x["mean"].mean(),
            "half_ge50": int((x.half >= 50).sum()),
            "round_ge50": int((x["round"] >= 50).sum()),
            "mean_trades": x.n.mean(),
        })

    s = pd.DataFrame(summary)
    print("ALL-233-MARKET SUMMARY")
    print("H  markets  mean_win  median_win  >=55%  mean_move  half>=50  round>=50  mean_trades")
    for _, r in s.iterrows():
        print(
            f"{int(r.horizon):<2d} {int(r.markets):>7d}  {r.mean_win:>8.2f}%  {r.median_win:>10.2f}%  "
            f"{int(r.mkt_ge55):>5d}  {r.mean_move:>+9.3f}%  {int(r.half_ge50):>9d}  "
            f"{int(r.round_ge50):>9d}  {r.mean_trades:>11.1f}"
        )

    print()
    print("SELECTED MAJOR MARKETS")
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]:
        x = out[out.symbol == symbol].sort_values("horizon")
        if x.empty:
            continue
        print(symbol)
        for _, r in x.iterrows():
            print(
                f"  H{int(r.horizon)} n={int(r.n):4d} win={r.win:6.2f}% "
                f"move={r['mean']:+.3f}% half={r.half:6.2f}% round={r['round']:6.2f}%"
            )

    print()
    print("TOP MARKETS BY HORIZON")
    for h in horizons:
        x = out[out.horizon == h].sort_values(["win", "n"], ascending=[False, False]).head(10)
        print(f"H{h}:")
        for _, r in x.iterrows():
            print(f"  {r.symbol:18s} n={int(r.n):4d} win={r.win:6.2f}% move={r['mean']:+.3f}% round={r['round']:6.2f}%")

    print()
    print(f"Detailed CSV: {args.output}")
    print("=" * 110)


if __name__ == "__main__":
    main()
