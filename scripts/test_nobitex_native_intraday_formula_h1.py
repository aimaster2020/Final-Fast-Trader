from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NO_SIGNAL_MARKETS = {"DOSUSDT", "GRVTUSDT", "SLVONUSDT"}
TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h")


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    return df.reset_index(drop=True)


def signals(df: pd.DataFrame) -> pd.DataFrame:
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
    out = df.copy()
    out["signal_dir"] = signal_dir
    return out


def stats(df: pd.DataFrame, half_fee: float, round_fee: float) -> dict[str, float | int]:
    move_pct = (df["close"].shift(-1) / df["close"] - 1.0) * 100.0
    signed = df["signal_dir"] * move_pct
    valid = df["signal_dir"].ne(0) & move_pct.notna() & move_pct.ne(0)
    signed = signed[valid]
    if signed.empty:
        return {"candles": len(df), "trades": 0, "win": np.nan, "mean": np.nan, "median": np.nan, "half": np.nan, "round": np.nan}
    return {
        "candles": int(len(df)),
        "trades": int(len(signed)),
        "win": float((signed > 0).mean() * 100.0),
        "mean": float(signed.mean()),
        "median": float(signed.median()),
        "half": float((signed >= half_fee).mean() * 100.0),
        "round": float((signed >= round_fee).mean() * 100.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Test the fixed formula with H1 horizon on native Nobitex 1m-1h data.")
    ap.add_argument("--data-dir", default="reports/nobitex_intraday_native")
    ap.add_argument("--half-fee", type=float, default=0.13)
    ap.add_argument("--round-fee", type=float, default=0.26)
    args = ap.parse_args()

    root = Path(args.data_dir)
    records: list[dict[str, object]] = []

    available_symbols = None
    for tf in TIMEFRAMES:
        tf_dir = root / tf
        files = sorted(tf_dir.glob(f"*_{tf}.csv"))
        symbols = {p.stem[: -len(f"_{tf}")].upper() for p in files if not p.stem[: -len(f"_{tf}")].upper().endswith("IRT")}
        available_symbols = symbols if available_symbols is None else available_symbols & symbols

    if available_symbols is None:
        raise SystemExit(f"No data found under {root}")
    symbols = sorted(available_symbols - NO_SIGNAL_MARKETS)

    for symbol in symbols:
        for tf in TIMEFRAMES:
            path = root / tf / f"{symbol}_{tf}.csv"
            if not path.exists():
                continue
            try:
                df = signals(load(path))
                s = stats(df, args.half_fee, args.round_fee)
                records.append({"symbol": symbol, "timeframe": tf, **s})
            except Exception as exc:
                print(f"{symbol} {tf}: ERROR {exc}")

    out = pd.DataFrame(records)
    output = root / "native_intraday_formula_h1.csv"
    out.to_csv(output, index=False)

    print("=" * 125)
    print("NOBITEX NON-IRT | NATIVE 1m-1h DATA | H1 HORIZON | FIXED FORMULA")
    print("=" * 125)
    print(f"Markets tested across all TFs : {len(symbols)}")
    print("Timeframes                    : 1m, 5m, 15m, 30m, 1h")
    print("Entry                         : current candle close")
    print("Exit                          : next candle close")
    print("Signal                        : fixed formula + ambiguity filter")
    print("Fees                          : half 0.13% | round-trip 0.26%")
    print()
    print("SUMMARY")
    print("TF    markets  mean_win  median_win  >=55  mean_move  half>=50  round>=50  mean_trades")
    for tf in TIMEFRAMES:
        x = out[out.timeframe == tf].copy()
        x = x[x.trades > 0]
        if x.empty:
            print(f"{tf:4s}  0        --         --        0     --          0          0          --")
            continue
        print(
            f"{tf:4s}  {len(x):7d}  {x.win.mean():8.2f}%  {x.win.median():9.2f}%"
            f"  {int((x.win >= 55).sum()):4d}  {x['mean'].mean():+9.3f}%"
            f"  {int((x.half >= 50).sum()):9d}  {int((x['round'] >= 50).sum()):9d}"
            f"  {x.trades.mean():10.1f}"
        )

    print()
    print("DETAIL — EACH MARKET")
    print("SYMBOL               | 1m WIN/MEAN/ROUND | 5m WIN/MEAN/ROUND | 15m WIN/MEAN/ROUND | 30m WIN/MEAN/ROUND | 1h WIN/MEAN/ROUND")
    print("-" * 125)
    for symbol in symbols:
        chunks = []
        for tf in TIMEFRAMES:
            r = out[(out.symbol == symbol) & (out.timeframe == tf)]
            if r.empty or int(r.iloc[0].trades) == 0:
                chunks.append(f"{tf} --/--/--")
            else:
                row = r.iloc[0]
                chunks.append(f"{tf} {row.win:4.1f}/{row['mean']:+.3f}/{row['round']:4.1f}")
        print(f"{symbol:20s} | " + " | ".join(chunks))

    print()
    print(f"Detailed CSV: {output}")
    print("=" * 125)


if __name__ == "__main__":
    main()
