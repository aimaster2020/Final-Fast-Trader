from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("reports/nobitex_intraday_native/1m")
FEE = 0.0013
NO_SIGNAL_MARKETS = {"DOSUSDT", "GRVTUSDT", "SLVONUSDT"}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def weak_reversal_direction(df: pd.DataFrame) -> np.ndarray:
    body = df["close"] - df["open"]
    hc = df["high"] - df["close"]
    ho = df["high"] - df["open"]
    lc = df["low"] - df["close"]
    score = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)

    # Weak short (score=1) -> enter LONG.
    # Weak long  (score=2) -> enter SHORT.
    # Strong signals (score=0 or 3) -> no trade.
    return np.where(score == 1, 1, np.where(score == 2, -1, 0)).astype(int)


def trade_return(entry: float, exit_price: float, direction: int) -> float:
    return ((exit_price - entry) / entry) if direction == 1 else ((entry - exit_price) / entry)


def simulate(df: pd.DataFrame, mode: str, n_opposite: int) -> dict[str, float | int]:
    dirs = weak_reversal_direction(df)
    n = len(df)
    capital = 1000.0
    trades = wins = 0
    gross_sum = hold_sum = 0.0
    i = 0

    while i < n - 1:
        entry_dir = int(dirs[i])
        if entry_dir == 0:
            i += 1
            continue

        entry = float(df.iloc[i].close)
        opposite_seen = 0
        consecutive = 0
        exit_idx = None
        j = i + 1
        while j < n:
            d = int(dirs[j])
            if d == -entry_dir:
                opposite_seen += 1
                consecutive += 1
            elif d == entry_dir or d == 0:
                consecutive = 0

            if mode == "consecutive" and consecutive >= n_opposite:
                exit_idx = j
                break
            if mode == "discrete" and opposite_seen >= n_opposite:
                exit_idx = j
                break
            j += 1

        if exit_idx is None:
            exit_idx = n - 1

        exit_price = float(df.iloc[exit_idx].close)
        gross = trade_return(entry, exit_price, entry_dir)
        capital *= max((1.0 + gross) * (1.0 - FEE) ** 2, 0.0)
        trades += 1
        wins += int(gross > 0)
        gross_sum += gross
        hold_sum += exit_idx - i
        i = exit_idx

    return {
        "final": capital,
        "return": capital / 1000.0 - 1.0,
        "trades": trades,
        "win": wins / trades if trades else 0.0,
        "avg_gross": gross_sum / trades if trades else 0.0,
        "avg_hold_min": hold_sum / trades if trades else 0.0,
    }


def main() -> None:
    rows = []
    loaded: dict[str, pd.DataFrame] = {}
    for path in sorted(DATA_DIR.glob("*_1m.csv")):
        symbol = path.stem[:-3] if path.stem.endswith("_1m") else path.stem
        symbol = symbol.upper()
        if symbol.endswith("IRT") or symbol in NO_SIGNAL_MARKETS:
            continue
        try:
            df = load(path)
            if len(df) >= 2:
                loaded[symbol] = df
        except Exception as exc:
            print(f"SKIP {symbol}: {exc}")

    print("=" * 120)
    print("NOBITEX NON-IRT | 1M | WEAK-SIGNAL REVERSAL ENTRY")
    print("=" * 120)
    print("Entry: score=1 SHORT -> LONG; score=2 LONG -> SHORT; score 0/3 -> NO TRADE")
    print(f"Fee: {FEE*100:.2f}% per side ({FEE*200:.2f}% round trip)")
    print(f"Markets: {len(loaded)}")
    print()

    for mode in ("consecutive", "discrete"):
        for k in (1, 2, 3):
            finals = trades = wins = 0
            gross = hold = 0.0
            per = []
            for symbol, df in loaded.items():
                r = simulate(df, mode, k)
                per.append((symbol, r))
                finals += r["final"]
                trades += int(r["trades"])
                wins += int(round(r["win"] * int(r["trades"])))
                gross += r["avg_gross"] * int(r["trades"])
                hold += r["avg_hold_min"] * int(r["trades"])
            ret = finals / (1000.0 * len(loaded)) - 1.0
            rows.append({"model": mode, "n_opposite": k, "return": ret, "trades": trades,
                         "win": wins / trades if trades else 0.0,
                         "avg_gross": gross / trades if trades else 0.0,
                         "avg_hold_min": hold / trades if trades else 0.0})
            print(f"{mode:12s} N={k} return={ret*100:+.2f}% trades={trades:7d} win={wins/trades*100 if trades else 0:5.1f}% avg_gross={gross/trades*100 if trades else 0:+.4f}% hold_min={hold/trades if trades else 0:.2f}")

    out = pd.DataFrame(rows)
    print()
    print("COMPARISON")
    for _, r in out.sort_values("return", ascending=False).iterrows():
        print(f"{r.model:12s} {int(r.n_opposite)} return={r['return']*100:+.2f}% trades={int(r.trades):7d} win={r.win*100:5.2f}% avg_gross={r.avg_gross*100:+.4f}% hold={r.avg_hold_min:.2f}m")

    detail_rows = []
    for symbol, df in loaded.items():
        for mode in ("consecutive", "discrete"):
            for k in (1, 2, 3):
                r = simulate(df, mode, k)
                detail_rows.append({"symbol": symbol, "model": mode, "n_opposite": k, **r})

    outdir = DATA_DIR
    out.to_csv(outdir / "weak_signal_reversal_1m_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(outdir / "weak_signal_reversal_1m_by_asset.csv", index=False)
    print(f"Summary CSV: {outdir / 'weak_signal_reversal_1m_summary.csv'}")
    print(f"Asset CSV  : {outdir / 'weak_signal_reversal_1m_by_asset.csv'}")
    print("=" * 120)


if __name__ == "__main__":
    main()
