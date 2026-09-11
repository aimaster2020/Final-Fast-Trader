from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

TIMEFRAMES = ("5m", "15m", "30m", "1h")
FEE = 0.0013
INITIAL = 1000.0
NO_SIGNAL = {"DOSUSDT", "GRVTUSDT", "SLVONUSDT"}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def score_series(df: pd.DataFrame) -> np.ndarray:
    body = df["close"] - df["open"]
    hc = df["high"] - df["close"]
    ho = df["high"] - df["open"]
    lc = df["low"] - df["close"]
    return ((hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)).to_numpy(int)


def trade_return(entry: float, exit_: float, direction: int) -> float:
    return (exit_ - entry) / entry if direction == 1 else (entry - exit_) / entry


def simulate(df: pd.DataFrame, mode: str, n_opp: int) -> dict[str, float | int]:
    scores = score_series(df)
    n = len(df)
    capital = INITIAL
    trades = wins = 0
    gross_sum = hold_sum = 0.0
    i = 0

    while i < n - 2:
        s1, s2 = int(scores[i]), int(scores[i + 1])
        entry_dir = 0
        if s1 == 1 and s2 == 1:
            entry_dir = 1       # two weak SHORT signals -> LONG
        elif s1 == 2 and s2 == 2:
            entry_dir = -1      # two weak LONG signals -> SHORT
        else:
            i += 1
            continue

        entry_idx = i + 1
        entry = float(df.iloc[entry_idx].close)
        seen = 0
        consec = 0
        exit_idx = None
        j = entry_idx + 1

        while j < n:
            s = int(scores[j])
            # Formula direction: 0/1 => SHORT; 2/3 => LONG.
            d = -1 if s <= 1 else 1
            if d == -entry_dir:
                seen += 1
                consec += 1
            else:
                if mode == "consecutive":
                    consec = 0

            if mode == "consecutive" and consec >= n_opp:
                exit_idx = j
                break
            if mode == "discrete" and seen >= n_opp:
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
        hold_sum += exit_idx - entry_idx
        i = exit_idx

    return {
        "final": capital,
        "return": capital / INITIAL - 1.0,
        "trades": trades,
        "win": wins / trades if trades else 0.0,
        "avg_gross": gross_sum / trades if trades else 0.0,
        "avg_hold": hold_sum / trades if trades else 0.0,
    }


def main() -> None:
    root = Path("reports/nobitex_intraday_native")
    all_rows = []

    print("=" * 126)
    print("NOBITEX NON-IRT | TWO CONSECUTIVE WEAK SIGNALS | NATIVE MULTI-TIMEFRAME")
    print("=" * 126)
    print("Entry LONG : score 1,1 on two consecutive candles")
    print("Entry SHORT: score 2,2 on two consecutive candles")
    print("Entry      : close of second weak-signal candle")
    print("Exit       : 1/2/3 opposite-direction frames; consecutive or discrete")
    print(f"Fee        : {FEE*100:.2f}% per side ({FEE*200:.2f}% round trip)")
    print()

    for tf in TIMEFRAMES:
        tf_dir = root / tf
        files = sorted(tf_dir.glob(f"*_{tf}.csv"))
        loaded = {}
        for path in files:
            symbol = path.stem[:-len(f"_{tf}")].upper()
            if symbol.endswith("IRT") or symbol in NO_SIGNAL:
                continue
            try:
                df = load(path)
                if len(df) >= 3:
                    loaded[symbol] = df
            except Exception as exc:
                print(f"SKIP {symbol} {tf}: {exc}")

        print(f"\nTIMEFRAME = {tf} | MARKETS = {len(loaded)}")
        print("model        N   return%   trades   win%   avg_gross%   avg_hold")
        rows = []
        for mode in ("consecutive", "discrete"):
            for k in (1, 2, 3):
                rs = [simulate(df, mode, k) for df in loaded.values()]
                final_total = sum(float(r["final"]) for r in rs)
                trades = sum(int(r["trades"]) for r in rs)
                wins = sum(int(round(float(r["win"]) * int(r["trades"]))) for r in rs)
                gross = sum(float(r["avg_gross"]) * int(r["trades"]) for r in rs)
                hold = sum(float(r["avg_hold"]) * int(r["trades"]) for r in rs)
                row = {
                    "timeframe": tf,
                    "model": mode,
                    "n_opp": k,
                    "return": final_total / (INITIAL * len(loaded)) - 1.0 if loaded else 0.0,
                    "trades": trades,
                    "win": wins / trades if trades else 0.0,
                    "avg_gross": gross / trades if trades else 0.0,
                    "avg_hold": hold / trades if trades else 0.0,
                }
                rows.append(row)
                all_rows.append(row)
                print(f"{mode:12s} {k:1d} {row['return']*100:+8.2f}% {trades:8d} {row['win']*100:6.2f}% {row['avg_gross']*100:+11.4f}% {row['avg_hold']:9.2f}")

        best = max(rows, key=lambda x: x["return"]) if rows else None
        if best:
            print(f"BEST {tf}: {best['model']} {best['n_opp']} | return={best['return']*100:+.2f}% trades={best['trades']} win={best['win']*100:.2f}%")

    out = pd.DataFrame(all_rows)
    out_dir = root / "multi_tf"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "two_same_weak_entry_summary.csv"
    out.to_csv(output, index=False)
    print() 
    print(f"Summary CSV: {output}")
    print("=" * 126)


if __name__ == "__main__":
    main()
