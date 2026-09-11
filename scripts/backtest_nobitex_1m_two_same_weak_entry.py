from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

NO_SIGNAL_MARKETS = {"DOSUSDT", "GRVTUSDT", "SLVONUSDT"}
DATA_DIR = Path("reports/nobitex_intraday_native/1m")
FEE = 0.0013
INITIAL_CAPITAL = 1000.0


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def direction_and_score(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    body = df["close"] - df["open"]
    hc = df["high"] - df["close"]
    ho = df["high"] - df["open"]
    lc = df["low"] - df["close"]
    score = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)
    direction = np.where(score <= 1, -1, np.where(score >= 2, 1, 0)).astype(int)
    return direction, score.to_numpy(int)


def trade_return(entry: float, exit_: float, direction: int) -> float:
    return (exit_ - entry) / entry if direction == 1 else (entry - exit_) / entry


def simulate(df: pd.DataFrame, entry_weak_count: int, exit_opp_count: int, mode: str) -> dict[str, float | int]:
    dirs, scores = direction_and_score(df)
    n = len(df)
    capital = INITIAL_CAPITAL
    trades = wins = 0
    gross_sum = hold_sum = 0.0
    i = 0

    while i < n - entry_weak_count:
        # Two consecutive weak signals:
        # score 1 = weak SHORT -> enter LONG
        # score 2 = weak LONG  -> enter SHORT
        entry_dir = 0
        entry_idx = None
        for k in range(i, i + entry_weak_count):
            pass

        if scores[i] == 1 and scores[i + 1] == 1:
            entry_dir = 1
            entry_idx = i + 1
        elif scores[i] == 2 and scores[i + 1] == 2:
            entry_dir = -1
            entry_idx = i + 1

        if entry_dir == 0:
            i += 1
            continue

        entry = float(df.iloc[entry_idx].close)
        opposite_seen = 0
        consecutive = 0
        exit_idx = None
        j = entry_idx + 1

        while j < n:
            d = int(dirs[j])
            if d == -entry_dir:
                opposite_seen += 1
                consecutive += 1
            elif d == entry_dir or d == 0:
                if mode == "consecutive":
                    consecutive = 0

            if mode == "consecutive" and consecutive >= exit_opp_count:
                exit_idx = j
                break
            if mode == "discrete" and opposite_seen >= exit_opp_count:
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
        "return": capital / INITIAL_CAPITAL - 1.0,
        "trades": trades,
        "win": wins / trades if trades else 0.0,
        "avg_gross": gross_sum / trades if trades else 0.0,
        "avg_hold": hold_sum / trades if trades else 0.0,
    }


def main() -> None:
    files = sorted(DATA_DIR.glob("*_1m.csv"))
    loaded: dict[str, pd.DataFrame] = {}
    for path in files:
        symbol = path.stem[:-3] if path.stem.endswith("_1m") else path.stem
        symbol = symbol.upper()
        if symbol.endswith("IRT") or symbol in NO_SIGNAL_MARKETS:
            continue
        try:
            df = load(path)
            if len(df) >= 3:
                loaded[symbol] = df
        except Exception as exc:
            print(f"SKIP {symbol}: {exc}")

    if not loaded:
        raise SystemExit(f"No valid 1m non-IRT files found under {DATA_DIR}")

    rows = []
    print("=" * 118)
    print("NOBITEX NON-IRT | 1M | TWO CONSECUTIVE WEAK SIGNAL ENTRY")
    print("=" * 118)
    print("Entry LONG  : score 1, score 1 on two consecutive candles")
    print("Entry SHORT : score 2, score 2 on two consecutive candles")
    print("Entry price : close of second weak-signal candle")
    print("Exit        : 1/2/3 opposite-direction frames, consecutive or discrete")
    print(f"Fee         : {FEE*100:.2f}% per side ({FEE*200:.2f}% round trip)")
    print(f"Markets     : {len(loaded)}")
    print()

    for mode in ("consecutive", "discrete"):
        print(f"MODEL = {mode.upper()}")
        print("N_OPPOSITE  final_total  return%  trades  win%  avg_gross%  avg_hold_min")
        for k in (1, 2, 3):
            per = [simulate(df, 2, k, mode) for df in loaded.values()]
            final_total = sum(float(r["final"]) for r in per)
            trades = sum(int(r["trades"]) for r in per)
            wins = sum(int(round(float(r["win"]) * int(r["trades"]))) for r in per)
            gross_sum = sum(float(r["avg_gross"]) * int(r["trades"]) for r in per)
            hold_sum = sum(float(r["avg_hold"]) * int(r["trades"]) for r in per)
            total_initial = INITIAL_CAPITAL * len(loaded)
            result = {
                "model": mode,
                "opposite_count": k,
                "return": final_total / total_initial - 1.0,
                "trades": trades,
                "win": wins / trades if trades else 0.0,
                "avg_gross": gross_sum / trades if trades else 0.0,
                "avg_hold": hold_sum / trades if trades else 0.0,
            }
            rows.append(result)
            print(f"{k:10d}  {final_total:11.2f}  {result['return']*100:+8.2f}%  {trades:6d}  {result['win']*100:5.1f}%  {result['avg_gross']*100:+10.4f}%  {result['avg_hold']:12.2f}")
        print()

    out = pd.DataFrame(rows)
    output = DATA_DIR / "two_same_weak_entry_1m_summary.csv"
    out.to_csv(output, index=False)

    print("COMPARISON")
    print("model        N   return%   trades   win%   avg_gross%   avg_hold_min")
    for _, r in out.sort_values("return", ascending=False).iterrows():
        print(f"{r['model']:12s} {int(r['opposite_count']):1d} {r['return']*100:+8.2f}% {int(r['trades']):8d} {r['win']*100:6.2f}% {r['avg_gross']*100:+11.4f}% {r['avg_hold']:14.2f}")
    print()
    print(f"Summary CSV: {output}")
    print("=" * 118)


if __name__ == "__main__":
    main()
