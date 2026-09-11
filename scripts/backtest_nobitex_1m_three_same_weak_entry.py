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
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def score_series(df: pd.DataFrame) -> np.ndarray:
    body = df["close"] - df["open"]
    hc = df["high"] - df["close"]
    ho = df["high"] - df["open"]
    lc = df["low"] - df["close"]
    return (
        (hc > body).astype(int)
        + (ho < hc).astype(int)
        + (lc > body).astype(int)
    ).to_numpy(dtype=int)


def trade_return(entry: float, exit_: float, direction: int) -> float:
    if direction == 1:
        return (exit_ - entry) / entry
    return (entry - exit_) / entry


def simulate_symbol(
    df: pd.DataFrame,
    scores: np.ndarray,
    mode: str,
    opposite_count: int,
) -> dict[str, float | int]:
    n = len(df)
    capital = INITIAL_CAPITAL
    trades = 0
    wins = 0
    gross_sum = 0.0
    hold_sum = 0

    i = 2
    while i < n - 1:
        # Three consecutive weak signals:
        # 1,1,1 (weak SHORT signals) -> enter LONG
        # 2,2,2 (weak LONG signals)  -> enter SHORT
        last_three = scores[i - 2 : i + 1]
        if np.all(last_three == 1):
            entry_dir = 1
        elif np.all(last_three == 2):
            entry_dir = -1
        else:
            i += 1
            continue

        entry_idx = i
        entry = float(df.iloc[entry_idx].close)
        opposite_seen = 0
        consecutive = 0
        exit_idx = None

        j = entry_idx + 1
        while j < n:
            # Direction represented by the score itself:
            # score <=1 = SHORT, score >=2 = LONG.
            d = int(scores[j])
            frame_dir = -1 if d <= 1 else 1
            opposite = frame_dir == -entry_dir

            if opposite:
                opposite_seen += 1
                consecutive += 1
            else:
                if mode == "consecutive":
                    consecutive = 0

            if mode == "consecutive" and consecutive >= opposite_count:
                exit_idx = j
                break
            if mode == "discrete" and opposite_seen >= opposite_count:
                exit_idx = j
                break

            j += 1

        if exit_idx is None:
            exit_idx = n - 1

        exit_price = float(df.iloc[exit_idx].close)
        gross = trade_return(entry, exit_price, entry_dir)
        net_factor = max((1.0 + gross) * (1.0 - FEE) ** 2, 0.0)
        capital *= net_factor

        trades += 1
        wins += int(gross > 0)
        gross_sum += gross
        hold_sum += exit_idx - entry_idx

        # Do not open another trade inside the active holding period.
        i = exit_idx + 1

    return {
        "final": capital,
        "return": capital / INITIAL_CAPITAL - 1.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "avg_gross": gross_sum / trades if trades else 0.0,
        "avg_hold_min": hold_sum / trades if trades else 0.0,
    }


def main() -> None:
    files = sorted(DATA_DIR.glob("*_1m.csv"))
    loaded: dict[str, pd.DataFrame] = {}
    scores_map: dict[str, np.ndarray] = {}

    for path in files:
        symbol = path.stem[:-3] if path.stem.endswith("_1m") else path.stem
        symbol = symbol.upper()
        if symbol.endswith("IRT") or symbol in NO_SIGNAL_MARKETS:
            continue
        try:
            df = load(path)
            if len(df) >= 4:
                loaded[symbol] = df
                scores_map[symbol] = score_series(df)
        except Exception as exc:
            print(f"SKIP {symbol}: {exc}")

    if not loaded:
        raise SystemExit(f"No valid 1m non-IRT files found under {DATA_DIR}")

    print("=" * 120)
    print("NOBITEX NON-IRT | 1M | THREE CONSECUTIVE WEAK SIGNAL ENTRY")
    print("=" * 120)
    print("Entry LONG  : score 1,1,1 on three consecutive candles")
    print("Entry SHORT : score 2,2,2 on three consecutive candles")
    print("Entry price : close of third weak-signal candle")
    print("Exit        : 1/2/3 opposite-direction frames, consecutive or discrete")
    print("Fee         : 0.13% per side (0.26% round trip)")
    print(f"Markets     : {len(loaded)}")
    print()

    rows: list[dict[str, object]] = []
    for mode in ("consecutive", "discrete"):
        print(f"MODEL = {mode.upper()}")
        print("N_OPPOSITE  final_total  return%  trades  win%  avg_gross%  avg_hold_min")
        for k in (1, 2, 3):
            per_symbol = [
                simulate_symbol(loaded[s], scores_map[s], mode, k)
                for s in loaded
            ]
            total_initial = INITIAL_CAPITAL * len(loaded)
            final_total = sum(float(r["final"]) for r in per_symbol)
            trades = sum(int(r["trades"]) for r in per_symbol)
            wins = sum(
                int(round(float(r["win_rate"]) * int(r["trades"])))
                for r in per_symbol
            )
            gross_sum = sum(
                float(r["avg_gross"]) * int(r["trades"])
                for r in per_symbol
            )
            hold_sum = sum(
                float(r["avg_hold_min"]) * int(r["trades"])
                for r in per_symbol
            )
            result = {
                "model": mode,
                "opposite_count": k,
                "markets": len(loaded),
                "final_total": final_total,
                "return": final_total / total_initial - 1.0,
                "trades": trades,
                "win_rate": wins / trades if trades else 0.0,
                "avg_gross": gross_sum / trades if trades else 0.0,
                "avg_hold_min": hold_sum / trades if trades else 0.0,
            }
            rows.append(result)
            print(
                f"{k:10d}  {final_total:11.2f}  {result['return']*100:+8.2f}%"
                f"  {trades:7d}  {result['win_rate']*100:5.1f}%"
                f"  {result['avg_gross']*100:+10.4f}%  {result['avg_hold_min']:12.2f}"
            )
        print()

    out = pd.DataFrame(rows)
    output = DATA_DIR / "three_same_weak_entry_1m_summary.csv"
    out.to_csv(output, index=False)

    print("COMPARISON")
    print("model        N   return%   trades   win%   avg_gross%   avg_hold_min")
    for _, r in out.sort_values("return", ascending=False).iterrows():
        print(
            f"{r['model']:12s} {int(r['opposite_count']):1d} "
            f"{r['return']*100:+8.2f}% {int(r['trades']):8d} "
            f"{r['win_rate']*100:6.2f}% {r['avg_gross']*100:+11.4f}% "
            f"{r['avg_hold_min']:14.2f}"
        )

    print()
    print(f"Summary CSV: {output}")
    print("=" * 120)


if __name__ == "__main__":
    main()
