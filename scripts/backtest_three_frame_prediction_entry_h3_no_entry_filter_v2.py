from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=lambda c: str(c).strip().lower() in {"open", "high", "low", "close"},
    )
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


def score_direction(o: float, h: float, l: float, c: float) -> tuple[int, int]:
    body = c - o
    hc = h - c
    ho = h - o
    lc = l - c
    score = int(hc > body) + int(ho < hc) + int(lc > body)
    direction = -1 if score == 0 else 1 if score == 3 else 0
    return score, direction


def predict_next(o: float, h: float, l: float, c: float) -> tuple[float, float, float, float, int]:
    """Create one synthetic next frame using only the supplied frame."""
    score, direction = score_direction(o, h, l, c)
    if direction == 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    upper = abs(h - o)
    lower = abs(o - l)
    trade_magnitude = upper if direction == 1 else lower
    if not np.isfinite(trade_magnitude) or trade_magnitude <= 1e-12:
        return np.nan, np.nan, np.nan, np.nan, 0

    # Preserve the supplied candle's normalized geometry. The predicted candle
    # uses the same trade-side excursion and opens at the supplied close.
    body_ratio = (c - o) / trade_magnitude
    upper_ratio = upper / trade_magnitude
    lower_ratio = lower / trade_magnitude

    po = c
    pc = po + body_ratio * trade_magnitude
    ph = po + upper_ratio * trade_magnitude
    pl = po - lower_ratio * trade_magnitude
    ph = max(po, pc, ph)
    pl = min(po, pc, pl)

    _, next_direction = score_direction(po, ph, pl, pc)
    return po, ph, pl, pc, next_direction


def three_frame_prediction(df: pd.DataFrame, i: int) -> tuple[int, int, int, bool]:
    """Predict t+1, then t+2 from predicted t+1, then t+3 from predicted t+2."""
    o, h, l, c = map(float, df.iloc[i][["open", "high", "low", "close"]])

    p1 = predict_next(o, h, l, c)
    if p1[4] == 0 or not np.isfinite(p1[3]):
        return 0, 0, 0, False

    p2 = predict_next(p1[0], p1[1], p1[2], p1[3])
    if p2[4] == 0 or not np.isfinite(p2[3]):
        return p1[4], 0, 0, False

    p3 = predict_next(p2[0], p2[1], p2[2], p2[3])
    if p3[4] == 0 or not np.isfinite(p3[3]):
        return p1[4], p2[4], 0, False

    agreement = p1[4] == p2[4] == p3[4]
    return p1[4], p2[4], p3[4], agreement


def evaluate_symbol(
    df: pd.DataFrame,
    initial_capital: float,
    fee: float,
    mode: str,
) -> dict[str, float | int]:
    capital = float(initial_capital)
    fee_factor = (1.0 - fee) ** 2

    eligible = agree = trades = wins = 0
    gross_sum = 0.0

    i = 0
    n = len(df)
    while i < n - 3:
        o, h, l, c = map(float, df.iloc[i][["open", "high", "low", "close"]])
        _, entry_dir = score_direction(o, h, l, c)

        if entry_dir == 0:
            i += 1
            continue
        if mode == "LONG" and entry_dir != 1:
            i += 1
            continue
        if mode == "SHORT" and entry_dir != -1:
            i += 1
            continue

        eligible += 1
        p1, p2, p3, agreement = three_frame_prediction(df, i)
        if agreement:
            agree += 1

        # Entry requires all three predicted frames to agree with the real entry signal.
        if not agreement or p1 != entry_dir or p2 != entry_dir or p3 != entry_dir:
            i += 1
            continue

        entry = c
        actual_exit = float(df.iloc[i + 3].close)
        gross_ret = (
            (actual_exit - entry) / entry
            if entry_dir == 1
            else (entry - actual_exit) / entry
        )

        capital *= max((1.0 + gross_ret) * fee_factor, 0.0)
        trades += 1
        wins += int(gross_ret > 0)
        gross_sum += gross_ret

        # One active position per symbol; signals are ignored during the 3-candle hold.
        i += 4

    return {
        "final": capital,
        "return": capital / initial_capital - 1.0,
        "eligible": eligible,
        "agree": agree,
        "agreement_rate": agree / eligible if eligible else 0.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "avg_gross_trade": gross_sum / trades if trades else 0.0,
    }


def pooled(
    loaded: dict[str, pd.DataFrame],
    symbols: list[str],
    initial_capital: float,
    fee: float,
    mode: str,
) -> dict[str, float | int]:
    results = [evaluate_symbol(loaded[s], initial_capital, fee, mode) for s in symbols]
    final = sum(float(r["final"]) for r in results)
    eligible = sum(int(r["eligible"]) for r in results)
    agree = sum(int(r["agree"]) for r in results)
    trades = sum(int(r["trades"]) for r in results)
    wins = sum(int(round(float(r["win_rate"]) * int(r["trades"]))) for r in results)
    gross = sum(float(r["avg_gross_trade"]) * int(r["trades"]) for r in results)
    initial = initial_capital * len(symbols)
    return {
        "final": final,
        "return": final / initial - 1.0,
        "eligible": eligible,
        "agree": agree,
        "agreement_rate": agree / eligible if eligible else 0.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "avg_gross_trade": gross / trades if trades else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="H3 recursive prediction agreement test without entry magnitude filter."
    )
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=0.0013)
    args = ap.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    root = Path(args.data_dir)
    loaded = {s: load(root / f"{s}_1h.csv") for s in symbols}

    print("=" * 112)
    print("H3 RECURSIVE PREDICTION AGREEMENT — NO ENTRY FILTER")
    print("=" * 112)
    print("Entry signal: current real candle SCORE 0=SHORT / SCORE 3=LONG")
    print("Entry magnitude filter: DISABLED")
    print("Prediction 1: synthetic t+1 from real t")
    print("Prediction 2: synthetic t+2 from predicted t+1")
    print("Prediction 3: synthetic t+3 from predicted t+2")
    print("Entry requires: pred(t+1) == pred(t+2) == pred(t+3) == current signal")
    print("Entry: current close | Exit: actual close at t+3")
    print("Signal exits: disabled | one active trade per symbol")
    print()

    for mode in ("SHORT", "LONG", "BOTH"):
        print(mode)
        print("fee      final       return      eligible  agree%   trades   win%   avg_gross")
        for fee in (0.0, args.fee):
            r = pooled(loaded, symbols, args.initial_capital, fee, mode)
            print(
                f"{fee*100:5.2f}%  {float(r['final']):10.2f}  "
                f"{float(r['return'])*100:+9.2f}%  "
                f"{int(r['eligible']):8d}  {float(r['agreement_rate'])*100:6.1f}%  "
                f"{int(r['trades']):6d}  {float(r['win_rate'])*100:5.1f}%  "
                f"{float(r['avg_gross_trade'])*100:+9.4f}%"
            )
        print()

    print("=" * 112)


if __name__ == "__main__":
    main()
