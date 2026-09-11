from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEE = 0.0013
THRESHOLD = 0.0225
NOISE = 1.0
MIN_CANDLES = 48


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"timestamp", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"missing columns: {sorted(required - set(df.columns))}")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(required)).sort_values("timestamp").reset_index(drop=True)

    body = df.close - df.open
    hc = df.high - df.close
    ho = df.high - df.open
    lc = df.low - df.close
    score = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)
    upper = (df.high - df.open).abs()
    lower = (df.open - df.low).abs()
    formula = np.where(lower > upper, 1, np.where(upper > lower, -1, 0))

    direction = np.where(score <= 1, -1, 1).astype(int)
    direction[(score == 1) & (formula == 1)] = 0
    direction[(score == 2) & (formula == -1)] = 0
    magnitude = np.where(direction == 1, upper, np.where(direction == -1, lower, np.nan))

    df["body"] = body
    df["direction"] = direction
    df["magnitude_pct"] = magnitude / df.close
    return df


def backtest(df: pd.DataFrame, direction_filter: int | None) -> dict[str, float | int]:
    fee_factor = (1.0 - FEE) ** 2
    capital = 1000.0
    trades = wins = 0
    i = 0
    n = len(df)

    while i < n - 1:
        row = df.iloc[i]
        d = int(row.direction)
        exp = float(row.magnitude_pct)
        body = abs(float(row.body))

        if direction_filter is not None and d != direction_filter:
            i += 1
            continue
        if d == 0 or not np.isfinite(exp) or exp < THRESHOLD or body <= 0:
            i += 1
            continue

        entry = float(row.close)
        first = 0.0
        opposite_count = 0
        exit_i = None
        j = i + 1

        while j < n:
            cur = df.iloc[j]
            pdirection = int(cur.direction)
            pmag = float(cur.magnitude_pct) if np.isfinite(cur.magnitude_pct) else np.nan
            if pdirection == 0 or not np.isfinite(pmag):
                j += 1
                continue

            move = abs(float(cur.close) - entry) / max(abs(entry), 1e-12)
            ref = max(move, body / max(abs(entry), 1e-12), 1e-12)

            if pdirection == d:
                opposite_count = 0
                first = 0.0
                j += 1
                continue

            opposite_count += 1
            if opposite_count == 1:
                first = pmag
                if pmag <= NOISE * ref:
                    j += 1
                    continue
                exit_i = j
                break
            if opposite_count == 2:
                if pmag < 2.0 * first or pmag <= 2.0 * NOISE * ref:
                    j += 1
                    continue
                exit_i = j
                break
            exit_i = j
            break

        if exit_i is None:
            break

        exit_price = float(df.iloc[exit_i].close)
        gross = (exit_price - entry) / entry if d == 1 else (entry - exit_price) / entry
        capital *= max((1.0 + gross) * fee_factor, 0.0)
        trades += 1
        wins += int(gross > 0)
        i = exit_i + 1

    return {
        "final": capital,
        "return": capital / 1000.0 - 1.0,
        "trades": trades,
        "wins": wins,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="OOS 2.25% threshold direction and market breadth test for non-IRT Nobitex markets.")
    ap.add_argument("--data-dir", default="reports/nobitex_1h")
    ap.add_argument("--oos-days", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.data_dir)
    files = sorted(p for p in root.glob("*_1h.csv") if "_IRT_" not in p.name.upper() and not p.stem.upper().endswith("IRT_1H"))
    loaded: dict[str, pd.DataFrame] = {}
    for path in files:
        symbol = path.stem[:-3] if path.stem.endswith("_1h") else path.stem
        if symbol.upper().endswith("IRT"):
            continue
        try:
            df = load(path)
            if len(df) >= MIN_CANDLES:
                loaded[symbol] = df
        except Exception:
            pass

    if not loaded:
        raise SystemExit("No valid non-IRT markets found.")

    global_max = min(int(df.timestamp.max()) for df in loaded.values())
    oos_end = global_max
    oos_start = oos_end - args.oos_days * 86400 + 3600

    print("=" * 112)
    print("NOBITEX NON-IRT | OOS 2.25% DIRECTION + MARKET BREADTH")
    print("=" * 112)
    print(f"Markets loaded : {len(loaded)}")
    print(f"OOS           : {pd.to_datetime(oos_start, unit='s', utc=True)} -> {pd.to_datetime(oos_end, unit='s', utc=True)}")
    print(f"Threshold     : {THRESHOLD * 100:.2f}%")
    print(f"Commission    : {FEE * 100:.2f}% per side ({FEE * 200:.2f}% round trip)")
    print()

    summaries = []
    for label, direction_filter in [("ALL", None), ("LONG", 1), ("SHORT", -1)]:
        finals = []
        trades = wins = 0
        positive = negative = flat = 0
        for symbol, df in loaded.items():
            seg = df[(df.timestamp >= oos_start) & (df.timestamp <= oos_end)].reset_index(drop=True)
            if len(seg) < MIN_CANDLES:
                continue
            r = backtest(seg, direction_filter)
            finals.append((symbol, r))
            trades += int(r["trades"])
            wins += int(r["wins"])
            if r["return"] > 0:
                positive += 1
            elif r["return"] < 0:
                negative += 1
            else:
                flat += 1

        pooled_initial = 1000.0 * len(finals)
        pooled_final = sum(float(r["final"]) for _, r in finals)
        pooled_return = pooled_final / pooled_initial - 1.0 if pooled_initial else 0.0
        returns = [float(r["return"]) for _, r in finals]
        mean_ret = float(np.mean(returns)) if returns else 0.0
        median_ret = float(np.median(returns)) if returns else 0.0
        win_rate = wins / trades if trades else 0.0
        summaries.append((label, pooled_return, mean_ret, median_ret, positive, negative, flat, trades, win_rate))

    print("DIRECTION")
    print("DIR     POOLED    MEAN    MEDIAN   PROFITABLE  LOSING  FLAT  TRADES  WIN RATE")
    for s in summaries:
        print(f"{s[0]:5s} {s[1]*100:+8.2f}% {s[2]*100:+7.2f}% {s[3]*100:+8.2f}% {s[4]:10d} {s[5]:7d} {s[6]:5d} {s[7]:7d} {s[8]*100:8.1f}%")

    all_results = []
    for symbol, df in loaded.items():
        seg = df[(df.timestamp >= oos_start) & (df.timestamp <= oos_end)].reset_index(drop=True)
        if len(seg) < MIN_CANDLES:
            continue
        r = backtest(seg, None)
        all_results.append({"symbol": symbol, **r})

    out = pd.DataFrame(all_results).sort_values("return", ascending=False).reset_index(drop=True)
    out_path = root / "nonirt_oos_225_breadth.csv"
    out.to_csv(out_path, index=False)

    print()
    print("ALL MARKET OOS BREADTH")
    print(f"profitable={int((out['return'] > 0).sum())} ({(out['return'] > 0).mean()*100:.1f}%)")
    print(f"losing    ={int((out['return'] < 0).sum())} ({(out['return'] < 0).mean()*100:.1f}%)")
    print(f"flat      ={int((out['return'] == 0).sum())} ({(out['return'] == 0).mean()*100:.1f}%)")
    print()
    print("TOP 10 OOS")
    for _, r in out.head(10).iterrows():
        print(f"{r.symbol:20s} return={r['return']*100:+7.2f}% trades={int(r['trades']):3d} win={r['wins']/r['trades']*100 if r['trades'] else 0:5.1f}%")
    print("BOTTOM 10 OOS")
    for _, r in out.tail(10).sort_values("return").iterrows():
        print(f"{r.symbol:20s} return={r['return']*100:+7.2f}% trades={int(r['trades']):3d} win={r['wins']/r['trades']*100 if r['trades'] else 0:5.1f}%")
    print()
    print(f"Detailed CSV: {out_path}")
    print("=" * 112)


if __name__ == "__main__":
    main()
