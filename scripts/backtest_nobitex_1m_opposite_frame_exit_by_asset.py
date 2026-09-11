from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NO_SIGNAL_MARKETS = {"DOSUSDT", "GRVTUSDT", "SLVONUSDT"}
DATA_DIR = "reports/nobitex_intraday_native/1m"
FEE = 0.0013
INITIAL_CAPITAL = 1000.0
MODES = (("consecutive", 1), ("consecutive", 2), ("consecutive", 3), ("discrete", 1), ("discrete", 2), ("discrete", 3))


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def direction_series(df: pd.DataFrame) -> np.ndarray:
    body = df["close"] - df["open"]
    hc = df["high"] - df["close"]
    ho = df["high"] - df["open"]
    lc = df["low"] - df["close"]
    score = (hc > body).astype(int) + (ho < hc).astype(int) + (lc > body).astype(int)
    return np.where(score <= 1, -1, np.where(score >= 2, 1, 0)).astype(int)


def simulate_symbol(df: pd.DataFrame, opposite_count: int, mode: str) -> dict[str, float | int]:
    dirs = direction_series(df)
    n = len(df)
    capital = INITIAL_CAPITAL
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
            elif mode == "consecutive":
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
        gross = ((exit_price - entry) / entry) if entry_dir == 1 else ((entry - exit_price) / entry)
        capital *= max((1.0 + gross) * (1.0 - FEE) ** 2, 0.0)

        trades += 1
        wins += int(gross > 0)
        gross_sum += gross
        hold_sum += exit_idx - i
        i = exit_idx

    return {
        "final": capital,
        "return_pct": (capital / INITIAL_CAPITAL - 1.0) * 100.0,
        "trades": trades,
        "win_pct": wins / trades * 100.0 if trades else np.nan,
        "avg_gross_pct": gross_sum / trades * 100.0 if trades else np.nan,
        "avg_hold_min": hold_sum / trades if trades else np.nan,
    }


def main() -> None:
    root = Path(DATA_DIR)
    files = sorted(root.glob("*_1m.csv"))
    loaded: dict[str, pd.DataFrame] = {}

    for path in files:
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

    if not loaded:
        raise SystemExit(f"No valid 1m non-IRT files found under {root}")

    records: list[dict[str, object]] = []
    for symbol in sorted(loaded):
        df = loaded[symbol]
        row: dict[str, object] = {"symbol": symbol, "candles": len(df)}
        for mode, k in MODES:
            r = simulate_symbol(df, k, mode)
            prefix = f"{mode[0].upper()}{k}"
            for key, value in r.items():
                row[f"{prefix}_{key}"] = value
            records.append({
                "symbol": symbol,
                "mode": mode,
                "n_opposite": k,
                "candles": len(df),
                **r,
            })

    detail = pd.DataFrame(records)
    detail_path = root / "backtest_opposite_frame_exit_1m_by_asset_detail.csv"
    detail.to_csv(detail_path, index=False)

    wide = detail.pivot(index="symbol", columns=["mode", "n_opposite"], values=["return_pct", "trades", "win_pct", "avg_gross_pct", "avg_hold_min"])
    # Flatten columns for easy Excel/CSV use.
    wide.columns = [f"{metric}_{'C' if mode == 'consecutive' else 'D'}{k}" for metric, mode, k in wide.columns]
    wide = wide.reset_index()

    for mode, k in MODES:
        ret_col = f"return_pct_{'C' if mode == 'consecutive' else 'D'}{k}"
        wide[f"rank_return_{'C' if mode == 'consecutive' else 'D'}{k}"] = wide[ret_col].rank(method="min", ascending=False).astype(int)

    wide_path = root / "backtest_opposite_frame_exit_1m_by_asset.csv"
    wide.to_csv(wide_path, index=False)

    print("=" * 150)
    print("NOBITEX NON-IRT | 1M SEQUENTIAL TRADING | PER-ASSET RESULTS")
    print("=" * 150)
    print("Entry: current frame formula direction, NO ambiguity filter | Entry: current close")
    print("Fee: 0.13% per side | 0.26% round trip")
    print("C1/C2/C3 = consecutive opposite frames | D1/D2/D3 = discrete opposite frames")
    print(f"Markets: {len(loaded)}")
    print()
    print("PER-ASSET RETURN%")
    print("SYMBOL               C1        C2        C3        D1        D2        D3        BEST")
    print("-" * 92)
    for _, r in wide.sort_values("return_pct_C3", ascending=False).iterrows():
        vals = {tag: float(r[f"return_pct_{tag}"]) for tag in ("C1", "C2", "C3", "D1", "D2", "D3")}
        best = max(vals, key=vals.get)
        print(
            f"{str(r['symbol']):20s} "
            f"{vals['C1']:>8.2f}% {vals['C2']:>8.2f}% {vals['C3']:>8.2f}% "
            f"{vals['D1']:>8.2f}% {vals['D2']:>8.2f}% {vals['D3']:>8.2f}%   {best}"
        )

    print()
    print("BEST 20 ASSETS BY C3 RETURN")
    for _, r in wide.sort_values("return_pct_C3", ascending=False).head(20).iterrows():
        print(
            f"{r['symbol']:20s} C3={r['return_pct_C3']:+8.2f}% trades={int(r['trades_C3']):6d} "
            f"win={r['win_pct_C3']:6.2f}% avg_gross={r['avg_gross_pct_C3']:+.4f}% hold={r['avg_hold_min_C3']:.1f}m"
        )

    print()
    print("WORST 20 ASSETS BY C3 RETURN")
    for _, r in wide.sort_values("return_pct_C3", ascending=True).head(20).iterrows():
        print(
            f"{r['symbol']:20s} C3={r['return_pct_C3']:+8.2f}% trades={int(r['trades_C3']):6d} "
            f"win={r['win_pct_C3']:6.2f}% avg_gross={r['avg_gross_pct_C3']:+.4f}% hold={r['avg_hold_min_C3']:.1f}m"
        )

    print()
    print(f"Wide CSV  : {wide_path}")
    print(f"Detail CSV: {detail_path}")
    print("=" * 150)


if __name__ == "__main__":
    main()
