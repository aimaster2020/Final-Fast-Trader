from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
DEFAULT_THRESHOLDS = "0.001,0.002,0.003,0.004,0.005,0.006,0.008"


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    notional: float


def load_data(symbol: str) -> pd.DataFrame:
    path = f"reports/1h/{symbol}_1h.csv"
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    df["J"] = df["close"] - df["open"]
    df["K"] = df["high"] - df["close"]
    df["L"] = df["close"] - df["low"]
    df["J_next"] = df["J"].shift(-1)
    df["U"] = df["J_next"] - df["J"]
    return df.dropna(subset=["J_next", "U"]).reset_index(drop=True)


def fit_model(train: pd.DataFrame) -> np.ndarray:
    x = np.column_stack([
        np.ones(len(train)),
        train[["J", "K", "L"]].to_numpy(float),
    ])
    y = train["U"].to_numpy(float)
    return np.linalg.lstsq(x, y, rcond=None)[0]


def predict(model: np.ndarray, row: pd.Series) -> float:
    x = np.array([1.0, float(row["J"]), float(row["K"]), float(row["L"])])
    return float(x @ model)


def run_month(
    data_by_symbol: dict[str, pd.DataFrame],
    month: str,
    leverage: float,
    long_threshold: float,
    short_threshold: float,
    initial_capital: float,
    position_fraction: float,
    commission_per_side: float,
) -> dict[str, float]:
    month_data: dict[str, pd.DataFrame] = {}
    models: dict[str, np.ndarray] = {}

    for symbol, df in data_by_symbol.items():
        train = df[df["timestamp"].dt.strftime("%Y-%m") < month]
        test = df[df["timestamp"].dt.strftime("%Y-%m") == month].copy().reset_index(drop=True)
        if len(train) < 50 or len(test) < 2:
            continue
        models[symbol] = fit_model(train)
        month_data[symbol] = test

    timeline = sorted(set(ts for df in month_data.values() for ts in df["timestamp"]))
    maps = {s: df.set_index("timestamp") for s, df in month_data.items()}

    capital = float(initial_capital)
    positions: dict[str, Position] = {}
    trades = 0
    wins = 0
    fees = 0.0
    gross_pnl = 0.0

    for ts in timeline:
        for symbol in list(positions):
            if ts not in maps[symbol].index:
                continue
            p = positions[symbol]
            row = maps[symbol].loc[ts]
            close = float(row["close"])
            ret = (close - p.entry_price) / p.entry_price if p.side == "LONG" else (p.entry_price - close) / p.entry_price
            gross = p.notional * ret
            exit_fee = p.notional * commission_per_side
            net = gross - exit_fee
            capital += net
            gross_pnl += gross
            fees += p.notional * commission_per_side + exit_fee
            trades += 1
            if net > 0:
                wins += 1
            del positions[symbol]

        for symbol, df in month_data.items():
            if symbol in positions:
                continue
            idx = df.index[df["timestamp"] == ts]
            if len(idx) == 0:
                continue
            i = int(idx[0])
            if i >= len(df) - 1:
                continue

            row = df.iloc[i]
            nxt = df.iloc[i + 1]
            predicted_body = float(row["J"]) + predict(models[symbol], row)
            entry = float(nxt["open"])
            predicted_pct = predicted_body / entry

            if predicted_pct >= long_threshold:
                side = "LONG"
            elif predicted_pct <= -short_threshold:
                side = "SHORT"
            else:
                continue

            margin = capital * position_fraction
            notional = margin * leverage
            entry_fee = notional * commission_per_side
            capital -= entry_fee
            fees += entry_fee
            positions[symbol] = Position(symbol, side, entry, notional)

    for symbol, p in list(positions.items()):
        row = month_data[symbol].iloc[-1]
        close = float(row["close"])
        ret = (close - p.entry_price) / p.entry_price if p.side == "LONG" else (p.entry_price - close) / p.entry_price
        gross = p.notional * ret
        exit_fee = p.notional * commission_per_side
        net = gross - exit_fee
        capital += net
        gross_pnl += gross
        fees += exit_fee
        trades += 1
        if net > 0:
            wins += 1

    return {
        "trades": float(trades),
        "win_pct": wins / trades * 100.0 if trades else 0.0,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "net_pnl": capital - initial_capital,
        "return_pct": (capital / initial_capital - 1.0) * 100.0,
        "capital": capital,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep the established body-magnitude entry threshold with 0 and 0.13% per-side commission.")
    p.add_argument("--month", required=True)
    p.add_argument("--leverage", type=float, choices=[1.0, 10.0], required=True)
    p.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    p.add_argument("--short-ratio", type=float, default=1.0375, help="short threshold = long threshold * ratio; preserves 0.0080/0.0083 canonical ratio")
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--position-fraction", type=float, default=0.10)
    args = p.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    data = {s: load_data(s) for s in SYMBOLS}

    print(f"BODY MAGNITUDE THRESHOLD SWEEP month={args.month} leverage={args.leverage:g}x")
    print("model=established J,K,L -> predicted J_next")
    print("entry=next open | hold=1 candle | initial=$1000 | position=10% | short_ratio=%.4f" % args.short_ratio)
    print("columns: threshold trades win return fees final")
    print("threshold  commission  trades  win%    return%    fees    final")
    print("----------------------------------------------------------------")

    for fee in [0.0, 0.0013]:
        label = f"{fee * 100:.2f}%/side"
        for threshold in thresholds:
            short_threshold = threshold * args.short_ratio
            r = run_month(
                data,
                args.month,
                args.leverage,
                threshold,
                short_threshold,
                args.initial_capital,
                args.position_fraction,
                fee,
            )
            print(
                f"{threshold * 100:8.3f}%  {label:>12}"
                f"  {int(r['trades']):6d}  {r['win_pct']:5.1f}%"
                f"  {r['return_pct']:+8.2f}%  {r['fees']:7.2f}  {r['capital']:8.2f}"
            )


if __name__ == "__main__":
    main()
