from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
COMMISSION_PER_SIDE = 0.0013
ROUND_TRIP_COMMISSION = COMMISSION_PER_SIDE * 2
DEFAULT_POSITION_FRACTION = 0.10
DEFAULT_INITIAL_CAPITAL = 1000.0
# Fine sweep around the useful 40%-80% range from the first test.
COVERAGES = [i / 100 for i in range(40, 81, 5)]


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    margin: float
    notional: float
    entry_fee: float
    entry_time: pd.Timestamp


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], unit="s", utc=True, errors="coerce"
    )
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    df["J"] = df["close"] - df["open"]
    df["K"] = df["high"] - df["close"]
    df["L"] = df["close"] - df["low"]
    df["J_next"] = df["J"].shift(-1)
    df["U"] = df["J_next"] - df["J"]

    return df.dropna(subset=["J_next", "U"]).reset_index(drop=True)


def fit_model(train: pd.DataFrame) -> np.ndarray:
    x = np.column_stack(
        [np.ones(len(train)), train[["J", "K", "L"]].to_numpy(float)]
    )
    y = train["U"].to_numpy(float)
    return np.linalg.lstsq(x, y, rcond=None)[0]


def predict(model: np.ndarray, row: pd.Series) -> float:
    x = np.array([1.0, float(row["J"]), float(row["K"]), float(row["L"])])
    return float(x @ model)


def price_return(side: str, entry: float, exit: float) -> float:
    if side == "LONG":
        return (exit - entry) / entry
    return (entry - exit) / entry


def run(
    data_by_symbol,
    month,
    leverage,
    coverage,
    initial_capital,
    position_fraction,
):
    month_data = {}
    models = {}

    for symbol, df in data_by_symbol.items():
        train = df[df["timestamp"].dt.strftime("%Y-%m") < month]
        test = (
            df[df["timestamp"].dt.strftime("%Y-%m") == month]
            .copy()
            .reset_index(drop=True)
        )
        if len(train) < 50 or len(test) < 2:
            raise ValueError(f"{symbol}: insufficient data for month {month}")
        models[symbol] = fit_model(train)
        month_data[symbol] = test

    threshold = coverage * ROUND_TRIP_COMMISSION
    timeline = sorted(
        set(ts for df in month_data.values() for ts in df["timestamp"])
    )
    maps = {s: df.set_index("timestamp") for s, df in month_data.items()}

    capital = float(initial_capital)
    positions = {}
    trades = []

    for ts in timeline:
        # Close the previous one-candle position at the current candle close.
        for symbol in list(positions):
            p = positions[symbol]
            if ts not in maps[symbol].index:
                continue
            row = maps[symbol].loc[ts]
            pnl = p.notional * price_return(
                p.side, p.entry_price, float(row["close"])
            )
            exit_fee = p.notional * COMMISSION_PER_SIDE
            net = pnl - exit_fee
            capital += net
            trades.append(
                (
                    symbol,
                    p.side,
                    p.entry_time,
                    ts,
                    p.margin,
                    p.notional,
                    net,
                    p.entry_fee + exit_fee,
                )
            )
            del positions[symbol]

        # Signal from candle t; enter at candle t+1 open.
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

            # Symmetric magnitude filter: direction comes from the sign,
            # while the threshold is a percentage of round-trip commission.
            if abs(predicted_pct) < threshold:
                continue
            side = "LONG" if predicted_pct > 0 else "SHORT"

            margin = capital * position_fraction
            notional = margin * leverage
            entry_fee = notional * COMMISSION_PER_SIDE
            capital -= entry_fee
            positions[symbol] = Position(
                symbol=symbol,
                side=side,
                entry_price=entry,
                margin=margin,
                notional=notional,
                entry_fee=entry_fee,
                entry_time=nxt["timestamp"],
            )

    # Close any remaining position at the final available close.
    for symbol, p in list(positions.items()):
        row = month_data[symbol].iloc[-1]
        ts = row["timestamp"]
        pnl = p.notional * price_return(
            p.side, p.entry_price, float(row["close"])
        )
        exit_fee = p.notional * COMMISSION_PER_SIDE
        net = pnl - exit_fee
        capital += net
        trades.append(
            (
                symbol,
                p.side,
                p.entry_time,
                ts,
                p.margin,
                p.notional,
                net,
                p.entry_fee + exit_fee,
            )
        )
        del positions[symbol]

    fees = sum(t[7] for t in trades)
    wins = sum(1 for t in trades if t[6] > 0)

    return {
        "coverage_pct": coverage * 100,
        "threshold_pct": threshold * 100,
        "capital": capital,
        "return_pct": (capital / initial_capital - 1) * 100,
        "trades": len(trades),
        "win_pct": wins / len(trades) * 100 if trades else 0.0,
        "fees": fees,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--month", required=True, help="YYYY-MM")
    p.add_argument("--leverage", type=float, choices=[1.0, 10.0], required=True)
    p.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    p.add_argument("--position-fraction", type=float, default=DEFAULT_POSITION_FRACTION)
    args = p.parse_args()

    data = {s: load_data(f"reports/1h/{s}_1h.csv") for s in SYMBOLS}
    results = [
        run(
            data,
            args.month,
            args.leverage,
            coverage,
            args.initial_capital,
            args.position_fraction,
        )
        for coverage in COVERAGES
    ]

    print(
        f"month={args.month} leverage={args.leverage:g}x "
        f"round_trip_commission={ROUND_TRIP_COMMISSION * 100:.3f}%"
    )
    print("coverage threshold trades win return fees capital")
    for r in results:
        print(
            f"{r['coverage_pct']:>7.0f}% "
            f"{r['threshold_pct']:>8.3f}% "
            f"{r['trades']:>6} "
            f"{r['win_pct']:>5.1f}% "
            f"{r['return_pct']:>+7.2f}% "
            f"{r['fees']:>7.2f} "
            f"{r['capital']:>8.2f}"
        )

    best = max(results, key=lambda r: r["capital"])
    print(
        f"BEST coverage={best['coverage_pct']:.0f}% "
        f"threshold={best['threshold_pct']:.3f}% "
        f"return={best['return_pct']:+.2f}% "
        f"trades={best['trades']} capital={best['capital']:.2f}"
    )


if __name__ == "__main__":
    main()
