from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_THRESHOLD_PCT = 0.26
DEFAULT_ROUND_TRIP_FEE_PCT = 0.26


def norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def read_ohlc(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = {norm(x): x for x in (reader.fieldnames or [])}
        required = ["open", "high", "low", "close"]
        missing = [x for x in required if x not in fields]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        rows: list[dict[str, float]] = []
        for raw in reader:
            try:
                rows.append({x: float(raw[fields[x]]) for x in required})
            except (TypeError, ValueError):
                continue

    if len(rows) < 4:
        raise ValueError("Need at least 4 valid OHLC rows")
    return rows


def b(value: bool) -> int:
    return 1 if value else 0


def build_f(rows: list[dict[str, float]]) -> list[float | None]:
    """Build U first, then the supplied Excel column-F prediction for each candle."""
    u: list[float] = []

    for row in rows:
        open_ = row["open"]
        high = row["high"]
        low = row["low"]
        close = row["close"]

        co = close - open_
        hc = high - close
        ho = high - open_

        n = b(hc > co)
        o = 0
        p = b(hc > ho)
        r = b(hc < co)
        s = 0
        t = b(hc < ho)

        # Exact supplied AD formula as written in Excel.
        if (n + o + p == 2) and ((high - open_) > 50):
            ad = 1
        elif r + s + t == 2:
            ad = -1
        else:
            ad = 0

        if ad == 1:
            value = high - open_ + close
        elif ad == -1:
            value = low - open_ + close
        else:
            value = open_

        u.append(value)

    f: list[float | None] = [None] * len(rows)

    # Exact supplied Excel F formula:
    # =IF(E25-B25>0,AVERAGE(U23:U25),IF(E25-B25<0,AVERAGE(D23:D25),E25))
    for i, row in enumerate(rows):
        if i < 2:
            continue

        close = row["close"]
        open_ = row["open"]

        if close - open_ > 0:
            f[i] = sum(u[j] for j in range(i - 2, i + 1)) / 3.0
        elif close - open_ < 0:
            f[i] = sum(rows[j]["low"] for j in range(i - 2, i + 1)) / 3.0
        else:
            f[i] = close

    return f


def evaluate(
    rows: list[dict[str, float]],
    predictions: list[float | None],
    horizon: int,
    threshold_pct: float,
    round_trip_fee_pct: float,
) -> dict[str, float | int]:
    test_rows: list[tuple[float, float, float]] = []

    for i, predicted_close in enumerate(predictions):
        if predicted_close is None:
            continue
        target_i = i + horizon
        if target_i >= len(rows):
            continue

        current_close = rows[i]["close"]
        actual_close = rows[target_i]["close"]
        predicted_move = float(predicted_close) - current_close
        actual_move = actual_close - current_close
        test_rows.append((predicted_move, actual_move, current_close))

    direction_rows = [r for r in test_rows if r[0] != 0 and r[1] != 0]
    magnitude_rows = [r for r in test_rows if r[2] != 0]

    direction_correct = sum(b((r[0] > 0) == (r[1] > 0)) for r in direction_rows)
    direction_accuracy = direction_correct / len(direction_rows) if direction_rows else 0.0

    magnitude_correct = 0
    threshold_rows = 0
    threshold_correct = 0
    pred_pct_sum = 0.0
    actual_pct_sum = 0.0

    gross_profit_sum = 0.0
    net_profit_sum = 0.0
    wins_net = 0

    for predicted_move, actual_move, current_close in magnitude_rows:
        pred_pct = abs(predicted_move) / abs(current_close) * 100.0
        actual_pct = abs(actual_move) / abs(current_close) * 100.0
        magnitude_correct += b((pred_pct >= threshold_pct) == (actual_pct >= threshold_pct))

        if pred_pct >= threshold_pct:
            threshold_rows += 1
            threshold_correct += b(actual_pct >= threshold_pct)
            pred_pct_sum += pred_pct
            actual_pct_sum += actual_pct

            # Trade in the direction predicted by F and hold for the selected horizon.
            direction = 1.0 if predicted_move > 0 else (-1.0 if predicted_move < 0 else 0.0)
            gross_return_pct = direction * (actual_move / current_close) * 100.0
            net_return_pct = gross_return_pct - round_trip_fee_pct

            gross_profit_sum += gross_return_pct
            net_profit_sum += net_return_pct
            wins_net += b(net_return_pct > 0)

    magnitude_accuracy = magnitude_correct / len(magnitude_rows) if magnitude_rows else 0.0
    coverage = threshold_rows / len(magnitude_rows) if magnitude_rows else 0.0
    precision = threshold_correct / threshold_rows if threshold_rows else 0.0
    avg_predicted = pred_pct_sum / threshold_rows if threshold_rows else 0.0
    avg_actual = actual_pct_sum / threshold_rows if threshold_rows else 0.0
    avg_gross_profit = gross_profit_sum / threshold_rows if threshold_rows else 0.0
    avg_net_profit = net_profit_sum / threshold_rows if threshold_rows else 0.0
    win_rate_net = wins_net / threshold_rows if threshold_rows else 0.0

    return {
        "test_n": len(test_rows),
        "direction_n": len(direction_rows),
        "direction_correct": direction_correct,
        "direction_accuracy": direction_accuracy,
        "magnitude_n": len(magnitude_rows),
        "magnitude_correct": magnitude_correct,
        "magnitude_accuracy": magnitude_accuracy,
        "threshold_n": threshold_rows,
        "threshold_coverage": coverage,
        "threshold_precision": precision,
        "avg_predicted": avg_predicted,
        "avg_actual": avg_actual,
        "avg_gross_profit": avg_gross_profit,
        "avg_net_profit": avg_net_profit,
        "total_net_profit": net_profit_sum,
        "win_rate_net": win_rate_net,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Test supplied Excel column-F prediction across horizons with profit metrics")
    ap.add_argument("--input", required=True)
    ap.add_argument("--threshold-pct", type=float, default=DEFAULT_THRESHOLD_PCT)
    ap.add_argument("--round-trip-fee-pct", type=float, default=DEFAULT_ROUND_TRIP_FEE_PCT)
    ap.add_argument("--horizons", default="1,2,3,4,5")
    args = ap.parse_args()

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError("--horizons must contain positive integers")

    rows = read_ohlc(Path(args.input))
    predictions = build_f(rows)

    print("=" * 150)
    print("NEW ALGORITHM FORMULA - COLUMN F - PROFIT HORIZON SWEEP")
    print("=" * 150)
    print(f"input={args.input}")
    print(f"OHLC rows={len(rows)}")
    print(f"threshold={args.threshold_pct:.4f}%")
    print(f"round-trip fee={args.round_trip_fee_pct:.4f}%")
    print("F is fixed; entry only when |F-Close| >= threshold; direction = sign(F-Close)")
    print("profit target = Close[t+horizon]; net return = directional return - round-trip fee")
    print("-" * 150)
    print(
        "H  TestN  ThrN  Coverage%  Precision%  DirAcc%  MagAcc%  "
        "AvgPred%  AvgActual%  AvgGross%  AvgNet%  WinRateNet%  TotalNet%"
    )

    for horizon in horizons:
        m = evaluate(rows, predictions, horizon, args.threshold_pct, args.round_trip_fee_pct)
        print(
            f"{horizon:>1}  "
            f"{m['test_n']:>6}  "
            f"{m['threshold_n']:>4}  "
            f"{m['threshold_coverage'] * 100:>9.2f}  "
            f"{m['threshold_precision'] * 100:>10.2f}  "
            f"{m['direction_accuracy'] * 100:>7.2f}  "
            f"{m['magnitude_accuracy'] * 100:>7.2f}  "
            f"{m['avg_predicted']:>9.6f}  "
            f"{m['avg_actual']:>10.6f}  "
            f"{m['avg_gross_profit']:>9.6f}  "
            f"{m['avg_net_profit']:>8.6f}  "
            f"{m['win_rate_net'] * 100:>11.2f}  "
            f"{m['total_net_profit']:>10.4f}"
        )

    print("=" * 150)


if __name__ == "__main__":
    main()
