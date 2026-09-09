from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path



def load(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    {
                        "timestamp": float(row["timestamp"]),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(rows, key=lambda r: r["timestamp"])



def evaluate(rows: list[dict[str, float]]) -> dict[str, float | int]:
    errors: list[float] = []
    sq_errors: list[float] = []
    correct = 0
    total = 0
    formula_sum = 0.0
    actual_sum = 0.0

    for i in range(len(rows) - 1):
        current = rows[i]
        nxt = rows[i + 1]
        close = current["close"]
        body = close - current["open"]

        # Exact formula represented by the final spreadsheet columns:
        # CO = C - O
        # 2CO = 2 * (C - O)
        predicted_delta = 2.0 * body
        actual_delta = nxt["close"] - close

        error = predicted_delta - actual_delta
        errors.append(abs(error))
        sq_errors.append(error * error)
        total += 1
        correct += int(
            (predicted_delta > 0 and actual_delta > 0)
            or (predicted_delta < 0 and actual_delta < 0)
            or (predicted_delta == 0 and actual_delta == 0)
        )
        formula_sum += predicted_delta
        actual_sum += actual_delta

    if not total:
        return {"rows": 0}

    mae = sum(errors) / total
    rmse = math.sqrt(sum(sq_errors) / total)
    directional_accuracy = correct / total * 100.0

    return {
        "rows": total,
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
        "avg_predicted_delta": formula_sum / total,
        "avg_actual_delta": actual_sum / total,
    }



def main() -> None:
    ap = argparse.ArgumentParser(
        description="Test the exact 2*(Close-Open) formula against the next candle Close delta."
    )
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    path = Path(args.input)
    rows = load(path)
    result = evaluate(rows)

    if not result.get("rows"):
        raise SystemExit("NO_VALID_ROWS")

    print(
        f"FORMULA_2CO | file={path.name} | rows={int(result['rows'])} | "
        f"MAE={float(result['mae']):.6f} | RMSE={float(result['rmse']):.6f} | "
        f"DIR_ACC={float(result['directional_accuracy']):.2f}% | "
        f"AVG_PRED_DELTA={float(result['avg_predicted_delta']):+.6f} | "
        f"AVG_ACTUAL_DELTA={float(result['avg_actual_delta']):+.6f}"
    )


if __name__ == "__main__":
    main()
