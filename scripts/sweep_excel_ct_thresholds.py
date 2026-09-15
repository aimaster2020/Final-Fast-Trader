from __future__ import annotations

import argparse
import csv
from pathlib import Path

from analyze_excel_prediction_columns import commission_events, calculate, load_rows


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep CT_UP and CT_DOWN thresholds over a grid.")
    p.add_argument("--input", required=True)
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=1000)
    p.add_argument("--step", type=int, default=50)
    p.add_argument("--commission", type=float, default=0.0013)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if args.start < 0 or args.end < args.start or args.step <= 0 or args.commission < 0:
        raise ValueError("start/end/step/commission values are invalid")

    rows = load_rows(Path(args.input), args.year)
    rows.sort(key=lambda row: row.get("Timestamp", ""))
    values = list(range(args.start, args.end + 1, args.step))

    out = []
    for ct_up in values:
        for ct_down in values:
            result = calculate(rows, ct_up, ct_down)
            valid = [r for r in result if r.get("pt") != ""]
            long_returns = [float(r["Long_Return"]) for r in valid if float(r["Long_Return"]) != 0.0]
            short_returns = [float(r["Short_Return"]) for r in valid if float(r["Short_Return"]) != 0.0]
            fees = commission_events(result, args.commission)
            gross = sum(long_returns) + sum(short_returns)
            net = gross - float(fees["commission_total"])

            out.append({
                "ct_up": ct_up,
                "ct_down": ct_down,
                "frames": len(valid),
                "long_signal_frames": len(long_returns),
                "short_signal_frames": len(short_returns),
                "entries": fees["entry_count"],
                "exits": fees["exit_count"],
                "long_entries": fees["long_entries"],
                "short_entries": fees["short_entries"],
                "long_exits": fees["long_exits"],
                "short_exits": fees["short_exits"],
                "completed_trades": fees["exit_count"],
                "gross_return_sum": gross,
                "commission_total": fees["commission_total"],
                "net_return_sum": net,
                "open_position_at_end": fees["open_position"],
            })

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(out[0].keys()) if out else ["ct_up", "ct_down"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)

    ranked = sorted(out, key=lambda r: float(r["net_return_sum"]), reverse=True)
    print(f"output={path}")
    print(f"input_rows_{args.year}={len(rows)}")
    print(f"combinations={len(out)}")
    print()
    print("TOP 20 BY NET_RETURN_SUM")
    print("ct_up,ct_down,entries,exits,completed_trades,gross_return_sum,commission_total,net_return_sum,open_position_at_end")
    for r in ranked[:20]:
        print(
            f"{r['ct_up']},{r['ct_down']},{r['entries']},{r['exits']},{r['completed_trades']},"
            f"{float(r['gross_return_sum']):.10f},{float(r['commission_total']):.10f},"
            f"{float(r['net_return_sum']):.10f},{r['open_position_at_end']}"
        )


if __name__ == "__main__":
    main()
