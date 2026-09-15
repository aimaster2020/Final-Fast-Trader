from __future__ import annotations

import argparse
import csv
from pathlib import Path

RAW_FIELDS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]


def num(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sign_asymmetric(value: float, up_threshold: float, down_threshold: float) -> int:
    if value > up_threshold:
        return 1
    if value < -down_threshold:
        return -1
    return 0


def looks_like_header(row: list[str]) -> bool:
    normalized = [str(x).strip().lower() for x in row]
    return {"timestamp", "open", "high", "low", "close"}.issubset(normalized)


def is_year(value: str, year: int) -> bool:
    text = str(value).strip()
    return text.startswith(str(year))


def load_rows(path: Path, year: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return []

        if looks_like_header(first):
            fields = [str(x).strip() for x in first]
            raw_rows = [
                {k: raw[idx] if idx < len(raw) else "" for idx, k in enumerate(fields)}
                for raw in reader
            ]
        else:
            if len(first) < 5:
                raise ValueError("Expected headerless order: Timestamp,Open,High,Low,Close[,Volume].")
            raw_rows = [
                {k: raw[idx] if idx < len(raw) else "" for idx, k in enumerate(RAW_FIELDS)}
                for raw in [first] + list(reader)
            ]

    rows = [row for row in raw_rows if is_year(row.get("Timestamp", ""), year)]
    if not rows:
        raise ValueError(f"No rows found for year {year}.")
    return rows


def calculate(rows, ct_up: float, ct_down: float) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    previous_pt: int | None = None

    for i in range(len(rows) - 1):
        current = rows[i]
        nxt = rows[i + 1]
        o = num(current.get("Open"))
        h = num(current.get("High"))
        low = num(current.get("Low"))
        c = num(current.get("Close"))
        next_c = num(nxt.get("Close"))

        if None in (o, h, low, c, next_c):
            result.append({"pt": "", "ct": "", "Long_Return": 0.0, "Short_Return": 0.0})
            continue

        predict: float | None = None
        ct: int | None = None
        pt: int | None = None
        pt_eq_ct = 0
        neg_pt_eq_ct = 0
        long_return = 0.0
        short_return = 0.0

        if i >= 5:
            window = rows[i - 5 : i + 1]
            closes = [num(x.get("Close")) for x in window]
            highs = [num(x.get("High")) for x in window]
            if all(v is not None for v in closes + highs):
                body = c - o
                if body > 0:
                    predict = sum(closes) / 6.0
                elif body < 0:
                    predict = sum(highs) / 6.0
                else:
                    predict = c

                ct = sign_asymmetric(next_c - c, ct_up, ct_down)
                pt = sign_asymmetric(predict - c, ct_up, ct_down)
                pt_eq_ct = int(pt == ct and pt == 1)
                neg_pt_eq_ct = int(pt == ct and pt == -1)

                if pt == 1 or previous_pt == 1:
                    long_return = (next_c - c) / c if c else 0.0
                if pt == -1 or previous_pt == -1:
                    short_return = (c - next_c) / next_c if next_c else 0.0

                previous_pt = pt

        result.append({
            "Timestamp": current.get("Timestamp", ""),
            "Open": o,
            "High": h,
            "Low": low,
            "Close": c,
            "predict": predict if predict is not None else "",
            "ct": ct if ct is not None else "",
            "pt": pt if pt is not None else "",
            "pt=ct": pt_eq_ct,
            "(-)pt=ct": neg_pt_eq_ct,
            "Long_Return": long_return,
            "Short_Return": short_return,
        })

    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Timestamp", "Open", "High", "Low", "Close",
        "predict", "ct", "pt", "pt=ct", "(-)pt=ct",
        "Long_Return", "Short_Return",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_completed_trades(
    rows: list[dict[str, object]],
    commission_per_side: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build actual position trades from PT signals.

    A position is opened on the first non-zero PT signal after being flat.
    Repeated same-direction PT signals do not re-enter and do not add fees.
    PT=0 does not exit an active position.
    An opposite PT signal closes the current trade at that candle's Close and
    immediately opens the opposite direction at the same Close. A position
    still open on the final row is reported separately and excluded from the
    completed-trade win/loss counts.
    """
    valid = [
        r for r in rows
        if r.get("pt") != "" and r.get("Close") not in (None, "")
    ]

    trades: list[dict[str, object]] = []
    position = 0
    entry_price = None
    entry_timestamp = None
    entry_index = None

    for idx, row in enumerate(valid):
        pt = int(row["pt"])
        close = float(row["Close"])
        timestamp = row.get("Timestamp", "")
        desired = pt if pt in (-1, 1) else 0

        if position == 0:
            if desired != 0:
                position = desired
                entry_price = close
                entry_timestamp = timestamp
                entry_index = idx
            continue

        if desired == 0 or desired == position:
            continue

        exit_price = close
        if position == 1:
            gross_return = (exit_price - entry_price) / entry_price
            direction = "LONG"
        else:
            gross_return = (entry_price - exit_price) / entry_price
            direction = "SHORT"

        net_return = gross_return - (2.0 * commission_per_side)
        trades.append({
            "direction": direction,
            "entry_timestamp": entry_timestamp,
            "exit_timestamp": timestamp,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_return": gross_return,
            "net_return": net_return,
            "win": net_return > 0,
            "bars_held": idx - entry_index,
        })

        # Reverse at the same close: old position exits and new position enters.
        position = desired
        entry_price = close
        entry_timestamp = timestamp
        entry_index = idx

    open_trade = None
    if position != 0 and entry_price is not None:
        direction = "LONG" if position == 1 else "SHORT"
        last = valid[-1]
        last_price = float(last["Close"])
        if position == 1:
            unrealized_return = (last_price - entry_price) / entry_price
        else:
            unrealized_return = (entry_price - last_price) / entry_price
        open_trade = {
            "direction": direction,
            "entry_timestamp": entry_timestamp,
            "last_timestamp": last.get("Timestamp", ""),
            "entry_price": entry_price,
            "last_price": last_price,
            "gross_unrealized_return": unrealized_return,
            "net_unrealized_after_entry_fee": unrealized_return - commission_per_side,
            "bars_held": len(valid) - 1 - entry_index,
        }

    return trades, {
        "completed_trades": len(trades),
        "open_trade": open_trade,
    }


def summarize_actual_trades(
    trades: list[dict[str, object]],
    open_trade: dict[str, object] | None,
    commission_per_side: float,
) -> None:
    long_trades = [t for t in trades if t["direction"] == "LONG"]
    short_trades = [t for t in trades if t["direction"] == "SHORT"]

    def print_side(name: str, side_trades: list[dict[str, object]]) -> None:
        wins = [t for t in side_trades if t["win"]]
        losses = [t for t in side_trades if not t["win"]]
        gross = sum(float(t["gross_return"]) for t in side_trades)
        net = sum(float(t["net_return"]) for t in side_trades)
        print(f"{name} trades={len(side_trades)} wins={len(wins)} losses={len(losses)}")
        print(
            f"{name} win_rate={len(wins) / len(side_trades) * 100:.3f}% "
            f"avg_net={net / len(side_trades) if side_trades else 0.0:.10f} "
            f"sum_gross={gross:.10f} sum_net={net:.10f}"
        )

    print()
    print("ACTUAL COMPLETED TRADES")
    print("commission is charged only on actual entry/exit events")
    print_side("LONG", long_trades)
    print_side("SHORT", short_trades)
    print(f"TOTAL completed_trades={len(trades)}")
    print(f"TOTAL wins={sum(1 for t in trades if t['win'])} losses={sum(1 for t in trades if not t['win'])}")
    print(f"COMMISSION_PER_SIDE={commission_per_side * 100:.4f}%")
    print(f"COMPLETED_GROSS_RETURN_SUM={sum(float(t['gross_return']) for t in trades):.10f}")
    print(f"COMPLETED_NET_RETURN_SUM={sum(float(t['net_return']) for t in trades):.10f}")

    if open_trade is None:
        print("OPEN_TRADE=none")
    else:
        print(
            f"OPEN_TRADE={open_trade['direction']} "
            f"gross_unrealized={float(open_trade['gross_unrealized_return']):.10f} "
            f"net_after_entry_fee={float(open_trade['net_unrealized_after_entry_fee']):.10f}"
        )


def commission_events(rows: list[dict[str, object]], commission_per_side: float) -> dict[str, float | int]:
    """Charge commission only when the actual position changes."""
    valid = [r for r in rows if r.get("pt") != ""]
    position = 0
    entry_count = 0
    exit_count = 0
    long_entries = 0
    short_entries = 0
    long_exits = 0
    short_exits = 0

    for row in valid:
        pt = int(row["pt"])
        desired = pt if pt in (-1, 1) else 0

        if position == 0:
            if desired != 0:
                position = desired
                entry_count += 1
                if position == 1:
                    long_entries += 1
                else:
                    short_entries += 1
        elif position == 1 and desired == -1:
            exit_count += 1
            long_exits += 1
            position = -1
            entry_count += 1
            short_entries += 1
        elif position == -1 and desired == 1:
            exit_count += 1
            short_exits += 1
            position = 1
            entry_count += 1
            long_entries += 1

    return {
        "entry_count": entry_count,
        "exit_count": exit_count,
        "long_entries": long_entries,
        "short_entries": short_entries,
        "long_exits": long_exits,
        "short_exits": short_exits,
        "open_position": position,
        "commission_total": (entry_count + exit_count) * commission_per_side,
    }


def summarize(rows: list[dict[str, object]], commission_per_side: float) -> None:
    valid = [r for r in rows if r.get("pt") != ""]
    up = [r for r in valid if r["pt"] == 1]
    down = [r for r in valid if r["pt"] == -1]
    hold = [r for r in valid if r["pt"] == 0]
    up_correct = sum(1 for r in up if r["ct"] == 1)
    down_correct = sum(1 for r in down if r["ct"] == -1)

    long_returns = [float(r["Long_Return"]) for r in rows if float(r["Long_Return"]) != 0.0]
    short_returns = [float(r["Short_Return"]) for r in rows if float(r["Short_Return"]) != 0.0]
    long_positive = [x for x in long_returns if x > 0]
    long_negative = [x for x in long_returns if x < 0]
    short_positive = [x for x in short_returns if x > 0]
    short_negative = [x for x in short_returns if x < 0]

    long_sum = sum(long_returns)
    short_sum = sum(short_returns)
    gross_sum = long_sum + short_sum

    fees = commission_events(rows, commission_per_side)
    net_sum = gross_sum - float(fees["commission_total"])

    actual_trades, trade_state = build_completed_trades(rows, commission_per_side)

    print("=" * 100)
    print("EXACT EXCEL SAMPLE / FORMULA TEST")
    print("=" * 100)
    print(f"frames={len(valid)}")
    print(f"YEAR={ARGS.year}")
    print(f"CT_UP={ARGS.ct_up:g}")
    print(f"CT_DOWN={ARGS.ct_down:g}")
    print(f"COMMISSION_PER_SIDE={commission_per_side * 100:.4f}%")
    print()
    print("pt summary")
    print(f"all={len(valid)}")
    print(f"up={len(up)} accuracy_vs_ct={up_correct / len(up) * 100:.3f}%" if up else "up=0")
    print(f"down={len(down)} accuracy_vs_ct={down_correct / len(down) * 100:.3f}%" if down else "down=0")
    print(f"hold={len(hold)}")
    print()
    print("final Long formula")
    print(
        f"signals={len(long_returns)} positive={len(long_positive)} negative={len(long_negative)} "
        f"hit_rate={len(long_positive) / len(long_returns) * 100:.3f}%" if long_returns else "signals=0"
    )
    print(f"avg_return={long_sum / len(long_returns) if long_returns else 0.0:.10f} total_return={long_sum:.10f}")
    print(f"avg_win={sum(long_positive) / len(long_positive) if long_positive else 0.0:.10f} avg_loss={sum(long_negative) / len(long_negative) if long_negative else 0.0:.10f}")
    print()
    print("final Short formula")
    print(
        f"signals={len(short_returns)} positive={len(short_positive)} negative={len(short_negative)} "
        f"hit_rate={len(short_positive) / len(short_returns) * 100:.3f}%" if short_returns else "signals=0"
    )
    print(f"avg_return={short_sum / len(short_returns) if short_returns else 0.0:.10f} total_return={short_sum:.10f}")
    print(f"avg_win={sum(short_positive) / len(short_positive) if short_positive else 0.0:.10f} avg_loss={sum(short_negative) / len(short_negative) if short_negative else 0.0:.10f}")
    print()
    print("actual position / commission")
    print(f"entries={fees['entry_count']} exits={fees['exit_count']}")
    print(f"long_entries={fees['long_entries']} short_entries={fees['short_entries']}")
    print(f"long_exits={fees['long_exits']} short_exits={fees['short_exits']}")
    print(f"open_position_at_end={fees['open_position']}")
    print(f"commission_total={float(fees['commission_total']):.10f}")
    print()
    print("combined")
    print(f"signal_count={len(long_returns) + len(short_returns)}")
    print(f"gross_total_return_sum={gross_sum:.10f}")
    print(f"net_total_return_sum={net_sum:.10f}")

    summarize_actual_trades(actual_trades, trade_state["open_trade"], commission_per_side)


ARGS = None


def main() -> None:
    global ARGS
    p = argparse.ArgumentParser(description="Exact implementation of the supplied Excel sample/formula logic")
    p.add_argument("--input", required=True)
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--ct-up", type=float, default=600.0)
    p.add_argument("--ct-down", type=float, default=100.0)
    p.add_argument("--commission", type=float, default=0.0013, help="commission per actual entry/exit side; default 0.13%%")
    p.add_argument("--output", required=True)
    ARGS = p.parse_args()

    if ARGS.ct_up < 0 or ARGS.ct_down < 0 or ARGS.commission < 0:
        raise ValueError("thresholds and commission must be >= 0")

    rows = load_rows(Path(ARGS.input), ARGS.year)
    rows.sort(key=lambda row: row.get("Timestamp", ""))
    result = calculate(rows, ARGS.ct_up, ARGS.ct_down)
    write_csv(Path(ARGS.output), result)
    print(f"output={ARGS.output}")
    print(f"input_rows_{ARGS.year}={len(rows)}")
    summarize(result, ARGS.commission)


if __name__ == "__main__":
    main()
