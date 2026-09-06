from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_pattern_trader.models import Candle
from scripts.july_r_composite_walkforward import components, TESTS, discover_symbols, find_month_file, load_binance, resample

BEST = "R16:I+R8:I:O+R6:I:O"


def raw_sig(c: Candle, rule: str) -> int:
    v = components(c)[rule]
    th = TESTS[rule]
    return 1 if v >= th else -1 if v <= -th else 0


def variant_sig(c: Candle, rule: str, variant: str) -> int:
    s = raw_sig(c, rule)
    return s if variant == "N" else -s


def chain_detail(cs: list[Candle], i: int, c: dict) -> tuple[bool, int, list[dict]]:
    trigger = variant_sig(cs[i], c["trigger"], c["variant"])
    steps = [{
        "step": 0, "rule": c["trigger"], "variant": c["variant"], "mode": "TRIGGER",
        "raw_signal": raw_sig(cs[i], c["trigger"]), "signal": trigger,
        "expected": trigger, "ok": bool(trigger), "timestamp": cs[i].timestamp,
    }]
    if not trigger:
        return False, 0, steps
    for k, x in enumerate(c["selected"], 1):
        j = i + k
        if j >= len(cs):
            return False, trigger, steps
        sraw = raw_sig(cs[j], x["rule"])
        s = sraw if x["variant"] == "N" else -sraw
        expected = trigger if x["mode"] == "SAME" else -trigger
        steps.append({
            "step": k, "rule": x["rule"], "variant": x["variant"], "mode": x["mode"],
            "raw_signal": sraw, "signal": s, "expected": expected,
            "ok": bool(s and s == expected), "timestamp": cs[j].timestamp,
        })
        if not s or s != expected:
            return False, trigger, steps
    return True, trigger, steps


def signal_times_15m_detailed(c15: list[Candle], c: dict) -> tuple[list[tuple[int, int, int]], list[dict]]:
    signals = []
    rows = []
    depth = len(c["selected"])
    for i in range(len(c15) - depth):
        ok, direction, steps = chain_detail(c15, i, c)
        signal_end = c15[i + depth].timestamp + 15 * 60
        signal_id = len(signals) + 1 if ok and direction else ""
        if ok and direction:
            signals.append((signal_end, direction, signal_id))
        for s in steps:
            rows.append({
                "signal_id": signal_id,
                "signal_confirmed": int(ok and direction),
                "direction": direction,
                "signal_end_ts": signal_end,
                **s,
            })
    return signals, rows


def run(series: dict[str, list[Candle]], fee_side_pct: float, trace_dir: Path) -> dict:
    initial = 1000.0
    per_symbol = initial / len(series)
    total = 0.0
    all_trade_rows = []
    all_60_rows = []
    all_15_rows = []
    trade_count = win_count = 0
    fees_total = 0.0
    global_trade_id = 0

    cfg = {
        "trigger": "R16",
        "variant": "I",
        "selected": [
            {"rule": "R8", "variant": "I", "mode": "OPPOSITE"},
            {"rule": "R6", "variant": "I", "mode": "OPPOSITE"},
        ],
    }

    for symbol, raw in series.items():
        local = per_symbol
        peak = local
        c15 = resample(raw, 15)
        c60 = resample(raw, 60)
        signals, signal_rows = signal_times_15m_detailed(c15, cfg)
        for r in signal_rows:
            r["symbol"] = symbol
        all_15_rows.extend(signal_rows)

        signal_ptr = 0
        j = 0
        while j < len(c60):
            candle = c60[j]
            ts = candle.timestamp

            # Non-overlap control: any signal that became confirmed before this
            # currently eligible 60m candle is stale. It would have been created
            # while a previous one-candle trade was active, so it is discarded.
            while signal_ptr < len(signals) and signals[signal_ptr][0] < ts:
                signal_ptr += 1

            signal_id = ""
            direction = 0
            signal_end = ""
            if signal_ptr < len(signals):
                se, sd, sid = signals[signal_ptr]
                if ts >= se:
                    signal_id, direction, signal_end = sid, sd, se
                    signal_ptr += 1

            action = "HOLD"
            entry = exit_price = move = gross = fee = 0.0
            before = local
            after = local
            win = ""
            if signal_id and j + 1 < len(c60):
                global_trade_id += 1
                entry = candle.close
                exit_price = c60[j + 1].close
                move = (exit_price / entry - 1.0) * direction if entry > 0 else 0.0
                gross = local * move
                fee = local * 2.0 * fee_side_pct / 100.0
                after = max(0.0, local + gross - fee)
                local = after
                fees_total += fee
                trade_count += 1
                is_win = gross > 0
                win_count += int(is_win)
                win = int(is_win)
                peak = max(peak, local)
                dd = 100.0 * (peak - local) / peak if peak else 0.0
                action = "BUY" if direction > 0 else "SELL"
                all_trade_rows.append({
                    "trade_id": global_trade_id, "symbol": symbol, "signal_id": signal_id,
                    "signal_end_ts": signal_end, "entry_ts": ts,
                    "exit_ts": c60[j + 1].timestamp, "direction": direction, "action": action,
                    "entry_price": entry, "exit_price": exit_price, "move_pct": move * 100.0,
                    "capital_before": before, "gross_pnl": gross, "fee": fee,
                    "capital_after": after, "win": win, "drawdown_pct": dd,
                })
                j += 2
            else:
                peak = max(peak, local)
                dd = 100.0 * (peak - local) / peak if peak else 0.0
                j += 1

            all_60_rows.append({
                "symbol": symbol, "timestamp": ts, "open": candle.open,
                "high": candle.high, "low": candle.low, "close": candle.close,
                "signal_id": signal_id, "signal_end_ts": signal_end,
                "direction": direction, "action": action, "entry_price": entry,
                "exit_price": exit_price, "move_pct": move * 100.0,
                "capital_before": before, "gross_pnl": gross, "fee": fee,
                "capital_after": after, "win": win, "drawdown_pct": dd,
            })

        total += local

    ret = 100.0 * (total / initial - 1.0)
    win_rate = 100.0 * win_count / trade_count if trade_count else 0.0
    max_dd = max((float(r["drawdown_pct"]) for r in all_trade_rows), default=0.0)
    trace_dir.mkdir(parents=True, exist_ok=True)
    write_csv(trace_dir / "best60m_nonoverlap_15m_decisions.csv", all_15_rows)
    write_csv(trace_dir / "best60m_nonoverlap_60m_frames.csv", all_60_rows)
    write_csv(trace_dir / "best60m_nonoverlap_trades.csv", all_trade_rows)
    return {"final": total, "ret": ret, "trades": trade_count, "win": win_rate, "dd": max_dd, "fees": fees_total}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_series(data_root: Path, month: str, symbols_arg: str) -> dict[str, list[Candle]]:
    symbols = sorted(discover_symbols(data_root, month)) if symbols_arg.upper() == "ALL" else [x.strip().upper() for x in symbols_arg.split(",") if x.strip()]
    out = {}
    for s in symbols:
        p = find_month_file(data_root, s, month)
        if p:
            cs = load_binance(p)
            if len(cs) >= 64:
                out[s] = cs
    if not out:
        raise RuntimeError("No test data found")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--test-month", default="2026-07")
    ap.add_argument("--symbols", default="ALL")
    ap.add_argument("--fee-side-pct", type=float, default=0.0)
    args = ap.parse_args()

    series = load_series(Path(args.input_dir), args.test_month, args.symbols)
    print(f"BEST_60M_NONOVERLAP candidate={BEST} architecture=B assets={len(series)} test={args.test_month} fee={args.fee_side_pct:.3f}%/side")
    print("15m: R16 inverted -> R8 inverted OPPOSITE -> R6 inverted OPPOSITE")
    print("60m: one trade per symbol at a time; signals confirmed before the next eligible candle are discarded")
    print("---")
    r = run(series, args.fee_side_pct, ROOT / "reports" / "july_r_best_60m_detail")
    print(f"RESULT FINAL={r['final']:.2f} RET={r['ret']:+.2f}% T={r['trades']} WIN={r['win']:.2f}% DD={r['dd']:.2f}% FEES={r['fees']:.2f}")
    print("SAVED reports/july_r_best_60m_detail/best60m_nonoverlap_15m_decisions.csv")
    print("SAVED reports/july_r_best_60m_detail/best60m_nonoverlap_60m_frames.csv")
    print("SAVED reports/july_r_best_60m_detail/best60m_nonoverlap_trades.csv")


if __name__ == "__main__":
    main()
