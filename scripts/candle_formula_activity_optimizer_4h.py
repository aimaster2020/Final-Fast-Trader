from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from fast_pattern_trader.candle_formula_strategy import decide
from fast_pattern_trader.models import Candle, Signal

FAST_TF = ("5m", "15m", "1h")
ALL_TF = ("5m", "15m", "1h", "4h")
DEFAULT_VALUES = tuple(float(x) for x in range(-200, 201, 25))
DEFAULT_MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")


def rows_for(path: Path, symbol: str) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r.get("symbol") == symbol]


def candle(row: dict[str, str]) -> Candle | None:
    try:
        return Candle(int(float(row["timestamp"])), float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]))
    except (KeyError, TypeError, ValueError):
        return None


def load_month(path: Path, month: str, symbol: str) -> list[Candle]:
    out = []
    for r in rows_for(path, symbol):
        if r.get("month") != month:
            continue
        c = candle(r)
        if c is not None:
            out.append(c)
    return sorted(out, key=lambda x: x.timestamp)


def month_from_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")


def build_4h(path: Path, symbol: str, month: str) -> list[Candle]:
    hourly = []
    for r in rows_for(path, symbol):
        c = candle(r)
        if c is not None:
            hourly.append(c)
    hourly.sort(key=lambda x: x.timestamp)
    buckets: dict[int, list[Candle]] = {}
    for c in hourly:
        b = c.timestamp - c.timestamp % 14_400
        buckets.setdefault(b, []).append(c)
    out = []
    for b in sorted(buckets):
        g = sorted(buckets[b], key=lambda x: x.timestamp)
        expected = [b + 3_600 * i for i in range(4)]
        if len(g) != 4 or [c.timestamp for c in g] != expected or month_from_ts(b) != month:
            continue
        out.append(Candle(b, g[0].open, max(c.high for c in g), min(c.low for c in g), g[-1].close))
    return out


def side(c: Candle) -> int:
    s = decide(c).signal
    return 1 if s == Signal.BUY else -1 if s == Signal.SELL else 0


def pnl(side_: int, entry: float, exit_: float) -> float:
    return side_ * (exit_ - entry) / entry if entry > 0 else 0.0


def run(candles: list[Candle], lower: float, upper: float, commission: float, initial: float) -> dict[str, float | int]:
    equity = initial
    pos: tuple[int, float, float] | None = None
    trades = wins = 0
    gross_profit = gross_loss = 0.0
    for c in candles:
        sig = side(c)
        body = c.close - c.open
        if pos is None:
            if sig and not (lower <= body <= upper):
                pos = (sig, c.close, equity)
            continue
        ps, entry, capital = pos
        opposite = bool(sig) and sig != ps
        exit_now = (ps > 0 and opposite and body <= upper) or (ps < 0 and opposite and body >= lower)
        if exit_now:
            gross = capital * pnl(ps, entry, c.close)
            net = gross - capital * commission * 2.0
            equity += net
            trades += 1
            if net > 0:
                wins += 1
                gross_profit += max(gross, 0.0)
            elif net < 0:
                gross_loss += min(gross, 0.0)
            pos = None
    if pos is not None and candles:
        ps, entry, capital = pos
        gross = capital * pnl(ps, entry, candles[-1].close)
        net = gross - capital * commission * 2.0
        equity += net
        trades += 1
        if net > 0:
            wins += 1
            gross_profit += max(gross, 0.0)
        elif net < 0:
            gross_loss += min(gross, 0.0)
    return {
        "return_pct": (equity / initial - 1.0) * 100.0,
        "trades": trades,
        "win_rate": wins / trades * 100.0 if trades else 0.0,
        "profit_factor": gross_profit / abs(gross_loss) if gross_loss < 0 else (float("inf") if gross_profit else 0.0),
    }


def compounded(xs: list[float]) -> float:
    f = 1.0
    for x in xs:
        f *= 1.0 + x / 100.0
    return (f - 1.0) * 100.0


def parse_values(raw: str) -> tuple[float, ...]:
    v = tuple(sorted({float(x.strip()) for x in raw.split(",") if x.strip()}))
    if len(v) < 2:
        raise SystemExit("Need at least two threshold values")
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description="Activity-aware common fast bias + independent 4h bias optimizer after commission.")
    ap.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--commission", type=float, default=0.0013)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--values", default=",".join(str(v) for v in DEFAULT_VALUES))
    ap.add_argument("--input-5m", default="reports/prepared_price_action_5m.csv")
    ap.add_argument("--input-15m", default="reports/prepared_price_action_15m.csv")
    ap.add_argument("--input-1h", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--min-5m-td", type=float, default=1.0)
    ap.add_argument("--min-15m-td", type=float, default=0.2)
    ap.add_argument("--min-1h-td", type=float, default=0.05)
    ap.add_argument("--min-4h-td", type=float, default=0.01)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    months = tuple(x.strip() for x in args.months.split(",") if x.strip())
    vals = parse_values(args.values)
    paths = {"5m": Path(args.input_5m), "15m": Path(args.input_15m), "1h": Path(args.input_1h)}
    data: dict[tuple[str, str], list[Candle]] = {}
    for tf in FAST_TF:
        for m in months:
            data[(tf, m)] = load_month(paths[tf], m, args.symbol)
            if not data[(tf, m)]:
                raise SystemExit(f"No usable {tf} candles for {args.symbol} {m}")
        
    for m in months:
        data[("4h", m)] = build_4h(paths["1h"], args.symbol, m)
        if not data[("4h", m)]:
            raise SystemExit(f"No usable 4h candles built from 1h for {args.symbol} {m}")

    fast_pairs = [(lo, hi) for lo in vals for hi in vals if lo < hi]
    cache: dict[tuple[str, str, float, float], dict[str, float | int]] = {}
    for tf in FAST_TF:
        for m in months:
            for lo, hi in fast_pairs:
                cache[(tf, m, lo, hi)] = run(data[(tf, m)], lo, hi, args.commission, args.initial_capital)

    candidates = []
    for flo, fhi in fast_pairs:
        fast = {}
        for tf in FAST_TF:
            ms = [cache[(tf, m, flo, fhi)] for m in months]
            rets = [float(x["return_pct"]) for x in ms]
            trades = sum(int(x["trades"]) for x in ms)
            days = len(months) and sum(len(data[(tf, m)]) for m in months)
            fast[tf] = {"combined": compounded(rets), "trades": trades, "td": trades / days if days else 0.0, "months": rets}
        if not (fast["5m"]["td"] >= args.min_5m_td and fast["15m"]["td"] >= args.min_15m_td and fast["1h"]["td"] >= args.min_1h_td):
            continue
        for hlo, hhi in fast_pairs:
            ms = [run(data[("4h", m)], hlo, hhi, args.commission, args.initial_capital) for m in months]
            rets = [float(x["return_pct"]) for x in ms]
            trades = sum(int(x["trades"]) for x in ms)
            days = sum(len(data[("4h", m)]) for m in months)
            h4 = {"combined": compounded(rets), "trades": trades, "td": trades / days if days else 0.0, "months": rets}
            if h4["td"] < args.min_4h_td:
                continue
            combined = [float(fast[tf]["combined"]) for tf in FAST_TF] + [float(h4["combined"])]
            positive_tf = sum(x > 0 for x in combined)
            positive_months = sum(x > 0 for x in list(fast["5m"]["months"]) + list(fast["15m"]["months"]) + list(fast["1h"]["months"]) + list(h4["months"]))
            avg_tf = sum(combined) / 4.0
            worst_tf = min(combined)
            candidates.append((positive_tf, worst_tf, avg_tf, positive_months, fast["5m"]["trades"] + fast["15m"]["trades"] + fast["1h"]["trades"] + h4["trades"], flo, fhi, hlo, hhi, fast, h4))

    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]), reverse=True)
    print(f"ACTIVITY_OPT_4H | symbol={args.symbol} | months={','.join(months)} | commission={args.commission*100:.2f}% per side | roundtrip={args.commission*200:.2f}%")
    print(f"MIN_T/D | 5m={args.min_5m_td:.2f} 15m={args.min_15m_td:.2f} 1h={args.min_1h_td:.2f} 4h={args.min_4h_td:.2f}")
    print("FAST_BIAS | 4H_BIAS | AVG_TF | WORST_TF | POS_TF | POS_MONTHS | TRADES | 5m_T/D | 15m_T/D | 1h_T/D | 4h_T/D | 5m | 15m | 1h | 4h")
    for row in candidates[: args.top]:
        _, _, avg_tf, pos_m, total, flo, fhi, hlo, hhi, fast, h4 = row
        print(f"[{flo:.0f},{fhi:.0f}] | [{hlo:.0f},{hhi:.0f}] | {avg_tf:+6.2f}% | {min(float(fast[t]['combined']) for t in FAST_TF + ('4h',) if t in fast or t=='4h') if False else row[1]:+7.2f}% | {row[0]}/4 | {pos_m}/{len(months)*4} | {total:6} | {float(fast['5m']['td']):6.2f} | {float(fast['15m']['td']):7.2f} | {float(fast['1h']['td']):6.2f} | {float(h4['td']):6.2f} | {float(fast['5m']['combined']):+6.2f}% | {float(fast['15m']['combined']):+7.2f}% | {float(fast['1h']['combined']):+6.2f}% | {float(h4['combined']):+6.2f}%")
    if candidates:
        best = candidates[0]
        _, worst, avg, pos_m, total, flo, fhi, hlo, hhi, fast, h4 = best
        print(f"ACTIVITY_BEST | fast=[{flo:.0f},{fhi:.0f}] 4h=[{hlo:.0f},{hhi:.0f}] avg_tf={avg:+.2f}% worst_tf={worst:+.2f}% positive_tf={best[0]}/4 positive_months={pos_m}/{len(months)*4} trades={total}")
    else:
        print("ACTIVITY_BEST | none (no candidate meets all minimum trades/day constraints)")


if __name__ == "__main__":
    main()
