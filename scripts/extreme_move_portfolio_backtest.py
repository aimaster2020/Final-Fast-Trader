from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_pattern_trader.models import Candle
from scripts.prepared_market_loader import load_prepared


@dataclass
class Position:
    symbol: str
    signal: str
    entry_ts: int
    exit_ts: int
    entry: float
    exit: float
    notional: float
    fee: float

    @property
    def side(self) -> int:
        return 1 if self.signal == "LOW" else -1

    @property
    def gross_ret(self) -> float:
        return ((self.exit / self.entry) - 1.0) * self.side

    @property
    def pnl(self) -> float:
        return self.notional * self.gross_ret - self.fee


def movement_budget(rows: list[Candle], i: int, bars: int) -> float:
    if bars <= 0 or i < bars:
        return float("inf")
    return sum(rows[j].close - rows[j - 1].close if False else abs(rows[j].close - rows[j - 1].close) for j in range(i - bars, i))


def extreme_signal(
    rows: list[Candle], i: int, window: int, move_bars: int, multiplier: float
) -> str:
    if i < max(window, move_bars):
        return ""
    previous = rows[i - window:i]
    prev_low = min(c.low for c in previous)
    prev_high = max(c.high for c in previous)
    budget = movement_budget(rows, i, move_bars) * multiplier
    low_break = rows[i].low < prev_low
    high_break = rows[i].high > prev_high
    low = low_break and (prev_low - rows[i].low) >= budget
    high = high_break and (rows[i].high - prev_high) >= budget
    if low and high:
        return ""
    if low:
        return "LOW"
    if high:
        return "HIGH"
    return ""


def build_candidates(
    symbol: str,
    rows: list[Candle],
    window: int,
    move_bars: int,
    multiplier: float,
    horizon: int,
    side_filter: str,
) -> list[dict]:
    out: list[dict] = []
    start = max(window, move_bars)
    for i in range(start, len(rows) - horizon - 1):
        sig = extreme_signal(rows, i, window, move_bars, multiplier)
        if sig not in ("LOW", "HIGH"):
            continue
        if side_filter != "ALL" and sig != side_filter:
            continue
        entry_i = i + 1
        exit_i = i + 1 + horizon
        if exit_i >= len(rows):
            continue
        entry = rows[entry_i].open
        exit_price = rows[exit_i].close
        if entry <= 0:
            continue
        out.append(
            {
                "symbol": symbol,
                "signal": sig,
                "entry_ts": rows[entry_i].timestamp,
                "exit_ts": rows[exit_i].timestamp,
                "entry": entry,
                "exit": exit_price,
            }
        )
    return out


def run_portfolio(
    candidates: list[dict],
    initial_capital: float,
    allocation: float,
    fee_rt_pct: float,
    max_positions: int,
) -> tuple[dict, list[dict]]:
    candidates.sort(key=lambda x: (x["entry_ts"], x["symbol"]))
    capital = initial_capital
    peak = initial_capital
    max_dd = 0.0
    active: dict[str, Position] = {}
    trades: list[dict] = []

    def close_position(pos: Position) -> None:
        nonlocal capital
        capital += pos.pnl
        trades.append(
            {
                "symbol": pos.symbol,
                "signal": pos.signal,
                "entry_ts": pos.entry_ts,
                "exit_ts": pos.exit_ts,
                "entry": pos.entry,
                "exit": pos.exit,
                "notional": pos.notional,
                "gross_ret_pct": pos.gross_ret * 100.0,
                "fee": pos.fee,
                "pnl": pos.pnl,
            }
        )

    for cand in candidates:
        for symbol, pos in list(active.items()):
            if pos.exit_ts <= cand["entry_ts"]:
                close_position(pos)
                del active[symbol]
                peak = max(peak, capital)
                max_dd = max(max_dd, (peak - capital) / peak * 100.0 if peak > 0 else 0.0)

        symbol = cand["symbol"]
        if symbol in active:
            continue
        if len(active) >= max_positions:
            continue

        notional = capital * allocation
        fee = notional * fee_rt_pct / 100.0
        active[symbol] = Position(
            symbol=symbol,
            signal=cand["signal"],
            entry_ts=cand["entry_ts"],
            exit_ts=cand["exit_ts"],
            entry=cand["entry"],
            exit=cand["exit"],
            notional=notional,
            fee=fee,
        )

    for symbol, pos in sorted(active.items(), key=lambda kv: kv[1].exit_ts):
        close_position(pos)
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak * 100.0 if peak > 0 else 0.0)

    wins = [r for r in trades if r["pnl"] > 0]
    losses = [r for r in trades if r["pnl"] < 0]
    gross_profit = sum(r["pnl"] for r in wins)
    gross_loss = -sum(r["pnl"] for r in losses)
    pf = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0)
    return (
        {
            "trades": len(trades),
            "wins": len(wins),
            "win_rate": len(wins) / len(trades) * 100.0 if trades else 0.0,
            "pnl": capital - initial_capital,
            "return_pct": (capital / initial_capital - 1.0) * 100.0 if initial_capital else 0.0,
            "final": capital,
            "pf": pf,
            "dd": max_dd,
            "fees": sum(r["fee"] for r in trades),
        },
        trades,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-account portfolio backtest for the extreme move strategy.")
    ap.add_argument("--prepared-file", default="reports/prepared_price_action_1h.csv")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--months", default="2026-05,2026-06,2026-07,2026-08")
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--move-bars", type=int, default=5)
    ap.add_argument("--multiplier", type=float, default=2.0)
    ap.add_argument("--horizons", default="1,2,3,4,6,12")
    ap.add_argument("--side", choices=("ALL", "LOW", "HIGH"), default="LOW")
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--allocation", type=float, default=0.30)
    ap.add_argument("--max-positions", type=int, default=3)
    ap.add_argument("--fee-roundtrip-pct", type=float, default=0.26)
    ap.add_argument("--output", default="reports/extreme_move_portfolio_trades.csv")
    args = ap.parse_args()

    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    months = [x.strip() for x in args.months.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    prepared = load_prepared(Path(args.prepared_file))

    print(
        f"EXTREME_MOVE_PORTFOLIO source=prepared W={args.window} M={args.move_bars} "
        f"X={args.multiplier:.2f} SIDE={args.side} alloc={args.allocation:.0%} "
        f"maxpos={args.max_positions} fee={args.fee_roundtrip_pct:.2f}%"
    )
    print("SINGLE $1000 ACCOUNT | entry=next open | exit=close after H | capital=P&L")
    print("H T WIN PNL RET FINAL PF DD FEES")
    print("---------------------------------------------")

    all_rows: list[dict] = []
    for horizon in horizons:
        candidates: list[dict] = []
        for symbol in symbols:
            rows: list[Candle] = []
            for month in months:
                rows.extend(prepared.get((symbol, month), []))
            rows.sort(key=lambda c: c.timestamp)
            candidates.extend(build_candidates(symbol, rows, args.window, args.move_bars, args.multiplier, horizon, args.side))

        summary, trades = run_portfolio(
            candidates,
            args.initial_capital,
            args.allocation,
            args.fee_roundtrip_pct,
            args.max_positions,
        )
        print(
            f"H={horizon:2d} T={summary['trades']:3d} WIN={summary['win_rate']:5.1f}% "
            f"PNL={summary['pnl']:+8.2f} RET={summary['return_pct']:+6.2f}% "
            f"FINAL={summary['final']:8.2f} PF={summary['pf']:.2f} "
            f"DD={summary['dd']:.2f}% FEES={summary['fees']:.2f}"
        )
        for t in trades:
            row = dict(t)
            row["horizon"] = horizon
            all_rows.append(row)

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "horizon", "symbol", "signal", "entry_ts", "exit_ts", "entry", "exit",
        "notional", "gross_ret_pct", "fee", "pnl",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"SAVED {out.relative_to(ROOT)} rows={len(all_rows)}")


if __name__ == "__main__":
    main()
