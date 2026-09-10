from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median

DEFAULT_LOOKBACK = 50
DEFAULT_MIN_HISTORY = 15
DEFAULT_CAPITAL = 1000.0
DEFAULT_COMMISSION = 0.0013
EPS = 1e-12


@dataclass
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float


@dataclass
class Signal:
    index: int
    timestamp: str
    symbol: str
    ad: int
    body: float
    move_ratio_pred: float
    next_body_pred: float
    side: str


@dataclass
class Trade:
    signal_index: int
    entry_timestamp: str
    exit_timestamp: str
    side: str
    ad: int
    current_body: float
    predicted_move_ratio: float
    predicted_next_body: float
    entry_price: float
    exit_price: float
    gross_return: float
    fee_return: float
    net_return: float
    capital_before: float
    capital_after: float


def load(path: Path) -> list[Candle]:
    rows: list[Candle] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(
                    Candle(
                        timestamp=str(r.get("timestamp", "")),
                        open=float(r["open"]),
                        high=float(r["high"]),
                        low=float(r["low"]),
                        close=float(r["close"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def ad_score(c: Candle) -> int:
    body = c.close - c.open
    hc = c.high - c.close
    ho = c.high - c.open
    lc = c.low - c.close
    return int(hc > body) + int(ho < hc) + int(lc > body)


def signed_body(c: Candle) -> float:
    return c.close - c.open


def move_to_zero_ratio(current_body: float, next_body: float) -> float | None:
    if abs(current_body) <= EPS:
        return None
    # Positive means the next signed body moved toward zero.
    return (math.copysign(1.0, current_body) * (next_body - current_body)) / abs(current_body)


def rolling_move_ratio(
    rows: list[Candle],
    i: int,
    score: int,
    lookback: int,
    min_history: int,
) -> tuple[float | None, int]:
    start = max(1, i - lookback)
    history: list[float] = []
    for j in range(start, i):
        if ad_score(rows[j]) != score:
            continue
        cur = signed_body(rows[j])
        nxt = signed_body(rows[j + 1])
        ratio = move_to_zero_ratio(cur, nxt)
        if ratio is not None and math.isfinite(ratio):
            history.append(ratio)
    if len(history) < min_history:
        return None, len(history)
    # Fully causal, local estimate: rolling median of the same AD regime.
    return median(history), len(history)


def predict_signal(
    rows: list[Candle],
    i: int,
    lookback: int,
    min_history: int,
) -> tuple[Signal | None, int]:
    c = rows[i]
    body = signed_body(c)
    if abs(body) <= EPS:
        return None, 0

    score = ad_score(c)
    pred_ratio, history_n = rolling_move_ratio(rows, i, score, lookback, min_history)
    if pred_ratio is None:
        return None, history_n

    predicted_next_body = body + math.copysign(abs(body) * pred_ratio, body)
    if abs(predicted_next_body) <= EPS:
        side = "FLAT"
    elif predicted_next_body > 0:
        side = "LONG"
    else:
        side = "SHORT"

    return (
        Signal(
            index=i,
            timestamp=c.timestamp,
            symbol="",
            ad=score,
            body=body,
            move_ratio_pred=pred_ratio,
            next_body_pred=predicted_next_body,
            side=side,
        ),
        history_n,
    )


def trade_once(
    rows: list[Candle],
    signal: Signal,
    initial_capital: float,
    commission: float,
) -> Trade | None:
    i = signal.index
    if i + 1 >= len(rows):
        return None

    # Enter at current close and exit at next close.
    # With continuous OHLC data, next-candle body approximates this close-to-close move.
    entry = rows[i].close
    exit_price = rows[i + 1].close
    if entry <= EPS:
        return None

    direction = 1.0 if signal.side == "LONG" else -1.0 if signal.side == "SHORT" else 0.0
    if direction == 0.0:
        return None

    gross_return = direction * (exit_price / entry - 1.0)
    fee_return = 2.0 * commission
    net_return = gross_return - fee_return
    capital_after = initial_capital * (1.0 + net_return)

    return Trade(
        signal_index=i,
        entry_timestamp=rows[i].timestamp,
        exit_timestamp=rows[i + 1].timestamp,
        side=signal.side,
        ad=signal.ad,
        current_body=signal.body,
        predicted_move_ratio=signal.move_ratio_pred,
        predicted_next_body=signal.next_body_pred,
        entry_price=entry,
        exit_price=exit_price,
        gross_return=gross_return,
        fee_return=fee_return,
        net_return=net_return,
        capital_before=initial_capital,
        capital_after=capital_after,
    )


def evaluate(
    rows: list[Candle],
    symbol: str,
    threshold: float,
    lookback: int,
    min_history: int,
    initial_capital: float,
    commission: float,
) -> tuple[dict[str, float | int | str], list[Trade]]:
    trades: list[Trade] = []
    capital = initial_capital
    skipped_no_history = 0
    skipped_flat = 0
    signals = 0
    correct_next_body_sign = 0
    total_move_ratio = []

    for i in range(1, len(rows) - 1):
        signal, history_n = predict_signal(rows, i, lookback, min_history)
        if signal is None:
            if history_n < min_history:
                skipped_no_history += 1
            else:
                skipped_flat += 1
            continue

        signal.symbol = symbol
        actual_next_body = signed_body(rows[i + 1])
        actual_ratio = move_to_zero_ratio(signal.body, actual_next_body)
        if actual_ratio is not None:
            total_move_ratio.append(actual_ratio)
            correct_next_body_sign += int((signal.next_body_pred > 0) == (actual_next_body > 0))
        signals += 1

        if signal.move_ratio_pred < threshold:
            continue
        if signal.side == "FLAT":
            skipped_flat += 1
            continue

        trade = trade_once(rows, signal, capital, commission)
        if trade is None:
            continue
        trades.append(trade)
        capital = trade.capital_after

    gross_capital = initial_capital
    net_capital = initial_capital
    gross_wins = 0
    net_wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    net_profit = 0.0
    net_loss = 0.0
    max_capital = initial_capital
    max_dd = 0.0

    for t in trades:
        gross_capital *= 1.0 + t.gross_return
        net_capital *= 1.0 + t.net_return
        gross_wins += int(t.gross_return > 0)
        net_wins += int(t.net_return > 0)
        if t.gross_return >= 0:
            gross_profit += t.gross_return
        else:
            gross_loss += t.gross_return
        if t.net_return >= 0:
            net_profit += t.net_return
        else:
            net_loss += t.net_return
        max_capital = max(max_capital, net_capital)
        dd = (max_capital - net_capital) / max_capital if max_capital > EPS else 0.0
        max_dd = max(max_dd, dd)

    gross_return_total = gross_capital / initial_capital - 1.0
    net_return_total = net_capital / initial_capital - 1.0
    avg_move_ratio = sum(total_move_ratio) / len(total_move_ratio) if total_move_ratio else 0.0
    sign_acc = correct_next_body_sign / signals * 100.0 if signals else 0.0
    gross_pf = (gross_profit / abs(gross_loss)) if gross_loss < -EPS else math.inf
    net_pf = (net_profit / abs(net_loss)) if net_loss < -EPS else math.inf

    report: dict[str, float | int | str] = {
        "symbol": symbol,
        "threshold": threshold,
        "lookback": lookback,
        "min_history": min_history,
        "commission": commission,
        "signals": signals,
        "trades": len(trades),
        "gross_wins": gross_wins,
        "net_wins": net_wins,
        "gross_win_rate": gross_wins / len(trades) * 100.0 if trades else 0.0,
        "net_win_rate": net_wins / len(trades) * 100.0 if trades else 0.0,
        "body_sign_accuracy": sign_acc,
        "avg_actual_move_to_zero_ratio": avg_move_ratio,
        "gross_return": gross_return_total,
        "net_return": net_return_total,
        "gross_pf": gross_pf,
        "net_pf": net_pf,
        "max_drawdown": max_dd,
        "final_capital": net_capital,
        "skipped_no_history": skipped_no_history,
        "skipped_flat": skipped_flat,
    }
    return report, trades


def write_trades(path: Path, trades: list[Trade]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "signal_index",
                "entry_timestamp",
                "exit_timestamp",
                "side",
                "ad",
                "current_body",
                "predicted_move_ratio",
                "predicted_next_body",
                "entry_price",
                "exit_price",
                "gross_return",
                "fee_return",
                "net_return",
                "capital_before",
                "capital_after",
            ]
        )
        for t in trades:
            writer.writerow(
                [
                    t.signal_index,
                    t.entry_timestamp,
                    t.exit_timestamp,
                    t.side,
                    t.ad,
                    f"{t.current_body:.12g}",
                    f"{t.predicted_move_ratio:.12g}",
                    f"{t.predicted_next_body:.12g}",
                    f"{t.entry_price:.12g}",
                    f"{t.exit_price:.12g}",
                    f"{t.gross_return:.12g}",
                    f"{t.fee_return:.12g}",
                    f"{t.net_return:.12g}",
                    f"{t.capital_before:.12g}",
                    f"{t.capital_after:.12g}",
                ]
            )


def fmt_pf(value: float) -> str:
    return "INF" if math.isinf(value) else f"{value:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Causal dynamic move-to-zero body strategy backtest."
    )
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    ap.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    ap.add_argument(
        "--thresholds",
        default="0.10,0.20,0.30,0.40,0.50,0.60,0.70",
        help="move-to-zero ratio thresholds",
    )
    ap.add_argument("--trade-log-dir", default="reports/move_to_zero_trades")
    args = ap.parse_args()

    lookback = max(5, args.lookback)
    min_history = max(3, args.min_history)
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    root = Path(args.input_dir)
    trade_root = Path(args.trade_log_dir)

    print(
        f"MODEL=causal rolling median(move_to_zero_ratio) | LOOKBACK={lookback} | "
        f"MIN_HISTORY={min_history} | CAPITAL={args.capital:.2f} | COMMISSION_EACH_SIDE={args.commission*100:.3f}%"
    )
    print("ENTRY=current close | EXIT=next close | position=100% capital | no overlap")
    print("move_to_zero_ratio = sign(current_body)*(next_body-current_body)/abs(current_body)")
    print()

    for symbol in symbols:
        rows = load(root / f"{symbol}_1h.csv")
        print(f"=== {symbol} rows={len(rows)} ===")
        if len(rows) < 20:
            print("SKIP=not enough rows")
            continue

        for commission in (0.0, args.commission):
            print(f"-- COMMISSION_EACH_SIDE={commission*100:.3f}% --")
            for threshold in thresholds:
                report, trades = evaluate(
                    rows,
                    symbol,
                    threshold,
                    lookback,
                    min_history,
                    args.capital,
                    commission,
                )
                log_name = (
                    f"{symbol}_thr{threshold:.2f}_fee{commission:.5f}.csv".replace(".", "_")
                    + ".csv"
                )
                write_trades(trade_root / log_name, trades)
                print(
                    f"T={threshold:.2f} SIG={int(report['signals'])} TR={int(report['trades'])} "
                    f"WR={report['net_win_rate']:.1f}% RET={report['net_return']*100:.2f}% "
                    f"FINAL={report['final_capital']:.2f} PF={fmt_pf(float(report['net_pf']))} "
                    f"DD={report['max_drawdown']*100:.2f}% BODY_SIGN={report['body_sign_accuracy']:.1f}% "
                    f"SKIP_H={int(report['skipped_no_history'])} LOG={trade_root / log_name}"
                )
        print()


if __name__ == "__main__":
    main()
