from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median

from test_dynamic_body_move_toward_zero import (
    DEFAULT_CAPITAL,
    DEFAULT_COMMISSION,
    DEFAULT_LOOKBACK,
    DEFAULT_MIN_HISTORY,
    EPS,
    Candle,
    Signal,
    Trade,
    ad_score,
    load,
    move_to_zero_ratio,
    predict_signal,
    signed_body,
    trade_once,
)


def write_trades(path: Path, trades: list[Trade]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "signal_index", "entry_timestamp", "exit_timestamp", "side", "ad",
                "current_body", "predicted_move_ratio", "predicted_next_body",
                "entry_price", "exit_price", "gross_return", "fee_return",
                "net_return", "capital_before", "capital_after",
            ]
        )
        for t in trades:
            w.writerow([
                t.signal_index, t.entry_timestamp, t.exit_timestamp, t.side, t.ad,
                f"{t.current_body:.12g}", f"{t.predicted_move_ratio:.12g}",
                f"{t.predicted_next_body:.12g}", f"{t.entry_price:.12g}",
                f"{t.exit_price:.12g}", f"{t.gross_return:.12g}",
                f"{t.fee_return:.12g}", f"{t.net_return:.12g}",
                f"{t.capital_before:.12g}", f"{t.capital_after:.12g}",
            ])


def run(
    rows: list[Candle],
    symbol: str,
    threshold: float,
    commission: float,
    lookback: int,
    min_history: int,
    capital0: float,
    wanted_ad: int | None,
) -> tuple[dict[str, float | int], list[Trade]]:
    capital = capital0
    trades: list[Trade] = []
    candidates = 0
    signals = 0
    skips_history = 0
    pred_sign_ok = 0
    actual_ratios: list[float] = []

    for i in range(1, len(rows) - 1):
        if ad_score(rows[i]) not in (3, 0):
            continue
        if wanted_ad is not None and ad_score(rows[i]) != wanted_ad:
            continue
        candidates += 1

        signal, hist_n = predict_signal(rows, i, lookback, min_history)
        if signal is None:
            if hist_n < min_history:
                skips_history += 1
            continue
        signal.symbol = symbol
        signals += 1

        actual_body = signed_body(rows[i + 1])
        actual_ratio = move_to_zero_ratio(signal.body, actual_body)
        if actual_ratio is not None:
            actual_ratios.append(actual_ratio)
            pred_sign_ok += int((signal.next_body_pred > 0) == (actual_body > 0))

        if signal.move_ratio_pred < threshold or signal.side == "FLAT":
            continue
        t = trade_once(rows, signal, capital, commission)
        if t is None:
            continue
        trades.append(t)
        capital = t.capital_after

    gross_capital = capital0
    net_capital = capital0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    peak = capital0
    max_dd = 0.0
    for t in trades:
        gross_capital *= 1.0 + t.gross_return
        net_capital *= 1.0 + t.net_return
        wins += int(t.net_return > 0)
        if t.gross_return >= 0:
            gross_profit += t.gross_return
        else:
            gross_loss += t.gross_return
        peak = max(peak, net_capital)
        max_dd = max(max_dd, (peak - net_capital) / peak if peak > EPS else 0.0)

    net_profit = sum(t.net_return for t in trades if t.net_return >= 0)
    net_loss = sum(t.net_return for t in trades if t.net_return < 0)
    pf = net_profit / abs(net_loss) if net_loss < -EPS else math.inf
    return {
        "candidates": candidates,
        "signals": signals,
        "trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades) * 100 if trades else 0.0,
        "gross_return": gross_capital / capital0 - 1,
        "net_return": net_capital / capital0 - 1,
        "final": net_capital,
        "pf": pf,
        "dd": max_dd,
        "pred_sign": pred_sign_ok / signals * 100 if signals else 0.0,
        "actual_move_ratio_mean": sum(actual_ratios) / len(actual_ratios) if actual_ratios else 0.0,
        "actual_move_ratio_med": median(actual_ratios) if actual_ratios else 0.0,
        "skip_history": skips_history,
    }, trades


def main() -> None:
    ap = argparse.ArgumentParser(description="AD3/AD0 dynamic move-to-zero strategy test")
    ap.add_argument("--input-dir", default="reports/1h")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    ap.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    ap.add_argument("--thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70")
    ap.add_argument("--trade-log-dir", default="reports/move_to_zero_trades_ad30")
    args = ap.parse_args()

    root = Path(args.input_dir)
    log_root = Path(args.trade_log_dir)
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    print(
        f"MODEL=rolling median move-to-zero ratio | LB={args.lookback} | MINH={args.min_history} | "
        f"CAPITAL={args.capital:.2f} | FEES=0 and {args.commission*100:.3f}%/side"
    )
    print("AD filter=3,0 | ENTRY=current close | EXIT=next close | position=100% | one-bar hold")
    print("move_ratio=sign(body)*(next_body-body)/abs(body); positive means toward zero")
    print()

    for symbol in symbols:
        rows = load(root / f"{symbol}_1h.csv")
        print(f"=== {symbol} rows={len(rows)} ===")
        if not rows:
            print("MISSING/EMPTY")
            continue
        for ad in (3, 0, None):
            label = "ALL_AD30" if ad is None else f"AD={ad}"
            print(f"[{label}]")
            for threshold in thresholds:
                parts = []
                for fee in (0.0, args.commission):
                    r, trades = run(
                        rows, symbol, threshold, fee, args.lookback, args.min_history,
                        args.capital, ad
                    )
                    fee_label = "F0" if fee == 0 else f"F{fee*100:.2f}%"
                    pf = "INF" if math.isinf(float(r["pf"])) else f"{float(r['pf']):.2f}"
                    parts.append(
                        f"{fee_label}:TR={int(r['trades'])} WR={float(r['win_rate']):.1f}% "
                        f"RET={float(r['net_return'])*100:.2f}% FIN={float(r['final']):.2f} "
                        f"PF={pf} DD={float(r['dd'])*100:.2f}%"
                    )
                    filename = f"{symbol}_{label.replace('=', '')}_thr{threshold:.2f}_{fee_label}.csv"
                    write_trades(log_root / filename, trades)
                print(
                    f"T={threshold:.2f} " + " | ".join(parts)
                )
            print()

    print("DEBUG NOTES")
    print("- Signal side is LONG if predicted next body > 0, SHORT if < 0.")
    print("- Detailed trade CSVs contain prediction, actual entry/exit, gross/net return and capital path.")
    print("- F0 is zero commission; F0.13% is 0.13% entry + 0.13% exit = 0.26% round trip.")


if __name__ == "__main__":
    main()
