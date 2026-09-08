from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd


RULES: dict[str, tuple[str, Callable[[pd.Series], bool]]] = {
    "R1": ("BUY", lambda c: (c.high - c.close) > (c.close - c.open)),
    "R2": ("BUY", lambda c: (c.high - c.open) < (c.high - c.close)),
    "R3": ("BUY", lambda c: (c.low - c.close) > (c.close - c.open)),
    "R4": ("SELL", lambda c: (c.high - c.close) < (c.close - c.open)),
    "R5": ("SELL", lambda c: (c.high - c.open) > (c.high - c.close)),
    "R6": ("SELL", lambda c: (c.low - c.close) < (c.close - c.open)),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure the standalone directional value of R1..R6.")
    p.add_argument("--input", default=r".\reports\prepared_price_action_1h.csv")
    p.add_argument("--commission", type=float, default=0.0, help="Commission per side, e.g. 0.0013")
    return p.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).strip().lower(): c for c in df.columns}

    def pick(*names: str) -> str:
        for name in names:
            if name in cols:
                return cols[name]
        raise KeyError(f"Missing column; tried {names}; got {list(df.columns)}")

    rename = {
        pick("timestamp", "time", "datetime", "date"): "timestamp",
        pick("symbol", "asset", "ticker"): "symbol",
        pick("open", "o"): "open",
        pick("high", "h"): "high",
        pick("low", "l"): "low",
        pick("close", "c"): "close",
    }
    return df.rename(columns=rename).copy()


def analyze_symbol(group: pd.DataFrame, commission: float) -> dict[str, dict[str, float]]:
    g = group.sort_values("timestamp").reset_index(drop=True)
    stats: dict[str, dict[str, float]] = {
        name: {"samples": 0, "correct": 0, "raw_pct": 0.0, "net_pct": 0.0}
        for name in RULES
    }

    for i in range(len(g) - 1):
        cur = g.iloc[i]
        nxt = g.iloc[i + 1]
        next_body = float(nxt.close - nxt.open)
        if next_body == 0:
            continue
        next_dir = 1 if next_body > 0 else -1
        for name, (side, predicate) in RULES.items():
            if not predicate(cur):
                continue
            direction = 1 if side == "BUY" else -1
            s = stats[name]
            s["samples"] += 1
            s["correct"] += int(direction == next_dir)
            raw = ((float(nxt.close) - float(cur.close)) / float(cur.close)) * direction
            s["raw_pct"] += raw * 100.0
            s["net_pct"] += (raw - 2.0 * commission) * 100.0

    for s in stats.values():
        n = s["samples"]
        s["accuracy_pct"] = (s["correct"] / n * 100.0) if n else 0.0
        s["avg_net_pct"] = (s["net_pct"] / n) if n else 0.0
        s["edge_vs_50_pct"] = s["accuracy_pct"] - 50.0
    return stats


def main() -> int:
    args = parse_args()
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")

    df = normalize_columns(pd.read_csv(path))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol", "open", "high", "low", "close"])

    fee = float(args.commission)
    print(
        "RULE_SCORE_ANALYSIS | tf=1h | standalone R1..R6 | "
        f"next-candle direction | fee={fee * 100:.2f}%/side"
    )
    print("Meaning: edge_vs_50 = accuracy - 50%; positive means directional value above random baseline.")

    all_rows: list[dict[str, object]] = []
    for symbol, group in df.groupby("symbol"):
        stats = analyze_symbol(group, fee)
        print(f"\n{symbol}")
        ranked = sorted(
            stats.items(),
            key=lambda kv: (kv[1]["edge_vs_50_pct"], kv[1]["avg_net_pct"]),
            reverse=True,
        )
        for rule, s in ranked:
            print(
                f"{rule} samples={int(s['samples']):5d} "
                f"acc={s['accuracy_pct']:6.2f}% "
                f"edge={s['edge_vs_50_pct']:+6.2f}pp "
                f"raw={s['raw_pct']:+8.3f}% "
                f"net={s['net_pct']:+8.3f}% "
                f"avg_net={s['avg_net_pct']:+.4f}%"
            )
            all_rows.append({"symbol": symbol, "rule": rule, **s})

        if ranked:
            r, s = ranked[0]
            print(f"BEST_SCORE {r} | edge={s['edge_vs_50_pct']:+.2f}pp | acc={s['accuracy_pct']:.2f}%")

    out = Path("reports/candle_formula_rule_score_analysis.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nCSV={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
