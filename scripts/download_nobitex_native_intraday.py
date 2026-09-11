from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://apiv2.nobitex.ir"
STATS_PATH = "/market/stats"
HISTORY_PATH = "/market/udf/history"
MAX_PER_REQUEST = 500
DEFAULT_RESOLUTIONS = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60"}
MIN_REQUEST_INTERVAL = 61.0 / 60.0


def get_json(session: requests.Session, path: str, params: dict[str, Any]) -> dict[str, Any]:
    response = session.get(
        BASE_URL + path,
        params=params,
        headers={
            "Accept": "application/json",
            "User-Agent": "TraderBot/Final-Fast-Trader-Nobitex-intraday-1.0",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected response type from {path}: {type(payload).__name__}")
    return payload


def normalize_symbol(raw_symbol: str) -> str:
    symbol = raw_symbol.replace("-", "").replace("/", "").upper()
    if symbol.endswith("RLS"):
        symbol = symbol[:-3] + "IRT"
    return symbol


def discover_nonirt_markets(session: requests.Session) -> list[str]:
    payload = get_json(session, STATS_PATH, {})
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        raise RuntimeError("Nobitex /market/stats response has no stats object")

    symbols = []
    for raw_symbol, value in stats.items():
        if not isinstance(raw_symbol, str) or not isinstance(value, dict):
            continue
        symbol = normalize_symbol(raw_symbol)
        if symbol and not symbol.endswith("IRT"):
            symbols.append(symbol)
    return sorted(set(symbols))


def fetch_history(
    session: requests.Session,
    symbol: str,
    resolution: str,
    start_ts: int,
    end_ts: int,
    request_sleep: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    page = 1
    first_request = True

    while True:
        if not first_request:
            time.sleep(request_sleep)
        first_request = False

        payload = get_json(
            session,
            HISTORY_PATH,
            {
                "symbol": symbol,
                "resolution": resolution,
                "from": start_ts,
                "to": end_ts,
                "countback": MAX_PER_REQUEST,
                "page": page,
            },
        )

        if payload.get("s") == "no_data" or payload.get("noData"):
            break
        if payload.get("s") != "ok":
            raise RuntimeError(str(payload.get("errmsg") or payload))

        ts = payload.get("t", [])
        opens = payload.get("o", [])
        highs = payload.get("h", [])
        lows = payload.get("l", [])
        closes = payload.get("c", [])
        volumes = payload.get("v", [])
        n = min(len(ts), len(opens), len(highs), len(lows), len(closes), len(volumes))
        if n == 0:
            break

        for i in range(n):
            rows.append(
                {
                    "timestamp": int(ts[i]),
                    "open": float(opens[i]),
                    "high": float(highs[i]),
                    "low": float(lows[i]),
                    "close": float(closes[i]),
                    "volume": float(volumes[i]),
                }
            )

        oldest = min(int(x) for x in ts)
        if oldest <= start_ts or n < MAX_PER_REQUEST:
            break
        page += 1

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download native Nobitex intraday candles for non-IRT markets.")
    ap.add_argument("--days", type=int, choices=[1, 3, 7, 30], default=7)
    ap.add_argument("--output-dir", default="reports/nobitex_intraday_native")
    ap.add_argument("--symbols", default="", help="Optional comma-separated market override.")
    ap.add_argument("--timeframes", default="1m,5m,15m,30m,1h")
    ap.add_argument("--sleep", type=float, default=MIN_REQUEST_INTERVAL)
    args = ap.parse_args()

    resolutions = DEFAULT_RESOLUTIONS.copy()
    requested = [x.strip() for x in args.timeframes.split(",") if x.strip()]
    unknown = [x for x in requested if x not in resolutions]
    if unknown:
        raise SystemExit(f"Unsupported timeframes: {unknown}; choose from {sorted(resolutions)}")

    now = int(time.time())
    start_ts = now - args.days * 86400
    end_ts = now
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    if args.symbols.strip():
        symbols = sorted(set(normalize_symbol(s) for s in args.symbols.split(",") if s.strip()))
        symbols = [s for s in symbols if not s.endswith("IRT")]
    else:
        symbols = discover_nonirt_markets(session)

    print(f"Non-IRT markets discovered: {len(symbols)}")
    print(f"Timeframes: {', '.join(requested)} | days: {args.days}")
    print(f"Output: {root}")
    print(f"Request interval: {args.sleep:.3f}s")

    summary: dict[str, Any] = {
        "source": BASE_URL,
        "history_endpoint": HISTORY_PATH,
        "stats_endpoint": STATS_PATH,
        "days": args.days,
        "from": start_ts,
        "to": end_ts,
        "timeframes": requested,
        "markets": symbols,
        "non_irt_only": True,
        "results": {},
    }

    for tf in requested:
        resolution = resolutions[tf]
        tf_dir = root / tf
        tf_dir.mkdir(parents=True, exist_ok=True)
        successful = 0
        skipped = []
        total_rows = 0

        print("=" * 90)
        print(f"TIMEFRAME {tf} | resolution={resolution}")
        for index, symbol in enumerate(symbols, start=1):
            try:
                df = fetch_history(session, symbol, resolution, start_ts, end_ts, args.sleep)
                if df.empty:
                    skipped.append({"symbol": symbol, "reason": "no_data"})
                    print(f"[{index}/{len(symbols)}] {symbol}: no_data")
                    continue
                path = tf_dir / f"{symbol}_{tf}.csv"
                df.to_csv(path, index=False)
                successful += 1
                total_rows += len(df)
                print(f"[{index}/{len(symbols)}] {symbol}: {len(df)} candles")
            except Exception as exc:
                skipped.append({"symbol": symbol, "reason": str(exc)})
                print(f"[{index}/{len(symbols)}] {symbol}: ERROR {exc}")

        summary["results"][tf] = {
            "resolution": resolution,
            "requested_markets": len(symbols),
            "successful_markets": successful,
            "skipped_markets": len(skipped),
            "total_candles": total_rows,
            "skipped": skipped,
        }
        print(f"DONE {tf}: successful={successful} skipped={len(skipped)} candles={total_rows}")

    (root / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 90)
    print("ALL TIMEFRAMES COMPLETE")


if __name__ == "__main__":
    main()
