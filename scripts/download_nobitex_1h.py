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
RESOLUTION = "60"
MAX_PER_REQUEST = 500
MIN_REQUEST_INTERVAL = 61.0 / 60.0


def get_json(session: requests.Session, path: str, params: dict[str, Any]) -> dict[str, Any]:
    response = session.get(
        BASE_URL + path,
        params=params,
        headers={
            "Accept": "application/json",
            "User-Agent": "TraderBot/Final-Fast-Trader-Nobitex-1.0.0",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected response type from {path}: {type(payload).__name__}")
    return payload


def fetch_markets(session: requests.Session) -> list[str]:
    payload = get_json(session, STATS_PATH, {})
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        raise RuntimeError(
            "Nobitex /market/stats response does not contain a 'stats' object. "
            f"Keys={list(payload.keys())}"
        )

    symbols: list[str] = []
    for raw_symbol, value in stats.items():
        if not isinstance(raw_symbol, str) or not isinstance(value, dict):
            continue

        # Nobitex market statistics are commonly keyed as btc-irt / btc-usdt.
        symbol = raw_symbol.replace("-", "").replace("/", "").upper()
        if not symbol:
            continue

        # Keep markets that expose a valid symbol and a normal market-stat object.
        # The UDF endpoint itself is used later as the final availability check.
        symbols.append(symbol)

    return sorted(set(symbols))


def fetch_history(
    session: requests.Session,
    symbol: str,
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
                "resolution": RESOLUTION,
                "from": start_ts,
                "to": end_ts,
                "countback": MAX_PER_REQUEST,
                "page": page,
            },
        )

        if payload.get("s") == "no_data" or payload.get("noData"):
            break
        if payload.get("s") != "ok":
            message = payload.get("errmsg") or payload
            raise RuntimeError(f"UDF error for {symbol}: {message}")

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

        # page=1 is the newest batch, page=2 the next older batch, etc.
        # Stop as soon as we have reached the requested start time or the API
        # returned less than the maximum batch size.
        oldest = min(int(x) for x in ts)
        if oldest <= start_ts or n < MAX_PER_REQUEST:
            break
        page += 1

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download all available Nobitex API v2 1h candles.")
    parser.add_argument("--days", type=int, choices=[7, 30], default=7, help="Lookback window in days.")
    parser.add_argument("--output-dir", default="reports/nobitex_1h")
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated override. Empty means discover all markets from Nobitex.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=MIN_REQUEST_INTERVAL,
        help="Delay between history requests. Default keeps below 60 requests/minute.",
    )
    args = parser.parse_args()

    now = int(time.time())
    start_ts = now - args.days * 24 * 60 * 60
    end_ts = now

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    if args.symbols.strip():
        symbols = sorted(set(s.strip().upper() for s in args.symbols.split(",") if s.strip()))
    else:
        symbols = fetch_markets(session)

    print(f"Nobitex markets discovered: {len(symbols)}")
    print(f"Resolution: 1h | days: {args.days} | output: {output_dir}")
    print(f"Request interval: {args.sleep:.3f}s")

    successful: list[str] = []
    skipped: list[dict[str, str]] = []
    total_rows = 0

    for index, symbol in enumerate(symbols, start=1):
        try:
            df = fetch_history(session, symbol, start_ts, end_ts, args.sleep)
            if df.empty:
                skipped.append({"symbol": symbol, "reason": "no_data"})
                print(f"[{index}/{len(symbols)}] {symbol}: no_data")
                continue

            path = output_dir / f"{symbol}_1h.csv"
            df.to_csv(path, index=False)
            successful.append(symbol)
            total_rows += len(df)
            print(f"[{index}/{len(symbols)}] {symbol}: {len(df)} candles")
        except Exception as exc:  # keep the full universe scan running
            skipped.append({"symbol": symbol, "reason": str(exc)})
            print(f"[{index}/{len(symbols)}] {symbol}: ERROR {exc}")

    metadata = {
        "source": BASE_URL,
        "history_endpoint": HISTORY_PATH,
        "stats_endpoint": STATS_PATH,
        "resolution": RESOLUTION,
        "days": args.days,
        "from": start_ts,
        "to": end_ts,
        "requested_markets": len(symbols),
        "successful_markets": len(successful),
        "skipped_markets": len(skipped),
        "total_candles": total_rows,
        "successful_symbols": successful,
        "skipped": skipped,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 90)
    print(f"DONE: requested={len(symbols)} successful={len(successful)} skipped={len(skipped)} candles={total_rows}")


if __name__ == "__main__":
    main()
