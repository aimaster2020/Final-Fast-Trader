#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


RAW_FIELDS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce the supplied Excel candle formulas exactly from OHLCV CSV data.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def fnum(value: str | float | int | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def looks_like_header(row: list[str]) -> bool:
    h = {str(x).strip().lower() for x in row}
    return {"timestamp", "open", "high", "low", "close"}.issubset(h)


def load_rows(path: Path) -> list[dict[str, float | str | None]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return []
        if looks_like_header(first):
            fields = [str(x).strip() for x in first]
            raw = [
                {k: row[idx] if idx < len(row) else "" for idx, k in enumerate(fields)}
                for row in reader
            ]
        else:
            raw = [
                {k: row[idx] if idx < len(row) else "" for idx, k in enumerate(RAW_FIELDS)}
                for row in [first] + list(reader)
            ]

    out: list[dict[str, float | str | None]] = []
    for r in raw:
        out.append(
            {
                "Timestamp": r.get("Timestamp", ""),
                "Open": fnum(r.get("Open")),
                "High": fnum(r.get("High")),
                "Low": fnum(r.get("Low")),
                "Close": fnum(r.get("Close")),
                "Volume": fnum(r.get("Volume")),
            }
        )
    return [r for r in out if None not in (r["Open"], r["High"], r["Low"], r["Close"])]


def flag(value: bool) -> int:
    return 1 if value else 0


def safe_max(values: list[float]) -> float:
    return max(values) if values else float("nan")


def safe_min(values: list[float]) -> float:
    return min(values) if values else float("nan")


def row_value(rows: list[dict], i: int, key: str) -> float:
    return float(rows[i][key])


def calculate(rows: list[dict]) -> list[dict]:
    out: list[dict] = []

    for i, r in enumerate(rows):
        b = row_value(rows, i, "Open")
        c = row_value(rows, i, "High")
        d = row_value(rows, i, "Low")
        e = row_value(rows, i, "Close")
        volume = r.get("Volume")

        # Exact basic formulas.
        g = abs(e - b)                              # =ABS(E-B)
        h = c - max(b, e)                           # =C-MAX(B,E)
        ii = min(b, e) - d                          # =MIN(B,E)-D
        j = c - d                                   # =C-D

        def p(n: int) -> dict:
            return rows[i - n]

        enough2 = i >= 1
        enough3 = i >= 2
        enough20 = i >= 20
        enough21 = i >= 21

        if enough2:
            b1, c1, d1, e1 = (float(p(1)[k]) for k in ("Open", "High", "Low", "Close"))
            g1 = abs(e1 - b1)
            j1 = c1 - d1
        else:
            b1 = c1 = d1 = e1 = g1 = j1 = 0.0

        if enough3:
            b2, c2, d2, e2 = (float(p(2)[k]) for k in ("Open", "High", "Low", "Close"))
            g2 = abs(e2 - b2)
            j2 = c2 - d2
        else:
            b2 = c2 = d2 = e2 = g2 = j2 = 0.0

        # K:Y bullish-side formulas, exactly as supplied.
        K = flag(g > 0 and ii >= 2 * g and h <= 0.5 * g)
        L = flag(g > 0 and h >= 2 * g and ii <= 0.5 * g)
        M = flag(j > 0 and ii >= 0.6 * j and h <= 0.05 * j)
        N = flag(e > b and h <= 0.05 * j and ii <= 0.05 * j)
        O = flag(enough2 and e1 < b1 and e > b and b <= e1 and e >= b1)
        P = flag(enough2 and e1 < b1 and e > b and b >= e1 and e <= b1 and g < g1)
        Q = flag(enough2 and e1 < b1 and e > b and e > (b1 + e1) / 2 and e < b1)
        R = flag(enough2 and e1 < b1 and e > b and abs(d - d1) <= 0.001 * max(d1, 0.000001))

        s_base = False
        if enough3:
            s_base = (
                e2 < b2
                and g1 <= 0.3 * j1
                and e > b
                and e > b2
                and e > e2 + (b2 - e2) / 2
            )
        S = flag(s_base)
        T = flag(
            enough3
            and e2 > b2
            and e1 > b1
            and e > b
            and e1 > e2
            and e > e1
            and g2 > 0.5 * j2
            and g1 > 0.5 * j1
            and g > 0.5 * j
        )
        U = flag(enough2 and c > c1 and d < d1 and e > b)
        V = flag(g > 0 and ii >= 2 * g and h <= 0.5 * g)

        prior20_highs = [float(x["High"]) for x in rows[i - 20 : i]] if enough20 else []
        W = flag(enough20 and e > safe_max(prior20_highs))
        X = flag(enough20 and c > safe_max(prior20_highs) and e < safe_max(prior20_highs))

        Y = 0
        if enough21:
            base_high = safe_max([float(x["High"]) for x in rows[i - 21 : i - 1]])
            Y = flag(e1 > base_high and d <= base_high and e > base_high)

        # Neutral/range formulas AA:AE.
        AA = flag(j > 0 and g <= 0.1 * j)
        AB = flag(j > 0 and h >= 0.4 * j and ii >= 0.4 * j and g <= 0.1 * j)
        AC = flag(j > 0 and g <= 0.3 * j and h > g and ii > g)
        AD = flag(enough2 and c < c1 and d > d1)
        AE = AA + AB + AC + AD

        # Bearish-side formulas AF:AT, exactly as supplied.
        AF = flag(enough2 and g > 0 and ii >= 2 * g and h <= 0.5 * g and e1 > b1)
        AG = flag(enough2 and g > 0 and h >= 2 * g and ii <= 0.5 * g and e1 > b1)
        AH = flag(j > 0 and h >= 0.6 * j and ii <= 0.05 * j)
        AI = flag(e < b and h <= 0.05 * j and ii <= 0.05 * j)
        AJ = flag(enough2 and e1 > b1 and e < b and b >= e1 and e <= b1)
        AK = flag(enough2 and e1 > b1 and e < b and b <= e1 and e >= b1 and g < g1)
        AL = flag(enough2 and e1 > b1 and e < b and e < (b1 + e1) / 2 and e > b)
        AM = flag(enough2 and e1 > b1 and e < b and abs(c - c1) <= 0.001 * max(c1, 0.000001))

        AN = 0
        if enough3:
            AN = flag(
                e2 > b2
                and g1 <= 0.3 * j1
                and e < b
                and e < b2
                and e < e2 - (e2 - b2) / 2
            )
        AO = flag(
            enough3
            and e2 < b2
            and e1 < b1
            and e < b
            and e1 < e2
            and e < e1
            and g2 > 0.5 * j2
            and g1 > 0.5 * j1
            and g > 0.5 * j
        )
        AP = flag(enough2 and c > c1 and d < d1 and e < b)
        AQ = flag(g > 0 and h >= 2 * g and ii <= 0.5 * g)

        prior20_lows = [float(x["Low"]) for x in rows[i - 20 : i]] if enough20 else []
        AR = flag(enough20 and e < safe_min(prior20_lows))
        AS = flag(enough20 and d < safe_min(prior20_lows) and e > safe_min(prior20_lows))

        AT = 0
        if enough21:
            base_low = safe_min([float(x["Low"]) for x in rows[i - 21 : i - 1]])
            AT = flag(e1 < base_low and c >= base_low and e < base_low)

        # Structured-reference score formulas rewritten literally.
        trend_up = None
        if i >= 4:
            prior4 = rows[i - 4 : i]
            avg_ohlc = mean(
                [
                    float(x[k])
                    for x in prior4
                    for k in ("Open", "High", "Low", "Close")
                ]
            )
            trend_up = flag(e > avg_ohlc)
        else:
            trend_up = 0

        bullish_score = trend_up + sum([K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y])
        bearish_score = -(trend_up + sum([AF, AG, AH, AI, AJ, AK, AL, AM, AN, AO, AP, AQ, AR, AS, AT]))

        if bullish_score + bearish_score > 0:
            final_score = bullish_score
        elif bullish_score + bearish_score < 0:
            final_score = bearish_score
        elif AE > bullish_score + bearish_score:
            final_score = 0
        else:
            final_score = 0

        trend = "صعودی" if trend_up == 1 else "نزولی"

        min4 = min(float(x["Low"]) for x in rows[i - 4 : i]) if i >= 4 else ""
        max4 = max(float(x["High"]) for x in rows[i - 4 : i]) if i >= 4 else ""
        avg_vol = mean(float(x["Volume"]) for x in rows[i - 4 : i] if x.get("Volume") is not None) if i >= 4 else None
        volume_flag = flag(volume is not None and avg_vol is not None and float(volume) > avg_vol)

        BA = "خرید" if K == 1 else ""
        BB = "خرید" if O == 1 else ""
        BC = "فروش" if AG == 1 else ""
        BD = "فروش" if AJ == 1 else ""
        BE = "خرید" if W == 1 else ""
        BF = "فروش" if AR == 1 else ""

        buys = sum(1 for x in (BA, BB, BE) if x == "خرید")
        sells = sum(1 for x in (BC, BD, BF) if x == "فروش")
        if buys > 0 and sells > 0:
            BG = ""
        elif buys > 0:
            BG = "خرید"
        elif sells > 0:
            BG = "فروش"
        else:
            BG = ""

        BH = e if BG != "" else ""
        BI = d - 0.5 if BG == "خرید" else c + 0.5 if BG == "فروش" else ""
        BJ = e + 10 if BG == "خرید" else e - 10 if BG == "فروش" else ""
        BK = ""
        BL = ""
        BM = ""
        BN = ""

        # Final movement columns corresponding to the supplied structured references.
        BO = e
        BP = 1 if final_score > 0 else -1 if final_score < 0 else 0
        if i >= 1:
            BQ = 1 if e - e1 > 0 else -1 if e - e1 < 0 else 0
        else:
            BQ = 0
        BR = flag(BQ == BP or BP == 0)

        row_out = {
            "Timestamp": r["Timestamp"],
            "B_Open": b, "C_High": c, "D_Low": d, "E_Close": e, "F_Volume": volume if volume is not None else "",
            "G": g, "H": h, "I": ii, "J": j,
            "K": K, "L": L, "M": M, "N": N, "O": O, "P": P, "Q": Q, "R": R, "S": S, "T": T, "U": U, "V": V, "W": W, "X": X, "Y": Y,
            "Z_BullScore": bullish_score,
            "AA": AA, "AB": AB, "AC": AC, "AD": AD, "AE_RangeScore": AE,
            "AF": AF, "AG": AG, "AH": AH, "AI": AI, "AJ": AJ, "AK": AK, "AL": AL, "AM": AM, "AN": AN, "AO": AO, "AP": AP, "AQ": AQ, "AR": AR, "AS": AS, "AT": AT,
            "AU_BearScore": bearish_score,
            "AV_FinalScore": final_score,
            "AW_Trend": trend,
            "AX_Min4": min4,
            "AY_Max4": max4,
            "AZ_VolumeFlag": volume_flag,
            "BA": BA, "BB": BB, "BC": BC, "BD": BD, "BE": BE, "BF": BF,
            "BG": BG, "BH": BH, "BI": BI, "BJ": BJ, "BK": BK, "BL": BL, "BM": BM, "BN": BN,
            "BO_Closing": BO, "BP_PredictedDirection": BP, "BQ_ActualDirection": BQ, "BR_DirectionMatch": BR,
        }
        out.append(row_out)

    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    target = Path(args.output)
    rows = load_rows(source)
    if not rows:
        raise SystemExit("No usable OHLC rows found")

    result = calculate(rows)
    write_csv(target, result)

    valid = result[20:] if len(result) > 20 else []
    matches = sum(int(r["BR_DirectionMatch"]) for r in valid)
    buys = sum(1 for r in valid if r["BG"] == "خرید")
    sells = sum(1 for r in valid if r["BG"] == "فروش")

    print(f"input={source}")
    print(f"output={target}")
    print(f"rows={len(result)}")
    print("FORMULAS=EXACT_SUPPLIED_EXCEL_LOGIC")
    print("commission=not applied")
    print(f"usable_rows={len(valid)}")
    print(f"buy_signals={buys}")
    print(f"sell_signals={sells}")
    print(f"direction_match={matches}/{len(valid)} accuracy={matches/len(valid)*100 if valid else 0.0:.4f}%")


if __name__ == "__main__":
    main()
