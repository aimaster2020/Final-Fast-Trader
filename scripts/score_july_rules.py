from __future__ import annotations

import argparse
import csv
from pathlib import Path

RULES = [f"R{i}" for i in range(1, 17)]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def score_row(r: dict[str, str]) -> dict[str, float | str]:
    """Fixed structural score; no P&L or parameter optimization is used.

    Components:
      activity  : rewards usable, non-extreme activation (10% ideal, 60% ceiling)
      persistence: rewards signals that persist instead of flipping constantly
      stability : rewards similar activation across assets/timeframes
      direction : rewards a repeatable directional bias, but only mildly

    This is a characterization score, not a claim of predictive edge.
    """
    activity = float(r["avg_signal_rate_pct"])
    persistence = float(r["avg_persistence_bars"])
    reversal = float(r["avg_reversal_rate_pct"])
    spread = float(r["asset_signal_rate_spread_pct"])
    bias = abs(float(r["avg_directional_bias"]))
    tf_rates = [
        float(r["tf_5m_signal_rate_pct"]),
        float(r["tf_15m_signal_rate_pct"]),
        float(r["tf_30m_signal_rate_pct"]),
        float(r["tf_60m_signal_rate_pct"]),
    ]

    # 0% activity is useless; very high activity is treated as noisy.
    activity_score = clamp01(activity / 10.0) if activity < 10.0 else clamp01(1.0 - (activity - 10.0) / 90.0)
    persistence_score = clamp01((persistence - 1.0) / 2.0)
    reversal_score = clamp01(1.0 - reversal / 100.0)
    stability_score = clamp01(1.0 - spread / max(activity, 1.0))
    tf_mean = mean(tf_rates)
    tf_spread = max(tf_rates) - min(tf_rates) if tf_rates else 0.0
    tf_stability_score = clamp01(1.0 - tf_spread / max(tf_mean, 1.0))
    direction_score = clamp01(bias)

    # Fixed weights. These are deliberately simple and not fitted to returns.
    structural = (
        0.25 * activity_score
        + 0.20 * persistence_score
        + 0.20 * reversal_score
        + 0.20 * stability_score
        + 0.10 * tf_stability_score
        + 0.05 * direction_score
    )
    score_100 = 100.0 * structural

    if score_100 >= 70:
        tier = "CORE"
    elif score_100 >= 50:
        tier = "SECONDARY"
    elif score_100 >= 30:
        tier = "EVENT"
    else:
        tier = "NOISY/RARE"

    return {
        "activity_score": activity_score,
        "persistence_score": persistence_score,
        "reversal_score": reversal_score,
        "asset_stability_score": stability_score,
        "tf_stability_score": tf_stability_score,
        "direction_score": direction_score,
        "behavior_score": score_100,
        "tier": tier,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Score R1-R16 from July behavioral characterization.")
    ap.add_argument("--input", default="reports/july_r_behavior_matrix.csv")
    ap.add_argument("--output", default="reports/july_r_behavior_scores.csv")
    args = ap.parse_args()

    src = Path(args.input)
    rows = list(csv.DictReader(src.open("r", encoding="utf-8", newline="")))
    by_rule = {r["rule"]: r for r in rows}
    missing = [r for r in RULES if r not in by_rule]
    if missing:
        raise SystemExit(f"Missing rules: {','.join(missing)}")

    out: list[dict] = []
    for rule in RULES:
        out.append({"rule": rule, **score_row(by_rule[rule])})

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        fields = list(out[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    print("R  SCORE TIER       ACT  PERS  REV  ASTAB TSTAB DIR")
    for r in out:
        print(
            f"{r['rule']:>3} {float(r['behavior_score']):5.1f} {str(r['tier']):<11} "
            f"{float(r['activity_score']):4.2f} {float(r['persistence_score']):4.2f} "
            f"{float(r['reversal_score']):4.2f} {float(r['asset_stability_score']):5.2f} "
            f"{float(r['tf_stability_score']):5.2f} {float(r['direction_score']):3.2f}"
        )
    print(f"SAVED {p} rows={len(out)}")


if __name__ == "__main__":
    main()
