"""
SIGNAL Phase 2 — run all five diagnostic modules and print results.
Saves Plotly charts as HTML files in data/charts/.

Usage: python3 run_diagnostics.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.io as pio

# Ensure signal/ is on the path regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))

from diagnostics import DiagnosticResult
from diagnostics.creative_fatigue import detect as detect_fatigue
from diagnostics.frequency_saturation import detect as detect_saturation
from diagnostics.daypart_analysis import detect as detect_daypart
from diagnostics.channel_cannibalization import detect as detect_cannibalization
from diagnostics.budget_efficiency import detect as detect_efficiency
from config import CAMPAIGN_CONFIG

AOV = CAMPAIGN_CONFIG.get("avg_order_value", 120)
DATA_DIR = Path(__file__).parent / "data"
CHARTS_DIR = DATA_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    campaigns = pd.read_csv(DATA_DIR / "campaigns.csv", parse_dates=["date"])
    creative  = pd.read_csv(DATA_DIR / "creative_performance.csv", parse_dates=["date"])
    daypart   = pd.read_csv(DATA_DIR / "daypart_performance.csv", parse_dates=["date"])
    return campaigns, creative, daypart


def _save_charts(results: list[DiagnosticResult]) -> None:
    for r in results:
        for i, fig in enumerate(r.charts):
            safe_name = r.name.replace(" ", "_")
            ch_safe = r.channel.replace(" ", "_").replace("→", "to")
            path = CHARTS_DIR / f"{safe_name}__{ch_safe}__{i}.html"
            pio.write_html(fig, str(path))
            print(f"  Chart saved → {path.name}")


def run() -> list[DiagnosticResult]:
    campaigns, creative, daypart = load_data()
    all_results: list[DiagnosticResult] = []

    print("=" * 70)
    print("SIGNAL Diagnostics Engine — Phase 2")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Creative Fatigue                                                  #
    # ------------------------------------------------------------------ #
    print("\n[1/5] Running creative fatigue detection...")
    fatigue_results = detect_fatigue(creative, campaigns, aov=AOV)
    if fatigue_results:
        for r in fatigue_results:
            r.print_summary()
        all_results.extend(fatigue_results)
    else:
        print("  No fatigued creatives detected.")

    # ------------------------------------------------------------------ #
    # 2. Frequency Saturation                                              #
    # ------------------------------------------------------------------ #
    print("\n[2/5] Running frequency saturation detection...")
    sat_result = detect_saturation(campaigns, aov=AOV)
    if sat_result:
        sat_result.print_summary()
        all_results.append(sat_result)
    else:
        print("  No frequency saturation detected.")

    # ------------------------------------------------------------------ #
    # 3. Daypart Opportunity                                               #
    # ------------------------------------------------------------------ #
    print("\n[3/5] Running daypart opportunity detection...")
    daypart_results = detect_daypart(daypart, campaigns, aov=AOV)
    if daypart_results:
        for r in daypart_results:
            r.print_summary()
        all_results.extend(daypart_results)
    else:
        print("  No daypart opportunities detected.")

    # ------------------------------------------------------------------ #
    # 4. Channel Cannibalization                                           #
    # ------------------------------------------------------------------ #
    print("\n[4/5] Running channel cannibalization detection...")
    cannibal_results = detect_cannibalization(campaigns, aov=AOV)
    if cannibal_results:
        for r in cannibal_results:
            r.print_summary()
        all_results.extend(cannibal_results)
    else:
        print("  No cannibalization signal detected.")

    # ------------------------------------------------------------------ #
    # 5. Budget Efficiency                                                 #
    # ------------------------------------------------------------------ #
    print("\n[5/5] Running budget efficiency analysis...")
    efficiency_results = detect_efficiency(campaigns, aov=AOV)
    if efficiency_results:
        for r in efficiency_results:
            r.print_summary()
        all_results.extend(efficiency_results)
    else:
        print("  No budget efficiency issues detected.")

    # ------------------------------------------------------------------ #
    # Summary table                                                        #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("CHECKPOINT 2 SUMMARY")
    print("=" * 70)
    print(f"{'Module':<28} {'Channel':<30} {'Severity':<10} {'Impact ($)':>12}")
    print("-" * 80)
    for r in all_results:
        print(
            f"{r.name:<28} {r.channel:<30} {r.severity:<10} "
            f"${r.estimated_impact_usd:>10,.0f}"
        )

    print(f"\nTotal results: {len(all_results)}")

    # Checkpoint 2 pass/fail
    print("\n" + "=" * 70)
    print("CHECKPOINT 2 — pass/fail indicators")

    def _has(name: str, channel_substr: str) -> bool:
        return any(
            r.name == name and channel_substr in r.channel
            for r in all_results
        )

    def _evidence(name: str, key: str) -> float | None:
        for r in all_results:
            if r.name == name and key in r.evidence:
                return r.evidence[key]
        return None

    checks = [
        (
            "creative_fatigue detects social_v1",
            any(
                r.name == "creative_fatigue"
                and r.evidence.get("creative_id") == "social_v1"
                for r in all_results
            ),
        ),
        (
            "creative_fatigue p < 0.05",
            any(
                r.name == "creative_fatigue"
                and r.evidence.get("p_value_lambda", 1.0) < 0.05
                for r in all_results
            ),
        ),
        (
            "frequency_saturation detects display",
            _has("frequency_saturation", "programmatic_display"),
        ),
        (
            "frequency_saturation breakpoint < current freq",
            (
                lambda bp, cf: bp is not None and cf is not None and bp < cf
            )(
                _evidence("frequency_saturation", "breakpoint_frequency"),
                _evidence("frequency_saturation", "current_avg_frequency_14d"),
            ),
        ),
        (
            "daypart_opportunity detects paid_search",
            _has("daypart_opportunity", "paid_search"),
        ),
        (
            "daypart prime/late_night flagged",
            any(
                r.name == "daypart_opportunity"
                and any(
                    dp in r.evidence.get("high_opportunity_dayparts", [])
                    for dp in ["prime", "late_night"]
                )
                for r in all_results
            ),
        ),
        (
            "channel_cannibalization r > 0.50",
            any(
                r.name == "channel_cannibalization"
                and r.evidence.get("pearson_r_contemporaneous", 0) > 0.50
                for r in all_results
            ),
        ),
        (
            "budget_efficiency flags youtube",
            _has("budget_efficiency", "youtube"),
        ),
        (
            "budget_efficiency recommends ctv",
            any(
                r.name == "budget_efficiency"
                and r.evidence.get("recommended_destination") == "ctv"
                for r in all_results
            ),
        ),
        (
            "All impacts in plausible range ($1K–$2M)",
            all(
                1_000 <= r.estimated_impact_usd <= 2_000_000
                for r in all_results
                if r.estimated_impact_usd > 0
            ),
        ),
    ]

    all_pass = True
    for label, passed in checks:
        status = "PASS ✓" if passed else "FAIL ✗"
        if not passed:
            all_pass = False
        print(f"  [{status}] {label}")

    print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    # Save charts
    print("\nSaving charts...")
    _save_charts(all_results)

    return all_results


if __name__ == "__main__":
    run()
