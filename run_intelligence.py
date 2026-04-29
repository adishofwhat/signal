"""
SIGNAL Phase 3 — LLM Intelligence Layer test runner.

Runs all five diagnostics, feeds results to the synthesizer, prints the
structured JSON output, and evaluates Checkpoint 3 pass/fail criteria.

Usage:
    python3 run_intelligence.py [--no-cache] [--stage2]

Environment:
    LLM_PROVIDER  — "claude" (default), "openai", or "gemini"
    LLM_API_KEY   — API key for chosen provider
    ANTHROPIC_API_KEY — fallback for Claude
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import CAMPAIGN_CONFIG, LLM_CONFIG
from diagnostics import DiagnosticResult
from diagnostics.creative_fatigue import detect as detect_fatigue
from diagnostics.frequency_saturation import detect as detect_saturation
from diagnostics.daypart_analysis import detect as detect_daypart
from diagnostics.channel_cannibalization import detect as detect_cannibalization
from diagnostics.budget_efficiency import detect as detect_efficiency
from intelligence.llm_client import LLMClient, LLMProvider
from intelligence.synthesizer import synthesize
from intelligence.cache import cache_stats


DATA_DIR = Path(__file__).parent / "data"
AOV = CAMPAIGN_CONFIG.get("avg_order_value", 120)


# ---------------------------------------------------------------------------
# Data + diagnostics
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    campaigns = pd.read_csv(DATA_DIR / "campaigns.csv", parse_dates=["date"])
    creative  = pd.read_csv(DATA_DIR / "creative_performance.csv", parse_dates=["date"])
    daypart   = pd.read_csv(DATA_DIR / "daypart_performance.csv", parse_dates=["date"])
    return campaigns, creative, daypart


def run_diagnostics(campaigns, creative, daypart) -> list[DiagnosticResult]:
    results: list[DiagnosticResult] = []
    results.extend(detect_fatigue(creative, campaigns, aov=AOV))
    sat = detect_saturation(campaigns, aov=AOV)
    if sat:
        results.append(sat)
    results.extend(detect_daypart(daypart, campaigns, aov=AOV))
    results.extend(detect_cannibalization(campaigns, aov=AOV))
    results.extend(detect_efficiency(campaigns, aov=AOV))
    return results


# ---------------------------------------------------------------------------
# Checkpoint 3 verification
# ---------------------------------------------------------------------------

def _extract_numbers_from_diagnostics(results: list[DiagnosticResult]) -> set[str]:
    """Collect all numeric strings that appear in the diagnostic output."""
    nums: set[str] = set()
    for r in results:
        # Pull numbers from finding text
        nums.update(re.findall(r"\d+\.?\d*", r.finding))
        # Pull numbers from evidence
        for v in r.evidence.values():
            if isinstance(v, (int, float)):
                nums.add(str(round(float(v), 4)))
            elif isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, (int, float)):
                        nums.add(str(round(float(vv), 4)))
    return nums


def check_checkpoint3(stage1: dict, results: list[DiagnosticResult]) -> list[tuple[str, bool]]:
    checks: list[tuple[str, bool]] = []

    # 1. JSON has required top-level keys
    required_keys = {"executive_summary", "findings", "cross_channel_insights", "action_plan"}
    checks.append((
        "JSON has all required top-level keys",
        required_keys.issubset(stage1.keys()),
    ))

    # 2. Executive summary references at least 2 specific metrics (numbers with % or $)
    exec_summary = stage1.get("executive_summary", "")
    metric_refs = re.findall(r"[\$\d][\d,.%×x]+", exec_summary)
    checks.append((
        "Executive summary references ≥2 specific metrics",
        len(metric_refs) >= 2,
    ))

    # 3. Findings array has ≥ 1 item with required fields
    findings = stage1.get("findings", [])
    required_finding_keys = {"rank", "title", "severity", "narrative", "recommendation", "estimated_impact", "confidence"}
    findings_valid = len(findings) >= 1 and all(
        required_finding_keys.issubset(f.keys()) for f in findings
    )
    checks.append(("Findings array is well-formed", findings_valid))

    # 4. Action plan has 3-5 items
    action_plan = stage1.get("action_plan", [])
    checks.append((
        "Action plan has 3-5 items",
        3 <= len(action_plan) <= 5,
    ))

    # 5. Each action plan item has required fields
    required_action_keys = {"priority", "action", "timeline", "owner"}
    action_valid = all(required_action_keys.issubset(a.keys()) for a in action_plan)
    checks.append(("Action plan items have required fields", action_valid))

    # 6. Findings ranked (rank field present and ordered)
    if findings:
        ranks = [f.get("rank") for f in findings]
        checks.append((
            "Findings are ranked in order",
            ranks == sorted(ranks),
        ))

    # 7. Each finding narrative references at least one number
    if findings:
        narr_nums = [
            bool(re.search(r"\d", f.get("narrative", "")))
            for f in findings
        ]
        checks.append((
            "All finding narratives reference at least one number",
            all(narr_nums),
        ))

    # 8. Severity values are valid
    if findings:
        valid_severities = {"critical", "warning", "info"}
        sev_valid = all(f.get("severity") in valid_severities for f in findings)
        checks.append(("All severities are valid values", sev_valid))

    # 9. Timeline values are valid
    valid_timelines = {"immediate", "this_week", "this_month"}
    if action_plan:
        timeline_valid = all(a.get("timeline") in valid_timelines for a in action_plan)
        checks.append(("All timelines are valid values", timeline_valid))

    # 10. Owner values are valid
    valid_owners = {"media_buyer", "creative_team", "analyst"}
    if action_plan:
        owner_valid = all(a.get("owner") in valid_owners for a in action_plan)
        checks.append(("All owners are valid values", owner_valid))

    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SIGNAL Phase 3 — LLM Intelligence Layer")
    parser.add_argument("--no-cache", action="store_true", help="Skip cache (always call LLM)")
    parser.add_argument("--stage2", action="store_true", help="Also run Stage 2 narrative report")
    args = parser.parse_args()

    use_cache = not args.no_cache

    print("=" * 70)
    print("SIGNAL Intelligence Engine — Phase 3")
    print("=" * 70)

    # -- Load data and run diagnostics --
    print("\nLoading data...")
    campaigns, creative, daypart = load_data()

    print("Running diagnostics...")
    results = run_diagnostics(campaigns, creative, daypart)
    print(f"  {len(results)} diagnostic findings collected.")

    # -- Set up LLM client --
    provider_str = LLM_CONFIG.get("provider", "claude")
    api_key = LLM_CONFIG.get("api_key", "")
    model = LLM_CONFIG.get("model")

    if not api_key:
        # Check cache — if populated, the LLM call won't be needed
        stats = cache_stats()
        if stats["count"] == 0 and use_cache:
            print(
                "\nWARNING: LLM_API_KEY not set and cache is empty. "
                "Set ANTHROPIC_API_KEY (or LLM_API_KEY) to call the LLM."
            )
            sys.exit(1)
        print(f"\nNo API key set — using cache ({stats['count']} cached responses).")

    client = LLMClient(
        provider=LLMProvider(provider_str),
        api_key=api_key,
        model=model,
    )

    print(f"\nProvider: {provider_str} | Model: {client.model} | Cache: {'ON' if use_cache else 'OFF'}")

    # -- Run synthesis --
    print("\nRunning LLM synthesis (Stage 1)...")
    output = synthesize(
        results=results,
        campaign_config=CAMPAIGN_CONFIG,
        llm_client=client,
        use_cache=use_cache,
        run_stage2=args.stage2,
    )

    stage1 = output["stage1"]

    # -- Print Stage 1 JSON --
    print("\n" + "=" * 70)
    print("STAGE 1 OUTPUT — Structured JSON")
    print("=" * 70)
    print(json.dumps(stage1, indent=2))

    # -- Print Stage 2 if run --
    if args.stage2 and output["stage2"]:
        print("\n" + "=" * 70)
        print("STAGE 2 OUTPUT — Narrative Report")
        print("=" * 70)
        print(output["stage2"])

    # -- Checkpoint 3 verification --
    print("\n" + "=" * 70)
    print("CHECKPOINT 3 — pass/fail indicators")
    print("=" * 70)

    checks = check_checkpoint3(stage1, results)
    all_pass = True
    for label, passed in checks:
        status = "PASS ✓" if passed else "FAIL ✗"
        if not passed:
            all_pass = False
        print(f"  [{status}] {label}")

    # Summary metrics
    print(f"\n  Findings count: {len(stage1.get('findings', []))}")
    print(f"  Action plan items: {len(stage1.get('action_plan', []))}")
    exec_numbers = re.findall(r"[\$\d][\d,.%×x]+", stage1.get("executive_summary", ""))
    print(f"  Metric references in exec summary: {len(exec_numbers)} → {exec_numbers}")

    stats = cache_stats()
    print(f"\n  Cache: {stats['count']} files, {stats['total_bytes'] / 1024:.1f} KB")

    print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    return output


if __name__ == "__main__":
    main()
