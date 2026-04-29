"""
What-If Simulator — projection engine.

Three levers, each grounded in the fitted statistical models from Phase 2:
  1. Budget Reallocation  — uses marginal CPA curves from budget_efficiency.py
  2. Frequency Cap        — uses piecewise CPA model from frequency_saturation.py
  3. Creative Rotation    — uses exponential decay model from creative_fatigue.py

All three functions return plain dicts. The LLM narrates; it does not compute.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ── Lever 1: Budget Reallocation ──────────────────────────────────────────────

def project_budget_reallocation(
    campaigns_df: pd.DataFrame,
    source_channel: str,
    dest_channel: str,
    shift_pct: float,               # 0.0 – 0.40
    all_channels_late_cpa: dict,    # from budget_efficiency evidence
    aov: float = 120.0,
) -> dict:
    """
    Project impact of shifting shift_pct of source_channel spend → dest_channel.

    Uses late-30d CPA as the marginal CPA proxy.  The math:
        net_extra_convs = shifted_spend / dest_late_cpa
                        - shifted_spend / source_late_cpa
    """
    src = campaigns_df[campaigns_df["channel"] == source_channel]
    dst = campaigns_df[campaigns_df["channel"] == dest_channel]

    src_total_spend = float(src["spend"].sum())
    src_total_convs = int(src["conversions"].sum())
    dst_total_spend = float(dst["spend"].sum())
    dst_total_convs = int(dst["conversions"].sum())

    # Fall back to overall CPA if late-CPA not in evidence
    src_late_cpa = float(
        all_channels_late_cpa.get(source_channel)
        or (src_total_spend / max(src_total_convs, 1))
    )
    dst_late_cpa = float(
        all_channels_late_cpa.get(dest_channel)
        or (dst_total_spend / max(dst_total_convs, 1))
    )

    shifted_spend = src_total_spend * shift_pct
    lost_convs    = shifted_spend / src_late_cpa if src_late_cpa > 0 else 0.0
    gained_convs  = shifted_spend / dst_late_cpa if dst_late_cpa > 0 else 0.0
    net_extra     = gained_convs - lost_convs

    campaign_spend = float(campaigns_df["spend"].sum())
    campaign_convs = int(campaigns_df["conversions"].sum())

    current_cpa   = campaign_spend / max(campaign_convs, 1)
    projected_cpa = campaign_spend / max(campaign_convs + net_extra, 1)
    cpa_delta_pct = (projected_cpa - current_cpa) / current_cpa * 100

    return {
        "source_channel":      source_channel,
        "dest_channel":        dest_channel,
        "shift_pct":           shift_pct,
        "shifted_spend_usd":   shifted_spend,
        "source_late_cpa":     src_late_cpa,
        "dest_late_cpa":       dst_late_cpa,
        "lost_convs_at_source":  lost_convs,
        "gained_convs_at_dest":  gained_convs,
        "net_extra_convs":     net_extra,
        "current_blended_cpa":    current_cpa,
        "projected_blended_cpa":  projected_cpa,
        "cpa_delta_pct":       cpa_delta_pct,
        "revenue_impact_usd":  net_extra * aov,
        "is_improvement":      net_extra > 0,
    }


# ── Lever 2: Frequency Cap Adjustment ─────────────────────────────────────────

def project_freq_cap(
    campaigns_df: pd.DataFrame,
    new_cap: float,
    breakpoint_freq: float,
    cpa_at_breakpoint: float,
    slope_segment1: float,
    slope_segment2: float,
    current_avg_freq: float,
    aov: float = 120.0,
) -> dict:
    """
    Project CPA improvement from capping programmatic_display frequency at new_cap.

    Uses the fitted piecewise linear model:
        CPA(f) = β0 + β1·f                    for f ≤ bp
        CPA(f) = val_at_bp + β2·(f − bp)      for f > bp

    β0 is recovered from the continuity constraint: val_at_bp = β0 + β1·bp.
    """
    bp         = breakpoint_freq
    val_at_bp  = cpa_at_breakpoint
    beta0      = val_at_bp - slope_segment1 * bp  # intercept of segment 1

    def _predict(f: float) -> float:
        if f <= bp:
            return max(1.0, beta0 + slope_segment1 * f)
        return max(1.0, val_at_bp + slope_segment2 * (f - bp))

    current_cpa   = _predict(current_avg_freq)
    projected_cpa = _predict(new_cap)

    # Model floor: never predict below the baseline (segment-1 at f=3)
    model_floor   = _predict(3.0)
    projected_cpa = max(projected_cpa, model_floor)

    cpa_reduction     = current_cpa - projected_cpa
    cpa_reduction_pct = cpa_reduction / current_cpa * 100 if current_cpa > 0 else 0.0

    disp = campaigns_df[campaigns_df["channel"] == "programmatic_display"]
    daily_spend = float(disp["spend"].mean())

    current_daily_convs   = daily_spend / current_cpa   if current_cpa   > 0 else 0.0
    projected_daily_convs = daily_spend / projected_cpa if projected_cpa > 0 else 0.0
    extra_daily_convs     = projected_daily_convs - current_daily_convs
    savings_30d           = extra_daily_convs * 30 * aov

    # Actual current CPA for display (from data, for "before" card)
    actual_current_cpa = float(disp["cpa"].tail(14).mean())

    return {
        "current_avg_freq":    current_avg_freq,
        "new_cap":             new_cap,
        "breakpoint_freq":     bp,
        "model_current_cpa":   current_cpa,
        "model_projected_cpa": projected_cpa,
        "actual_current_cpa":  actual_current_cpa,
        "cpa_reduction":       cpa_reduction,
        "cpa_reduction_pct":   cpa_reduction_pct,
        "extra_daily_convs":   extra_daily_convs,
        "savings_30d_usd":     savings_30d,
        "is_improvement":      projected_cpa < current_cpa,
    }


# ── Lever 3: Creative Rotation Timing ─────────────────────────────────────────

def project_creative_rotation(
    creative_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
    rotation_day: int,
    decay_lambda: float,
    a_amplitude: float,
    c_floor: float,
    onset_day: int,
    initial_ctr_obs: float,
    creative_id: str = "social_v1",
    channel: str = "paid_social",
    aov: float = 120.0,
) -> dict:
    """
    Project impact of rotating creative_id at `rotation_day` (campaign day 0-based).

    Pre-rotation days  (0 … rotation_day−1): actual observed CTR.
    Post-rotation days (rotation_day … end): CTR resets to initial_ctr_obs
                                             (fresh replacement creative).
    Compared against what actually happened (never rotated).
    """
    cdata = (
        creative_df[
            (creative_df["channel"] == channel)
            & (creative_df["creative_id"] == creative_id)
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if cdata.empty:
        return {"error": f"No data found for {creative_id}"}

    n_days     = len(cdata)
    ctr_actual = cdata["ctr"].values
    impr       = cdata["impressions"].values

    ch_cvr = float(campaigns_df[campaigns_df["channel"] == channel]["cvr"].mean())

    rot = int(np.clip(rotation_day, 0, n_days - 1))

    # ── Actual (never rotated) ────────────────────────────────────────────────
    actual_clicks = float(np.sum(impr * ctr_actual))
    actual_convs  = actual_clicks * ch_cvr
    actual_avg_ctr = float(np.mean(ctr_actual))

    # ── Counterfactual (rotated at day `rot`) ─────────────────────────────────
    cf_ctr = np.concatenate([
        ctr_actual[:rot],
        np.full(n_days - rot, initial_ctr_obs),
    ])
    cf_clicks = float(np.sum(impr * cf_ctr))
    cf_convs  = cf_clicks * ch_cvr
    cf_avg_ctr = float(np.mean(cf_ctr))

    extra_clicks = cf_clicks - actual_clicks
    extra_convs  = cf_convs  - actual_convs
    revenue_impact = extra_convs * aov

    return {
        "rotation_day":            rot,
        "onset_day":               onset_day,
        "n_campaign_days":         n_days,
        "days_remaining_post_rot": n_days - rot,
        "actual_avg_ctr_pct":      actual_avg_ctr  * 100,
        "cf_avg_ctr_pct":          cf_avg_ctr      * 100,
        "ctr_lift_pp":             (cf_avg_ctr - actual_avg_ctr) * 100,
        "initial_ctr_pct":         initial_ctr_obs * 100,
        "extra_clicks":            extra_clicks,
        "extra_conversions":       extra_convs,
        "revenue_impact_usd":      revenue_impact,
        "is_improvement":          extra_convs > 0,
    }


# ── LLM Narrative ─────────────────────────────────────────────────────────────

_WHATIF_SYSTEM = (
    "You are a senior media strategist explaining a scenario analysis to a marketing "
    "director. Write 3-4 sentences of clear, direct prose. Reference the exact numbers "
    "provided. Identify which lever delivers the most value and why. No bullet points. "
    "No jargon."
)


def build_whatif_prompt(
    budget: dict,
    freq: dict,
    creative: dict,
) -> str:
    src  = budget["source_channel"].replace("_", " ").title()
    dst  = budget["dest_channel"].upper()
    return (
        f"WHAT-IF SCENARIO ANALYSIS\n\n"
        f"Lever 1 — Budget Reallocation: Shift {budget['shift_pct']*100:.0f}% of {src} "
        f"budget (${budget['shifted_spend_usd']:,.0f}) to {dst}.\n"
        f"  {src} late CPA: ${budget['source_late_cpa']:.0f} | {dst} late CPA: "
        f"${budget['dest_late_cpa']:.0f}\n"
        f"  Net additional conversions: {budget['net_extra_convs']:+.1f}\n"
        f"  Blended CPA: ${budget['current_blended_cpa']:.2f} → "
        f"${budget['projected_blended_cpa']:.2f} ({budget['cpa_delta_pct']:+.1f}%)\n"
        f"  Revenue impact: ${budget['revenue_impact_usd']:+,.0f}\n\n"
        f"Lever 2 — Frequency Cap: Cap Programmatic Display at {freq['new_cap']:.0f} "
        f"(from current avg {freq['current_avg_freq']:.1f}; saturation breakpoint "
        f"{freq['breakpoint_freq']:.1f}).\n"
        f"  Model CPA: ${freq['model_current_cpa']:.2f} → "
        f"${freq['model_projected_cpa']:.2f} ({freq['cpa_reduction_pct']:+.1f}%)\n"
        f"  Estimated 30-day savings: ${freq['savings_30d_usd']:+,.0f}\n\n"
        f"Lever 3 — Creative Rotation: Rotate social_v1 at campaign day "
        f"{creative['rotation_day']} (decay onset: day {creative['onset_day']}; "
        f"never rotated in reality).\n"
        f"  Campaign avg CTR: {creative['actual_avg_ctr_pct']:.2f}% → "
        f"{creative['cf_avg_ctr_pct']:.2f}% (+{creative['ctr_lift_pp']:.2f}pp)\n"
        f"  Extra conversions: {creative['extra_conversions']:+.1f} | "
        f"Revenue impact: ${creative['revenue_impact_usd']:+,.0f}\n\n"
        f"Write a 3-4 sentence strategic summary of this combined scenario."
    )


def _extract_narrative_text(raw: str) -> str:
    """
    Some providers (Gemini with JSON mode) wrap prose in a JSON envelope.
    Try to unwrap; fall back to returning the raw string if parsing fails.
    """
    raw = raw.strip()
    # Strip markdown fences
    import re
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if m:
        raw = m.group(1).strip()
    try:
        obj = json.loads(raw)
        # Return the first string value found in the JSON object
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, str) and len(v) > 20:
                    return v
        if isinstance(obj, str):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


def generate_whatif_narrative(
    budget: dict,
    freq: dict,
    creative: dict,
    llm_client,
    use_cache: bool = True,
) -> str:
    """
    Call the LLM (with cache) to narrate the what-if scenario.
    Returns plain text narrative, or an error message if the call fails.
    """
    from intelligence.cache import get_cached, save_to_cache

    system_prompt = _WHATIF_SYSTEM
    user_prompt   = build_whatif_prompt(budget, freq, creative)

    if use_cache:
        cached = get_cached(system_prompt, user_prompt)
        if cached:
            return _extract_narrative_text(cached)

    try:
        raw = llm_client.generate(system_prompt, user_prompt, max_tokens=512)
    except Exception as exc:
        return f"[LLM call failed: {exc}]"

    if use_cache:
        save_to_cache(system_prompt, user_prompt, raw)

    return _extract_narrative_text(raw)
