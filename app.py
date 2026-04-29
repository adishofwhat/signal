"""
SIGNAL — Campaign Diagnostic Intelligence Engine
Streamlit UI — Phase 4 + 5

Tabs:
  1. Executive Overview
  2. Diagnostic Deep Dive
  3. What-If Simulator
  4. Action Plan
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import CAMPAIGN_CONFIG, LLM_CONFIG, UI_CONFIG
from diagnostics.creative_fatigue import detect as detect_fatigue
from diagnostics.frequency_saturation import detect as detect_saturation
from diagnostics.daypart_analysis import detect as detect_daypart
from diagnostics.channel_cannibalization import detect as detect_cannibalization
from diagnostics.budget_efficiency import detect as detect_efficiency
from intelligence.llm_client import LLMClient, LLMProvider
from intelligence.whatif import (
    project_budget_reallocation,
    project_freq_cap,
    project_creative_rotation,
    generate_whatif_narrative,
)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SIGNAL — Campaign Diagnostic Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Root dark theme ── */
.stApp {
    background-color: #0a0a0a !important;
    color: #e8e8e8 !important;
}
.main .block-container {
    background-color: #0a0a0a;
    padding-top: 0;
    padding-bottom: 4rem;
    max-width: 1400px;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0a0a0a !important;
    border-bottom: 1px solid #1e1e1e !important;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: #555555 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 12px 24px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px;
    transition: color 0.15s, border-color 0.15s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #aaaaaa !important;
    background-color: #0f0f0f !important;
}
.stTabs [aria-selected="true"] {
    color: #c8a55a !important;
    border-bottom: 2px solid #c8a55a !important;
    background-color: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background-color: #0a0a0a !important;
    padding-top: 32px;
}

/* ── Expanders ── */
details {
    background-color: #0d0d0d !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 6px !important;
    margin-bottom: 10px;
}
details summary {
    background-color: #111111 !important;
    color: #cccccc !important;
    padding: 12px 16px !important;
    border-radius: 6px;
    font-size: 14px !important;
    font-weight: 500;
    cursor: pointer;
}
details[open] summary {
    border-radius: 6px 6px 0 0;
    border-bottom: 1px solid #1e1e1e;
}
details > div {
    padding: 16px !important;
}

/* ── Streamlit expander override ── */
[data-testid="stExpander"] {
    background-color: #0d0d0d !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 6px !important;
    margin-bottom: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #cccccc !important;
}
[data-testid="stExpanderDetails"] {
    background-color: #0a0a0a !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    background-color: #0d0d0d !important;
}
.dvn-scroller {
    background-color: #0d0d0d !important;
}

/* ── Metrics (native st.metric) ── */
[data-testid="metric-container"] {
    background-color: #111111 !important;
    border: 1px solid #222222 !important;
    border-radius: 8px !important;
    padding: 20px 24px !important;
}
[data-testid="stMetricLabel"] p {
    color: #666666 !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}
[data-testid="stMetricValue"] {
    color: #e8e8e8 !important;
    font-size: 28px !important;
    font-weight: 600 !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #c8a55a !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 24px !important;
    letter-spacing: 0.5px;
}
.stButton > button:hover {
    background-color: #d4b570 !important;
    color: #0a0a0a !important;
}

/* ── Alerts / warnings ── */
[data-testid="stAlert"] {
    background-color: #1a1200 !important;
    border-color: #f39c12 !important;
    color: #e8e8e8 !important;
    border-radius: 6px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #c8a55a !important;
}

/* ── Horizontal rule ── */
hr {
    border-color: #1e1e1e !important;
    margin: 24px 0 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d0d0d !important;
    border-right: 1px solid #1e1e1e !important;
}

/* Remove any lingering blue focus rings */
*:focus { outline-color: #c8a55a !important; }
a { color: #c8a55a !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR   = ROOT / "data"
CHARTS_DIR = DATA_DIR / "charts"
AOV        = CAMPAIGN_CONFIG.get("avg_order_value", 120)

SEVERITY_COLORS = {
    "critical": "#e74c3c",
    "warning":  "#f39c12",
    "info":     "#2ecc71",
}
OWNER_LABELS = {
    "media_buyer":   "Media Buyer",
    "creative_team": "Creative Team",
    "analyst":       "Analyst",
}
TIMELINE_LABELS = {
    "immediate":  "Immediate",
    "this_week":  "This Week",
    "this_month": "This Month",
}
TIMELINE_COLORS = {
    "immediate":  "#e74c3c",
    "this_week":  "#f39c12",
    "this_month": "#2ecc71",
}

# Keywords used to match LLM findings → diagnostic types
DIAGNOSTIC_KEYWORDS = {
    "creative_fatigue":      ["fatigue", "creative", "social_v1", "ctr", "rotate", "social"],
    "frequency_saturation":  ["frequency", "saturation", "cap", "display"],
    "daypart_opportunity":   ["daypart", "prime", "late night", "cvr", "bid multiplier"],
    "channel_cannibalization": ["cannib", "overlap", "attribution", "search spend"],
    "budget_efficiency":     ["diminishing", "youtube", "ctv", "marginal cpa", "reallocate"],
}

# Ordered list of diagnostics for Tab 2
DIAGNOSTIC_META = [
    {
        "key":        "creative_fatigue",
        "label":      "Creative Fatigue",
        "channel":    "Paid Social",
        "chart_file": "creative_fatigue__paid_social__0.html",
    },
    {
        "key":        "frequency_saturation",
        "label":      "Frequency Saturation",
        "channel":    "Programmatic Display",
        "chart_file": "frequency_saturation__programmatic_display__0.html",
    },
    {
        "key":        "daypart_opportunity",
        "label":      "Daypart Opportunity",
        "channel":    "Paid Search",
        "chart_file": "daypart_opportunity__paid_search__0.html",
    },
    {
        "key":        "channel_cannibalization",
        "label":      "Channel Cannibalization",
        "channel":    "Paid Search → Display",
        "chart_file": "channel_cannibalization__paid_search_to_programmatic_display__0.html",
    },
    {
        "key":        "budget_efficiency",
        "label":      "Budget Efficiency",
        "channel":    "YouTube / CTV",
        "chart_file": "budget_efficiency__youtube__0.html",
    },
]


# ── Data & diagnostic loading (cached) ───────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_campaigns() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "campaigns.csv", parse_dates=["date"])


@st.cache_data(show_spinner=False)
def _load_creative() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "creative_performance.csv", parse_dates=["date"])


@st.cache_data(show_spinner=False)
def _load_daypart() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "daypart_performance.csv", parse_dates=["date"])


@st.cache_resource(show_spinner=False)
def _run_diagnostics():
    """
    Run all five diagnostic modules once and cache the results for the session.
    Uses cache_resource so Plotly figure objects are not pickled.
    """
    camps    = _load_campaigns()
    creative = _load_creative()
    daypart  = _load_daypart()

    results = []
    results.extend(detect_fatigue(creative, camps, aov=AOV))
    sat = detect_saturation(camps, aov=AOV)
    if sat:
        results.append(sat)
    results.extend(detect_daypart(daypart, camps, aov=AOV))
    results.extend(detect_cannibalization(camps, aov=AOV))
    results.extend(detect_efficiency(camps, aov=AOV))
    return results


def _load_llm_cache() -> dict | None:
    """
    Read the Stage-1 LLM output from the on-disk cache.
    Returns the parsed JSON dict, or None if no cache exists.
    """
    cache_dir = ROOT / "cache" / "llm_responses"
    if not cache_dir.exists():
        return None
    json_files = sorted(cache_dir.glob("*.json"))
    if not json_files:
        return None
    # Take the most recently modified file
    latest = max(json_files, key=lambda f: f.stat().st_mtime)
    try:
        raw = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return None
    response_text = raw.get("response", "")
    if not response_text:
        return None
    # The response field is the raw LLM output (already JSON or wrapped in fences)
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", response_text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except Exception:
                pass
    return None


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _severity_badge(severity: str) -> str:
    color = SEVERITY_COLORS.get(severity, "#888888")
    return (
        f'<span style="background-color:{color};color:#0a0a0a;'
        f'font-size:10px;font-weight:700;padding:3px 9px;border-radius:3px;'
        f'letter-spacing:0.5px;text-transform:uppercase;">'
        f'{severity}</span>'
    )


def _owner_badge(owner: str) -> str:
    label = OWNER_LABELS.get(owner, owner)
    return (
        f'<span style="background-color:#1a1a1a;color:#888888;'
        f'font-size:11px;font-weight:500;padding:3px 10px;'
        f'border-radius:10px;border:1px solid #2a2a2a;">'
        f'{label}</span>'
    )


def _conf_badge(confidence: str) -> str:
    color = {"high": "#2ecc71", "medium": "#f39c12", "low": "#e74c3c"}.get(confidence, "#888")
    return (
        f'<span style="color:{color};font-size:12px;font-weight:500;">'
        f'Confidence: {confidence}</span>'
    )


# ── Section helpers ───────────────────────────────────────────────────────────

def _section_label(text: str):
    st.markdown(
        f'<div style="font-size:10px;color:#555555;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:10px;">{text}</div>',
        unsafe_allow_html=True,
    )


def _render_header():
    st.markdown(
        """
        <div style="padding:32px 0 24px 0;border-bottom:1px solid #1a1a1a;margin-bottom:32px;">
            <div style="font-size:40px;font-weight:700;color:#c8a55a;letter-spacing:8px;line-height:1;">
                SIGNAL
            </div>
            <div style="font-size:12px;color:#666666;letter-spacing:3px;text-transform:uppercase;margin-top:8px;">
                Campaign Diagnostic Intelligence
            </div>
            <div style="font-size:13px;color:#3a3a3a;margin-top:6px;">
                Meridian Outdoor Co. &mdash; Spring 2026 Performance Campaign
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, sub: str = "", sub_positive: bool = True):
    sub_color = "#2ecc71" if sub_positive else "#e74c3c"
    sub_html  = (
        f'<div style="font-size:12px;color:{sub_color};margin-top:6px;">{sub}</div>'
        if sub else ""
    )
    st.markdown(
        f"""
        <div style="background-color:#111111;border:1px solid #1e1e1e;border-radius:8px;
                    padding:20px 24px;height:115px;">
            <div style="font-size:10px;color:#555555;text-transform:uppercase;
                        letter-spacing:1.5px;margin-bottom:10px;">{label}</div>
            <div style="font-size:28px;font-weight:600;color:#e8e8e8;line-height:1;">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_finding_cards(findings: list[dict]):
    for f in sorted(findings, key=lambda x: x.get("rank", 99)):
        sev    = f.get("severity", "info")
        color  = SEVERITY_COLORS.get(sev, "#888888")
        title  = f.get("title", "")
        impact = f.get("estimated_impact", "")
        conf   = f.get("confidence", "")
        narr   = f.get("narrative", "")
        rec    = f.get("recommendation", "")

        with st.expander(f"#{f.get('rank', '?')}  —  {title}", expanded=False):
            badge_col, impact_col, conf_col = st.columns([2, 1, 1])
            with badge_col:
                st.markdown(_severity_badge(sev), unsafe_allow_html=True)
            with impact_col:
                st.markdown(
                    f'<span style="color:#c8a55a;font-size:13px;font-weight:600;">{impact}</span>',
                    unsafe_allow_html=True,
                )
            with conf_col:
                st.markdown(_conf_badge(conf), unsafe_allow_html=True)

            st.markdown(
                f'<div style="font-size:13px;color:#aaaaaa;line-height:1.75;'
                f'margin-top:14px;margin-bottom:16px;">{narr}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background-color:#0f0f0f;border-left:3px solid {color};
                            border-radius:0 4px 4px 0;padding:12px 16px;">
                    <div style="font-size:10px;color:#444;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:4px;">Recommendation</div>
                    <div style="font-size:13px;color:#e0e0e0;line-height:1.6;">{rec}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _match_llm_finding(key: str, llm_findings: list[dict]) -> dict | None:
    """Return the LLM finding that best matches the given diagnostic key."""
    keywords = DIAGNOSTIC_KEYWORDS.get(key, [])
    for lf in llm_findings:
        text = (lf.get("title", "") + " " + lf.get("narrative", "")).lower()
        if any(kw in text for kw in keywords):
            return lf
    return None


def _render_evidence_table(evidence: dict):
    # Coerce all values to JSON-safe types so st.json() renders cleanly
    safe: dict = {}
    for k, v in evidence.items():
        label = k.replace("_", " ").title()
        if v is None:
            safe[label] = None
        elif isinstance(v, float):
            safe[label] = round(v, 6)
        elif isinstance(v, (int, bool, str)):
            safe[label] = v
        elif isinstance(v, dict):
            safe[label] = {str(dk): (round(dv, 6) if isinstance(dv, float) else dv)
                           for dk, dv in v.items()}
        elif isinstance(v, list):
            safe[label] = [round(x, 6) if isinstance(x, float) else x for x in v]
        else:
            safe[label] = str(v)
    st.json(safe, expanded=True)


# ═════════════════════════════════════════════════════════════════════════════
# What-If Simulator Tab
# ═════════════════════════════════════════════════════════════════════════════

def _wi_delta_html(label: str, current: str, projected: str, delta: str,
                   is_improvement: bool, lower_is_better: bool = False) -> str:
    """
    Render a single before/after metric card.
    lower_is_better=True  → green arrow points ▼ when is_improvement (e.g. CPA)
    lower_is_better=False → green arrow points ▲ when is_improvement (e.g. revenue)
    """
    d_color = "#2ecc71" if is_improvement else "#e74c3c"
    if lower_is_better:
        arrow = "▼" if is_improvement else "▲"
    else:
        arrow = "▲" if is_improvement else "▼"
    return (
        f"<div style='background-color:#0d0d0d;border:1px solid #1a1a1a;"
        f"border-radius:6px;padding:14px 16px;'>"
        f"<div style='font-size:10px;color:#444444;text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:8px;'>{label}</div>"
        f"<div style='font-size:12px;color:#666666;margin-bottom:2px;'>Current: "
        f"<span style='color:#aaaaaa;'>{current}</span></div>"
        f"<div style='font-size:12px;color:#666666;margin-bottom:6px;'>Projected: "
        f"<span style='color:#e0e0e0;font-weight:600;'>{projected}</span></div>"
        f"<div style='font-size:14px;font-weight:700;color:{d_color};'>"
        f"{arrow} {delta}</div>"
        f"</div>"
    )


def _wi_lever_header(icon: str, title: str, subtitle: str):
    st.markdown(
        f"<p style='font-size:11px;color:#555555;text-transform:uppercase;"
        f"letter-spacing:1.5px;margin-bottom:4px;'>{icon} {title}</p>"
        f"<p style='font-size:11px;color:#3a3a3a;margin-bottom:12px;'>{subtitle}</p>",
        unsafe_allow_html=True,
    )


def _wi_section_divider():
    st.markdown(
        "<div style='height:1px;background-color:#1a1a1a;margin:20px 0;'></div>",
        unsafe_allow_html=True,
    )


def _render_whatif_tab(
    diag_results: list,
    campaigns_df: pd.DataFrame,
    creative_df: pd.DataFrame,
):
    AOV = CAMPAIGN_CONFIG.get("avg_order_value", 120)

    # ── Extract model parameters from cached DiagnosticResult evidence ─────────
    by_name: dict = {}
    for r in diag_results:
        by_name.setdefault(r.name, []).append(r)

    budget_result = (by_name.get("budget_efficiency") or [None])[0]
    freq_result   = (by_name.get("frequency_saturation") or [None])[0]
    fat_result    = (by_name.get("creative_fatigue") or [None])[0]

    # Budget evidence
    if budget_result:
        all_ch_late_cpa: dict = budget_result.evidence.get("all_channels_late_cpa", {})
    else:
        all_ch_late_cpa = {}

    # Frequency saturation evidence
    if freq_result:
        bp_freq  = float(freq_result.evidence.get("breakpoint_frequency", 7.1))
        val_bp   = float(freq_result.evidence.get("cpa_at_breakpoint", 45.0))
        slope1   = float(freq_result.evidence.get("slope_segment1", 1.5))
        slope2   = float(freq_result.evidence.get("slope_segment2", 8.0))
        cur_freq = float(freq_result.evidence.get("current_avg_frequency_14d", 9.7))
    else:
        bp_freq, val_bp, slope1, slope2, cur_freq = 7.1, 45.0, 1.5, 8.0, 9.7

    # Creative fatigue evidence
    if fat_result:
        decay_lam   = float(fat_result.evidence.get("decay_lambda", 0.048))
        a_amp       = float(fat_result.evidence.get("a_amplitude", 0.015))
        c_fl        = float(fat_result.evidence.get("c_floor", 0.012))
        onset_day   = int(fat_result.evidence.get("decay_onset_day", 33))
        init_ctr    = float(fat_result.evidence.get("initial_ctr_observed", 0.028))
        opt_rot_day = fat_result.evidence.get("optimal_rotation_campaign_day")
        opt_rot_day = int(opt_rot_day) if opt_rot_day else 46
        creative_id = str(fat_result.evidence.get("creative_id", "social_v1"))
    else:
        decay_lam, a_amp, c_fl, onset_day = 0.048, 0.015, 0.012, 33
        init_ctr, opt_rot_day, creative_id = 0.028, 46, "social_v1"

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:22px;font-weight:700;color:#e8e8e8;margin-bottom:4px;'>"
        "What-If Simulator</p>"
        "<p style='font-size:12px;color:#555555;margin-bottom:28px;'>"
        "Adjust levers — projections update instantly. Click "
        "<strong style='color:#c8a55a;'>Generate Strategic Analysis</strong> "
        "for an LLM-generated narrative grounded in these numbers.</p>",
        unsafe_allow_html=True,
    )

    # ── Two-column layout: sliders (40%) | results (60%) ──────────────────────
    left_col, right_col = st.columns([4, 6], gap="large")

    # ───────────────────────────── LEFT: SLIDERS ──────────────────────────────
    with left_col:

        # ── Lever 1: Budget Reallocation ──────────────────────────────────────
        _wi_lever_header(
            "📦", "Budget Reallocation",
            f"Shift YouTube spend → CTV  "
            f"(YouTube late CPA: ${all_ch_late_cpa.get('youtube', 151):.0f}  |  "
            f"CTV late CPA: ${all_ch_late_cpa.get('ctv', 73):.0f})",
        )
        shift_pct = st.slider(
            "% of YouTube budget to shift to CTV",
            min_value=0, max_value=40, value=20, step=1,
            key="wi_shift_pct",
            format="%d%%",
        )

        _wi_section_divider()

        # ── Lever 2: Frequency Cap ────────────────────────────────────────────
        _wi_lever_header(
            "📡", "Frequency Cap — Programmatic Display",
            f"Current avg: {cur_freq:.1f}  |  Saturation breakpoint: {bp_freq:.1f}",
        )
        new_cap = st.slider(
            "Set frequency cap",
            min_value=3.0, max_value=12.0, value=float(round(bp_freq)),
            step=0.5, key="wi_new_cap",
        )

        _wi_section_divider()

        # ── Lever 3: Creative Rotation ────────────────────────────────────────
        _wi_lever_header(
            "🎨", "Creative Rotation — social_v1",
            f"Decay onset: day {onset_day}  |  Optimal rotation: day {opt_rot_day}  "
            f"|  Initial CTR: {init_ctr*100:.2f}%",
        )
        rotation_day = st.slider(
            "Rotate social_v1 at campaign day",
            min_value=20, max_value=80, value=opt_rot_day, step=1,
            key="wi_rotation_day",
        )

        _wi_section_divider()

        # ── Generate narrative button ─────────────────────────────────────────
        generate_clicked = st.button(
            "⚡  Generate Strategic Analysis",
            key="wi_generate_btn",
            use_container_width=True,
        )

    # ───────────────────────────── RIGHT: RESULTS ─────────────────────────────
    with right_col:

        # ── Compute projections ───────────────────────────────────────────────
        budget_proj = project_budget_reallocation(
            campaigns_df,
            source_channel="youtube",
            dest_channel="ctv",
            shift_pct=shift_pct / 100.0,
            all_channels_late_cpa=all_ch_late_cpa,
            aov=AOV,
        )
        freq_proj = project_freq_cap(
            campaigns_df,
            new_cap=new_cap,
            breakpoint_freq=bp_freq,
            cpa_at_breakpoint=val_bp,
            slope_segment1=slope1,
            slope_segment2=slope2,
            current_avg_freq=cur_freq,
            aov=AOV,
        )
        creative_proj = project_creative_rotation(
            creative_df,
            campaigns_df,
            rotation_day=rotation_day,
            decay_lambda=decay_lam,
            a_amplitude=a_amp,
            c_floor=c_fl,
            onset_day=onset_day,
            initial_ctr_obs=init_ctr,
            creative_id=creative_id,
            aov=AOV,
        )

        # ── Lever 1 results ───────────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:11px;color:#c8a55a;text-transform:uppercase;"
            "letter-spacing:1.5px;margin-bottom:10px;'>📦 Budget Reallocation</p>",
            unsafe_allow_html=True,
        )
        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown(
                _wi_delta_html(
                    "Net Conversions",
                    f"{int(campaigns_df['conversions'].sum()):,}",
                    f"{int(campaigns_df['conversions'].sum() + budget_proj['net_extra_convs']):,}",
                    f"{budget_proj['net_extra_convs']:+.0f}",
                    budget_proj["is_improvement"],
                ),
                unsafe_allow_html=True,
            )
        with b2:
            cpa_chg = budget_proj["projected_blended_cpa"] - budget_proj["current_blended_cpa"]
            st.markdown(
                _wi_delta_html(
                    "Blended CPA",
                    f"${budget_proj['current_blended_cpa']:.2f}",
                    f"${budget_proj['projected_blended_cpa']:.2f}",
                    f"{cpa_chg:+.2f} ({budget_proj['cpa_delta_pct']:+.1f}%)",
                    budget_proj["cpa_delta_pct"] < 0,
                    lower_is_better=True,
                ),
                unsafe_allow_html=True,
            )
        with b3:
            st.markdown(
                _wi_delta_html(
                    "Revenue Impact",
                    "Baseline",
                    f"${abs(budget_proj['revenue_impact_usd']):,.0f}",
                    f"${budget_proj['revenue_impact_usd']:+,.0f}",
                    budget_proj["is_improvement"],
                ),
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<p style='font-size:11px;color:#3a3a3a;margin-top:8px;'>"
            f"Shifting ${budget_proj['shifted_spend_usd']:,.0f} from YouTube "
            f"(late CPA ${budget_proj['source_late_cpa']:.0f}) to CTV "
            f"(late CPA ${budget_proj['dest_late_cpa']:.0f})</p>",
            unsafe_allow_html=True,
        )

        _wi_section_divider()

        # ── Lever 2 results ───────────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:11px;color:#c8a55a;text-transform:uppercase;"
            "letter-spacing:1.5px;margin-bottom:10px;'>📡 Frequency Cap</p>",
            unsafe_allow_html=True,
        )
        f1, f2, f3 = st.columns(3)
        with f1:
            f_cpa_chg = freq_proj["model_projected_cpa"] - freq_proj["model_current_cpa"]
            st.markdown(
                _wi_delta_html(
                    "Display CPA (model)",
                    f"${freq_proj['model_current_cpa']:.2f}",
                    f"${freq_proj['model_projected_cpa']:.2f}",
                    f"{f_cpa_chg:+.2f} ({-freq_proj['cpa_reduction_pct']:+.1f}%)",
                    freq_proj["is_improvement"],
                    lower_is_better=True,
                ),
                unsafe_allow_html=True,
            )
        with f2:
            st.markdown(
                _wi_delta_html(
                    "Extra Daily Convs",
                    "Baseline",
                    f"+{freq_proj['extra_daily_convs']:.1f}/day",
                    f"+{freq_proj['extra_daily_convs']:.1f}",
                    freq_proj["is_improvement"],
                ),
                unsafe_allow_html=True,
            )
        with f3:
            st.markdown(
                _wi_delta_html(
                    "30-Day Value",
                    "Baseline",
                    f"${abs(freq_proj['savings_30d_usd']):,.0f}",
                    f"${freq_proj['savings_30d_usd']:+,.0f}",
                    freq_proj["is_improvement"],
                ),
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<p style='font-size:11px;color:#3a3a3a;margin-top:8px;'>"
            f"Cap frequency at {new_cap:.1f} vs current {cur_freq:.1f} "
            f"(saturation breakpoint: {bp_freq:.1f})</p>",
            unsafe_allow_html=True,
        )

        _wi_section_divider()

        # ── Lever 3 results ───────────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:11px;color:#c8a55a;text-transform:uppercase;"
            "letter-spacing:1.5px;margin-bottom:10px;'>🎨 Creative Rotation</p>",
            unsafe_allow_html=True,
        )
        c1c, c2c, c3c = st.columns(3)
        with c1c:
            st.markdown(
                _wi_delta_html(
                    "Campaign Avg CTR",
                    f"{creative_proj['actual_avg_ctr_pct']:.2f}%",
                    f"{creative_proj['cf_avg_ctr_pct']:.2f}%",
                    f"+{creative_proj['ctr_lift_pp']:.2f}pp",
                    creative_proj["is_improvement"],
                ),
                unsafe_allow_html=True,
            )
        with c2c:
            st.markdown(
                _wi_delta_html(
                    "Extra Conversions",
                    "Baseline",
                    f"+{creative_proj['extra_conversions']:.0f}",
                    f"+{creative_proj['extra_conversions']:.0f}",
                    creative_proj["is_improvement"],
                ),
                unsafe_allow_html=True,
            )
        with c3c:
            st.markdown(
                _wi_delta_html(
                    "Revenue Impact",
                    "Baseline",
                    f"${abs(creative_proj['revenue_impact_usd']):,.0f}",
                    f"${creative_proj['revenue_impact_usd']:+,.0f}",
                    creative_proj["is_improvement"],
                ),
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<p style='font-size:11px;color:#3a3a3a;margin-top:8px;'>"
            f"Rotate {creative_id} at day {rotation_day} — replacement performs "
            f"at initial CTR {init_ctr*100:.2f}% for remaining "
            f"{creative_proj['days_remaining_post_rot']} days</p>",
            unsafe_allow_html=True,
        )

    # ── LLM Strategic Analysis ─────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='height:1px;background-color:#1a1a1a;margin:4px 0 24px 0;'></div>",
        unsafe_allow_html=True,
    )

    if generate_clicked:
        provider_str = LLM_CONFIG.get("provider", "claude")
        api_key      = LLM_CONFIG.get("api_key", "")
        model        = LLM_CONFIG.get("model")

        if not api_key:
            # Try cache first before complaining about the key
            from intelligence.cache import get_cached
            from intelligence.whatif import _WHATIF_SYSTEM, build_whatif_prompt
            sys_p = _WHATIF_SYSTEM
            usr_p = build_whatif_prompt(budget_proj, freq_proj, creative_proj)
            cached = get_cached(sys_p, usr_p)
            if cached:
                st.session_state["wi_narrative"] = cached
            else:
                st.session_state["wi_narrative"] = (
                    "⚠️  Set `LLM_API_KEY` (or `ANTHROPIC_API_KEY`) environment variable "
                    "to generate live analysis. The statistical projections above are "
                    "computed directly from the fitted models."
                )
        else:
            with st.spinner("Generating strategic analysis..."):
                try:
                    client = LLMClient(
                        provider=LLMProvider(provider_str),
                        api_key=api_key,
                        model=model,
                    )
                    narrative = generate_whatif_narrative(
                        budget_proj, freq_proj, creative_proj,
                        llm_client=client,
                        use_cache=True,
                    )
                    st.session_state["wi_narrative"] = narrative
                except Exception as e:
                    st.session_state["wi_narrative"] = f"[Error generating analysis: {e}]"

    if "wi_narrative" in st.session_state and st.session_state["wi_narrative"]:
        _section_label("Strategic Analysis")
        # Guard against JSON-wrapped responses from any LLM provider
        raw_narr = st.session_state["wi_narrative"]
        try:
            parsed = json.loads(raw_narr)
            if isinstance(parsed, dict):
                display_narr = next(
                    (v for v in parsed.values() if isinstance(v, str) and len(v) > 20),
                    raw_narr,
                )
            else:
                display_narr = raw_narr
        except (json.JSONDecodeError, ValueError):
            display_narr = raw_narr

        st.markdown(
            f"<div style='background-color:#111111;border:1px solid #1e1e1e;"
            f"border-left:3px solid #c8a55a;border-radius:0 8px 8px 0;"
            f"padding:18px 22px;font-size:14px;line-height:1.75;color:#cccccc;'>"
            f"{display_narr}</div>",
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Main app
# ═════════════════════════════════════════════════════════════════════════════

def main():
    _render_header()

    # ── Load data (CSV) ───────────────────────────────────────────────────────
    with st.spinner("Loading data..."):
        campaigns = _load_campaigns()

    # ── Run diagnostics (cached on first load) ────────────────────────────────
    with st.spinner("Running diagnostics..."):
        diag_results = _run_diagnostics()

    # ── Load LLM cache ────────────────────────────────────────────────────────
    llm_output = _load_llm_cache()

    # ── Summary metrics ───────────────────────────────────────────────────────
    total_spend       = float(campaigns["spend"].sum())
    total_conversions = int(campaigns["conversions"].sum())
    blended_cpa       = total_spend / total_conversions if total_conversions else 0.0
    blended_roas      = (total_conversions * AOV) / total_spend if total_spend else 0.0

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "  📊  Executive Overview  ",
        "  🔬  Diagnostic Deep Dive  ",
        "  🎛  What-If Simulator  ",
        "  🗂  Action Plan  ",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Executive Overview
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _metric_card("Total Spend", f"${total_spend:,.0f}")
        with c2:
            _metric_card("Total Conversions", f"{total_conversions:,}")
        with c3:
            _metric_card("Blended CPA", f"${blended_cpa:.2f}")
        with c4:
            _metric_card(
                "Blended ROAS",
                f"{blended_roas:.2f}x",
                sub="Above break-even" if blended_roas >= 1 else "Below break-even",
                sub_positive=blended_roas >= 1,
            )

        st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)

        if llm_output is None:
            st.warning(
                "No cached LLM response found. "
                "Run `python3 run_intelligence.py` to generate the analysis, then refresh."
            )
        else:
            # Executive summary
            _section_label("AI Executive Summary")
            st.markdown(
                f"""
                <div style="background-color:#111111;border:1px solid #1e1e1e;
                            border-left:3px solid #c8a55a;border-radius:0 8px 8px 0;
                            padding:20px 24px;font-size:15px;line-height:1.75;
                            color:#cccccc;margin-bottom:32px;">
                    {llm_output.get('executive_summary', '')}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Cross-channel insights
            cross = llm_output.get("cross_channel_insights", "")
            if cross:
                _section_label("Cross-Channel Dynamics")
                st.markdown(
                    f"""
                    <div style="background-color:#0d0d0d;border:1px solid #1a1a1a;
                                border-radius:8px;padding:16px 20px;font-size:13px;
                                line-height:1.75;color:#888888;margin-bottom:32px;">
                        {cross}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Ranked findings
            _section_label("Ranked Findings — by Estimated Impact")
            _render_finding_cards(llm_output.get("findings", []))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Diagnostic Deep Dive
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        # Build lookup: diagnostic name → list of DiagnosticResult
        by_name: dict[str, list] = {}
        for r in diag_results:
            by_name.setdefault(r.name, []).append(r)

        llm_findings = llm_output.get("findings", []) if llm_output else []

        for meta in DIAGNOSTIC_META:
            key        = meta["key"]
            label      = meta["label"]
            channel    = meta["channel"]
            chart_file = meta["chart_file"]

            diag_list = by_name.get(key, [])
            diag      = diag_list[0] if diag_list else None

            severity   = diag.severity if diag else "info"
            sev_color  = SEVERITY_COLORS.get(severity, "#888888")
            impact_str = f"${diag.estimated_impact_usd:,.0f}" if diag else "—"

            with st.expander(
                f"{label}  ·  {channel}  ·  {severity.upper()}  ·  Impact: {impact_str}",
                expanded=False,
            ):
                if diag:
                    # One-liner finding
                    st.markdown(
                        f"""
                        <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
                            {_severity_badge(severity)}
                            <span style="font-size:13px;color:#888888;">{diag.finding}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # LLM narrative & recommendation
                matched = _match_llm_finding(key, llm_findings)
                if matched:
                    st.markdown(
                        f"""
                        <div style="background-color:#111111;border-left:3px solid {sev_color};
                                    border-radius:0 6px 6px 0;padding:16px 18px;
                                    font-size:13px;line-height:1.75;color:#bbbbbb;
                                    margin-bottom:16px;">
                            {matched.get('narrative', '')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                        <div style="background-color:#0d0d0d;border:1px solid #1a1a1a;
                                    border-radius:6px;padding:12px 16px;margin-bottom:20px;">
                            <div style="font-size:10px;color:#444444;text-transform:uppercase;
                                        letter-spacing:1px;margin-bottom:6px;">Recommendation</div>
                            <div style="font-size:13px;color:#e0e0e0;line-height:1.6;">
                                {matched.get('recommendation', '')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Chart
                _section_label("Diagnostic Chart")

                chart_rendered = False

                # Prefer live Plotly figure from DiagnosticResult
                if diag and diag.charts:
                    fig = diag.charts[0]
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0a0a0a",
                        plot_bgcolor="#0a0a0a",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    chart_rendered = True

                # Fall back to pre-saved HTML
                if not chart_rendered:
                    html_path = CHARTS_DIR / chart_file
                    if html_path.exists():
                        html_content = html_path.read_text(encoding="utf-8")
                        components.html(html_content, height=480, scrolling=False)
                    else:
                        st.info("Chart not available. Ensure data/charts/ contains the saved HTML files.")

                # Statistical evidence
                if diag:
                    with st.expander("📐  Statistical Evidence", expanded=False):
                        _render_evidence_table(diag.evidence)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — What-If Simulator (Placeholder)
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        _render_whatif_tab(diag_results, campaigns, _load_creative())

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — Action Plan
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        if llm_output is None:
            st.warning(
                "No cached LLM response found. "
                "Run `python3 run_intelligence.py` to generate the analysis, then refresh."
            )
        else:
            action_plan = llm_output.get("action_plan", [])
            if not action_plan:
                st.info("No action plan found in the cached LLM output.")
            else:
                _section_label("Prioritized Action Plan · Generated from diagnostic findings")
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

                # Group by timeline
                grouped: dict[str, list] = {
                    "immediate":  [],
                    "this_week":  [],
                    "this_month": [],
                }
                for item in action_plan:
                    tl = item.get("timeline", "this_month")
                    grouped.setdefault(tl, []).append(item)

                for tl_key in ["immediate", "this_week", "this_month"]:
                    items = grouped.get(tl_key, [])
                    if not items:
                        continue

                    tl_color = TIMELINE_COLORS[tl_key]
                    tl_label = TIMELINE_LABELS[tl_key]

                    # Timeline divider
                    st.markdown(
                        f"""
                        <div style="display:flex;align-items:center;gap:12px;
                                    margin-top:28px;margin-bottom:16px;">
                            <div style="width:8px;height:8px;border-radius:50%;
                                        background-color:{tl_color};flex-shrink:0;"></div>
                            <div style="font-size:11px;text-transform:uppercase;
                                        letter-spacing:2px;color:{tl_color};font-weight:600;">
                                {tl_label}
                            </div>
                            <div style="flex:1;height:1px;background-color:#1a1a1a;"></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    for item in sorted(items, key=lambda x: x.get("priority", 99)):
                        priority    = item.get("priority", "")
                        action_text = item.get("action", "")
                        owner       = item.get("owner", "")

                        st.markdown(
                            f"""
                            <div style="background-color:#111111;border:1px solid #1a1a1a;
                                        border-radius:8px;padding:18px 20px;margin-bottom:10px;
                                        display:flex;align-items:flex-start;gap:16px;">
                                <div style="width:32px;height:32px;border-radius:50%;
                                            background-color:{tl_color};display:flex;
                                            align-items:center;justify-content:center;
                                            font-size:13px;font-weight:700;color:#0a0a0a;
                                            flex-shrink:0;">
                                    {priority}
                                </div>
                                <div style="flex:1;">
                                    <div style="font-size:14px;color:#e0e0e0;line-height:1.55;
                                                margin-bottom:10px;">
                                        {action_text}
                                    </div>
                                    {_owner_badge(owner)}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


if __name__ == "__main__":
    main()
