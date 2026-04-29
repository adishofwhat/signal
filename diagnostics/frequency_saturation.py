"""
Frequency saturation detection for programmatic_display.

Fits a piecewise linear CPA model over frequency bins using a grid search
over candidate breakpoints. Flags when current avg frequency exceeds the
saturation breakpoint and CPA above threshold is >20% higher.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import plotly.graph_objects as go

from diagnostics import DiagnosticResult

_DARK = "#111111"
_GOLD = "#c8a55a"


# ---------------------------------------------------------------------------
# Piecewise linear fit
# ---------------------------------------------------------------------------

def _fit_piecewise(freq: np.ndarray, cpa: np.ndarray, bp: float):
    """
    Fit a continuous piecewise linear model with a single breakpoint at bp.
    Returns (coeffs1, slope2, val_at_bp, sse).
    """
    mask1 = freq <= bp
    mask2 = freq > bp
    if mask1.sum() < 5 or mask2.sum() < 5:
        return None

    f1, c1 = freq[mask1], cpa[mask1]
    f2, c2 = freq[mask2], cpa[mask2]

    # Segment 1: c = β0 + β1*f  (OLS)
    A1 = np.column_stack([np.ones_like(f1), f1])
    coeffs1, _, _, _ = np.linalg.lstsq(A1, c1, rcond=None)
    val_at_bp = coeffs1[0] + coeffs1[1] * bp

    # Segment 2 (constrained continuous at bp): c = val_at_bp + β2*(f - bp)
    A2 = (f2 - bp).reshape(-1, 1)
    coeffs2, _, _, _ = np.linalg.lstsq(A2, c2 - val_at_bp, rcond=None)
    slope2 = float(coeffs2[0])

    pred1 = A1 @ coeffs1
    pred2 = val_at_bp + slope2 * (f2 - bp)
    sse = float(np.sum((c1 - pred1) ** 2) + np.sum((c2 - pred2) ** 2))

    return coeffs1, slope2, val_at_bp, sse


def _grid_search_breakpoint(freq: np.ndarray, cpa: np.ndarray, n_candidates: int = 80):
    """Return the breakpoint that minimises total piecewise SSE."""
    lo = np.percentile(freq, 10)
    hi = np.percentile(freq, 90)
    candidates = np.linspace(lo, hi, n_candidates)

    best = None
    for bp in candidates:
        result = _fit_piecewise(freq, cpa, bp)
        if result is None:
            continue
        coeffs1, slope2, val_at_bp, sse = result
        if best is None or sse < best[-1]:
            best = (bp, coeffs1, slope2, val_at_bp, sse)

    return best


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _make_chart(
    disp: pd.DataFrame,
    bp: float,
    coeffs1: np.ndarray,
    slope2: float,
    val_at_bp: float,
) -> go.Figure:
    freq = disp["frequency"].values
    cpa = disp["cpa"].values
    current_avg_freq = disp["frequency"].tail(14).mean()

    mask_below = freq <= bp
    mask_above = freq > bp

    fig = go.Figure()

    # Scatter: below saturation (blue)
    fig.add_trace(go.Scatter(
        x=freq[mask_below], y=cpa[mask_below],
        mode="markers",
        name="freq ≤ breakpoint",
        marker=dict(color="#2c7bb6", size=6, opacity=0.65),
    ))

    # Scatter: above saturation (red)
    fig.add_trace(go.Scatter(
        x=freq[mask_above], y=cpa[mask_above],
        mode="markers",
        name="freq > breakpoint (saturation zone)",
        marker=dict(color="#e74c3c", size=6, opacity=0.65),
    ))

    # Fitted piecewise curve
    f_lo = freq.min() - 0.2
    f_hi = freq.max() + 0.2
    f_seg1 = np.linspace(f_lo, bp, 100)
    f_seg2 = np.linspace(bp, f_hi, 100)

    cpa_seg1 = coeffs1[0] + coeffs1[1] * f_seg1
    cpa_seg2 = val_at_bp + slope2 * (f_seg2 - bp)

    fig.add_trace(go.Scatter(
        x=np.concatenate([f_seg1, f_seg2]),
        y=np.concatenate([cpa_seg1, cpa_seg2]),
        mode="lines",
        name="Piecewise fit",
        line=dict(color="white", width=2.5, dash="solid"),
    ))

    # Breakpoint vertical line
    fig.add_vline(
        x=bp, line_dash="dash", line_color=_GOLD,
        annotation_text=f"Breakpoint f={bp:.1f}", annotation_position="top left",
    )

    # Current avg frequency line
    fig.add_vline(
        x=current_avg_freq, line_dash="dot", line_color="#e74c3c",
        annotation_text=f"Current avg f={current_avg_freq:.1f}", annotation_position="top right",
    )

    # Saturation shading
    fig.add_vrect(
        x0=bp, x1=f_hi + 0.5, fillcolor="#e74c3c", opacity=0.08,
        layer="below", line_width=0,
        annotation_text="Saturation zone", annotation_position="top right",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_DARK, plot_bgcolor=_DARK,
        title=dict(text="Frequency Saturation — Display: Frequency vs CPA", font=dict(color="white")),
        xaxis=dict(title="Daily Avg Frequency (impressions/unique user)", color="white"),
        yaxis=dict(title="CPA ($)", color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
        height=450,
    )
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect(
    campaigns_df: pd.DataFrame,
    channel: str = "programmatic_display",
    min_obs_per_segment: int = 10,
    cpa_uplift_threshold: float = 0.20,
    aov: float = 120.0,
) -> DiagnosticResult | None:
    """
    Detect frequency saturation for a given channel.
    Returns DiagnosticResult or None if no saturation found.
    """
    ch_data = campaigns_df[campaigns_df["channel"] == channel].copy()
    if len(ch_data) < 30:
        return None

    freq = ch_data["frequency"].values
    cpa = ch_data["cpa"].values

    result = _grid_search_breakpoint(freq, cpa)
    if result is None:
        return None

    bp, coeffs1, slope2, val_at_bp, _ = result

    # Segment CPA
    cpa_below = float(cpa[freq <= bp].mean())
    cpa_above = float(cpa[freq > bp].mean())
    obs_below = int((freq <= bp).sum())
    obs_above = int((freq > bp).sum())

    # Current state (last 14 days)
    current_avg_freq = float(ch_data["frequency"].tail(14).mean())

    cpa_uplift = (cpa_above - cpa_below) / cpa_below if cpa_below > 0 else 0.0

    # Pearson r (overall freq vs CPA) for evidence
    r_overall, p_overall = pearsonr(freq, cpa)

    # Decision rule
    if not (
        bp < current_avg_freq
        and cpa_uplift > cpa_uplift_threshold
        and obs_below >= min_obs_per_segment
        and obs_above >= min_obs_per_segment
        and slope2 > coeffs1[1]  # slope must accelerate above breakpoint
    ):
        return None

    # Estimated impact
    # Excess CPA paid on conversions in the saturation zone
    excess_cpa = cpa_above - val_at_bp  # above-trend CPA
    convs_in_saturation = ch_data.loc[ch_data["frequency"] > bp, "conversions"].sum()
    estimated_impact = float(max(0.0, excess_cpa * convs_in_saturation))

    severity = "critical" if (current_avg_freq - bp) > 2.0 else "warning"

    fig = _make_chart(ch_data, bp, coeffs1, slope2, val_at_bp)

    return DiagnosticResult(
        name="frequency_saturation",
        severity=severity,
        channel=channel,
        finding=(
            f"Saturation breakpoint at f={bp:.1f}; "
            f"current avg f={current_avg_freq:.1f} "
            f"({current_avg_freq - bp:.1f} units past threshold). "
            f"CPA {cpa_uplift*100:.0f}% higher above breakpoint."
        ),
        evidence={
            "breakpoint_frequency": float(bp),
            "current_avg_frequency_14d": float(current_avg_freq),
            "frequency_above_threshold": float(current_avg_freq - bp),
            "cpa_below_breakpoint": float(cpa_below),
            "cpa_above_breakpoint": float(cpa_above),
            "cpa_at_breakpoint": float(val_at_bp),
            "cpa_uplift_pct": float(cpa_uplift * 100),
            "slope_segment1": float(coeffs1[1]),
            "slope_segment2": float(slope2),
            "obs_below_breakpoint": obs_below,
            "obs_above_breakpoint": obs_above,
            "pearson_r_freq_cpa": float(r_overall),
            "pearson_p": float(p_overall),
        },
        estimated_impact_usd=estimated_impact,
        charts=[fig],
    )
