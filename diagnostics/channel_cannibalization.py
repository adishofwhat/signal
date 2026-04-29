"""
Cross-channel cannibalization detection.

Computes Pearson correlation between weekly spend changes (channel A) and
weekly CPA changes (channel B). Adds Granger causality test for directional
evidence. Flags when search spend spikes correspond to display CPA spikes.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from diagnostics import DiagnosticResult

_DARK = "#111111"
_GOLD = "#c8a55a"

# Channel pairs to test: (channel_a, channel_b)
# A's spend increases → B's CPA increases
CANDIDATE_PAIRS = [
    ("paid_search", "programmatic_display"),
]


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _make_chart(
    combined: pd.DataFrame,
    ps_weekly: pd.Series,
    disp_weekly: pd.Series,
    spike_weeks: list[int],
) -> go.Figure:
    x = list(combined.index)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        shared_xaxes=True,
        subplot_titles=[
            "Weekly Paid Search Spend vs Display CPA",
            "WoW % Changes (correlation view)",
        ],
        vertical_spacing=0.12,
    )

    # Row 1: dual-axis spend + CPA
    fig.add_trace(
        go.Bar(
            x=x, y=ps_weekly.loc[x].values / 1_000,
            name="PS Weekly Spend ($K)",
            marker_color=[
                "#e74c3c" if w in spike_weeks else "#2c7bb6"
                for w in x
            ],
            opacity=0.65,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x, y=disp_weekly.loc[x].values,
            name="Display CPA ($)",
            mode="lines+markers",
            line=dict(color="#f39c12", width=2.5),
            marker=dict(size=6, color="#f39c12"),
            yaxis="y2",
        ),
        row=1, col=1,
    )

    # Row 2: WoW% changes scatter for correlation
    fig.add_trace(
        go.Scatter(
            x=combined["ps_wow"].values * 100,
            y=combined["disp_wow"].values * 100,
            mode="markers",
            name="WoW% correlation",
            marker=dict(
                color=[
                    "#e74c3c" if w in spike_weeks else "#2c7bb6"
                    for w in combined.index
                ],
                size=8,
                opacity=0.75,
            ),
            text=[f"Wk {w}" for w in combined.index],
        ),
        row=2, col=1,
    )

    # Add regression line on scatter
    ps_wow = combined["ps_wow"].values
    disp_wow = combined["disp_wow"].values
    if len(ps_wow) > 2:
        m, b = np.polyfit(ps_wow, disp_wow, 1)
        x_line = np.linspace(ps_wow.min(), ps_wow.max(), 50)
        fig.add_trace(
            go.Scatter(
                x=x_line * 100, y=(m * x_line + b) * 100,
                mode="lines",
                name="OLS fit",
                line=dict(color="white", width=1.5, dash="dash"),
            ),
            row=2, col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_DARK, plot_bgcolor=_DARK,
        title=dict(text="Channel Cannibalization — Paid Search → Display", font=dict(color="white")),
        height=600,
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
        yaxis2=dict(overlaying="y", side="right", color="#f39c12"),
    )
    fig.update_xaxes(title_text="Campaign Week", color="white", row=1, col=1)
    fig.update_xaxes(title_text="Search WoW Spend Change (%)", color="white", row=2, col=1)
    fig.update_yaxes(title_text="PS Spend ($K)", color="white", row=1, col=1)
    fig.update_yaxes(title_text="Display WoW CPA Change (%)", color="white", row=2, col=1)

    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect(
    campaigns_df: pd.DataFrame,
    pairs: list[tuple[str, str]] | None = None,
    corr_threshold: float = 0.50,
    p_threshold: float = 0.05,
    aov: float = 120.0,
) -> list[DiagnosticResult]:
    """
    Detect cross-channel cannibalization for specified (channel_a, channel_b) pairs.
    """
    results: list[DiagnosticResult] = []
    pairs = pairs or CANDIDATE_PAIRS

    camps = campaigns_df.copy()
    camps["camp_week"] = ((camps["date"] - camps["date"].min()).dt.days // 7)

    for ch_a, ch_b in pairs:
        a_weekly = camps[camps["channel"] == ch_a].groupby("camp_week").agg(
            spend=("spend", "sum"),
        )
        b_weekly = camps[camps["channel"] == ch_b].groupby("camp_week").agg(
            cpa=("cpa", "mean"),
        )

        # WoW % changes (campaign-week aligned)
        ps_wow = a_weekly["spend"].pct_change()
        disp_wow = b_weekly["cpa"].pct_change()

        combined = pd.DataFrame({
            "ps_wow":   ps_wow,
            "disp_wow": disp_wow,
        }).dropna()

        if len(combined) < 8:
            continue

        # --- Pearson correlation ---
        r_contemporaneous, p_contemporaneous = pearsonr(
            combined["ps_wow"], combined["disp_wow"]
        )

        # --- Lag-1 correlation (search WoW predicts next week's display CPA change) ---
        if len(combined) >= 4:
            r_lag1, p_lag1 = pearsonr(
                combined["ps_wow"].iloc[:-1],
                combined["disp_wow"].iloc[1:],
            )
        else:
            r_lag1, p_lag1 = 0.0, 1.0

        # --- Granger causality test ---
        # H0: ps_wow does NOT granger-cause disp_wow
        # Array: column 0 = dependent (disp), column 1 = independent (ps)
        gc_p = 1.0
        try:
            gc_data = combined[["disp_wow", "ps_wow"]].values
            gc_result = grangercausalitytests(gc_data, maxlag=1, verbose=False)
            # SSR chi2 test p-value
            gc_p = float(gc_result[1][0]["ssr_chi2test"][1])
        except Exception:
            gc_p = 1.0

        # Decision rule: contemporaneous OR lag-1 correlation must be significant
        is_cannibalization = (
            (r_contemporaneous > corr_threshold and p_contemporaneous < p_threshold)
            or (r_lag1 > corr_threshold and p_lag1 < p_threshold)
        )
        if not is_cannibalization:
            continue

        # Identify spike weeks (search WoW > +15%)
        spike_weeks = combined.index[combined["ps_wow"] > 0.15].tolist()

        # --- Estimated impact ---
        if spike_weeks:
            b_spike_cpa = b_weekly.loc[b_weekly.index.isin(spike_weeks), "cpa"].mean()
            b_normal_cpa = b_weekly.loc[~b_weekly.index.isin(spike_weeks), "cpa"].mean()
            excess_cpa_per_week = max(0.0, float(b_spike_cpa - b_normal_cpa))
            b_daily_convs = float(
                camps.loc[camps["channel"] == ch_b, "conversions"].mean()
            )
            estimated_impact = float(excess_cpa_per_week * b_daily_convs * 7 * len(spike_weeks))
        else:
            estimated_impact = 0.0

        severity = "warning" if r_contemporaneous < 0.65 else "critical"

        fig = _make_chart(combined, a_weekly["spend"], b_weekly["cpa"], spike_weeks)

        results.append(DiagnosticResult(
            name="channel_cannibalization",
            severity=severity,
            channel=f"{ch_a} → {ch_b}",
            finding=(
                f"Search spend WoW changes correlate positively with display CPA changes "
                f"(r={r_contemporaneous:.2f}, p={p_contemporaneous:.3f}). "
                f"Audience overlap: scaling search steals attributed conversions from display."
            ),
            evidence={
                "channel_a": ch_a,
                "channel_b": ch_b,
                "pearson_r_contemporaneous": float(r_contemporaneous),
                "p_value_contemporaneous": float(p_contemporaneous),
                "pearson_r_lag1": float(r_lag1),
                "p_value_lag1": float(p_lag1),
                "granger_causality_p": float(gc_p),
                "granger_causal": bool(gc_p < 0.10),
                "n_weekly_observations": int(len(combined)),
                "n_spike_weeks": int(len(spike_weeks)),
                "avg_display_cpa_spike_weeks": float(
                    b_weekly.loc[b_weekly.index.isin(spike_weeks), "cpa"].mean()
                ) if spike_weeks else None,
                "avg_display_cpa_normal_weeks": float(
                    b_weekly.loc[~b_weekly.index.isin(spike_weeks), "cpa"].mean()
                ) if spike_weeks else None,
            },
            estimated_impact_usd=estimated_impact,
            charts=[fig],
        ))

    return results
