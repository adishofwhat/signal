"""
Budget efficiency frontier — marginal CPA analysis.

Computes rolling 14-day average CPA per channel. Compares first-30-day vs
last-30-day marginal CPA. Flags channels in diminishing returns and recommends
reallocation toward most efficient channel.
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import plotly.graph_objects as go

from diagnostics import DiagnosticResult

_DARK = "#111111"
_GOLD = "#c8a55a"

_CHANNEL_COLORS = {
    "paid_search":          "#2c7bb6",
    "paid_social":          "#7b5ea7",
    "programmatic_display": "#4dac26",
    "ctv":                  "#1a9641",
    "youtube":              "#e74c3c",
}


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _make_chart(ch_rolling: dict[str, pd.DataFrame], flagged: list[str]) -> go.Figure:
    fig = go.Figure()

    for ch, df in ch_rolling.items():
        color = _CHANNEL_COLORS.get(ch, "#aaaaaa")
        width = 2.8 if ch in flagged else 1.5
        dash = "solid" if ch in flagged else "dot"
        opacity = 1.0 if ch in flagged else 0.55

        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["rolling_cpa"],
            name=ch + (" ⚠ DIM RETURNS" if ch in flagged else ""),
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            opacity=opacity,
        ))

    # Shade first 30 / last 30 day zones
    all_dates = list(ch_rolling.values())[0]["date"]
    d_start = all_dates.iloc[0]
    d_end = all_dates.iloc[-1]
    d30 = d_start + pd.Timedelta(days=30)
    d60 = d_end - pd.Timedelta(days=30)

    fig.add_vrect(
        x0=d_start, x1=d30, fillcolor="#2ecc71", opacity=0.06,
        layer="below", line_width=0,
        annotation_text="First 30d", annotation_position="top left",
    )
    fig.add_vrect(
        x0=d60, x1=d_end, fillcolor="#e74c3c", opacity=0.06,
        layer="below", line_width=0,
        annotation_text="Last 30d", annotation_position="top right",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_DARK, plot_bgcolor=_DARK,
        title=dict(
            text="Budget Efficiency — Rolling 14-Day Avg CPA by Channel",
            font=dict(color="white"),
        ),
        xaxis=dict(title="Date", color="white"),
        yaxis=dict(title="CPA ($)", color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
        height=480,
    )
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect(
    campaigns_df: pd.DataFrame,
    window_days: int = 14,
    early_days: int = 30,
    late_days: int = 30,
    ratio_threshold: float = 1.50,
    cross_channel_ratio_threshold: float = 2.0,
    aov: float = 120.0,
) -> list[DiagnosticResult]:
    """
    Detect budget efficiency issues and produce reallocation recommendations.
    Returns one DiagnosticResult per flagged channel (plus a summary result).
    """
    results: list[DiagnosticResult] = []

    # Compute rolling metrics per channel
    ch_data: dict[str, pd.DataFrame] = {}
    ch_metrics: dict[str, dict] = {}

    for ch in campaigns_df["channel"].unique():
        df = campaigns_df[campaigns_df["channel"] == ch].sort_values("date").copy()

        # Rolling 14-day average CPA (proxy for marginal CPA)
        df["rolling_spend"] = df["spend"].rolling(window_days, min_periods=max(1, window_days // 2)).sum()
        df["rolling_convs"] = df["conversions"].rolling(window_days, min_periods=max(1, window_days // 2)).sum()
        df["rolling_cpa"] = df["rolling_spend"] / df["rolling_convs"].clip(1)

        early = df.iloc[:early_days]
        late = df.iloc[-late_days:]

        early_cpa = float(
            early["spend"].sum() / early["conversions"].sum()
        ) if early["conversions"].sum() > 0 else np.inf

        late_cpa = float(
            late["spend"].sum() / late["conversions"].sum()
        ) if late["conversions"].sum() > 0 else np.inf

        ratio = late_cpa / early_cpa if early_cpa > 0 and np.isfinite(late_cpa) else np.inf

        # ROAS
        total_spend = float(df["spend"].sum())
        total_convs = int(df["conversions"].sum())
        roas = (total_convs * aov) / total_spend if total_spend > 0 else 0.0

        # T-test: late CPA vs early CPA (daily)
        early_daily_cpas = (early["spend"] / early["conversions"].clip(1)).values
        late_daily_cpas = (late["spend"] / late["conversions"].clip(1)).values
        if len(early_daily_cpas) > 1 and len(late_daily_cpas) > 1:
            t_stat, t_p = ttest_ind(late_daily_cpas, early_daily_cpas, equal_var=False)
        else:
            t_stat, t_p = 0.0, 1.0

        ch_data[ch] = df
        ch_metrics[ch] = {
            "early_cpa": early_cpa,
            "late_cpa": late_cpa,
            "ratio": ratio,
            "roas": roas,
            "total_spend": total_spend,
            "total_convs": total_convs,
            "t_stat": float(t_stat),
            "t_p": float(t_p),
        }

    # Identify channels with finite metrics
    finite = {ch: m for ch, m in ch_metrics.items() if np.isfinite(m["late_cpa"])}
    if not finite:
        return results

    worst_ch = max(finite, key=lambda c: finite[c]["late_cpa"])
    # Globally best for cross-ratio denominator only
    _global_best = min(finite, key=lambda c: finite[c]["late_cpa"])
    cross_ratio = finite[worst_ch]["late_cpa"] / finite[_global_best]["late_cpa"]

    # Flagged channels: ratio > threshold OR ROAS < 1.0
    flagged = [
        ch for ch, m in finite.items()
        if m["ratio"] > ratio_threshold or m["roas"] < 1.0
    ]

    # One DiagnosticResult per flagged channel
    ch_rolling = {ch: ch_data[ch] for ch in ch_data}

    for ch in flagged:
        m = ch_metrics[ch]

        # Best destination: stable/improving channel (lowest ratio) that is NOT flagged
        # Prefer channels not in diminishing returns; break ties by late CPA.
        candidate_dests = {
            c: v for c, v in finite.items()
            if c != ch and v["ratio"] < ratio_threshold
        }
        if not candidate_dests:
            candidate_dests = {c: v for c, v in finite.items() if c != ch}

        best_ch = min(candidate_dests, key=lambda c: finite[c]["ratio"])

        # Reallocation impact: shift 20% of spend from flagged → best destination
        shift_spend = m["total_spend"] * 0.20
        dest_late_cpa = finite[best_ch]["late_cpa"]
        extra_convs = shift_spend / dest_late_cpa
        reduced_convs = shift_spend / m["late_cpa"]   # conversions lost at source
        net_extra_convs = extra_convs - reduced_convs
        estimated_impact = float(max(0.0, net_extra_convs * aov))

        severity = "critical" if m["ratio"] > 2.0 or m["roas"] < 1.0 else "warning"

        # Only emit chart for the first flagged channel to avoid duplication
        charts_for_result = []
        if ch == flagged[0]:
            fig = _make_chart(ch_rolling, flagged)
            charts_for_result = [fig]

        results.append(DiagnosticResult(
            name="budget_efficiency",
            severity=severity,
            channel=ch,
            finding=(
                f"{ch} marginal CPA ratio (last30/first30) = {m['ratio']:.2f}× "
                f"(${m['early_cpa']:.0f} → ${m['late_cpa']:.0f}); "
                f"ROAS={m['roas']:.2f}. "
                f"Recommend shifting budget toward {best_ch} "
                f"(ratio={finite[best_ch]['ratio']:.2f}×, ${dest_late_cpa:.0f} late CPA)."
            ),
            evidence={
                "channel": ch,
                "early_30d_cpa": float(m["early_cpa"]),
                "late_30d_cpa": float(m["late_cpa"]),
                "cpa_ratio_late_over_early": float(m["ratio"]),
                "roas": float(m["roas"]),
                "total_spend_90d": float(m["total_spend"]),
                "total_conversions_90d": int(m["total_convs"]),
                "t_stat_late_vs_early": float(m["t_stat"]),
                "t_p_value": float(m["t_p"]),
                "recommended_destination": best_ch,
                "destination_late_cpa": float(finite[best_ch]["late_cpa"]),
                "cross_channel_cpa_ratio_worst_best": float(cross_ratio),
                "all_channels_late_cpa": {
                    c: round(float(v["late_cpa"]), 2)
                    for c, v in finite.items()
                },
            },
            estimated_impact_usd=estimated_impact,
            charts=charts_for_result,
        ))

    # Emit a separate chart-only result if no individual channel was flagged
    # but cross-channel ratio exceeds threshold
    if not flagged and cross_ratio > cross_channel_ratio_threshold:
        fig = _make_chart(ch_rolling, [worst_ch])
        results.append(DiagnosticResult(
            name="budget_efficiency",
            severity="info",
            channel=worst_ch,
            finding=(
                f"Cross-channel CPA spread: {cross_ratio:.1f}× difference between "
                f"{best_ch} (${finite[best_ch]['late_cpa']:.0f}) and "
                f"{worst_ch} (${finite[worst_ch]['late_cpa']:.0f})."
            ),
            evidence={
                "best_channel": best_ch,
                "worst_channel": worst_ch,
                "cross_channel_ratio": float(cross_ratio),
                "all_channels_late_cpa": {
                    c: round(float(v["late_cpa"]), 2) for c, v in finite.items()
                },
            },
            estimated_impact_usd=0.0,
            charts=[fig],
        ))

    # Always emit the chart for the complete picture if we have flagged channels
    if flagged and all(len(r.charts) == 0 for r in results[1:]):
        # Chart is already on results[0]; nothing more needed
        pass

    return results
