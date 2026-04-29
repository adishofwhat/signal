"""
Daypart opportunity detection.

Uses Kruskal-Wallis H-test across dayparts on CVR per channel, then pairwise
Mann-Whitney U post-hoc tests (Bonferroni corrected) to identify which dayparts
differ. Flags channels with high-CVR, low-impression-share dayparts.
"""
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
import plotly.graph_objects as go

from diagnostics import DiagnosticResult

_DARK = "#111111"
_GOLD = "#c8a55a"

DAYPART_ORDER = [
    "early_morning", "morning", "midday",
    "afternoon", "evening", "prime", "late_night",
]

DAYPART_HOURS = {
    "early_morning": "5–8h",
    "morning":       "8–11h",
    "midday":        "11–14h",
    "afternoon":     "14–17h",
    "evening":       "17–20h",
    "prime":         "20–23h",
    "late_night":    "23–5h",
}


# ---------------------------------------------------------------------------
# Opportunity score
# ---------------------------------------------------------------------------

def _opportunity_score(cvr: float, avg_cvr: float, impr_share: float) -> float:
    """
    opportunity = (daypart_cvr - avg_cvr) / avg_cvr * (1 - impr_share)
    Positive = under-invested with high CVR.
    """
    if avg_cvr <= 0:
        return 0.0
    return float((cvr - avg_cvr) / avg_cvr * (1 - impr_share))


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _make_chart(
    channel: str,
    dp_stats: pd.DataFrame,
    avg_cvr: float,
    high_opp_dps: list[str],
) -> go.Figure:
    dp_stats = dp_stats.reindex(DAYPART_ORDER).copy()
    labels = [f"{dp}\n{DAYPART_HOURS.get(dp, '')}" for dp in dp_stats.index]

    # Bar colors
    colors = []
    for dp in dp_stats.index:
        if dp in high_opp_dps:
            colors.append("#2ecc71")   # opportunity
        elif dp_stats.loc[dp, "cvr"] < avg_cvr * 0.85 and dp_stats.loc[dp, "impr_share"] > 0.18:
            colors.append("#e74c3c")   # overinvested + weak
        else:
            colors.append("#2c7bb6")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=DAYPART_ORDER,
        y=dp_stats["cvr"].values * 100,
        name="CVR (%)",
        marker_color=colors,
        opacity=0.80,
        yaxis="y1",
    ))

    fig.add_hline(
        y=avg_cvr * 100, line_dash="dash", line_color=_GOLD,
        annotation_text=f"Channel avg CVR {avg_cvr*100:.2f}%",
        annotation_position="bottom right",
    )

    fig.add_trace(go.Scatter(
        x=DAYPART_ORDER,
        y=dp_stats["impr_share"].values * 100,
        name="Impression Share (%)",
        mode="lines+markers",
        marker=dict(size=8, symbol="diamond", color="#f39c12"),
        line=dict(color="#f39c12", width=2),
        yaxis="y2",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_DARK, plot_bgcolor=_DARK,
        title=dict(
            text=f"Daypart Opportunity — {channel}: CVR vs Impression Share",
            font=dict(color="white"),
        ),
        xaxis=dict(title="Daypart", color="white", tickangle=-30),
        yaxis=dict(title="CVR (%)", color="white", side="left"),
        yaxis2=dict(
            title="Impression Share (%)", color="#f39c12",
            overlaying="y", side="right",
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
        height=450,
        annotations=[
            dict(
                x=dp, y=dp_stats.loc[dp, "cvr"] * 100 + 0.05,
                text="★ OPP", font=dict(color="#2ecc71", size=9),
                showarrow=False, yref="y",
            )
            for dp in high_opp_dps if dp in dp_stats.index
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect(
    daypart_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
    target_channels: list[str] | None = None,
    kw_p_threshold: float = 0.05,
    high_cvr_ratio: float = 1.30,
    low_impr_share: float = 0.15,
    aov: float = 120.0,
) -> list[DiagnosticResult]:
    """
    Detect daypart opportunities for each channel.
    Returns DiagnosticResult for each channel where significant daypart variance exists.
    """
    results: list[DiagnosticResult] = []
    channels = target_channels or daypart_df["channel"].unique().tolist()

    for ch in channels:
        ch_dp = daypart_df[daypart_df["channel"] == ch].copy()
        if ch_dp.empty:
            continue

        # Aggregate CVR, impressions per daypart
        dp_stats = (
            ch_dp.groupby("daypart")
            .agg(
                cvr=("cvr", "mean"),
                impressions=("impressions", "sum"),
                clicks=("clicks", "sum"),
                conversions=("conversions", "sum"),
            )
            .reindex(DAYPART_ORDER)
            .dropna(subset=["cvr"])
        )
        if len(dp_stats) < 3:
            continue

        total_impr = dp_stats["impressions"].sum()
        dp_stats["impr_share"] = dp_stats["impressions"] / total_impr
        ch_avg_cvr = float(dp_stats["cvr"].mean())

        dp_stats["opp_score"] = dp_stats.apply(
            lambda r: _opportunity_score(r["cvr"], ch_avg_cvr, r["impr_share"]),
            axis=1,
        )

        # Kruskal-Wallis H-test across dayparts
        groups = [
            ch_dp.loc[ch_dp["daypart"] == dp, "cvr"].values
            for dp in dp_stats.index
            if ch_dp["daypart"].eq(dp).sum() > 0
        ]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 3:
            continue

        kw_stat, kw_p = kruskal(*groups)
        if kw_p >= kw_p_threshold:
            continue

        # High-opportunity dayparts: CVR > 1.30× avg AND impression share < 15%
        high_opp = dp_stats[
            (dp_stats["cvr"] > ch_avg_cvr * high_cvr_ratio)
            & (dp_stats["impr_share"] < low_impr_share)
        ]
        if high_opp.empty:
            continue

        # Pairwise Mann-Whitney U (Bonferroni corrected) for high-opp vs rest
        n_tests = len(DAYPART_ORDER) - 1
        alpha_bonf = 0.05 / n_tests
        pairwise = {}
        for dp_name in high_opp.index:
            dp_cvrs = ch_dp.loc[ch_dp["daypart"] == dp_name, "cvr"].values
            rest_cvrs = ch_dp.loc[ch_dp["daypart"] != dp_name, "cvr"].values
            if len(dp_cvrs) < 2 or len(rest_cvrs) < 2:
                continue
            _, p_mw = mannwhitneyu(dp_cvrs, rest_cvrs, alternative="greater")
            pairwise[f"{dp_name}_vs_rest_p"] = float(p_mw)
            pairwise[f"{dp_name}_vs_rest_sig_bonferroni"] = bool(p_mw < alpha_bonf)

        # Estimated impact
        # Shift 15% of daily clicks from lowest-CVR daypart to best-opportunity daypart.
        # CVR = conversions/clicks, so we must work in click space, not impression space.
        # 30-day horizon (conservative; one calendar month of optimisation).
        best_dp = high_opp["cvr"].idxmax()
        best_cvr = float(high_opp.loc[best_dp, "cvr"])
        ch_camps = campaigns_df[campaigns_df["channel"] == ch]
        daily_clicks = float(ch_camps["clicks"].mean())
        realloc_clicks_daily = daily_clicks * 0.15
        extra_daily_convs = realloc_clicks_daily * (best_cvr - ch_avg_cvr)
        estimated_impact = float(max(0.0, extra_daily_convs * aov * 30))

        severity = "critical" if high_opp["opp_score"].max() > 0.50 else "warning"

        fig = _make_chart(ch, dp_stats, ch_avg_cvr, high_opp.index.tolist())

        results.append(DiagnosticResult(
            name="daypart_opportunity",
            severity=severity,
            channel=ch,
            finding=(
                f"{', '.join(high_opp.index.tolist())} have "
                f"{high_opp['cvr'].mean() / ch_avg_cvr:.2f}× avg CVR but only "
                f"{high_opp['impr_share'].mean()*100:.1f}% of impressions — "
                f"significant reallocation opportunity."
            ),
            evidence={
                "kruskal_wallis_h": float(kw_stat),
                "kruskal_wallis_p": float(kw_p),
                "channel_avg_cvr": float(ch_avg_cvr),
                "high_opportunity_dayparts": high_opp.index.tolist(),
                "best_daypart": str(best_dp),
                "best_daypart_cvr": float(best_cvr),
                "best_daypart_cvr_ratio": float(best_cvr / ch_avg_cvr),
                "best_daypart_impr_share": float(
                    dp_stats.loc[best_dp, "impr_share"] if best_dp in dp_stats.index else 0
                ),
                "best_opportunity_score": float(high_opp["opp_score"].max()),
                "daypart_cvr_table": {
                    dp: round(float(row["cvr"]), 4)
                    for dp, row in dp_stats.iterrows()
                },
                "daypart_impr_share_table": {
                    dp: round(float(row["impr_share"]), 4)
                    for dp, row in dp_stats.iterrows()
                },
                **pairwise,
            },
            estimated_impact_usd=estimated_impact,
            charts=[fig],
        ))

    return results
