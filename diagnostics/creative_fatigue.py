"""
Creative fatigue detection.

Fits CTR(t) = a * exp(-λ * t) + c to each creative's decaying phase.
Detects the onset of decay before fitting so that plateau-then-decay patterns
(e.g., social_v1 fresh for 25 days then fatiguing) are correctly identified.
Decision rule uses observed CTR statistics, model fit provides the λ and half-life.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist
import plotly.graph_objects as go

from diagnostics import DiagnosticResult

_PLOT_COLORS = {
    "social_v1": "#e74c3c",
    "social_v2": "#7f8c8d",
    "social_v3": "#95a5a6",
    "social_v4": "#bdc3c7",
    "default":   "#aaaaaa",
}

_DARK = "#111111"
_GOLD = "#c8a55a"


def _decay(t: np.ndarray, a: float, lam: float, c: float) -> np.ndarray:
    return a * np.exp(-lam * t) + c


def _detect_decay_onset(ctr: np.ndarray, window: int = 7) -> int:
    """
    Return the index where decay becomes apparent.
    Onset = first position where the rolling mean drops below 90% of the
    initial level (first `window` days). Returns 0 if no onset detected.
    """
    if len(ctr) < window * 3:
        return 0
    initial_level = float(np.mean(ctr[:window]))
    threshold = initial_level * 0.90
    rolling = pd.Series(ctr).rolling(window, min_periods=window // 2).mean().values
    for i in range(window, len(rolling) - window):
        if rolling[i] < threshold:
            # Onset is ≈ half a window before the rolling mean crossed
            return max(0, i - window // 2)
    return 0


def _fit_decay(t: np.ndarray, ctr: np.ndarray):
    """Fit CTR(t) = a*exp(-λ*t) + c. Bounds from spec."""
    p0 = [max(0.001, ctr.max() - ctr.min()), 0.035, max(0.001, ctr.min())]
    popt, pcov = curve_fit(
        _decay, t, ctr,
        p0=p0,
        bounds=([0, 1e-6, 0], [0.10, 0.20, 0.05]),
        maxfev=20_000,
    )
    return popt, pcov


def _p_value_lambda(popt, pcov, n: int) -> float:
    """One-sided t-test: H1 λ > 0."""
    try:
        se_lam = float(np.sqrt(max(0, pcov[1, 1])))
        if se_lam < 1e-12:
            return 0.0  # essentially zero variance → near-certain
        t_stat = popt[1] / se_lam
        return float(t_dist.sf(t_stat, df=max(n - 3, 1)))
    except Exception:
        return 1.0


def _r_squared(t, ctr, popt) -> float:
    fitted = _decay(t, *popt)
    ss_res = float(np.sum((ctr - fitted) ** 2))
    ss_tot = float(np.sum((ctr - ctr.mean()) ** 2))
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def _optimal_rotation_day(popt, onset: int) -> float | None:
    """Campaign day when CTR first drops to 80% of its pre-decay level."""
    a, lam, c = popt
    initial_ctr_model = float(a + c)
    target = 0.80 * initial_ctr_model
    ratio = (target - c) / a if a > 1e-9 else -1
    if 0 < ratio < 1:
        return float(-np.log(ratio) / lam) + onset
    return None


def _make_chart(
    channel: str,
    ch_creative_df: pd.DataFrame,
    fatigued_cid: str,
    popt,
    onset_day: int,
) -> go.Figure:
    start_date = ch_creative_df["date"].min()
    fig = go.Figure()

    for cid, grp in ch_creative_df.groupby("creative_id"):
        grp = grp.sort_values("date")
        days = (grp["date"] - start_date).dt.days.values

        if cid == fatigued_cid:
            fig.add_trace(go.Scatter(
                x=days, y=grp["ctr"].values * 100,
                mode="markers",
                name=f"{cid} (actual)",
                marker=dict(color=_PLOT_COLORS.get(cid, "#e74c3c"), size=5, opacity=0.6),
            ))
            # Fitted curve shown only over the decay portion
            t_range = np.linspace(0, len(days) - onset_day - 1, 300)
            ctr_fit = _decay(t_range, *popt) * 100
            fig.add_trace(go.Scatter(
                x=t_range + onset_day, y=ctr_fit,
                mode="lines",
                name=f"{cid} (fitted decay)",
                line=dict(color=_PLOT_COLORS.get(cid, "#e74c3c"), width=2.5),
            ))
            # Optimal rotation marker
            t_opt = _optimal_rotation_day(popt, onset_day)
            if t_opt is not None and t_opt <= days.max():
                ctr_at_opt = _decay(t_opt - onset_day, *popt) * 100
                fig.add_trace(go.Scatter(
                    x=[t_opt], y=[ctr_at_opt],
                    mode="markers+text",
                    name="Optimal rotation point",
                    text=["↑ Rotate creative"],
                    textposition="top center",
                    marker=dict(color=_GOLD, size=14, symbol="star"),
                ))
        else:
            fig.add_trace(go.Scatter(
                x=days, y=grp["ctr"].values * 100,
                mode="lines",
                name=cid,
                line=dict(color=_PLOT_COLORS.get(cid, "#888888"), width=1.2, dash="dot"),
                opacity=0.45,
            ))

    if onset_day > 0:
        fig.add_vline(
            x=onset_day, line_dash="dash", line_color="#888888",
            annotation_text=f"Decay onset (day {onset_day})",
            annotation_position="top right",
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_DARK, plot_bgcolor=_DARK,
        title=dict(
            text=f"Creative Fatigue — {channel}: CTR by Creative",
            font=dict(color="white"),
        ),
        xaxis=dict(title="Campaign Day", color="white"),
        yaxis=dict(title="CTR (%)", color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
        height=450,
    )
    return fig


def detect(
    creative_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
    min_days: int = 30,
    half_life_threshold: float = 30.0,
    ctr_ratio_threshold: float = 0.60,
    p_threshold: float = 0.05,
    aov: float = 120.0,
) -> list[DiagnosticResult]:
    """
    Detect creative fatigue for every channel + creative combination.
    Uses onset detection so plateau-then-decay shapes are handled correctly.
    """
    results: list[DiagnosticResult] = []

    for ch in creative_df["channel"].unique():
        ch_df = creative_df[creative_df["channel"] == ch]

        for cid, cdata in ch_df.groupby("creative_id"):
            cdata = cdata.sort_values("date").copy()
            if len(cdata) < min_days:
                continue

            cdata["day"] = (cdata["date"] - cdata["date"].min()).dt.days
            t_all = cdata["day"].values.astype(float)
            ctr_all = cdata["ctr"].values

            # ── Observed initial / current CTR for decision rule ──────────
            obs_window = max(5, min(10, len(ctr_all) // 8))
            initial_ctr_obs = float(np.mean(ctr_all[:obs_window]))
            current_ctr_obs = float(np.mean(ctr_all[-obs_window:]))
            ctr_ratio_obs = current_ctr_obs / initial_ctr_obs if initial_ctr_obs > 1e-9 else 1.0

            # Quick short-circuit: if CTR hasn't dropped much, skip fitting
            if ctr_ratio_obs >= ctr_ratio_threshold:
                continue

            # ── Detect onset and fit model over decaying segment ──────────
            onset_idx = _detect_decay_onset(ctr_all, window=7)
            if onset_idx > 0 and len(ctr_all) - onset_idx >= 15:
                t_fit = t_all[onset_idx:] - t_all[onset_idx]
                ctr_fit = ctr_all[onset_idx:]
            else:
                t_fit = t_all
                ctr_fit = ctr_all
                onset_idx = 0

            try:
                popt, pcov = _fit_decay(t_fit, ctr_fit)
            except RuntimeError:
                continue

            a, lam, c_floor = popt
            if lam < 1e-6:
                continue

            half_life = float(np.log(2) / lam)
            p_val = _p_value_lambda(popt, pcov, len(t_fit))
            r2 = _r_squared(t_fit, ctr_fit, popt)

            # Model-derived initial / current for reporting
            initial_ctr_model = float(a + c_floor)
            current_ctr_model = float(_decay(t_fit[-1], *popt))

            # ── Decision rule (uses observed CTR, not model prediction) ──
            is_fatigued = (
                half_life < half_life_threshold
                and ctr_ratio_obs < ctr_ratio_threshold
                and p_val < p_threshold
            )
            if not is_fatigued:
                continue

            # ── Estimated impact ──────────────────────────────────────────
            # Additional conversions lost in last 30 days of decay:
            # = daily_impr * (initial_ctr_obs - current_ctr_obs) * ch_avg_ctr_to_cvr * AOV * 30
            ch_camps = campaigns_df[campaigns_df["channel"] == ch]
            cdata_last30 = cdata.tail(30)
            avg_daily_impr = float(cdata_last30["impressions"].mean())
            ch_avg_cvr = float(ch_camps["cvr"].mean())
            ctr_gap = initial_ctr_obs - current_ctr_obs
            lost_daily_convs = avg_daily_impr * ctr_gap * ch_avg_cvr
            estimated_impact = float(max(0.0, lost_daily_convs * aov * 30))

            t_opt = _optimal_rotation_day(popt, onset_idx)
            severity = "critical" if (half_life < 20 and ctr_ratio_obs < 0.50) else "warning"

            fig = _make_chart(ch, ch_df, cid, popt, onset_idx)

            results.append(DiagnosticResult(
                name="creative_fatigue",
                severity=severity,
                channel=ch,
                finding=(
                    f"{cid} CTR decayed {(1 - ctr_ratio_obs)*100:.0f}% "
                    f"({initial_ctr_obs*100:.2f}% → {current_ctr_obs*100:.2f}%); "
                    f"half-life {half_life:.1f}d (λ={lam:.4f}), onset ~day {onset_idx}"
                ),
                evidence={
                    "creative_id": cid,
                    "decay_onset_day": int(onset_idx),
                    "decay_lambda": float(lam),
                    "half_life_days": float(half_life),
                    "initial_ctr_observed": float(initial_ctr_obs),
                    "current_ctr_observed": float(current_ctr_obs),
                    "initial_ctr_model": float(initial_ctr_model),
                    "current_ctr_model": float(current_ctr_model),
                    "ctr_ratio_observed": float(ctr_ratio_obs),
                    "p_value_lambda": float(p_val),
                    "r_squared_decay_fit": float(r2),
                    "a_amplitude": float(a),
                    "c_floor": float(c_floor),
                    "optimal_rotation_campaign_day": float(t_opt) if t_opt else None,
                    "n_days_in_decay_fit": int(len(t_fit)),
                },
                estimated_impact_usd=estimated_impact,
                charts=[fig],
            ))

    return results
