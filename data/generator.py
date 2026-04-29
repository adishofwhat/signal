"""
SIGNAL synthetic campaign data generator.

Generates three tables — campaigns, creative_performance, daypart_performance —
with five planted failure modes the diagnostic engine will detect.
"""
import numpy as np
import pandas as pd
from pathlib import Path


SEED = 42

# Paid Search weekly spend multipliers over 13 weeks.
# Spike weeks (WoW increase > +15%): weeks 2, 5, 8, 11.
#   w1→w2: +26%  w4→w5: +36%  w7→w8: +42%  w10→w11: +30%
_PS_WEEK_MULTS = np.array([
    1.00, 0.95, 1.20, 1.05, 0.90, 1.22, 0.98,
    0.88, 1.25, 1.05, 0.92, 1.20, 1.05,
])
_SPIKE_WEEKS = {2, 5, 8, 11}

# Daypart ordering
DAYPARTS = [
    "early_morning", "morning", "midday",
    "afternoon", "evening", "prime", "late_night",
]

# Creative IDs per channel (social_v1 is the fatigued creative)
CREATIVES = {
    "paid_search":          ["search_v1", "search_v2", "search_v3"],
    "paid_social":          ["social_v1", "social_v2", "social_v3", "social_v4"],
    "programmatic_display": ["display_v1", "display_v2", "display_v3"],
    "ctv":                  ["ctv_v1", "ctv_v2"],
    "youtube":              ["yt_v1", "yt_v2", "yt_v3"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dow_multipliers(dates: pd.DatetimeIndex) -> np.ndarray:
    """Day-of-week spend/performance multiplier (weekends softer)."""
    mults = {0: 1.05, 1: 1.08, 2: 1.06, 3: 1.07, 4: 1.10, 5: 0.90, 6: 0.85}
    return np.array([mults[d.weekday()] for d in dates])


# ---------------------------------------------------------------------------
# Campaign-level table
# ---------------------------------------------------------------------------

def _generate_campaigns(rng: np.random.Generator, dates: pd.DatetimeIndex) -> pd.DataFrame:
    N = len(dates)
    dow = _dow_multipliers(dates)
    week_idx = np.arange(N) // 7
    spike_mask = np.array([week_idx[i] in _SPIKE_WEEKS for i in range(N)])
    rows: list[dict] = []

    # ------------------------------------------------------------------ #
    # PAID SEARCH                                                          #
    # ------------------------------------------------------------------ #
    ps_spend = (
        4_000 * _PS_WEEK_MULTS[week_idx] * dow
        * rng.lognormal(0, 0.07, N)
    )
    ps_cpm = rng.uniform(22, 32, N)
    ps_ctr = rng.normal(0.040, 0.004, N).clip(0.025, 0.065)
    # CPA-first: consistent ~$28 with mild noise (no failure mode for search CPA)
    ps_cpa = rng.normal(28, 3.5, N).clip(18, 48)

    ps_impr = (ps_spend / ps_cpm * 1_000).round().astype(int)
    ps_clicks = (ps_impr * ps_ctr).round().astype(int).clip(1)
    ps_convs = (ps_spend / ps_cpa).round().astype(int).clip(0)
    ps_cvr = (ps_convs / ps_clicks).clip(0.02, 0.12)
    ps_freq = rng.uniform(1.8, 3.2, N)
    ps_reach = (ps_impr / ps_freq).round().astype(int)

    for i in range(N):
        rows.append({
            "date": dates[i], "channel": "paid_search",
            "impressions": int(ps_impr[i]),
            "clicks":      int(ps_clicks[i]),
            "conversions": int(ps_convs[i]),
            "spend":       round(float(ps_spend[i]), 2),
            "cpm":         round(float(ps_cpm[i]), 2),
            "cpc":         round(float(ps_spend[i] / ps_clicks[i]), 2),
            "cpa":         round(float(ps_spend[i] / max(int(ps_convs[i]), 1)), 2),
            "ctr":         round(float(ps_ctr[i]), 4),
            "cvr":         round(float(ps_cvr[i]), 4),
            "frequency":   round(float(ps_freq[i]), 2),
            "reach":       int(ps_reach[i]),
        })

    # ------------------------------------------------------------------ #
    # PAID SOCIAL                                                          #
    # Creative fatigue signal: channel-level CTR trends down because       #
    # social_v1 (dominant creative) is fatiguing. Full decay in             #
    # creative_performance table.                                           #
    # ------------------------------------------------------------------ #
    soc_spend = 5_000 * dow * rng.lognormal(0, 0.08, N)
    soc_cpm = rng.uniform(9, 13, N)
    soc_ctr = (np.linspace(0.022, 0.016, N) + rng.normal(0, 0.0025, N)).clip(0.010, 0.035)
    soc_cpa = rng.normal(55, 7, N).clip(35, 90)

    soc_impr = (soc_spend / soc_cpm * 1_000).round().astype(int)
    soc_clicks = (soc_impr * soc_ctr).round().astype(int).clip(1)
    soc_convs = (soc_spend / soc_cpa).round().astype(int).clip(0)
    soc_cvr = (soc_convs / soc_clicks).clip(0.01, 0.08)
    soc_freq = rng.uniform(2.5, 4.5, N)
    soc_reach = (soc_impr / soc_freq).round().astype(int)

    for i in range(N):
        rows.append({
            "date": dates[i], "channel": "paid_social",
            "impressions": int(soc_impr[i]),
            "clicks":      int(soc_clicks[i]),
            "conversions": int(soc_convs[i]),
            "spend":       round(float(soc_spend[i]), 2),
            "cpm":         round(float(soc_cpm[i]), 2),
            "cpc":         round(float(soc_spend[i] / soc_clicks[i]), 2),
            "cpa":         round(float(soc_spend[i] / max(int(soc_convs[i]), 1)), 2),
            "ctr":         round(float(soc_ctr[i]), 4),
            "cvr":         round(float(soc_cvr[i]), 4),
            "frequency":   round(float(soc_freq[i]), 2),
            "reach":       int(soc_reach[i]),
        })

    # ------------------------------------------------------------------ #
    # PROGRAMMATIC DISPLAY                                                 #
    # Failure mode 2: frequency saturation (freq 4→10, CPA hockey-stick)  #
    # Failure mode 3: cannibalization (CPA +8-12% in paid_search spike wks)
    # ------------------------------------------------------------------ #
    disp_spend = 3_000 * dow * rng.lognormal(0, 0.08, N)
    disp_cpm = rng.uniform(5.0, 7.5, N)
    disp_ctr = rng.normal(0.005, 0.001, N).clip(0.002, 0.009)

    # Frequency: linear drift 4 → 10 over 90 days
    disp_freq = (np.linspace(4.0, 10.0, N) + rng.normal(0, 0.40, N)).clip(3.0, 12.0)

    # Piecewise CPA: flat ~$28 until freq=7, then +$7/unit above 7
    freq_excess = (disp_freq - 7.0).clip(0)
    disp_base_cpa = 28.0 + 7.0 * freq_excess + rng.normal(0, 3.0, N)

    # Cannibalization: spike weeks boost display CPA by 12-18%
    cannibal_mult = np.where(
        spike_mask,
        1.0 + rng.uniform(0.12, 0.18, N),
        1.0 + rng.uniform(0.00, 0.02, N),
    )
    disp_cpa = (disp_base_cpa * cannibal_mult).clip(14.0)

    disp_impr = (disp_spend / disp_cpm * 1_000).round().astype(int)
    disp_clicks = (disp_impr * disp_ctr).round().astype(int).clip(1)
    disp_convs = (disp_spend / disp_cpa).round().astype(int).clip(0)
    disp_cvr = (disp_convs / disp_clicks).clip(0.002, 0.018)
    disp_reach = (disp_impr / disp_freq).round().astype(int)

    for i in range(N):
        rows.append({
            "date": dates[i], "channel": "programmatic_display",
            "impressions": int(disp_impr[i]),
            "clicks":      int(disp_clicks[i]),
            "conversions": int(disp_convs[i]),
            "spend":       round(float(disp_spend[i]), 2),
            "cpm":         round(float(disp_cpm[i]), 2),
            "cpc":         round(float(disp_spend[i] / disp_clicks[i]), 2),
            "cpa":         round(float(disp_cpa[i]), 2),
            "ctr":         round(float(disp_ctr[i]), 4),
            "cvr":         round(float(disp_cvr[i]), 4),
            "frequency":   round(float(disp_freq[i]), 2),
            "reach":       int(disp_reach[i]),
        })

    # ------------------------------------------------------------------ #
    # CTV — stable returns, slight improvement (contrast to YouTube)       #
    # ------------------------------------------------------------------ #
    ctv_spend = 2_000 * dow * rng.lognormal(0, 0.09, N)
    ctv_cpm = rng.uniform(28, 38, N)
    ctv_ctr = rng.normal(0.003, 0.0008, N).clip(0.001, 0.006)
    # CPA slowly improving: $85 → $72 (linear) with noise
    ctv_cpa = (85 - 0.14 * np.arange(N) + rng.normal(0, 8, N)).clip(50, 125)

    ctv_impr = (ctv_spend / ctv_cpm * 1_000).round().astype(int)
    ctv_clicks = (ctv_impr * ctv_ctr).round().astype(int).clip(1)
    ctv_convs = (ctv_spend / ctv_cpa).round().astype(int).clip(0)
    ctv_cvr = (ctv_convs / ctv_clicks).clip(0.02, 0.60)
    ctv_freq = rng.uniform(2.5, 4.5, N)
    ctv_reach = (ctv_impr / ctv_freq).round().astype(int)

    for i in range(N):
        rows.append({
            "date": dates[i], "channel": "ctv",
            "impressions": int(ctv_impr[i]),
            "clicks":      int(ctv_clicks[i]),
            "conversions": int(ctv_convs[i]),
            "spend":       round(float(ctv_spend[i]), 2),
            "cpm":         round(float(ctv_cpm[i]), 2),
            "cpc":         round(float(ctv_spend[i] / ctv_clicks[i]), 2),
            "cpa":         round(float(ctv_cpa[i]), 2),
            "ctr":         round(float(ctv_ctr[i]), 4),
            "cvr":         round(float(ctv_cvr[i]), 4),
            "frequency":   round(float(ctv_freq[i]), 2),
            "reach":       int(ctv_reach[i]),
        })

    # ------------------------------------------------------------------ #
    # YOUTUBE — failure mode 5: diminishing returns                        #
    # Logistic CPA curve: ~$40 early, ~$115 late (≈2.5-3x)               #
    # ------------------------------------------------------------------ #
    yt_spend = 3_000 * dow * rng.lognormal(0, 0.08, N)
    yt_cpm = rng.uniform(12, 18, N)
    yt_ctr = rng.normal(0.010, 0.002, N).clip(0.004, 0.020)
    # S-curve CPA increase: inflects around day 55
    t = np.arange(N, dtype=float)
    yt_cpa = (38 + 85 / (1 + np.exp(-0.07 * (t - 55))) + rng.normal(0, 7, N)).clip(28, 180)

    yt_impr = (yt_spend / yt_cpm * 1_000).round().astype(int)
    yt_clicks = (yt_impr * yt_ctr).round().astype(int).clip(1)
    yt_convs = (yt_spend / yt_cpa).round().astype(int).clip(0)
    yt_cvr = (yt_convs / yt_clicks).clip(0.002, 0.08)
    yt_freq = rng.uniform(3.0, 5.5, N)
    yt_reach = (yt_impr / yt_freq).round().astype(int)

    for i in range(N):
        rows.append({
            "date": dates[i], "channel": "youtube",
            "impressions": int(yt_impr[i]),
            "clicks":      int(yt_clicks[i]),
            "conversions": int(yt_convs[i]),
            "spend":       round(float(yt_spend[i]), 2),
            "cpm":         round(float(yt_cpm[i]), 2),
            "cpc":         round(float(yt_spend[i] / yt_clicks[i]), 2),
            "cpa":         round(float(yt_cpa[i]), 2),
            "ctr":         round(float(yt_ctr[i]), 4),
            "cvr":         round(float(yt_cvr[i]), 4),
            "frequency":   round(float(yt_freq[i]), 2),
            "reach":       int(yt_reach[i]),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "channel"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Creative-performance table
# ---------------------------------------------------------------------------

def _generate_creative_performance(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    campaigns: pd.DataFrame,
) -> pd.DataFrame:
    N = len(dates)

    # Spend split per channel across creatives
    spend_splits = {
        "paid_search":          [0.50, 0.30, 0.20],
        "paid_social":          [0.45, 0.25, 0.20, 0.10],  # social_v1 dominant
        "programmatic_display": [0.40, 0.35, 0.25],
        "ctv":                  [0.55, 0.45],
        "youtube":              [0.40, 0.35, 0.25],
    }

    # Stable base CTR per creative (no fatigue unless noted)
    base_ctr = {
        "search_v1": 0.042, "search_v2": 0.038, "search_v3": 0.034,
        "social_v1": 0.028,  # DECAYING — handled below
        "social_v2": 0.019, "social_v3": 0.016, "social_v4": 0.015,
        "display_v1": 0.0055, "display_v2": 0.0048, "display_v3": 0.0040,
        "ctv_v1": 0.0030, "ctv_v2": 0.0027,
        "yt_v1": 0.0110, "yt_v2": 0.0095, "yt_v3": 0.0100,
    }

    base_cvr = {
        "search_v1": 0.058, "search_v2": 0.052, "search_v3": 0.047,
        "social_v1": 0.030, "social_v2": 0.027, "social_v3": 0.024, "social_v4": 0.021,
        "display_v1": 0.011, "display_v2": 0.010, "display_v3": 0.009,
        "ctv_v1": 0.015, "ctv_v2": 0.013,
        "yt_v1": 0.020, "yt_v2": 0.018, "yt_v3": 0.016,
    }

    # social_v1 exponential decay parameters
    # CTR(t) = a * exp(-λ * t) + c,  t = days since day 25
    _lambda = np.log(2) / 20   # half-life = 20 days → λ ≈ 0.0347
    _a = 0.028 - 0.011         # initial amplitude above floor
    _c = 0.011                  # floor CTR

    cam_by_ch = {
        ch: df.set_index("date")
        for ch, df in campaigns.groupby("channel")
    }

    rows: list[dict] = []

    for ch, creatives in CREATIVES.items():
        splits = spend_splits[ch]
        ch_df = cam_by_ch[ch]

        for j, cid in enumerate(creatives):
            ch_spend_by_day = ch_df["spend"].values
            ch_cpm_by_day = ch_df["cpm"].values

            for i in range(N):
                crt_spend = float(ch_spend_by_day[i]) * splits[j]
                crt_cpm = float(ch_cpm_by_day[i]) * rng.uniform(0.95, 1.05)

                # CTR: social_v1 decays after day 25; all others stable ± noise
                if cid == "social_v1" and i >= 25:
                    t = float(i - 25)
                    true_ctr = _a * np.exp(-_lambda * t) + _c
                    crt_ctr = float(np.clip(rng.normal(true_ctr, 0.0015), 0.008, 0.035))
                else:
                    crt_ctr = float(np.clip(
                        rng.normal(base_ctr[cid], base_ctr[cid] * 0.08),
                        0.001, 0.12,
                    ))

                crt_cvr = float(np.clip(
                    rng.normal(base_cvr[cid], base_cvr[cid] * 0.10),
                    0.003, 0.15,
                ))

                impressions = max(0, int(crt_spend / crt_cpm * 1_000))
                clicks = max(0, int(impressions * crt_ctr))
                conversions = max(0, int(clicks * crt_cvr))

                rows.append({
                    "date":        dates[i],
                    "channel":     ch,
                    "creative_id": cid,
                    "impressions": impressions,
                    "clicks":      clicks,
                    "ctr":         round(crt_ctr, 5),
                    "conversions": conversions,
                    "spend":       round(crt_spend, 2),
                })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "channel", "creative_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Daypart-performance table
# ---------------------------------------------------------------------------

def _generate_daypart_performance(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    campaigns: pd.DataFrame,
) -> pd.DataFrame:
    # Impression share per daypart per channel
    # paid_search: prime & late_night underinvested (failure mode 4)
    impression_shares = {
        "paid_search": {
            "early_morning": 0.13, "morning": 0.22, "midday": 0.22,
            "afternoon": 0.22, "evening": 0.11, "prime": 0.05, "late_night": 0.05,
        },
        "paid_social": {
            "early_morning": 0.07, "morning": 0.14, "midday": 0.18,
            "afternoon": 0.24, "evening": 0.25, "prime": 0.08, "late_night": 0.04,
        },
        "programmatic_display": {
            "early_morning": 0.09, "morning": 0.14, "midday": 0.21,
            "afternoon": 0.24, "evening": 0.20, "prime": 0.08, "late_night": 0.04,
        },
        "ctv": {
            "early_morning": 0.04, "morning": 0.09, "midday": 0.11,
            "afternoon": 0.14, "evening": 0.30, "prime": 0.24, "late_night": 0.08,
        },
        "youtube": {
            "early_morning": 0.07, "morning": 0.14, "midday": 0.19,
            "afternoon": 0.21, "evening": 0.23, "prime": 0.12, "late_night": 0.04,
        },
    }

    # CVR multipliers per daypart per channel
    # paid_search: prime & late_night are 40-60% above average (failure mode 4)
    cvr_mults = {
        "paid_search": {
            "early_morning": 0.80, "morning": 1.05, "midday": 0.95,
            "afternoon": 1.00, "evening": 1.10, "prime": 1.58, "late_night": 1.52,
        },
        "paid_social": {
            "early_morning": 0.82, "morning": 1.00, "midday": 0.95,
            "afternoon": 1.12, "evening": 1.22, "prime": 1.18, "late_night": 0.75,
        },
        "programmatic_display": {
            "early_morning": 0.78, "morning": 1.00, "midday": 1.05,
            "afternoon": 1.10, "evening": 1.08, "prime": 0.95, "late_night": 0.65,
        },
        "ctv": {
            "early_morning": 0.68, "morning": 0.85, "midday": 0.90,
            "afternoon": 1.00, "evening": 1.22, "prime": 1.28, "late_night": 0.88,
        },
        "youtube": {
            "early_morning": 0.78, "morning": 0.95, "midday": 1.00,
            "afternoon": 1.05, "evening": 1.18, "prime": 1.22, "late_night": 0.92,
        },
    }

    # Base CTR per channel (used to derive daypart clicks from impressions)
    ch_base_ctr = {
        "paid_search": 0.040,
        "paid_social": 0.018,
        "programmatic_display": 0.005,
        "ctv": 0.003,
        "youtube": 0.010,
    }

    cam_by_ch = {
        ch: df.set_index("date")
        for ch, df in campaigns.groupby("channel")
    }

    rows: list[dict] = []

    for ch in campaigns["channel"].unique():
        ch_df = cam_by_ch[ch]
        base_cvr = float(ch_df["cvr"].mean())
        base_ctr = ch_base_ctr[ch]
        shares = impression_shares[ch]
        mults = cvr_mults[ch]

        for d in dates:
            ch_impr = int(ch_df.loc[d, "impressions"]) if d in ch_df.index else 200_000
            ch_spend = float(ch_df.loc[d, "spend"]) if d in ch_df.index else 3_000.0

            for dp in DAYPARTS:
                share = shares[dp]
                dp_impr = max(0, int(ch_impr * share * float(rng.lognormal(0, 0.06))))
                dp_spend = max(0.0, float(ch_spend * share * float(rng.lognormal(0, 0.06))))
                dp_clicks = max(0, int(dp_impr * base_ctr * float(rng.lognormal(0, 0.10))))
                dp_cvr = float(np.clip(
                    base_cvr * mults[dp] + rng.normal(0, base_cvr * 0.08),
                    0.001, 0.25,
                ))
                dp_convs = max(0, int(dp_clicks * dp_cvr))

                rows.append({
                    "date":        d,
                    "channel":     ch,
                    "daypart":     dp,
                    "impressions": dp_impr,
                    "clicks":      dp_clicks,
                    "conversions": dp_convs,
                    "spend":       round(dp_spend, 2),
                    "cvr":         round(dp_cvr, 5),
                })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "channel", "daypart"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_data(save_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate all three tables, save as CSVs, return the DataFrames."""
    rng = np.random.default_rng(SEED)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2026-01-15", "2026-04-14", freq="D")

    print("Generating campaigns table...")
    campaigns = _generate_campaigns(rng, dates)

    print("Generating creative_performance table...")
    creative = _generate_creative_performance(rng, dates, campaigns)

    print("Generating daypart_performance table...")
    daypart = _generate_daypart_performance(rng, dates, campaigns)

    campaigns.to_csv(save_path / "campaigns.csv", index=False)
    creative.to_csv(save_path / "creative_performance.csv", index=False)
    daypart.to_csv(save_path / "daypart_performance.csv", index=False)

    print(f"\nSaved to {save_path}/")
    print(f"  campaigns:            {campaigns.shape}")
    print(f"  creative_performance: {creative.shape}")
    print(f"  daypart_performance:  {daypart.shape}")

    return campaigns, creative, daypart


if __name__ == "__main__":
    generate_data()
