"""
SIGNAL Phase 1 sanity check.
Prints campaigns.describe() and produces five failure-mode verification plots.
Run from the signal/ directory: python sanity_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

# Generate data if CSVs don't exist
DATA_DIR = Path(__file__).parent / "data"
_csvs = ["campaigns.csv", "creative_performance.csv", "daypart_performance.csv"]

if not all((DATA_DIR / f).exists() for f in _csvs):
    sys.path.insert(0, str(Path(__file__).parent))
    from data.generator import generate_data
    generate_data(str(DATA_DIR))

campaigns = pd.read_csv(DATA_DIR / "campaigns.csv", parse_dates=["date"])
creative  = pd.read_csv(DATA_DIR / "creative_performance.csv", parse_dates=["date"])
daypart   = pd.read_csv(DATA_DIR / "daypart_performance.csv", parse_dates=["date"])

# ---------------------------------------------------------------------------
# Shape & basic quality checks
# ---------------------------------------------------------------------------
print("=" * 70)
print("SHAPE CHECK")
print(f"  campaigns            : {campaigns.shape}  (expect ~450 rows)")
print(f"  creative_performance : {creative.shape}  (expect 1350-2250 rows)")
print(f"  daypart_performance  : {daypart.shape}  (expect ~3150 rows)")

print("\nDATA QUALITY")
for name, df in [("campaigns", campaigns), ("creative", creative), ("daypart", daypart)]:
    nulls = df.isnull().sum().sum()
    neg = (df.select_dtypes(include="number") < 0).sum().sum()
    print(f"  {name:24s}  nulls={nulls}  negatives={neg}")

print("\nCAMPAIGNS .describe()")
print(campaigns.describe().to_string())
print()

# ---------------------------------------------------------------------------
# Five-panel failure-mode plots
# ---------------------------------------------------------------------------
DAYPART_ORDER = ["early_morning", "morning", "midday", "afternoon", "evening", "prime", "late_night"]
START = pd.Timestamp("2026-01-15")
END   = pd.Timestamp("2026-04-14")

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("#f4f4f4")
fig.suptitle(
    "SIGNAL — Phase 1 Sanity Check: Five Failure Mode Verification\n"
    "Meridian Outdoor Co. · Spring 2026 Performance Campaign",
    fontsize=13, fontweight="bold", y=0.98,
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

COLORS = {
    "primary":  "#2c7bb6",
    "accent":   "#d7191c",
    "green":    "#1a9641",
    "orange":   "#fdae61",
    "purple":   "#762a83",
    "gray":     "#aaaaaa",
}

# ============================================================
# Plot 1 — Creative fatigue: social_v1 CTR decay
# ============================================================
ax1 = fig.add_subplot(gs[0, 0])
social = creative[creative["channel"] == "paid_social"].copy()
social["day"] = (social["date"] - START).dt.days

style = {
    "social_v1": dict(color=COLORS["accent"], lw=2.5, ls="-", alpha=1.0, zorder=5),
    "social_v2": dict(color=COLORS["gray"],   lw=1.0, ls="--", alpha=0.6, zorder=3),
    "social_v3": dict(color=COLORS["gray"],   lw=1.0, ls=":",  alpha=0.6, zorder=3),
    "social_v4": dict(color=COLORS["gray"],   lw=1.0, ls="-.", alpha=0.6, zorder=3),
}
for cid, grp in social.groupby("creative_id"):
    grp_sorted = grp.sort_values("date")
    label = f"{cid} ← FATIGUING" if cid == "social_v1" else cid
    ax1.plot(grp_sorted["day"], grp_sorted["ctr"] * 100, label=label, **style.get(cid, {}))

ax1.axvline(25, color=COLORS["accent"], ls=":", lw=1.2, alpha=0.8, label="Decay onset (day 25)")
ax1.set_title("Plot 1 · Creative Fatigue\nPaid Social CTR by Creative", fontweight="bold")
ax1.set_xlabel("Campaign Day")
ax1.set_ylabel("CTR (%)")
ax1.legend(fontsize=7.5, loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 89)

# ============================================================
# Plot 2 — Frequency saturation: display freq vs CPA hockey-stick
# ============================================================
ax2 = fig.add_subplot(gs[0, 1])
disp = campaigns[campaigns["channel"] == "programmatic_display"].copy()
disp["day"] = (disp["date"] - START).dt.days

# Color by phase: before / after saturation threshold (freq=7)
early = disp[disp["frequency"] <= 7]
late  = disp[disp["frequency"] > 7]

ax2.scatter(early["frequency"], early["cpa"], c=COLORS["primary"],  alpha=0.55, s=22, label="freq ≤ 7")
ax2.scatter(late["frequency"],  late["cpa"],  c=COLORS["accent"],   alpha=0.55, s=22, label="freq > 7 (saturation)")

# Fit and plot a reference quadratic
z = np.polyfit(disp["frequency"], disp["cpa"], 2)
f_range = np.linspace(disp["frequency"].min(), disp["frequency"].max(), 200)
ax2.plot(f_range, np.polyval(z, f_range), color="#333333", lw=1.5, ls="--", label="quadratic fit", zorder=6)
ax2.axvline(7.0, color=COLORS["accent"], ls=":", lw=1.5, label="Saturation threshold (f=7)")

ax2.set_title("Plot 2 · Frequency Saturation\nDisplay: Frequency vs CPA", fontweight="bold")
ax2.set_xlabel("Daily Frequency (avg impr/user)")
ax2.set_ylabel("CPA ($)")
ax2.legend(fontsize=7.5)
ax2.grid(True, alpha=0.3)

# ============================================================
# Plot 3 — Cannibalization: paid_search spend vs display CPA (weekly)
# Use campaign weeks (day // 7) so spike blocks align cleanly with aggregation.
# ============================================================
ax3 = fig.add_subplot(gs[0, 2])

camps_w = campaigns.copy()
camps_w["camp_week"] = ((camps_w["date"] - START).dt.days // 7)

ps_wk   = camps_w[camps_w["channel"] == "paid_search"         ].groupby("camp_week")["spend"].sum()
disp_wk = camps_w[camps_w["channel"] == "programmatic_display"].groupby("camp_week")["cpa"].mean()

# WoW % change in paid_search spend
ps_wow_pct = ps_wk.pct_change() * 100
spike_wks  = ps_wow_pct[ps_wow_pct > 15].index.tolist()

x = np.arange(len(ps_wk))
ax3.bar(x, ps_wk.values / 1_000, color=COLORS["primary"], alpha=0.40, label="PS Spend ($K)")
for sw in spike_wks:
    ax3.bar(sw, ps_wk.iloc[sw] / 1_000, color=COLORS["primary"], alpha=0.90,
            edgecolor=COLORS["accent"], linewidth=2.0, label="_")

ax3t = ax3.twinx()
ax3t.plot(x, disp_wk.values, color=COLORS["accent"], marker="o", ms=5, lw=2.2, label="Display CPA ($)")
for sw in spike_wks:
    ax3t.axvspan(sw - 0.4, sw + 0.4, alpha=0.12, color=COLORS["accent"])

ax3.set_title("Plot 3 · Channel Cannibalization\nSearch Spend vs Display CPA (campaign weeks)", fontweight="bold")
ax3.set_xlabel("Campaign Week")
ax3.set_ylabel("PS Weekly Spend ($K)", color=COLORS["primary"])
ax3t.set_ylabel("Display CPA ($)", color=COLORS["accent"])
ax3.tick_params(axis="y", colors=COLORS["primary"])
ax3t.tick_params(axis="y", colors=COLORS["accent"])

lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3t.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc="upper left")
ax3.text(0.02, 0.02, "Shaded = search spike week (WoW >+15%)", transform=ax3.transAxes,
         fontsize=7, color=COLORS["accent"])
ax3.grid(True, alpha=0.3)

# ============================================================
# Plot 4 — Daypart opportunity: paid_search CVR & impression share
# ============================================================
ax4 = fig.add_subplot(gs[1, 0])

ps_dp = (
    daypart[daypart["channel"] == "paid_search"]
    .groupby("daypart")
    .agg(cvr=("cvr", "mean"), impressions=("impressions", "sum"))
    .reindex(DAYPART_ORDER)
    .reset_index()
)
ps_dp["impr_share"] = ps_dp["impressions"] / ps_dp["impressions"].sum()
avg_cvr = ps_dp["cvr"].mean()

bar_colors = []
for _, row in ps_dp.iterrows():
    if row["cvr"] > avg_cvr * 1.30 and row["impr_share"] < 0.10:
        bar_colors.append(COLORS["green"])    # high opportunity
    elif row["cvr"] < avg_cvr * 0.90 and row["impr_share"] > 0.15:
        bar_colors.append(COLORS["accent"])   # over-invested + weak CVR
    else:
        bar_colors.append(COLORS["primary"])

ax4.bar(ps_dp["daypart"], ps_dp["cvr"] * 100, color=bar_colors, alpha=0.75, zorder=3)
ax4.axhline(avg_cvr * 100, color="#333333", ls="--", lw=1.2, label=f"Channel avg CVR ({avg_cvr*100:.2f}%)", zorder=5)

ax4t = ax4.twinx()
ax4t.plot(ps_dp["daypart"], ps_dp["impr_share"] * 100, color=COLORS["orange"],
          marker="D", ms=6, lw=2, label="Impression Share (%)", zorder=6)

ax4.set_title("Plot 4 · Daypart Opportunity\nPaid Search CVR vs Impression Share", fontweight="bold")
ax4.set_xlabel("Daypart")
ax4.set_ylabel("CVR (%)")
ax4t.set_ylabel("Impression Share (%)", color=COLORS["orange"])
ax4.tick_params(axis="x", rotation=35)
ax4t.tick_params(axis="y", colors=COLORS["orange"])
ax4.grid(True, alpha=0.3, zorder=0)

from matplotlib.patches import Patch
legend_elements = [
    Patch(color=COLORS["green"],   label="High CVR, low share → opportunity"),
    Patch(color=COLORS["accent"],  label="Low CVR, high share → over-invested"),
    Patch(color=COLORS["primary"], label="Balanced"),
]
lines4t, labs4t = ax4t.get_legend_handles_labels()
ax4.legend(handles=legend_elements + lines4t,
           labels=[p.get_label() for p in legend_elements] + labs4t,
           fontsize=6.5, loc="upper left")

# ============================================================
# Plot 5 — Budget efficiency: YouTube vs CTV rolling CPA
# ============================================================
ax5 = fig.add_subplot(gs[1, 1])

for ch, color, label in [
    ("youtube", COLORS["accent"],  "YouTube (diminishing returns)"),
    ("ctv",     COLORS["primary"], "CTV (stable / improving)"),
]:
    ch_data = campaigns[campaigns["channel"] == ch].sort_values("date").copy()
    ch_data["rolling_cpa"] = ch_data["cpa"].rolling(14, min_periods=5).mean()
    ax5.plot(ch_data["date"], ch_data["rolling_cpa"], color=color, lw=2.2, label=label)

# Mark the first-30 / last-30 day zones
d30 = START + pd.Timedelta(days=30)
d60 = END   - pd.Timedelta(days=30)
ax5.axvspan(START, d30, alpha=0.06, color=COLORS["green"], label="First 30 days")
ax5.axvspan(d60, END,   alpha=0.06, color=COLORS["accent"], label="Last 30 days")

ax5.set_title("Plot 5 · Budget Efficiency\nYouTube vs CTV Rolling CPA (14-day)", fontweight="bold")
ax5.set_xlabel("Date")
ax5.set_ylabel("CPA ($)")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)
ax5.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
ax5.tick_params(axis="x", rotation=30)

# Add first/last 30d ratio annotation
yt = campaigns[campaigns["channel"] == "youtube"].sort_values("date").copy()
yt_early = yt.iloc[:30]["cpa"].mean()
yt_late  = yt.iloc[-30:]["cpa"].mean()
ax5.text(0.03, 0.93,
         f"YT marginal CPA ratio (last30/first30): {yt_late/yt_early:.1f}×",
         transform=ax5.transAxes, fontsize=8, color=COLORS["accent"],
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["accent"], alpha=0.8))

# Hide unused subplot
fig.add_subplot(gs[1, 2]).axis("off")

out_path = DATA_DIR / "sanity_check.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nPlots saved → {out_path}")
plt.close()

# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FAILURE MODE SUMMARY")

# 1. Creative fatigue
sv1 = creative[(creative["creative_id"] == "social_v1")].sort_values("date")
early_ctr = sv1.iloc[:10]["ctr"].mean()
late_ctr  = sv1.iloc[-10:]["ctr"].mean()
print(f"\n1. Creative fatigue (social_v1)")
print(f"   CTR days 0-9:  {early_ctr*100:.2f}%   CTR days 80-89: {late_ctr*100:.2f}%"
      f"   decay: {(1 - late_ctr/early_ctr)*100:.0f}%")

# 2. Frequency saturation
disp_lo = disp[disp["frequency"] <= 7]["cpa"].mean()
disp_hi = disp[disp["frequency"] > 7 ]["cpa"].mean()
print(f"\n2. Frequency saturation (programmatic_display)")
print(f"   Mean CPA (freq≤7): ${disp_lo:.2f}   Mean CPA (freq>7): ${disp_hi:.2f}"
      f"   uplift: {(disp_hi/disp_lo - 1)*100:.0f}%")

# 3. Cannibalization — WoW% changes correlated (campaign-week aligned)
ps_wow_pct   = ps_wk.pct_change().dropna()
disp_wow_pct = disp_wk.pct_change().dropna()
corr = float(ps_wow_pct.corr(disp_wow_pct))
print(f"\n3. Cannibalization (search WoW%spend vs display WoW%CPA, campaign-week aligned)")
print(f"   Pearson r = {corr:.3f}  (positive = search spikes → display CPA spikes)")

# 4. Daypart opportunity
prime_cvr = ps_dp[ps_dp["daypart"] == "prime"]["cvr"].values[0]
ln_cvr    = ps_dp[ps_dp["daypart"] == "late_night"]["cvr"].values[0]
prime_share = ps_dp[ps_dp["daypart"] == "prime"]["impr_share"].values[0]
ln_share    = ps_dp[ps_dp["daypart"] == "late_night"]["impr_share"].values[0]
print(f"\n4. Daypart opportunity (paid_search)")
print(f"   prime CVR: {prime_cvr*100:.2f}% ({prime_cvr/avg_cvr:.1f}× avg),  share: {prime_share*100:.1f}%")
print(f"   late_night CVR: {ln_cvr*100:.2f}% ({ln_cvr/avg_cvr:.1f}× avg),  share: {ln_share*100:.1f}%")

# 5. YouTube vs CTV efficiency
ctv = campaigns[campaigns["channel"] == "ctv"].sort_values("date")
ctv_early = ctv.iloc[:30]["cpa"].mean()
ctv_late  = ctv.iloc[-30:]["cpa"].mean()
print(f"\n5. Budget efficiency")
print(f"   YouTube CPA first30: ${yt_early:.2f}  last30: ${yt_late:.2f}  ratio: {yt_late/yt_early:.2f}×")
print(f"   CTV    CPA first30: ${ctv_early:.2f}  last30: ${ctv_late:.2f}  ratio: {ctv_late/ctv_early:.2f}×")

print("\n" + "=" * 70)
print("CHECKPOINT 1 — pass/fail indicators")
checks = [
    ("Creative fatigue visible (decay >30%)",    (1 - late_ctr/early_ctr) > 0.30),
    ("Frequency saturation visible (>20% CPA↑)", (disp_hi/disp_lo - 1) > 0.20),
    ("Cannibalization signal (WoW% r>0.40)",       corr > 0.40),
    ("Daypart prime CVR >1.30× avg",              prime_cvr / avg_cvr > 1.30),
    ("Daypart prime impr share <10%",             prime_share < 0.10),
    ("YouTube last30/first30 CPA ratio >2.0×",    yt_late / yt_early > 2.0),
    ("CTV CPA stable (ratio <1.3×)",              ctv_late / ctv_early < 1.30),
]
for label, passed in checks:
    status = "PASS ✓" if passed else "FAIL ✗"
    print(f"  [{status}] {label}")
