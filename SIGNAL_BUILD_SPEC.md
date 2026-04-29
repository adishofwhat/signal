# SIGNAL — Campaign Diagnostic Intelligence Engine
## Full Build Specification

### What This Is
An LLM-powered multi-channel media performance audit tool that mirrors what Known's MOA (Media Opportunity Analysis) does: ingest campaign data, run statistical diagnostics, surface optimization opportunities, and synthesize findings into a strategic narrative with actionable recommendations.

Built to demonstrate: statistical rigor + LLM reasoning + product thinking for the Known GenAI DS role.

### Stack
- Python 3.11+
- pandas, numpy (data)
- scipy, statsmodels (statistical methods)
- scikit-learn (clustering/segmentation where needed)
- anthropic SDK (LLM layer)
- plotly (interactive charts)
- streamlit (UI)

### Project Structure
```
signal/
├── README.md
├── requirements.txt
├── app.py                      # Streamlit entry point
├── data/
│   └── generator.py            # Synthetic campaign data generation
├── diagnostics/
│   ├── __init__.py
│   ├── creative_fatigue.py     # Exponential decay fitting
│   ├── frequency_saturation.py # Piecewise CPA modeling
│   ├── channel_cannibalization.py # Cross-channel correlation
│   ├── daypart_analysis.py     # Time-of-day opportunity detection
│   └── budget_efficiency.py    # Marginal ROI / efficiency frontier
├── intelligence/
│   ├── __init__.py
│   ├── synthesizer.py          # LLM prompt chain + response parsing
│   └── whatif.py               # What-if simulation engine
├── ui/
│   ├── __init__.py
│   ├── theme.py                # Styling / CSS
│   ├── components.py           # Reusable UI components
│   └── charts.py               # Plotly chart builders
└── config.py                   # Constants, thresholds, API config
```

---

## PHASE 1: Data Generation (`data/generator.py`)

### Goal
Generate realistic multi-channel campaign performance data with **planted failure modes** that the diagnostics layer will detect. The data should look like what Known's Skeptic ingests from platform APIs.

### Schema

**campaigns table** — one row per channel per day

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Daily, 90-day window |
| channel | str | One of: paid_search, paid_social, programmatic_display, ctv, youtube |
| impressions | int | Daily impressions |
| clicks | int | Daily clicks |
| conversions | int | Daily conversions |
| spend | float | Daily spend in USD |
| cpm | float | Cost per thousand impressions |
| cpc | float | Cost per click |
| cpa | float | Cost per acquisition |
| ctr | float | Click-through rate |
| cvr | float | Conversion rate |
| frequency | float | Avg impressions per unique user (daily) |
| reach | int | Unique users reached |

**creative_performance table** — one row per creative variant per channel per day

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Daily |
| channel | str | Channel name |
| creative_id | str | e.g., "search_v1", "social_v3" |
| impressions | int | |
| clicks | int | |
| ctr | float | |
| conversions | int | |
| spend | float | |

**daypart_performance table** — one row per channel per daypart per day

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Daily |
| channel | str | |
| daypart | str | "early_morning" (5-8), "morning" (8-11), "midday" (11-14), "afternoon" (14-17), "evening" (17-20), "prime" (20-23), "late_night" (23-5) |
| impressions | int | |
| clicks | int | |
| conversions | int | |
| spend | float | |
| cvr | float | |

### Failure Modes to Plant (these are what the diagnostics will find)

1. **Creative fatigue in paid_social**: creative "social_v1" should show a clear exponential CTR decay starting around day 25. CTR starts at ~2.8%, decays to ~1.1% by day 90 with a half-life of ~20 days. Other social creatives should show mild or no decay for contrast.

2. **Frequency saturation in programmatic_display**: Frequency should drift upward from ~4 to ~10 over the 90 days. CPA should remain flat until frequency hits ~7, then increase sharply (piecewise relationship). This models the real problem of over-serving the same users.

3. **Cross-channel cannibalization**: When paid_search spend increases by >15% week-over-week, programmatic_display CPA should increase by ~8-12% in the same period (audience overlap — the same users are being reached by both channels, and attribution assigns the conversion to search, making display look worse).

4. **Daypart opportunity in paid_search**: The "prime" (20-23) and "late_night" (23-5) dayparts should have 40-60% higher CVR than the average for paid_search, but should only receive ~10% of impressions. This represents an untapped opportunity — the MOA case study about time-of-day optimization.

5. **Budget inefficiency in youtube**: YouTube should be past the point of diminishing returns. Marginal CPA in the last 30 days should be 2-3x the marginal CPA in the first 30 days at similar spend levels. Meanwhile, CTV should show linear or improving returns — budget should shift from YouTube to CTV.

### Data Generation Approach
- Use numpy random with seeded RNG for reproducibility
- Base each channel on realistic industry benchmarks:
  - Paid Search: CPM ~$20-35, CTR ~3-5%, CVR ~4-7%, daily spend ~$3-5K
  - Paid Social: CPM ~$8-15, CTR ~1-3%, CVR ~2-4%, daily spend ~$4-6K
  - Programmatic Display: CPM ~$4-8, CTR ~0.3-0.8%, CVR ~0.5-1.5%, daily spend ~$2-4K
  - CTV: CPM ~$25-40, CTR ~N/A (use VCR ~85-95%), CVR ~1-2%, daily spend ~$1.5-3K
  - YouTube: CPM ~$10-20, CTR ~0.5-1.5%, CVR ~1-3%, daily spend ~$2.5-4K
- Add Gaussian noise (5-15% coefficient of variation) to all metrics for realism
- Add day-of-week seasonality (weekends slightly different)
- Total daily spend should be ~$15-22K (realistic for a mid-market client Known would serve)
- Total 90-day budget: ~$1.5-2M

### Config for the demo
```python
# config.py
CAMPAIGN_CONFIG = {
    "client_name": "Meridian Outdoor Co.",  # Fictional outdoor/lifestyle brand
    "campaign_name": "Spring 2026 Performance Campaign",
    "date_range": ("2026-01-15", "2026-04-14"),  # 90 days
    "total_budget": 1_800_000,
    "objective": "Drive online conversions (e-commerce purchases) while maintaining brand awareness",
    "channels": ["paid_search", "paid_social", "programmatic_display", "ctv", "youtube"],
}
```

---

## PHASE 2: Statistical Diagnostics (`diagnostics/`)

Each diagnostic module should return a standardized result dict:

```python
@dataclass
class DiagnosticResult:
    name: str                    # e.g., "creative_fatigue"
    severity: str                # "critical", "warning", "info"
    channel: str                 # which channel(s) affected
    finding: str                 # one-line summary
    evidence: dict               # statistical details (test stats, p-values, parameters)
    estimated_impact_usd: float  # rough dollar impact estimate
    charts: list                 # list of plotly figure objects
```

### 2a. Creative Fatigue Detection (`creative_fatigue.py`)

**Method**: For each creative variant with >30 days of data, fit an exponential decay model to daily CTR:

```
CTR(t) = a * exp(-λ * t) + c
```

Where:
- `a` = initial CTR amplitude above floor
- `λ` = decay rate (higher = faster fatigue)
- `c` = floor CTR (what it decays toward)
- `t` = days since creative launch

Use `scipy.optimize.curve_fit` with bounds:
- a: [0, 0.10]
- λ: [0, 0.20]
- c: [0, 0.05]

Compute **half-life** = ln(2) / λ

**Decision rule**: Flag as "fatigued" if:
- half_life < 30 days AND
- current CTR < 60% of initial CTR AND
- p-value of decay fit (via residual analysis) < 0.05

**Estimated impact**: 
```
impact = (initial_ctr - current_ctr) / initial_ctr * channel_spend_last_30d * avg_cpa_ratio
```
Rough estimate: "if you had rotated this creative at the optimal point, you would have saved approximately $X in wasted spend"

**Chart**: Line chart showing actual CTR over time with the fitted decay curve overlaid. Mark the "optimal rotation point" (where marginal CTR loss exceeds threshold). Show other creatives on the same chart for comparison.

### 2b. Frequency Saturation (`frequency_saturation.py`)

**Method**: Bin daily observations by frequency (e.g., 0-2, 2-4, 4-6, 6-8, 8-10, 10+). Compute mean CPA per bin. Fit a piecewise linear model (segmented regression) with one breakpoint:

```
CPA(f) = β0 + β1*f                    for f ≤ breakpoint
CPA(f) = β0 + β1*bp + β2*(f - bp)     for f > breakpoint
```

Use `scipy.optimize.minimize` or a grid search over candidate breakpoints to find the one that minimizes total squared error.

**Alternative**: Use `numpy.polyfit` with degree 2 (quadratic) and find the inflection point analytically. The quadratic is simpler and may be more robust.

**Decision rule**: Flag if:
- Breakpoint frequency < current average frequency (meaning you're past the saturation point)
- CPA above breakpoint is >20% higher than CPA below breakpoint
- At least 10 observations in each segment

**Estimated impact**:
```
days_above_threshold = count of days where frequency > breakpoint
excess_cpa = mean_cpa_above - mean_cpa_at_breakpoint
impact = excess_cpa * conversions_in_those_days
```

**Chart**: Scatter plot of frequency vs. CPA with the fitted piecewise/quadratic curve. Mark the breakpoint. Shade the "saturation zone" in red.

### 2c. Cross-Channel Cannibalization (`channel_cannibalization.py`)

**Method**: For each pair of channels, compute:
1. Weekly spend changes (% change week-over-week) for channel A
2. Weekly CPA changes for channel B
3. Pearson correlation between (1) and (2) with a 0 or 1-week lag

**Decision rule**: Flag if:
- Correlation > 0.5 AND p-value < 0.05
- The direction is: when channel A spend ↑, channel B CPA ↑ (positive correlation)
- This suggests audience overlap — scaling A steals attributed conversions from B

Also compute: Granger causality test (from statsmodels) at lag 1 to test whether channel A spend changes *predict* channel B CPA changes. This adds causal directionality.

**Estimated impact**:
```
When channel A spend increases by X%, channel B CPA increases by Y%
At current spend levels, this cannibalization costs approximately $Z/month
```

**Chart**: Dual-axis time series showing channel A weekly spend and channel B weekly CPA. Highlight the correlated periods. Maybe a scatter with regression line.

### 2d. Daypart Opportunity (`daypart_analysis.py`)

**Method**: For each channel, run Kruskal-Wallis H-test across dayparts on CVR (conversion rate). If significant (p < 0.05), run pairwise Dunn's post-hoc tests to identify which dayparts differ.

Compute an "opportunity score" for each daypart:
```
opportunity = (daypart_cvr - channel_avg_cvr) / channel_avg_cvr * (1 - daypart_impression_share)
```

High opportunity = high CVR but low impression share (underinvested).
Low opportunity = low CVR but high impression share (overinvested).

**Decision rule**: Flag if:
- Kruskal-Wallis p < 0.05 AND
- At least one daypart has CVR > 30% above average AND impression share < 15%
- Or: at least one daypart has CVR > 20% below average AND impression share > 20%

**Estimated impact**:
```
If impressions were reallocated to match CVR ranking (proportional to CVR),
estimated additional conversions = X
estimated revenue impact = X * avg_order_value
```

**Chart**: Grouped bar chart — CVR by daypart (bars) with impression share overlay (line). Color-code high-opportunity dayparts in green, overinvested in red.

### 2e. Budget Efficiency Frontier (`budget_efficiency.py`)

**Method**: For each channel, compute rolling 14-day marginal CPA:
```
marginal_cpa = Δspend / Δconversions (over 14-day windows)
```

Compare marginal CPA in the first 30 days vs. last 30 days. If marginal CPA in the last 30 days is significantly higher (>30%), the channel is in diminishing returns territory.

Also compute channel-level ROAS (return on ad spend) assuming a fixed AOV (average order value, set in config):
```
roas = (conversions * aov) / spend
```

**Decision rule**: Flag if:
- Any channel's marginal CPA in last 30 days > 1.5x marginal CPA in first 30 days
- Any channel has ROAS < 1.0 (spending more than earning)
- Cross-channel comparison shows >2x difference in marginal CPA between best and worst channel

**Estimated impact**:
```
If $X shifted from [worst marginal CPA channel] to [best marginal CPA channel],
estimated additional conversions at same spend = Y
```

**Chart**: Line chart of rolling marginal CPA by channel over time. Highlight divergence. Maybe a waterfall chart showing the reallocation recommendation.

---

## PHASE 3: LLM Intelligence Layer (`intelligence/`)

### 3a. Synthesizer (`synthesizer.py`)

**Architecture**: Two-stage prompt chain.

**Stage 1: Finding Synthesis**
- Input: All DiagnosticResult objects serialized as JSON
- System prompt: "You are a senior media strategist at a data-driven advertising agency. You are reviewing the output of a quantitative media audit conducted on a client's paid media campaign. Your job is to synthesize these statistical findings into a clear, actionable strategic narrative."
- User prompt: Contains the campaign config (client name, objective, budget, date range) + all diagnostic results
- Output format (JSON):
```json
{
  "executive_summary": "2-3 sentence overview of the campaign's health and top opportunities",
  "findings": [
    {
      "rank": 1,
      "title": "Creative fatigue in Paid Social is eroding performance",
      "severity": "critical",
      "narrative": "3-4 sentences explaining what's happening and why it matters",
      "recommendation": "Specific, actionable recommendation",
      "estimated_impact": "$XX,XXX",
      "confidence": "high/medium/low"
    }
  ],
  "cross_channel_insights": "2-3 sentences about how findings connect across channels",
  "action_plan": [
    {"priority": 1, "action": "...", "timeline": "immediate/this_week/this_month", "owner": "media_buyer/creative_team/analyst"}
  ]
}
```

**Stage 2: Narrative Report** (optional, for the expanded view)
- Input: Stage 1 output + original diagnostic evidence
- Prompt: "Write a concise but thorough campaign diagnostic report based on these findings. Write it as if presenting to a CMO — clear, direct, no jargon. Reference specific numbers from the data."
- Output: Markdown-formatted report

### 3b. What-If Simulator (`whatif.py`)

**Concept**: User adjusts one of several levers:
1. Budget reallocation: shift X% from channel A to channel B
2. Frequency cap adjustment: set new cap for a channel
3. Creative rotation: simulate rotating the fatigued creative at day X
4. Daypart reallocation: shift impressions to high-CVR dayparts

**Implementation**:
- For each lever, compute the projected impact using the statistical models already fitted in Phase 2:
  - Budget shift: use the marginal CPA curves to project new conversion counts
  - Frequency cap: use the saturation model to project CPA at new frequency
  - Creative rotation: use the decay model to project CTR if rotated earlier
  - Daypart reallocation: use CVR differences to project conversion lift
- The projection is computed in Python (deterministic)
- The LLM receives the projection + context and generates a natural language explanation of the expected impact

**UI**: Sliders for each lever. Results update in real-time (stats are pre-computed; LLM call happens on "Generate Analysis" button click).

---

## PHASE 4: Streamlit UI (`app.py`, `ui/`)

### Design Principles
- Dark, editorial aesthetic (matches the persona project's approach — this is a Known-ready tool)
- Minimal color palette: dark bg (#0a0a0a or #111111), white text, gold accent (#c8a55a), red for critical findings (#e74c3c), green for opportunities (#2ecc71)
- Typography: clean sans-serif (Inter or similar available via Google Fonts in Streamlit)
- No Streamlit default blue anywhere

### Layout

**Header**: 
```
SIGNAL
Campaign Diagnostic Intelligence
```
Small subtext: "Meridian Outdoor Co. — Spring 2026 Performance Campaign"

**Tab 1: Executive Overview**
- Campaign summary stats (total spend, total conversions, blended CPA, blended ROAS) in 4 metric cards
- AI-generated executive summary (from LLM)
- Finding cards ranked by severity/impact — each card shows: icon, title, severity badge, estimated impact, 1-line summary. Click to expand.

**Tab 2: Diagnostic Deep Dive**
- One expandable section per diagnostic
- Each section: finding narrative + interactive Plotly chart + statistical evidence (collapsible)
- Statistical evidence shows: test statistics, p-values, model parameters, confidence intervals — for the PhDs who want to see the math

**Tab 3: What-If Simulator**
- Left panel: adjustment sliders
- Right panel: projected impact (before/after comparison)
- Bottom: "Generate Strategic Analysis" button → LLM-generated explanation of the what-if scenario
- Show the delta: "Projected additional conversions: +X | Projected CPA reduction: -Y% | Projected savings: $Z"

**Tab 4: Action Plan**
- LLM-generated prioritized action plan
- Timeline view: immediate / this week / this month
- Each action tagged with owner (media buyer, creative team, analyst)

### Streamlit Custom CSS
```python
# ui/theme.py
CUSTOM_CSS = """
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0a0a0a;
        color: #e8e8e8;
    }
    /* ... (full CSS in implementation) */
</style>
"""
```

---

## PHASE 5: README.md

### Critical Framing (what to include)

**What it is**: A prototype of an LLM-powered media diagnostic tool that demonstrates how generative AI can accelerate campaign performance audits — the kind of work Known's MOA team does for Fortune 500 clients using Skeptic.

**What the statistical layer does**: Five diagnostic modules that detect real optimization opportunities in campaign data: creative fatigue, frequency saturation, cross-channel cannibalization, daypart inefficiency, and budget allocation problems. Each module uses specific statistical methods (exponential decay fitting, piecewise regression, Granger causality, Kruskal-Wallis H-tests, marginal CPA analysis).

**What the LLM layer does**: Synthesizes statistical findings into a strategic narrative — the kind of insight report a senior media strategist would write. The LLM explains and recommends; it doesn't make the statistical calls. This architectural separation (quant layer computes, LLM layer narrates) is deliberate.

**What the what-if simulator shows**: Interactive scenario exploration where adjusting a parameter (budget, frequency cap, creative rotation timing) produces projected impact grounded in the statistical models. This mirrors the simulation capability of tools like Skeptic.

**Honest limitations**:
- Data is synthetic. In production, the interesting problems are: handling missing/inconsistent data from platform APIs, dealing with attribution model discrepancies across channels, validating that detected anomalies are real and not artifacts of seasonality or external factors.
- The statistical models are fitted to clean data with planted patterns. Real campaign data has confounders — promotions, competitor activity, news cycles — that require more robust causal inference methods.
- The what-if projections assume ceteris paribus. Real budget shifts trigger platform algorithm responses (auction dynamics, learning phases) that this model doesn't capture.

**What would change in production**:
- Data ingestion from platform APIs (Meta Marketing API, Google Ads API, The Trade Desk API)
- Bayesian structural time series (CausalImpact) instead of simple correlation for cannibalization detection
- Integration with MMM (marketing mix model) outputs for budget allocation
- Feedback loop: track whether implemented recommendations actually improved performance

**Stack**: Python, scipy, statsmodels, scikit-learn, Anthropic API (Claude), Plotly, Streamlit

---

## BUILD ORDER

### Step 1: Data generation (2-3 hours)
Build `data/generator.py` and `config.py`. Generate the three tables. Save as CSV for inspection. Verify the planted failure modes are visible in simple plots.

### Step 2: Diagnostic modules (3-4 hours)
Build each diagnostic module one at a time. Test each independently against the generated data. Verify each one correctly detects the planted failure mode. Start with creative_fatigue (most visually impressive) → frequency_saturation → daypart_analysis → channel_cannibalization → budget_efficiency.

### Step 3: Charts (1-2 hours)
Build `ui/charts.py` with Plotly chart functions for each diagnostic. Make them interactive (hover, zoom). Use the dark theme color palette.

### Step 4: LLM integration (2-3 hours)
Build `intelligence/synthesizer.py`. Test the prompt chain. Iterate on the prompt until the output is clean, specific, and references actual numbers from the data. Parse the JSON response robustly.

### Step 5: Streamlit UI (2-3 hours)
Build `app.py` with tabs. Wire up the diagnostics + charts + LLM output. Apply custom CSS.

### Step 6: What-if simulator (2-3 hours)
Build `intelligence/whatif.py` and the Tab 3 UI. This is the creative differentiator — spend time making the projections feel real and the UI feel responsive.

### Step 7: README + Polish (1-2 hours)
Write the README with the framing above. Deploy to Streamlit Community Cloud or a public URL. Test the deployed version.

**Total estimated time: 14-20 hours**

---

## DEPLOYMENT

Option 1: **Streamlit Community Cloud** (free, easy, public URL)
- Push to GitHub repo
- Connect to share.streamlit.io
- Set ANTHROPIC_API_KEY as a secret

Option 2: **Railway / Render** (more control, still free tier available)

The deployed URL is what goes in the message to Sibi. It must be clickable and working.

---

## NAMING

**SIGNAL** — Campaign Diagnostic Intelligence

Alternatively: **AUDIT** or **PULSE** or **MERIDIAN** (after the fictional client)

Keep it clean, one word, all caps. It should read like an internal tool name at an agency.