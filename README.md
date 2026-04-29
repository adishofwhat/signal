# SIGNAL — Campaign Diagnostic Intelligence Engine

SIGNAL is a prototype of an LLM-powered media performance diagnostic tool that demonstrates how generative AI can accelerate paid media campaign audits. It was built to mirror the kind of work Known's MOA (Media Opportunity Analysis) team does with Skeptic — ingest campaign data, run statistical diagnostics, surface optimization opportunities, and synthesize findings into a strategic narrative with actionable recommendations. The demo client is a fictional outdoor/lifestyle brand (Meridian Outdoor Co.) running a 90-day, $1.8M multi-channel performance campaign across paid search, paid social, programmatic display, CTV, and YouTube.

---

## What the statistical layer does

Five diagnostic modules, each implementing a specific statistical method against the campaign data:

| Diagnostic | Method | Failure mode detected |
|---|---|---|
| **Creative Fatigue** | Exponential decay fitting (`scipy.optimize.curve_fit`) to model CTR(t) = a·e^(−λt) + c. Onset detection isolates the plateau-then-decay pattern. | social_v1 CTR decayed 53% (2.80% → 1.32%) with a half-life of 14.5 days |
| **Frequency Saturation** | Piecewise linear regression via grid-search breakpoint over daily frequency vs CPA. | Programmatic display past saturation at frequency 7.1; current avg 9.7 → CPA 41% higher above breakpoint |
| **Channel Cannibalization** | Pearson correlation between weekly paid search spend changes and programmatic display CPA changes, with Granger causality test for directional evidence. | r = 0.65 (p = 0.02): search spend spikes correlate with display CPA inflation |
| **Daypart Opportunity** | Kruskal-Wallis H-test across dayparts on CVR; Dunn's post-hoc tests for pairwise differences. Opportunity score = (daypart CVR − avg CVR) / avg CVR × (1 − impression share). | Prime (20–23h) and Late Night (23–5h) have 1.38× average CVR but receive only 5–7% of impressions |
| **Budget Efficiency** | Rolling 14-day marginal CPA per channel; first-30d vs last-30d comparison; Welch's t-test for significance. | YouTube marginal CPA ratio 2.65× (first-30d $57 → last-30d $151); CTV showing stable/improving returns |

The LLM explains and recommends — it does not make the statistical calls. All flagging decisions use predetermined thresholds (e.g., half-life < 30 days, CPA uplift > 20% above breakpoint, correlation > 0.5 at p < 0.05). The LLM receives the pre-computed evidence and synthesizes a strategic narrative grounded in those numbers.

---

## What the What-If simulator does

Three interactive levers, each grounded in the fitted statistical models:

- **Budget Reallocation** — shifts N% of YouTube spend to CTV. Uses late-30d CPA as a proxy for marginal CPA: `net_extra_convs = shifted_spend/dest_late_cpa − shifted_spend/src_late_cpa`. Projects blended CPA delta and revenue impact.

- **Frequency Cap Adjustment** — reconstructs the piecewise linear CPA model from its fitted parameters (breakpoint, slopes, continuity constraint) and evaluates CPA at the new cap frequency. Projects daily conversion lift and 30-day savings.

- **Creative Rotation Timing** — builds a counterfactual CTR series: actual observed CTR up to the rotation day, then initial CTR (fresh creative) for the remainder. Compares against the never-rotated reality to project extra clicks, conversions, and revenue.

Projections update in real-time as sliders move. The "Generate Strategic Analysis" button sends the specific parameter values and computed projection numbers to the LLM, which narrates the scenario in 3–4 sentences. The math happens in Python; the LLM explains what the math means.

---

## Honest limitations

**Synthetic data.** The campaign data was generated with planted failure modes and controlled noise. In production, the interesting problems are: handling missing and inconsistent data from platform APIs (Meta Marketing API response gaps, Google Ads sampling, The Trade Desk reporting latency); reconciling attribution model discrepancies across channels where the same conversion is claimed by multiple touchpoints; and validating that a detected anomaly is a real media problem rather than an artifact of seasonality, a promotion, or an external event.

**Clean-data models.** The statistical models are fitted to data generated to exhibit the patterns they detect. Real campaign data has confounders — competitor activity, news cycles, inventory constraints, algorithmic platform changes — that require more robust causal inference methods before you can attribute a CPA shift to audience saturation rather than to an external factor.

**Ceteris paribus projections.** The what-if simulator assumes all else remains equal. Real budget shifts trigger platform algorithm responses: YouTube and Meta run learning phases when spend changes significantly, auction dynamics shift when a channel's budget increases, and incrementality is not the same as attributed conversions. A 20% budget reallocation in the real world does not simply produce `shifted_spend / dest_late_cpa` additional conversions.

---

## What would change in production

- **Data ingestion**: Replace synthetic CSVs with API connectors to Meta Marketing API, Google Ads API, and The Trade Desk Reporting API. Normalize attribution across platforms using a first-touch / last-touch / data-driven reconciliation layer.

- **Cannibalization detection**: Replace Pearson correlation + Granger causality with Bayesian structural time series (CausalImpact) to properly control for seasonality and external factors before attributing CPA changes to cross-channel audience overlap.

- **Budget allocation**: Integrate with a marketing mix model (MMM) output to get long-run channel response curves, rather than extrapolating from a 90-day marginal CPA proxy.

- **Feedback loop**: Track whether implemented recommendations actually improved performance. The diagnostic that fired, the recommendation made, the action taken, and the measured outcome should be logged to evaluate model quality over time.

---

## Stack

Python 3.11+ · pandas · numpy · scipy · statsmodels · scikit-learn · Google Gemini API · Plotly · Streamlit

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic campaign data
python3 data/generator.py

# Run statistical diagnostics and populate LLM cache
# (set LLM_PROVIDER and LLM_API_KEY environment variables first)
LLM_PROVIDER=gemini LLM_API_KEY=your_key python3 run_intelligence.py

# Launch the app (works with cached responses — no live API key required)
streamlit run app.py
```

The app reads from `cache/llm_responses/` by default. Run `run_intelligence.py` once locally to populate the cache; the deployed app serves from cache without requiring a live API key.

---

## Project structure

```
signal/
├── app.py                          # Streamlit entry point (4 tabs)
├── config.py                       # Campaign config, thresholds, LLM settings
├── run_intelligence.py             # CLI: run diagnostics + LLM synthesis
├── data/
│   ├── generator.py                # Synthetic campaign data generation
│   ├── campaigns.csv               # Generated (gitignored)
│   ├── creative_performance.csv    # Generated (gitignored)
│   └── daypart_performance.csv     # Generated (gitignored)
├── diagnostics/
│   ├── creative_fatigue.py         # Exponential decay fitting
│   ├── frequency_saturation.py     # Piecewise CPA regression
│   ├── channel_cannibalization.py  # Pearson + Granger causality
│   ├── daypart_analysis.py         # Kruskal-Wallis H-test
│   └── budget_efficiency.py        # Marginal CPA analysis
├── intelligence/
│   ├── synthesizer.py              # Two-stage LLM prompt chain
│   ├── whatif.py                   # What-if projection engine
│   ├── llm_client.py               # Unified Claude / OpenAI / Gemini client
│   └── cache.py                    # File-based LLM response cache
└── cache/
    └── llm_responses/              # Cached LLM responses (gitignored)
```
