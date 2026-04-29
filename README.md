# SIGNAL

Automates the kind of multi-channel media audit that MOA does manually. Ingests campaign data, runs statistical diagnostics, flags inefficiencies, estimates dollar impact, and uses an LLM to write the strategic narrative.

Built on a fictional 90-day, $1.8M campaign (Meridian Outdoor Co.) across paid search, paid social, programmatic display, CTV, and YouTube. Data is synthetic with planted failure modes. Methods are real.

## Architecture decision

The LLM narrates. It doesn't compute. Every flag is a predetermined threshold (half-life < 30d, CPA uplift > 20%, correlation > 0.5 at p < 0.05). The LLM receives evidence that's already been computed and writes a report grounded in those numbers. Swap the model and the findings don't change.

## Diagnostics

**Creative fatigue** — Exponential decay fit on CTR over time. Onset detection first (social_v1 was flat for 25 days before decaying), then fit only the decaying portion. Half-life 14.5d, 53% CTR collapse. Estimates $130K lost over 30 days if not rotated.

**Frequency saturation** — Piecewise linear regression on CPA vs frequency, breakpoint found via grid search. Breakpoint at 7.1, current avg 9.7. CPA 41% higher above the break. Recommendation: cap at 7 in the DSP.

**Cannibalization** — Pearson correlation between weekly search spend changes and display CPA changes (r=0.65, p=0.02). Granger causality not significant at lag 1, so this is correlational, not causal. In production this would be CausalImpact, not Pearson. The app says that.

**Daypart opportunity** — Kruskal-Wallis on CVR across dayparts, Dunn's post-hoc with Bonferroni correction. Prime and late night have 1.38× average CVR but get 5-7% of impressions. Set bid multipliers.

**Budget efficiency** — Rolling 14-day marginal CPA, first-30d vs last-30d comparison, Welch's t-test. YouTube ratio 2.65× ($57 → $151). CTV stable at 0.91×. Shift budget.

## What-if simulator

Three levers, each grounded in the fitted model from the corresponding diagnostic:

- Budget reallocation — shift 0-40% of YouTube to CTV using late-30d CPAs
- Frequency cap — piecewise model predicts CPA at new frequency
- Creative rotation — counterfactual CTR series assuming replacement performs at initial level

Numbers update as sliders move. LLM generates a narrative on button click but the math is already done.

## What's not here

The data is synthetic and the failure modes are planted. The interesting production problems this skips:

- API data quality (Meta response gaps, Google Ads sampling, TTD reporting latency)
- Attribution reconciliation across platforms claiming the same conversion
- Confounders — competitor activity, promos, platform algorithm changes all look like the patterns these diagnostics detect
- Platform learning phases when you shift budget (ceteris paribus doesn't hold)

Production version would use CausalImpact for cannibalization, MMM integration for budget allocation, and a feedback loop tracking whether recommendations actually moved the numbers.

## Stack

Python · scipy · statsmodels · scikit-learn · Gemini API · Plotly · Streamlit

## Setup

```bash
pip install -r requirements.txt
python3 data/generator.py
LLM_PROVIDER=gemini LLM_API_KEY=your_key python3 run_intelligence.py
streamlit run app.py
```

Runs from cached LLM responses after the first `run_intelligence.py` call. No live key needed for the app.
