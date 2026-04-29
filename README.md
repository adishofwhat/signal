# SIGNAL

Prototype that automates multi-channel media audits. Runs five statistical diagnostics against campaign data, flags inefficiencies with dollar estimates, and uses an LLM to write the report. Inspired by MOA.

Synthetic data on a fictional client (Meridian Outdoor Co., 90 days, $1.8M, five channels). The failure modes are planted. The methods are standard.

## How it works

Hard separation between the stats and the LLM. Python does all the detection — exponential decay fitting for creative fatigue, piecewise regression for frequency saturation, Pearson + Granger for cross-channel cannibalization, Kruskal-Wallis for daypart CVR differences, rolling marginal CPA for budget efficiency. Each module has its own thresholds. The LLM gets the results after everything's already been flagged and writes a narrative around the numbers. You could swap the model and nothing changes about what gets detected.

I spent the most time on creative fatigue and cannibalization. Fatigue needed onset detection because social_v1 was flat for 25 days before it started decaying — fitting the full series produced garbage. Cannibalization is the weakest of the five and I know it. Pearson correlation with Granger at lag 1 is a signal, not proof. CausalImpact would be the right tool but felt like overkill for a prototype on synthetic data where I already know the answer.

The what-if simulator lets you adjust budget allocation, frequency caps, and creative rotation timing. Projections are computed from the fitted models and update as you move sliders. The LLM just explains the scenario on a button click.

## What this doesn't handle

The synthetic data dodges the hardest production problems — API data quality across platforms, attribution reconciliation, and confounders that look like the patterns these diagnostics detect (a competitor launching a sale looks a lot like creative fatigue in the data). The what-if projections assume ceteris paribus, which doesn't hold when platforms enter learning phases after budget changes.

## Stack

Python · scipy · statsmodels · Gemini API · Plotly · Streamlit

## Setup

```
pip install -r requirements.txt
python3 data/generator.py
LLM_PROVIDER=gemini LLM_API_KEY=your_key python3 run_intelligence.py
streamlit run app.py
```

Runs from cached LLM responses after first `run_intelligence.py`. No live key needed for the app.
