# SIGNAL — Addendum: Checkpoints & Multi-Provider LLM Setup

---

## MULTI-PROVIDER LLM SETUP

### Architecture

```python
# intelligence/llm_client.py

from enum import Enum

class LLMProvider(Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"

class LLMClient:
    """
    Unified interface for LLM calls.
    Swap provider without touching any other code.
    """
    def __init__(self, provider: LLMProvider, api_key: str, model: str = None):
        self.provider = provider
        self.api_key = api_key
        # Sensible defaults per provider
        self.model = model or {
            LLMProvider.CLAUDE: "claude-sonnet-4-20250514",
            LLMProvider.OPENAI: "gpt-4o",
            LLMProvider.GEMINI: "gemini-2.0-flash",
        }[provider]
    
    def generate(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        """Returns raw text response. JSON parsing happens in synthesizer.py."""
        if self.provider == LLMProvider.CLAUDE:
            return self._call_claude(system_prompt, user_prompt)
        elif self.provider == LLMProvider.OPENAI:
            return self._call_openai(system_prompt, user_prompt, json_mode)
        elif self.provider == LLMProvider.GEMINI:
            return self._call_gemini(system_prompt, user_prompt)
    
    def _call_claude(self, system_prompt, user_prompt):
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text
    
    def _call_openai(self, system_prompt, user_prompt, json_mode=False):
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **kwargs
        )
        return response.choices[0].message.content
    
    def _call_gemini(self, system_prompt, user_prompt):
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt
        )
        response = model.generate_content(user_prompt)
        return response.text
```

### Config Setup

```python
# config.py (add to existing)

import os

LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "claude"),  # "claude", "openai", "gemini"
    "api_key": os.getenv("LLM_API_KEY", ""),
    "model": os.getenv("LLM_MODEL", None),  # None = use default for provider
}
```

### Cached Fallback

```python
# intelligence/cache.py

import json
import hashlib
from pathlib import Path

CACHE_DIR = Path("cache/llm_responses")

def get_cache_key(system_prompt: str, user_prompt: str) -> str:
    content = system_prompt + user_prompt
    return hashlib.md5(content.encode()).hexdigest()

def get_cached(system_prompt: str, user_prompt: str) -> str | None:
    key = get_cache_key(system_prompt, user_prompt)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())["response"]
    return None

def save_to_cache(system_prompt: str, user_prompt: str, response: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = get_cache_key(system_prompt, user_prompt)
    cache_file = CACHE_DIR / f"{key}.json"
    cache_file.write_text(json.dumps({
        "system_prompt": system_prompt[:200],  # truncate for readability
        "response": response
    }))
```

### Usage in synthesizer.py

```python
def synthesize(diagnostic_results, campaign_config, llm_client, use_cache=True):
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(diagnostic_results, campaign_config)
    
    if use_cache:
        cached = get_cached(system_prompt, user_prompt)
        if cached:
            return json.loads(cached)
    
    response = llm_client.generate(system_prompt, user_prompt, json_mode=True)
    save_to_cache(system_prompt, user_prompt, response)
    return json.loads(response)
```

**For deployment**: Run the full pipeline once locally with your API key. This populates the cache. The deployed app reads from cache by default. If the cache misses (e.g., what-if scenario), it falls back to live API. This way Sibi gets a working demo even if your API key has issues.

---

## CHECKPOINTS

Each checkpoint has: what to verify, what to send me, and pass/fail criteria.

---

### CHECKPOINT 1: Data Generation
**When**: After `data/generator.py` and `config.py` are built

**What to verify**:
1. Three DataFrames generated without errors
2. Shape check: campaigns should be ~450 rows (90 days × 5 channels), creative_performance ~1,350-2,250 rows (90 × 5 channels × 3-5 variants), daypart_performance ~3,150 rows (90 × 5 × 7 dayparts)
3. All metrics are in realistic ranges (no negative values, no CTR > 100%, CPA in reasonable bounds)
4. **The five failure modes are visible in raw plots**

**What to show me**:
- DataFrame `.describe()` output for the campaigns table
- Five quick plots (one per failure mode):
  1. social_v1 CTR over time (should show clear decay)
  2. programmatic_display frequency vs CPA scatter (should show hockey stick)
  3. paid_search weekly spend overlaid with programmatic_display weekly CPA (should show correlation)
  4. paid_search CVR by daypart (bar chart — prime/late_night should pop)
  5. YouTube vs CTV marginal CPA over time (YouTube should be climbing, CTV stable)

**Pass criteria**: 
- All five plots show the intended pattern visibly but not cartoonishly (there should be noise, but the signal should be clear enough that you'd notice it in an exploratory analysis)
- Metrics are in realistic industry ranges
- No data quality issues (nulls, negative values, impossible percentages)

**Fail signals**:
- Failure mode is invisible in the noise → increase signal strength
- Failure mode is too clean (no noise at all) → add more variance
- Metrics are unrealistic (e.g., $0.10 CPA for CTV, 15% CTR on display) → adjust base rates

---

### CHECKPOINT 2: Diagnostics Engine
**When**: After all five diagnostic modules are built and tested

**What to verify**:
1. Each module runs without errors on the generated data
2. Each module correctly identifies the planted failure mode
3. Each module does NOT false-positive on the channels/metrics where no failure was planted
4. Statistical evidence is sound (p-values are below threshold for real findings, above for non-findings)
5. DiagnosticResult objects are well-formed

**What to show me**:
- For each diagnostic, the output DiagnosticResult: finding text, severity, statistical evidence dict, estimated impact
- One chart per diagnostic (the Plotly figure)
- Any cases where a diagnostic flagged something unexpected (could be interesting — real cross-talk between planted failure modes)

**Pass criteria**:
- 5/5 planted failure modes detected correctly
- False positive rate: at most 1 unexpected finding across all modules (some cross-talk is fine and even realistic)
- Statistical evidence includes: test statistic, p-value, effect size, model parameters
- Estimated impact is in a plausible range (not $0, not $10M for a $1.8M campaign)
- Charts are readable and clearly show the finding

**Fail signals**:
- A module doesn't detect its target failure mode → check the detection thresholds, may need to strengthen the signal in the data or loosen the threshold
- Too many false positives → tighten thresholds or add minimum effect size requirements
- curve_fit fails to converge → check initial parameter guesses and bounds
- Charts are noisy/unreadable → consider smoothing or aggregating before plotting

---

### CHECKPOINT 3: LLM Integration
**When**: After `intelligence/llm_client.py` and `intelligence/synthesizer.py` are built and tested

**What to verify**:
1. LLM client works with at least one provider
2. Synthesizer produces valid JSON with the specified schema
3. The narrative references specific numbers from the diagnostics (not generic marketing speak)
4. The action plan is specific and prioritized
5. Cache works (second call with same input returns cached response)

**What to show me**:
- The raw JSON output from the synthesizer (Stage 1)
- The narrative report (Stage 2) if built
- Which provider you tested with and any issues

**Pass criteria**:
- JSON parses cleanly into the expected schema
- Executive summary references at least 2 specific metrics (e.g., "CTR decayed from 2.8% to 1.1%")
- Each finding's recommendation is specific enough that a media buyer could act on it without further analysis
- Action plan has 3-5 items with priorities and timelines
- No hallucinated metrics (everything should trace back to the diagnostic output)

**Fail signals**:
- JSON parsing fails → add explicit "respond ONLY with valid JSON, no markdown backticks" to the prompt. For OpenAI, use json_mode=True
- Generic output ("consider optimizing your campaigns") → add more diagnostic data to the prompt, include the actual numbers
- Hallucinated numbers → add instruction "only reference numbers that appear in the data provided below"
- One provider doesn't work → try another, note the issue, move on

---

### CHECKPOINT 4: UI Shell
**When**: After `app.py` and the basic tab structure are wired up with theme applied

**What to verify**:
1. Streamlit runs locally without errors
2. Dark theme is applied (no default Streamlit blue/white)
3. All four tabs render with placeholder or real content
4. Charts are interactive (hover, zoom work)
5. Metric cards show correct campaign summary stats

**What to show me**:
- Screenshots of each tab (or a screen recording)
- Any styling issues or Streamlit quirks encountered

**Pass criteria**:
- Looks professional and editorial, not like a default Streamlit demo
- Charts render correctly in dark theme (Plotly template matches)
- Navigation between tabs works
- Data flows correctly: generator → diagnostics → charts → LLM output all connected

**Fail signals**:
- Default Streamlit styling leaks through → more CSS overrides needed
- Charts look wrong in dark theme → set plotly template to "plotly_dark" and customize colors
- Performance issues (slow load) → cache the diagnostic computation, don't re-run on every interaction

---

### CHECKPOINT 5: What-If Simulator
**When**: After Tab 3 (What-If) is functional

**What to verify**:
1. Sliders adjust and projected impact updates
2. Projections are mathematically grounded (not random numbers)
3. LLM generates a coherent explanation of the scenario
4. Before/after comparison is clear

**What to show me**:
- Screenshot/recording of adjusting each lever and seeing the projected impact
- One example LLM explanation for a what-if scenario
- The projection math for one scenario (e.g., "if frequency cap reduced from 9 to 6, projected CPA reduction is X% because the saturation model shows...")

**Pass criteria**:
- At least 3 levers work (budget shift, frequency cap, creative rotation)
- Projections change in the expected direction (lower frequency → lower CPA, not higher)
- Projections include uncertainty/caveats ("assuming current trends continue")
- The LLM explanation references the specific adjustment made

**Fail signals**:
- Projections don't change when sliders move → wiring issue
- Projections are implausibly large or small → check the model extrapolation
- LLM explanation is generic → pass the specific slider values and projection numbers into the prompt

---

### CHECKPOINT 6: Final Polish & Deploy
**When**: After README is written and app is deployed

**What to verify**:
1. Deployed URL loads and works end-to-end
2. README has the framing from the spec (honest about limitations, discusses production considerations)
3. GitHub repo is clean (no API keys, no pycache, has .gitignore)
4. App works without live API key (cached responses)
5. Load time is acceptable (<10 seconds to first render)

**What to show me**:
- The deployed URL
- The GitHub repo link
- The README

**Pass criteria**:
- Cold load (first visit) works without errors
- All tabs render with real data and LLM output
- README reads like a senior engineer wrote it (not like a tutorial or class project)
- No API keys or secrets in the repo
- The what-if tab works even with cached LLM responses

---

## WORKING RHYTHM

Here's how to structure the sessions:

1. Build a phase with Claude Code / Cursor
2. Run the checkpoint verification yourself
3. Send me the checkpoint outputs (screenshots, code output, data summaries)
4. I review and either green-light the next phase or flag issues
5. Move to next phase

Don't skip ahead. Each phase depends on the previous one being solid. The data generation is the foundation — if that's wrong, everything downstream is wrong.

Estimated total: 14-20 hours of focused build time across 3-4 sessions.