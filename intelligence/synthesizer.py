"""
Two-stage LLM prompt chain for synthesizing diagnostic findings.

Stage 1 → structured JSON (executive summary, ranked findings, action plan)
Stage 2 → optional markdown narrative report

Key design: the LLM narrates, it does not compute. All numbers come from the
statistical layer (DiagnosticResult.evidence). Prompts are written to force
reference to specific numbers from the provided data.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from diagnostics import DiagnosticResult
from intelligence.cache import get_cached, save_to_cache
from intelligence.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _format_evidence(evidence: dict) -> str:
    lines = []
    for k, v in evidence.items():
        if isinstance(v, float):
            lines.append(f"  - {k}: {v:.6g}")
        elif isinstance(v, dict):
            lines.append(f"  - {k}:")
            for dk, dv in v.items():
                lines.append(f"      {dk}: {dv}")
        elif isinstance(v, list):
            lines.append(f"  - {k}: {', '.join(str(x) for x in v)}")
        else:
            lines.append(f"  - {k}: {v}")
    return "\n".join(lines)


def _serialize_results(results: list[DiagnosticResult]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        block = (
            f"FINDING {i}: {r.name.replace('_', ' ').title()}\n"
            f"Channel: {r.channel}\n"
            f"Severity: {r.severity}\n"
            f"One-line finding: {r.finding}\n"
            f"Estimated financial impact: ${r.estimated_impact_usd:,.0f}\n"
            f"Statistical evidence:\n{_format_evidence(r.evidence)}"
        )
        blocks.append(block)
    return "\n\n" + ("-" * 60 + "\n\n").join(blocks)


def _serialize_campaign_config(config: dict) -> str:
    return (
        f"Client: {config.get('client_name', 'N/A')}\n"
        f"Campaign: {config.get('campaign_name', 'N/A')}\n"
        f"Date range: {config.get('date_range', ('N/A', 'N/A'))[0]} to "
        f"{config.get('date_range', ('N/A', 'N/A'))[1]}\n"
        f"Total budget: ${config.get('total_budget', 0):,}\n"
        f"Objective: {config.get('objective', 'N/A')}\n"
        f"Channels: {', '.join(config.get('channels', []))}"
    )


# ---------------------------------------------------------------------------
# JSON extraction (robust against markdown code-block wrapping)
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Strip markdown fences if present and return raw JSON string."""
    text = text.strip()
    # ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        return m.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# Stage 1: structured JSON synthesis
# ---------------------------------------------------------------------------

_STAGE1_SYSTEM = """You are a senior media strategist at a data-driven advertising agency. \
You are reviewing the statistical output of a quantitative media audit conducted on a \
client's paid media campaign. Your job is to synthesize these findings into a clear, \
strategic, and specific narrative.

RULES YOU MUST FOLLOW:
1. Respond ONLY with a single valid JSON object — no markdown, no prose, no code fences.
2. Every number you cite MUST appear in the diagnostic data provided. Do not invent or \
round numbers beyond what is shown. Do not extrapolate.
3. Recommendations must be specific enough that a media buyer could act on them today \
without needing additional information.
4. Rank findings by estimated financial impact (highest first).
5. The action_plan must have exactly 3 to 5 items.
6. Use the following JSON schema exactly — do not add or remove top-level keys:

{
  "executive_summary": "<2-3 sentences covering overall campaign health and top opportunities>",
  "findings": [
    {
      "rank": 1,
      "title": "<short descriptive title>",
      "severity": "<critical|warning|info>",
      "narrative": "<3-4 sentences explaining what is happening, the statistical evidence, and why it matters>",
      "recommendation": "<specific, actionable recommendation — include channel names, thresholds, and timelines>",
      "estimated_impact": "<$XX,XXX formatted string>",
      "confidence": "<high|medium|low>"
    }
  ],
  "cross_channel_insights": "<2-3 sentences connecting findings across channels>",
  "action_plan": [
    {
      "priority": 1,
      "action": "<concrete action>",
      "timeline": "<immediate|this_week|this_month>",
      "owner": "<media_buyer|creative_team|analyst>"
    }
  ]
}"""


def _build_stage1_user_prompt(
    results: list[DiagnosticResult],
    campaign_config: dict,
) -> str:
    return (
        "CAMPAIGN CONTEXT\n"
        + "=" * 60 + "\n"
        + _serialize_campaign_config(campaign_config)
        + "\n\n"
        + "DIAGNOSTIC FINDINGS\n"
        + "=" * 60
        + _serialize_results(results)
        + "\n\n"
        + "=" * 60 + "\n"
        + "Synthesize the above diagnostic findings into the JSON schema specified in the "
        "system prompt. Cite specific metrics (p-values, decay rates, CPA ratios, impact "
        "estimates) drawn directly from the evidence above. Do not add numbers not present "
        "in the data."
    )


# ---------------------------------------------------------------------------
# Stage 2: narrative report (optional)
# ---------------------------------------------------------------------------

_STAGE2_SYSTEM = """You are a senior media strategist writing a diagnostic report to \
present to a CMO. Write in clear, direct prose — no jargon, no filler. Every claim must \
reference a specific number from the data. Use markdown formatting: use ## for section \
headers and bold for key numbers."""


def _build_stage2_user_prompt(
    stage1_json: dict,
    results: list[DiagnosticResult],
) -> str:
    return (
        "STRUCTURED FINDINGS (from Stage 1 synthesis)\n"
        + "=" * 60 + "\n"
        + json.dumps(stage1_json, indent=2)
        + "\n\n"
        + "RAW DIAGNOSTIC EVIDENCE\n"
        + "=" * 60
        + _serialize_results(results)
        + "\n\n"
        + "=" * 60 + "\n"
        + "Write a concise but thorough campaign diagnostic report based on these findings. "
        "Structure it as: Executive Summary → Key Findings (one ## section per finding, "
        "most impactful first) → Cross-Channel Dynamics → Recommended Action Plan. "
        "Reference specific numbers throughout. Keep the total length under 600 words."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def synthesize(
    results: list[DiagnosticResult],
    campaign_config: dict,
    llm_client: LLMClient,
    use_cache: bool = True,
    run_stage2: bool = False,
) -> dict:
    """
    Run the two-stage synthesis chain.

    Returns a dict with keys:
      - stage1: parsed JSON dict matching the schema above
      - stage2: markdown string (only if run_stage2=True, else None)
      - stage1_raw: raw LLM response string for debugging
    """
    sys1 = _STAGE1_SYSTEM
    usr1 = _build_stage1_user_prompt(results, campaign_config)

    # Stage 1 — structured JSON
    if use_cache:
        cached = get_cached(sys1, usr1)
    else:
        cached = None

    if cached is not None:
        stage1_raw = cached
    else:
        stage1_raw = llm_client.generate(sys1, usr1, json_mode=True)
        if use_cache:
            save_to_cache(sys1, usr1, stage1_raw)

    stage1 = json.loads(_extract_json(stage1_raw))

    # Stage 2 — narrative report (optional)
    stage2_text: str | None = None
    if run_stage2:
        sys2 = _STAGE2_SYSTEM
        usr2 = _build_stage2_user_prompt(stage1, results)

        if use_cache:
            cached2 = get_cached(sys2, usr2)
        else:
            cached2 = None

        if cached2 is not None:
            stage2_text = cached2
        else:
            stage2_text = llm_client.generate(sys2, usr2)
            if use_cache:
                save_to_cache(sys2, usr2, stage2_text)

    return {
        "stage1": stage1,
        "stage2": stage2_text,
        "stage1_raw": stage1_raw,
    }
