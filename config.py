import os

CAMPAIGN_CONFIG = {
    "client_name": "Meridian Outdoor Co.",
    "campaign_name": "Spring 2026 Performance Campaign",
    "date_range": ("2026-01-15", "2026-04-14"),
    "total_budget": 1_800_000,
    "objective": "Drive online conversions (e-commerce purchases) while maintaining brand awareness",
    "channels": ["paid_search", "paid_social", "programmatic_display", "ctv", "youtube"],
    "avg_order_value": 120,
}

LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "claude"),
    "api_key": os.getenv("LLM_API_KEY", os.getenv("ANTHROPIC_API_KEY", "")),
    "model": os.getenv("LLM_MODEL", None),
}

DIAGNOSTIC_THRESHOLDS = {
    "creative_fatigue": {
        "min_days": 30,
        "half_life_threshold": 30,       # flag if half-life < 30 days
        "ctr_decay_threshold": 0.60,     # flag if current CTR < 60% of initial
        "p_value_threshold": 0.05,
    },
    "frequency_saturation": {
        "cpa_increase_threshold": 0.20,  # flag if CPA above breakpoint > 20% higher
        "min_observations": 10,
    },
    "cannibalization": {
        "correlation_threshold": 0.50,
        "p_value_threshold": 0.05,
    },
    "daypart": {
        "p_value_threshold": 0.05,
        "high_cvr_threshold": 1.30,      # 30% above average
        "low_impression_share": 0.15,
        "low_cvr_threshold": 0.80,       # 20% below average
        "high_impression_share": 0.20,
    },
    "budget_efficiency": {
        "marginal_cpa_ratio_threshold": 1.50,
        "roas_floor": 1.0,
        "cross_channel_ratio_threshold": 2.0,
        "window_days": 14,
    },
}

UI_CONFIG = {
    "dark_bg": "#0a0a0a",
    "text_color": "#e8e8e8",
    "gold_accent": "#c8a55a",
    "critical_red": "#e74c3c",
    "opportunity_green": "#2ecc71",
    "muted_text": "#888888",
}
