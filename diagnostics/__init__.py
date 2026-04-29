from dataclasses import dataclass, field


@dataclass
class DiagnosticResult:
    name: str                    # e.g., "creative_fatigue"
    severity: str                # "critical", "warning", "info"
    channel: str                 # affected channel(s)
    finding: str                 # one-line summary
    evidence: dict               # statistical details
    estimated_impact_usd: float  # rough dollar impact
    charts: list = field(default_factory=list)  # plotly figures

    def print_summary(self) -> None:
        badges = {"critical": "CRITICAL", "warning": "WARNING", "info": "INFO"}
        badge = badges.get(self.severity, self.severity.upper())
        print(f"\n[{badge}] {self.name}")
        print(f"  Channel : {self.channel}")
        print(f"  Finding : {self.finding}")
        print(f"  Impact  : ${self.estimated_impact_usd:,.0f} (estimated)")
        print(f"  Evidence:")
        for k, v in self.evidence.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            elif isinstance(v, list):
                print(f"    {k}: {v}")
            else:
                print(f"    {k}: {v}")
