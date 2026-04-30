# state.py
from typing import TypedDict, Optional


class AgentState(TypedDict):
    # --- Input ---
    query: str
    conversation_history: str   # formatted prior turns injected for context

    # --- Parsed intent (filled by parse_query node) ---
    market: str
    sector: str
    query_type: str        # "single_market" | "market_comparison" | "risk_assessment" | "opportunity_scan"
    time_horizon: str      # "short_term" | "medium_term" | "long_term"
    investor_type: str     # "fdi" | "portfolio" | "startup" | "general"

    # --- Research outputs (filled by parallel nodes) ---
    macro_data: Optional[dict]
    political_data: Optional[dict]
    sector_data: Optional[dict]

    # --- Regulatory output ---
    regulatory_data: Optional[dict]

    # --- Investor intelligence ---
    exit_data: Optional[dict]      # Exit landscape, comparable deals, M&A/IPO activity
    fx_data: Optional[dict]        # Currency risk, FX trends, capital controls
    timing_data: Optional[dict]    # Market maturity, entry timing signal, sector cycle stage

    # --- Draft + reflection ---
    draft_report: Optional[str]
    reflection_notes: Optional[str]

    # --- Final outputs ---
    final_report: Optional[str]
    chart_data: Optional[dict]

    # --- Internal bookkeeping ---
    errors: list[str]
