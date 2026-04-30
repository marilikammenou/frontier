# nodes.py — optimised for speed
# Changes from v1:
#   Fix 1: parse_query skips scope check (router already did it) — saves 1 LLM call
#   Fix 2: reflect_node removed — brief_node writes final report directly — saves 2 LLM calls
#   Fix 3: brief_node merges report + chart into ONE LLM call — saves 1 LLM call
#   Fix 4: Tavily max_results reduced 3→2 across all nodes — saves token processing time
#   Total: 4 fewer sequential LLM calls per pipeline query

import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient
from state import AgentState
from tools import get_macro_indicators, get_political_risk, get_sector_data, get_regulatory_environment


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.3,
    )


def get_tavily():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None
    return TavilyClient(api_key=api_key)


# Fix 4: max_results reduced to 2
def tavily_search(query: str, max_results: int = 2) -> str:
    try:
        client = get_tavily()
        if not client:
            return ""
        results = client.search(query=query, max_results=max_results)
        snippets = []
        for r in results.get("results", []):
            title   = r.get("title", "")
            content = r.get("content", "")
            url     = r.get("url", "")
            snippets.append(f"Source: {title} ({url})\n{content}")
        return "\n\n---\n\n".join(snippets)
    except Exception as e:
        print(f"    [Tavily error: {e}]")
        return ""


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", str(content))
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def clean_json(raw: str) -> str:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


FRONTIER_SYSTEM_PROMPT = """You are Frontier, an institutional-grade emerging markets intelligence platform built by AskMarketIQ.

Your sole purpose is to provide rigorous, data-driven analysis on:
- Emerging market investment opportunities (India, Brazil, Nigeria, Vietnam, Indonesia, Mexico, Egypt, Pakistan, Kenya, Bangladesh, Philippines, Ethiopia, Colombia, Argentina, China, Southeast Asia, Sub-Saharan Africa, Latin America, Middle East, and more)
- Macroeconomic analysis, GDP trends, inflation, monetary policy
- Sector opportunities: fintech, SaaS/cloud, e-commerce, energy, consumer, infrastructure
- Political risk, regulatory environments, FX/currency risk
- Market timing, maturity curves, entry signals
- M&A activity, comparable deals, exit landscapes
- Investment strategy for FDI, portfolio investors, and startups
- Developed markets when discussed in the context of EM investment strategy or comparison

Always respond as a senior analyst at a top-tier emerging markets investment firm — precise, evidence-based, and actionable."""


# ---------------------------------------------------------------------------
# NODE 1: parse_query
# Fix 1: Skip scope check — router already handled it. One LLM call only.
# ---------------------------------------------------------------------------

def parse_query(state: AgentState) -> dict:
    print("\n[Node 1] Extracting intent (scope already cleared by router)...")
    llm = get_llm()
    query   = state["query"]
    history = state.get("conversation_history", "")
    history_block = f"\n\nCONVERSATION HISTORY:\n{history}" if history else ""

    intent_messages = [
        SystemMessage(content=f"""{FRONTIER_SYSTEM_PROMPT}{history_block}

Extract structured intent from the query. Use conversation history to resolve references like "this market" or "that sector".
Respond with ONLY valid JSON:
{{
  "market": "Indonesia",
  "sector": "fintech / digital payments",
  "query_type": "single_market",
  "time_horizon": "medium_term",
  "investor_type": "fdi"
}}

market: specific country or "Emerging Markets" if broad
sector: infer from context; "general" only as last resort
query_type: "single_market" | "market_comparison" | "risk_assessment" | "opportunity_scan"
time_horizon: "short_term" | "medium_term" (default) | "long_term"
investor_type: "fdi" | "portfolio" | "startup" | "general"
"""),
        HumanMessage(content=query)
    ]

    response = llm.invoke(intent_messages)
    content  = clean_json(extract_text(response.content))

    try:
        parsed        = json.loads(content)
        market        = parsed.get("market", "Emerging Markets")
        sector        = parsed.get("sector", "general")
        query_type    = parsed.get("query_type", "single_market")
        time_horizon  = parsed.get("time_horizon", "medium_term")
        investor_type = parsed.get("investor_type", "general")
    except Exception:
        market, sector = "Emerging Markets", "general"
        query_type, time_horizon, investor_type = "single_market", "medium_term", "general"

    print(f"    Market: {market} | Sector: {sector} | Type: {query_type}")
    return {
        "market": market, "sector": sector,
        "query_type": query_type, "time_horizon": time_horizon,
        "investor_type": investor_type,
        "errors": state.get("errors", [])
    }


# ---------------------------------------------------------------------------
# NODE 2a: macro_node — Fix 4: max_results=2
# ---------------------------------------------------------------------------

def macro_node(state: AgentState) -> dict:
    print("[Node 2a] Macro data...")
    market       = state.get("market", "Emerging Markets")
    time_horizon = state.get("time_horizon", "medium_term")
    horizon_map  = {
        "short_term":  "2025 economic outlook GDP inflation interest rates",
        "medium_term": "GDP growth forecast inflation economic stability 2025 2026",
        "long_term":   "long-term economic growth demographics infrastructure investment",
    }
    search_query = f"{market} {horizon_map.get(time_horizon, horizon_map['medium_term'])}"
    live_data    = tavily_search(search_query, max_results=2)

    if live_data:
        return {"macro_data": {"source": "tavily:live", "market": market, "live_search_results": live_data, "search_query": search_query}}
    return {"macro_data": get_macro_indicators(market)}


# ---------------------------------------------------------------------------
# NODE 2b: political_node — Fix 4: max_results=2
# ---------------------------------------------------------------------------

def political_node(state: AgentState) -> dict:
    print("[Node 2b] Political risk...")
    market        = state.get("market", "Emerging Markets")
    investor_type = state.get("investor_type", "general")
    investor_map  = {
        "fdi":       "FDI policy foreign investment restrictions political stability governance",
        "portfolio": "capital controls currency risk political risk sovereign rating",
        "startup":   "startup ecosystem entrepreneurship policy government support",
        "general":   "political stability governance risk regulatory environment",
    }
    search_query = f"{market} {investor_map.get(investor_type, investor_map['general'])} 2025"
    live_data    = tavily_search(search_query, max_results=2)

    if live_data:
        return {"political_data": {"source": "tavily:live", "market": market, "live_search_results": live_data, "search_query": search_query}}
    return {"political_data": get_political_risk(market)}


# ---------------------------------------------------------------------------
# NODE 2c: sector_node — Fix 4: max_results=2
# ---------------------------------------------------------------------------

def sector_node(state: AgentState) -> dict:
    print("[Node 2c] Sector opportunity...")
    market     = state.get("market", "Emerging Markets")
    sector     = state.get("sector", "general")
    query_type = state.get("query_type", "single_market")
    qmap = {
        "market_comparison": f"best emerging markets {sector} investment opportunity comparison 2025",
        "risk_assessment":   f"{market} {sector} risks challenges barriers to entry 2025",
        "opportunity_scan":  f"top emerging markets {sector} growth opportunity whitespace 2025",
    }
    search_query = qmap.get(query_type, f"{market} {sector} market size growth opportunity 2025")
    live_data    = tavily_search(search_query, max_results=2)

    if live_data:
        return {"sector_data": {"source": "tavily:live", "market": market, "sector": sector, "live_search_results": live_data, "search_query": search_query}}
    return {"sector_data": get_sector_data(market, sector)}


# ---------------------------------------------------------------------------
# NODE 2d: exit_node — Fix 4: max_results=2
# ---------------------------------------------------------------------------

def exit_node(state: AgentState) -> dict:
    print("[Node 2d] Exit landscape...")
    market        = state.get("market", "Emerging Markets")
    sector        = state.get("sector", "general")
    investor_type = state.get("investor_type", "general")

    if investor_type == "portfolio":
        search_query = f"{market} {sector} IPO listings stock exchange liquidity exit 2024 2025"
    elif investor_type == "startup":
        search_query = f"{market} {sector} startup acquisition M&A venture exit deals 2024 2025"
    else:
        search_query = f"{market} {sector} M&A deals acquisitions exits comparable transactions 2024 2025"

    live_data = tavily_search(search_query, max_results=2)

    if live_data:
        return {"exit_data": {"source": "tavily:live", "market": market, "sector": sector, "live_search_results": live_data, "search_query": search_query}}
    return {"exit_data": {
        "source": "stub", "market": market, "sector": sector,
        "notes": "Limited exit data available. Market may be pre-liquidity stage.",
        "comparable_deals": [], "ipo_pipeline": "Nascent", "ma_activity": "Early stage"
    }}


# ---------------------------------------------------------------------------
# NODE 2e: fx_node — Fix 4: max_results=2
# ---------------------------------------------------------------------------

def fx_node(state: AgentState) -> dict:
    print("[Node 2e] FX risk...")
    market       = state.get("market", "Emerging Markets")
    search_query = f"{market} currency exchange rate trend capital controls USD 2025"
    live_data    = tavily_search(search_query, max_results=2)

    if live_data:
        return {"fx_data": {"source": "tavily:live", "market": market, "live_search_results": live_data, "search_query": search_query}}
    return {"fx_data": {
        "source": "stub", "market": market,
        "currency": "Local currency", "usd_trend": "Stable",
        "volatility": "Medium", "capital_controls": "Some restrictions apply",
        "repatriation_risk": "Medium"
    }}


# ---------------------------------------------------------------------------
# NODE 2f: timing_node — Fix 4: max_results=2
# ---------------------------------------------------------------------------

def timing_node(state: AgentState) -> dict:
    print("[Node 2f] Market timing...")
    market       = state.get("market", "Emerging Markets")
    sector       = state.get("sector", "general")
    search_query = f"{market} {sector} market maturity stage early growth saturation investment timing 2025"
    live_data    = tavily_search(search_query, max_results=2)

    if live_data:
        return {"timing_data": {"source": "tavily:live", "market": market, "sector": sector, "live_search_results": live_data, "search_query": search_query}}
    return {"timing_data": {
        "source": "stub", "market": market, "sector": sector,
        "maturity_stage": "Growth", "entry_window": "Open",
        "cycle_position": "Mid-cycle",
        "notes": "Market showing growth-phase characteristics."
    }}


# ---------------------------------------------------------------------------
# NODE 3: aggregate_node
# ---------------------------------------------------------------------------

def aggregate_node(state: AgentState) -> dict:
    print("[Node 3] Aggregating research...")
    errors = state.get("errors", [])
    for field, label in [
        ("macro_data","macro"), ("political_data","political"),
        ("sector_data","sector"), ("exit_data","exit"),
        ("fx_data","fx"), ("timing_data","timing")
    ]:
        if not state.get(field):
            errors.append(f"Warning: {label} data missing")
    return {"errors": errors}


# ---------------------------------------------------------------------------
# NODE 4: regulatory_node
# ---------------------------------------------------------------------------

def regulatory_node(state: AgentState) -> dict:
    print("[Node 4] Regulatory environment...")
    market        = state.get("market", "Emerging Markets")
    sector        = state.get("sector", "general")
    investor_type = state.get("investor_type", "general")

    if investor_type == "fdi":
        search_query = f"{market} {sector} FDI rules foreign ownership limits licensing requirements 2025"
    elif investor_type == "startup":
        search_query = f"{market} {sector} startup registration licensing fintech sandbox regulations 2025"
    else:
        search_query = f"{market} {sector} regulation compliance requirements market entry rules 2025"

    live_data = tavily_search(search_query, max_results=2)

    if live_data:
        return {"regulatory_data": {"source": "tavily:live", "market": market, "sector": sector, "live_search_results": live_data, "search_query": search_query}}
    return {"regulatory_data": get_regulatory_environment(market, sector)}


# ---------------------------------------------------------------------------
# NODE 5: brief_node
# Fix 2: No separate reflect_node — this produces the FINAL report directly.
# Fix 3: Report + chart data in ONE LLM call using a structured JSON wrapper.
# ---------------------------------------------------------------------------

def brief_node(state: AgentState) -> dict:
    print("[Node 5] Writing final brief + chart data (single LLM call)...")
    llm = get_llm()

    market        = state.get("market", "Unknown")
    sector        = state.get("sector", "Unknown")
    query_type    = state.get("query_type", "single_market")
    time_horizon  = state.get("time_horizon", "medium_term")
    investor_type = state.get("investor_type", "general")
    query         = state.get("query", "")

    def fmt(d):
        if isinstance(d, dict) and d.get("source") == "tavily:live":
            # Truncate live results slightly to reduce token count
            results = d.get("live_search_results", "")[:2000]
            return f"[Live: '{d.get('search_query', '')}']\n{results}"
        return json.dumps(d, indent=2)[:1500]

    context = f"""QUERY: {query}
MARKET: {market} | SECTOR: {sector} | TYPE: {query_type} | HORIZON: {time_horizon} | INVESTOR: {investor_type}

MACRO: {fmt(state.get("macro_data",{}))}
POLITICAL: {fmt(state.get("political_data",{}))}
SECTOR: {fmt(state.get("sector_data",{}))}
REGULATORY: {fmt(state.get("regulatory_data",{}))}
EXIT: {fmt(state.get("exit_data",{}))}
FX: {fmt(state.get("fx_data",{}))}
TIMING: {fmt(state.get("timing_data",{}))}"""

    investor_guidance = {
        "fdi":       "Focus on operational considerations: local partnerships, ownership structures, workforce, and regulatory compliance.",
        "portfolio": "Focus on return profile, valuation benchmarks, exit liquidity, currency hedging, and sovereign risk.",
        "startup":   "Focus on product-market fit signals, local competition, talent availability, and regulatory sandbox opportunities.",
        "general":   "Provide a balanced overview suitable for a strategic decision-maker.",
    }

    structure_note = {
        "market_comparison": "Produce a ranked comparison of the top 3-5 markets. Use a comparison table. End with a clear #1 recommendation.",
        "risk_assessment":   "Lead with risk analysis. Be direct about red flags.",
        "opportunity_scan":  "Lead with opportunity size. Identify attractive whitespace and concrete entry vectors.",
    }.get(query_type, "This is a single-market deep dive. Be specific and data-driven throughout.")

    # Fix 3: Single prompt returns BOTH the markdown report AND the chart JSON
    # wrapped in a simple delimiter so we can split them reliably
    combined_messages = [
        SystemMessage(content=f"""{FRONTIER_SYSTEM_PROMPT}

You must respond with EXACTLY this structure — nothing else:

<REPORT>
[Your full markdown market entry brief here]
</REPORT>
<CHARTS>
[Your chart JSON here]
</CHARTS>

━━━ REPORT INSTRUCTIONS ━━━
INVESTOR LENS: {investor_guidance.get(investor_type, investor_guidance['general'])}
STRUCTURE: {structure_note}

Write the brief using EXACTLY these markdown headers:

# Market Entry Brief: [{market}] — [{sector}]

## Executive Summary
2-3 sentences. Opportunity size + overall verdict.

## Macroeconomic Context
Key economic factors with specific numbers. Flag tailwinds and headwinds.

## Sector Opportunity
Market size, growth rate, key players, whitespace.

## Exit Landscape & Comparable Deals
Recent M&A, IPO pipeline, comparable transactions. Clear exit path?

## Currency & FX Risk
Currency trend, volatility, capital controls, repatriation, hedging.

## Market Timing
Where on the maturity curve? Right moment to enter?

## Risk Assessment
| Risk Factor | Rating | Commentary |
|---|---|---|
Rows: Political Stability, Currency Risk, Regulatory Risk, Competitive Intensity, Infrastructure, Exit Liquidity
Ratings: Low / Medium / High

## Regulatory Landscape
Licences, timeline, key restrictions for a {investor_type} investor.

## Recommendation
**Verdict: [PROCEED / PROCEED WITH CAUTION / HIGH RISK — HOLD]**
3-4 sentences. Specific and actionable.

━━━ CHARTS INSTRUCTIONS ━━━
Output valid JSON only — no markdown, no backticks:
{{
  "risk_chart": {{
    "title": "Risk Assessment",
    "labels": ["Political Stability","Currency Risk","Regulatory Risk","Competitive Intensity","Infrastructure","Exit Liquidity"],
    "scores": [65,45,70,80,40,55]
  }},
  "market_size_chart": {{
    "title": "Market Size (USD bn)",
    "markets": ["Market A","Market B"],
    "values": [2.5,4.1]
  }},
  "gdp_chart": {{
    "title": "GDP Growth Rate (%)",
    "markets": ["Market A","Market B"],
    "values": [5.1,6.4]
  }},
  "fx_chart": {{
    "title": "Currency Risk Factors",
    "labels": ["Volatility","Capital Controls","Repatriation Risk","USD Correlation","Inflation Risk"],
    "scores": [40,30,45,60,55]
  }},
  "timing_chart": {{
    "market": "{market}",
    "sector": "{sector}",
    "maturity_stage": "Growth",
    "stage_index": 2,
    "stages": ["Nascent","Early","Growth","Mature","Saturated"],
    "entry_signal": "STRONG BUY",
    "entry_rationale": "Short justification for the signal.",
    "comparable_deals": [
      {{"name":"Deal A","amount":"$45M","year":"2024","type":"Series B"}}
    ]
  }}
}}
Rules: scores = integers 0-100 | stage_index 0-4 | entry_signal: STRONG BUY|BUY|HOLD|WAIT|AVOID | use real data where available"""),
        HumanMessage(content=context)
    ]

    response     = llm.invoke(combined_messages)
    raw          = extract_text(response.content)

    # Split on our delimiters
    report_text  = ""
    chart_data   = None

    try:
        report_start = raw.index("<REPORT>") + len("<REPORT>")
        report_end   = raw.index("</REPORT>")
        report_text  = raw[report_start:report_end].strip()
    except ValueError:
        # Fallback: treat whole response as report
        report_text = raw.strip()

    try:
        chart_start = raw.index("<CHARTS>") + len("<CHARTS>")
        chart_end   = raw.index("</CHARTS>")
        chart_raw   = clean_json(raw[chart_start:chart_end].strip())
        chart_data  = json.loads(chart_raw)
        print("    Report + chart data extracted.")
    except Exception as e:
        print(f"    Chart extraction failed: {e} — report still good.")

    return {
        "draft_report":     report_text,
        "reflection_notes": "",      # No separate reflect pass
        "final_report":     report_text,
        "chart_data":       chart_data,
    }


# ---------------------------------------------------------------------------
# NODE 6: reflect_node — Fix 2: Now a no-op pass-through.
# Kept so graph.py doesn't need changing, but does zero LLM work.
# ---------------------------------------------------------------------------

def reflect_node(state: AgentState) -> dict:
    print("[Node 6] Reflect skipped (brief_node already wrote final report).")
    return {}
