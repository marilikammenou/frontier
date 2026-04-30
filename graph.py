# graph.py
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send

from state import AgentState
from nodes import (
    parse_query,
    macro_node,
    political_node,
    sector_node,
    exit_node,
    fx_node,
    timing_node,
    aggregate_node,
    regulatory_node,
    brief_node,
    reflect_node,
)


def dispatch_parallel_research(state: AgentState):
    return [
        Send("macro_node",     state),
        Send("political_node", state),
        Send("sector_node",    state),
        Send("exit_node",      state),
        Send("fx_node",        state),
        Send("timing_node",    state),
    ]


def build_graph():
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("parse_query",     parse_query)
    graph.add_node("macro_node",      macro_node)
    graph.add_node("political_node",  political_node)
    graph.add_node("sector_node",     sector_node)
    graph.add_node("exit_node",       exit_node)
    graph.add_node("fx_node",         fx_node)
    graph.add_node("timing_node",     timing_node)
    graph.add_node("aggregate_node",  aggregate_node)
    graph.add_node("regulatory_node", regulatory_node)
    graph.add_node("brief_node",      brief_node)
    graph.add_node("reflect_node",    reflect_node)

    # START → parse_query → fan out to 6 parallel research nodes
    graph.add_edge(START, "parse_query")
    graph.add_conditional_edges(
        "parse_query",
        dispatch_parallel_research,
        ["macro_node", "political_node", "sector_node", "exit_node", "fx_node", "timing_node"]
    )

    # All 6 parallel nodes → aggregate → regulatory → brief → reflect → END
    graph.add_edge("macro_node",      "aggregate_node")
    graph.add_edge("political_node",  "aggregate_node")
    graph.add_edge("sector_node",     "aggregate_node")
    graph.add_edge("exit_node",       "aggregate_node")
    graph.add_edge("fx_node",         "aggregate_node")
    graph.add_edge("timing_node",     "aggregate_node")
    graph.add_edge("aggregate_node",  "regulatory_node")
    graph.add_edge("regulatory_node", "brief_node")
    graph.add_edge("brief_node",      "reflect_node")
    graph.add_edge("reflect_node",    END)

    return graph.compile()
