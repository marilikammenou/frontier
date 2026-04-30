import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from graph import build_graph
from state import AgentState
from router import classify_query, direct_answer

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

executor = ThreadPoolExecutor(max_workers=4)


class QueryRequest(BaseModel):
    query: str
    conversation_history: list = []


def extract_report(r):
    if isinstance(r, str): return r
    if isinstance(r, dict): return r.get("text", str(r))
    if isinstance(r, list):
        return "\n".join([i.get("text", str(i)) if isinstance(i, dict) else str(i) for i in r])
    return str(r)


def format_history(history: list) -> str:
    if not history:
        return ""
    lines = []
    for msg in history[-6:]:
        role    = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")[:400]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def run_pipeline(query: str, history_str: str, queue: asyncio.Queue, loop):
    """Runs in a thread. Puts status events + final result onto the asyncio queue."""
    try:
        agent = build_graph()
        state: AgentState = {
            "query": query,
            "conversation_history": history_str,
            "market": "", "sector": "",
            "query_type": "", "time_horizon": "", "investor_type": "",
            "macro_data": None, "political_data": None, "sector_data": None,
            "regulatory_data": None, "exit_data": None, "fx_data": None, "timing_data": None,
            "draft_report": None, "reflection_notes": None,
            "final_report": None, "chart_data": None, "errors": []
        }

        node_labels = {
            "parse_query":     "Classifying intent...",
            "macro_node":      "Pulling macroeconomic data...",
            "political_node":  "Scanning political risk...",
            "sector_node":     "Analysing sector opportunity...",
            "exit_node":       "Researching exit landscape...",
            "fx_node":         "Analysing FX & currency risk...",
            "timing_node":     "Assessing market timing...",
            "aggregate_node":  "Aggregating research...",
            "regulatory_node": "Checking regulatory environment...",
            "brief_node":      "Writing market brief...",
            "reflect_node":    "Finalising...",
        }

        final_state = {}

        # Stream node-by-node progress
        for chunk in agent.stream(state, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                label = node_labels.get(node_name, f"Running {node_name}...")
                asyncio.run_coroutine_threadsafe(
                    queue.put(json.dumps({"type": "status", "text": label}) + "\n"),
                    loop
                )
                if isinstance(node_output, dict):
                    final_state.update(node_output)

        # Send done event
        asyncio.run_coroutine_threadsafe(
            queue.put(json.dumps({
                "type": "done",
                "data": {
                    "report":           extract_report(final_state.get("final_report", "")),
                    "draft_report":     extract_report(final_state.get("draft_report", "")),
                    "reflection_notes": extract_report(final_state.get("reflection_notes", "")),
                    "chart_data":       final_state.get("chart_data"),
                    "market":           final_state.get("market", ""),
                    "sector":           final_state.get("sector", ""),
                    "query_type":       final_state.get("query_type", ""),
                    "investor_type":    final_state.get("investor_type", ""),
                    "time_horizon":     final_state.get("time_horizon", ""),
                    "errors":           final_state.get("errors", []),
                    "agent":            "askmarketiq",
                    "route":            "pipeline",
                    "route_reason":     ""
                }
            }) + "\n"),
            loop
        )
    except Exception as e:
        asyncio.run_coroutine_threadsafe(
            queue.put(json.dumps({"type": "error", "text": str(e)}) + "\n"),
            loop
        )
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)


@app.post("/analyse")
async def analyse(request: QueryRequest):
    history_str = format_history(request.conversation_history)

    async def event_stream():
        # Step 1: Classify
        yield json.dumps({"type": "status", "text": "Classifying query..."}) + "\n"
        routing = classify_query(request.query, history_str)

        # Fast path — direct or out of scope
        if routing["route"] in ("direct", "out_of_scope"):
            yield json.dumps({"type": "status", "text": "Generating response..."}) + "\n"
            answer = direct_answer(
                request.query,
                routing.get("query_type", "general_knowledge"),
                history_str
            )
            yield json.dumps({
                "type": "done",
                "data": {
                    "report": answer, "draft_report": "", "reflection_notes": "",
                    "chart_data": None, "market": "", "sector": "",
                    "query_type": routing.get("query_type", "general_knowledge"),
                    "investor_type": "", "time_horizon": "", "errors": [],
                    "agent": "askmarketiq",
                    "route": routing["route"],
                    "route_reason": routing.get("reason", "")
                }
            }) + "\n"
            return

        # Full pipeline — run in thread, stream progress events back
        yield json.dumps({"type": "status", "text": "Starting research pipeline..."}) + "\n"

        loop  = asyncio.get_event_loop()
        queue = asyncio.Queue()

        loop.run_in_executor(
            executor,
            run_pipeline,
            request.query, history_str, queue, loop
        )

        # Drain queue until None sentinel
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/health")
async def health():
    return {"status": "ok"}
