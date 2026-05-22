"""
agent6.py — Session 6 orchestrator.

Wires the four cognitive layers (Memory, Perception, Decision, Action) into
one loop and routes every LLM call through llm_gatewayV3 at
http://localhost:8101.

  $ python agent6.py "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me ..."

Before running:
  1. Start llm_gatewayV3:    cd llm_gatewayV3 && ./run.sh
  2. Ensure mcp_server.py's deps are installed (ddgs, crawl4ai, tavily, mcp,
     python-dotenv, httpx).
  3. Ensure .env contains GEMINI_API_KEY and NVIDIA_API_KEY (and optionally
     TAVILY_API_KEY for the web_search tool).

Layout:
  schemas.py       Pydantic v2 contracts (every boundary is typed).
  artifacts.py     File-backed store for large tool payloads.
  memory.py        Durable, file-backed keyword memory (state/memory.json).
  perception.py    Goal decomposition + stable goal ids (structured output).
  decision.py      Pick one ToolCall OR one final answer (tool_choice=auto).
  action.py        MCP dispatch + 4 KB artifact threshold + art: handle guard.
  agent6.py        This file — the loop.
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import action
import artifacts
import decision
import memory
import perception
from schemas import Goal, PerceptionOutput

GATEWAY_URL = "http://localhost:8101"
MAX_ITERATIONS = 12


# ────────────────────────────────────────────────────────────────────────────
# Gateway / MCP bootstrap
# ────────────────────────────────────────────────────────────────────────────

def ensure_gateway() -> None:
    """Fail fast with a clear message if the gateway is not running."""
    try:
        r = httpx.get(f"{GATEWAY_URL}/v1/capabilities", timeout=4.0)
        r.raise_for_status()
    except Exception as e:
        raise SystemExit(
            f"\n[fatal] LLM Gateway V3 is not reachable at {GATEWAY_URL}.\n"
            f"        Start it with:  cd llm_gatewayV3 && ./run.sh\n"
            f"        (reason: {e})\n"
        )


def _mcp_tools_for_decision(mcp_tools) -> list[dict]:
    """Reshape MCP tool descriptors into the gateway's canonical ToolDef form."""
    out = []
    for t in mcp_tools:
        out.append({
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema or {"type": "object", "properties": {}},
        })
    return out


# ────────────────────────────────────────────────────────────────────────────
# Pretty-printing helpers (operational visibility)
# ────────────────────────────────────────────────────────────────────────────

def _hr(label: str) -> None:
    print(f"\n─── {label} " + "─" * (70 - len(label)))


def _show_perception(obs) -> None:
    for g in obs.goals:
        attach = f"  attach={g.attach_artifact_id}" if g.attach_artifact_id else ""
        rtype = f"  <{g.reasoning_type}>" if g.reasoning_type else ""
        print(f"[perception]    [{g.status}] {g.id} {g.description}{rtype}{attach}")
    if obs.notes:
        print(f"[perception]    notes: {obs.notes}")


def _show_decision(out) -> None:
    if out.is_answer:
        snippet = (out.answer or "").strip()
        if len(snippet) > 240:
            snippet = snippet[:240] + " ..."
        print(f"[decision]      ANSWER: {snippet}")
    else:
        tc = out.tool_call
        print(f"[decision]      TOOL_CALL: {tc.name}({json.dumps(tc.arguments)})")


def _final_answer_from(history: list[dict]) -> str:
    """The final answer is the last 'answer' event the loop recorded."""
    for ev in reversed(history):
        if ev.get("kind") == "answer" and ev.get("text"):
            return ev["text"]
    return "(no final answer produced)"


# ────────────────────────────────────────────────────────────────────────────
# The loop
# ────────────────────────────────────────────────────────────────────────────

async def run(query: str) -> str:
    ensure_gateway()
    run_id = uuid.uuid4().hex[:8]
    history: list[dict] = []
    prior_goals: list[Goal] = []

    print("═" * 78)
    print(f"agent6.py — Session 6 four-role agent loop")
    print(f"run_id   : {run_id}")
    print(f"query    : {query}")
    print("═" * 78)

    # Durable memory: classify the user's query first so facts/preferences in
    # it survive into future runs.
    try:
        saved = memory.remember(query, source="user_query", run_id=run_id)
    except httpx.HTTPStatusError as e:
        print(f"[memory.remember]  ERROR: gateway {e.response.status_code} — skipped")
        saved = None
    if saved:
        print(f"[memory.remember]  saved {saved.kind}: {saved.text!r}")
    else:
        print(f"[memory.remember]  query not classified as memorable")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).with_name("mcp_server.py"))],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools_list = (await session.list_tools()).tools
            tools = _mcp_tools_for_decision(mcp_tools_list)
            print(f"[mcp] tools loaded: {[t.name for t in mcp_tools_list]}")

            for it in range(1, MAX_ITERATIONS + 1):
                _hr(f"iter {it}")

                # 1. Memory: keyword-recall relevant rows.
                hits = memory.read(query, history)
                print(f"[memory.read]   {len(hits)} hits")

                # 2. Perception: decompose / update goals.
                try:
                    obs = perception.observe(query, hits, history, prior_goals, run_id)
                except httpx.HTTPStatusError as e:
                    print(f"[perception]    ERROR: gateway {e.response.status_code} — "
                          f"reusing prior goals")
                    if not prior_goals:
                        prior_goals = [Goal(id="g1", description=query, status="open")]
                    obs = PerceptionOutput(goals=prior_goals, notes="(perception unavailable)")
                else:
                    prior_goals = obs.goals
                _show_perception(obs)

                if obs.all_done:
                    print("\n[done] all goals satisfied")
                    break

                goal = obs.next_unfinished()
                if goal is None:
                    print("\n[done] no unfinished goal")
                    break

                # 3. Attachment: hydrate artifact bytes if Perception requested it
                #    AND the artifact still exists (defence vs hallucinated handles).
                attached: list[tuple[str, bytes]] = []
                if goal.attach_artifact_id and artifacts.exists(goal.attach_artifact_id):
                    raw = artifacts.get_bytes(goal.attach_artifact_id)
                    attached.append((goal.attach_artifact_id, raw))
                    print(f"[attach]        {goal.attach_artifact_id} ({len(raw)} bytes)")

                # 4. Decision: one LLM call -> answer OR exactly one ToolCall.
                try:
                    out = decision.next_step(goal, hits, attached, history, tools)
                except httpx.HTTPStatusError as e:
                    print(f"[decision]      ERROR: gateway {e.response.status_code} — "
                          f"recording failure and continuing")
                    history.append({
                        "iter": it,
                        "kind": "action",
                        "goal_id": goal.id,
                        "tool": "(decision-call)",
                        "arguments": {},
                        "result_descriptor": f"ERROR: gateway returned {e.response.status_code}. "
                                             "All workers unavailable or rate-limited.",
                        "artifact_id": None,
                    })
                    continue
                _show_decision(out)

                if out.is_answer:
                    history.append({
                        "iter": it,
                        "kind": "answer",
                        "goal_id": goal.id,
                        "text": out.answer,
                    })
                    continue

                # 5. Action: dispatch the tool call.
                tc = out.tool_call
                result_text, art_id = await action.execute(session, tc, run_id=run_id)
                print(f"[action]        -> {result_text[:200]}"
                      + (" ..." if len(result_text) > 200 else ""))

                # 6. Memory: record the outcome (no LLM classification here).
                memory.record_outcome(
                    tool_call=tc,
                    result_text=result_text,
                    artifact_id=art_id,
                    run_id=run_id,
                    goal_id=goal.id,
                )

                history.append({
                    "iter": it,
                    "kind": "action",
                    "goal_id": goal.id,
                    "tool": tc.name,
                    "arguments": tc.arguments,
                    "result_descriptor": result_text[:300],
                    "artifact_id": art_id,
                })
            else:
                print(f"\n[stopped] reached MAX_ITERATIONS={MAX_ITERATIONS}")

    final = _final_answer_from(history)
    print("\n" + "═" * 78)
    print("FINAL ANSWER:")
    print(final)
    print("═" * 78)
    return final


# ────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ────────────────────────────────────────────────────────────────────────────

DEFAULT_QUERIES = {
    "A": (
        "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his "
        "birth date, death date, and three key contributions to information theory."
    ),
    "B": (
        "Find 3 family-friendly things to do in Tokyo this weekend. Check "
        "Saturday's weather forecast there and tell me which one is most "
        "appropriate."
    ),
    "C1": (
        "My mom's birthday is 15 May 2026. Remember that and give me a "
        "calendar reminder for two weeks before and on the day."
    ),
    "C2": "When is mom's birthday?",
    "D": (
        "Search for 'Python asyncio best practices', read the top 3 results, "
        "and give me a short numbered list of the advice they agree on."
    ),
}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python agent6.py \"<your query>\"")
        print("  python agent6.py --query A|B|C1|C2|D")
        sys.exit(1)
    if sys.argv[1] == "--query":
        if len(sys.argv) < 3 or sys.argv[2] not in DEFAULT_QUERIES:
            print(f"Unknown --query, choose one of: {list(DEFAULT_QUERIES)}")
            sys.exit(1)
        query = DEFAULT_QUERIES[sys.argv[2]]
    else:
        query = " ".join(sys.argv[1:])
    asyncio.run(run(query))


if __name__ == "__main__":
    main()
