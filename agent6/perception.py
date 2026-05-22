"""
perception.py — decompose the user goal into a stable list of sub-goals.

Perception runs once per loop iteration. It receives:
  - the original user query
  - the relevant memory hits
  - the current history of action/answer events
  - the prior_goals list (the goals it returned last iteration, so their ids
    stay stable across iterations)
  - a run_id (for trace correlation)

It returns a PerceptionOutput containing the up-to-date goals list. The same
goal ids are reused across iterations so the orchestrator can recognise the
same step from one turn to the next. Goals already known to be satisfied are
marked status="done"; new sub-goals are appended; the "first open" goal is
what the loop tackles next.

Safety nets (defence in depth — none of these depend on the model being smart):
  - position-based artifact_index: when memory hits include artifact rows, we
    list them by position in the prompt and ask Perception to refer to them
    by integer index, then map back to the real ``art:...`` handle. This
    avoids the model hallucinating a handle.
  - sticky-done: once a goal is marked done in a prior iteration, the
    orchestrator carries that status forward unless Perception explicitly
    re-opens it.
  - force-attach for synthesis goals: if the first open goal has synthesis
    keywords (synthesise / extract / list / compare / choose / decide /
    select / recommend) and any artifact_id is available in the memory hits
    or recent history, attach the most-recent matching artifact handle. This
    is the rule the class doc calls out for Query D.

LLM routing
───────────
Pinned to ``provider="g"`` and ``auto_route="perception"``. Temperature is
1.0 because Gemini 3.x flash-lite loops at temperature=0 on structured-
output requests (documented in the class page).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from schemas import Goal, MemoryItem, PerceptionOutput

_GATEWAY = Path(__file__).resolve().parent.parent / "llm_gatewayV3"
if str(_GATEWAY) not in sys.path:
    sys.path.insert(0, str(_GATEWAY))
from client import LLM  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# Wire-format schema (uses artifact_index instead of art:... handle)
# ──────────────────────────────────────────────────────────────────────────

class _WireGoal(BaseModel):
    id: str
    description: str
    status: str = "open"          # validated to {open,done} below
    # Position into the ATTACHED ARTIFACT INDEX list shown in the prompt, or
    # null when no attach is needed. The orchestrator maps this back to a
    # real handle.
    artifact_index: Optional[int] = None
    # REQUIRED. One-word tag from a closed set telling downstream layers
    # what kind of cognitive step this goal is. Made required so Gemini
    # can't silently omit it.
    reasoning_type: str


class _WirePerception(BaseModel):
    goals: list[_WireGoal] = Field(default_factory=list)
    # A one-sentence rationale: how Perception arrived at the current goal
    # list, what changed since last iteration, and what it expects next.
    # This is the closest thing the structured-output schema has to a
    # reasoning trace — keep it terse but substantive.
    notes: str = ""


# ──────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────

PERCEPTION_SYSTEM = """\
You are the Perception layer of an agent loop.

Your job is to break the user's query into a STABLE, COARSE list of sub-goals
and keep that list up to date as the agent works.

HOW TO REASON (do this privately before emitting JSON)
------------------------------------------------------
Before you produce the final JSON, mentally walk through these steps in order.
Do NOT include the walkthrough in the output — only the resulting JSON ships.

  1. UNDERSTAND. Restate the user's query in your head. Identify the verbs
     (fetch / search / extract / list / create / compare / answer / ...).
     Count how many distinct outcomes the user is actually asking for.

  2. REVIEW. Look at PRIOR GOALS and HISTORY. For each prior goal, decide:
     - Is there a successful tool result or substantive answer for it in
       HISTORY? If yes, it stays "done".
     - Is there an ERROR for it in HISTORY? If yes, it stays "open" and
       you note the failure.
     - Is it still relevant given what's now known? If no, leave it but
       describe why in notes.

  3. PLAN. Decide the minimal coarse goal list (1-4 goals) that covers all
     the user's outcomes. Re-use prior goal ids wherever possible — do NOT
     renumber.

  4. CLASSIFY. For each goal, pick the single best reasoning_type from the
     closed list below.

  5. SELF-CHECK before outputting:
     (a) Does every "done" goal have evidence in HISTORY?
     (b) Does every open goal have a clear next action it implies?
     (c) Are there no "store in memory" / "remember X" goals? (those are
         handled before Perception runs)
     (d) Are there no artifact handles ("art:1", "art:2", ...) anywhere in
         your output — neither in any description nor in notes?
     (e) Is the goals list 1-4 items, no more than 5?
     If any check fails, revise BEFORE emitting the JSON.

REASONING_TYPE — closed set
---------------------------
Tag each goal with exactly one of:
  - "lookup"        : retrieve a single fact from memory or a known source
  - "search"        : open-ended search / discovery (web_search-style)
  - "fetch"         : retrieve the bytes of a specific named URL or file
  - "extraction"    : pull facts from already-known text (attached artifact, snippet)
  - "synthesis"     : combine multiple pieces into one substantive answer
  - "comparison"    : pick the best option from a set against a criterion
  - "tool_action"   : a state-changing tool call (create_file, make_dir, ...)
  - "memory_recall" : answer is already in MEMORY HITS; no external work needed

GRANULARITY (very important)
----------------------------
Aim for 1-4 sub-goals. NEVER more than 5. The right number is determined by
the abstract PATTERN of the query, not by the topic. Apply these patterns:

PATTERN 1 — "Fetch ONE source, then extract MULTIPLE facts from it"
   Two goals: one fetch + one combined extraction.
   The extraction is ONE goal even if the user lists 5 facts to pull. Never
   emit one extraction goal per fact.
     Generic shape:
       g1 = "Fetch <the named source>" — reasoning_type: fetch
       g2 = "Extract <all requested facts> from the fetched content"
            — reasoning_type: extraction
     Triggers: query names a specific URL/file AND asks for multiple facts.

PATTERN 2 — "Discover candidates + check a constraint + pick the best"
   Three goals: discovery + constraint lookup + comparison.
     Generic shape:
       g1 = "Search/discover candidates that satisfy <topic>" — search
       g2 = "Look up the value of <constraint variable>"     — lookup or fetch
       g3 = "Compare candidates against the constraint and choose one"
                                                              — comparison
     Triggers: "find N <things>, check <condition>, pick which is best".

PATTERN 3 — "Search the web on a topic and summarise what sources agree on"
   Two goals: one search + one synthesis. Snippets already summarise each
   result, so "reading top N" is what the synthesis step does over the
   snippet text — do NOT emit a separate fetch-each-URL goal unless the
   user EXPLICITLY says "fetch each URL" or "give me the full text".
     Generic shape:
       g1 = "Search the web for <topic> (snippets summarise the top results)"
            — search
       g2 = "Synthesize the <answer shape> the top sources agree on, using
             the search snippets" — synthesis

PATTERN 4 — "Imperative actions that produce files / state changes"
   One goal per state-changing action. Group multiple files of the same
   type only if they share content; otherwise one goal per file.
     Generic shape:
       g_i = "Create / update / write <a specific file or record> with
              <its purpose>" — tool_action
     Special architectural rule (THIS IS NOT NEGOTIABLE for this agent):
     If the query mixes a DECLARATIVE FACT with an IMPERATIVE REQUEST
     (e.g. "X is true. Remember that AND do Y") — the "remember" half is
     handled BEFORE Perception runs by memory.remember(query) at the top
     of the loop. DO NOT emit any goal whose only purpose is "store X in
     memory" / "remember Y". Plan only the imperative half.

PATTERN 5 — "Question whose answer is already in MEMORY HITS"
   One goal: a single memory-backed answer. No tool calls.
     Generic shape:
       g1 = "Answer <the question> using MEMORY HITS as the source of truth"
            — memory_recall
     Triggers: query is a question AND a relevant fact appears in
     MEMORY HITS in this iteration.

WORKED CONCRETE EXAMPLES (these instantiate the patterns above)
---------------------------------------------------------------
Use these to ground the patterns, but recognise that the same pattern
applies to ANY topic — Wikipedia / restaurants / stocks / sports scores /
internal docs — not just the topics shown here.

  PATTERN 1: "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell
  me his birth date, death date, and three contributions"
    → g1 fetch the URL, g2 extract all three facts from the fetched page.

  PATTERN 2: "Find 3 family-friendly things to do in Tokyo this weekend,
  check Saturday's weather, pick the most appropriate"
    → g1 search activities, g2 lookup Saturday's weather, g3 compare and choose.

  PATTERN 3: "Search 'Python asyncio best practices', read the top 3
  results, give me a numbered list of the advice they agree on"
    → g1 web_search the topic, g2 synthesize agreed advice from snippets.

  PATTERN 4: "My mom's birthday is 15 May 2026. Remember that and give
  me a calendar reminder for two weeks before and on the day"
    → memory handled by remember(query). Then two file-creation goals:
      g1 create a reminder file for 1 May 2026, g2 for 15 May 2026.

  PATTERN 5: "When is mom's birthday?"
    → g1 answer from MEMORY HITS.

RULES
-----
1. On the first iteration, decompose the query into 1-4 atomic-but-coarse goals.
   Use ids "g1", "g2", "g3", "g4" in order.
2. NEVER emit a goal whose only purpose is "store X in memory" or "remember Y".
   Memory is handled by a separate layer at run start; Perception's job is to
   plan ACTIONS (tool calls) and ANSWERS, not state writes.
3. Each goal must be either:
   - an ACTION goal (verb: fetch, search, create, read, update, get_time, ...)
     that maps to ONE tool call, OR
   - an ANSWER/SYNTHESIS goal (verb: extract, list, compare, choose, decide,
     summarise, answer) that maps to a Decision text answer.
4. On later iterations, RE-USE the same ids for the same logical goal in
   PRIOR GOALS. Only append new ids if a genuinely new step emerged.
5. Mark status="done" when HISTORY shows the goal is satisfied:
   - an ACTION goal is done as soon as a successful tool call for it appears
     in HISTORY (kind=action with a non-error result_descriptor),
   - an ANSWER goal is done as soon as Decision emitted a substantive answer
     for it (kind=answer).
6. If a recent action in HISTORY shows "ERROR" or "Error executing tool", do
   NOT mark the related goal done. Mention the failure in notes so Decision
   can pick a different action next iteration.
7. NEVER mention artifact handles like "art:1" / "art:2" ANYWHERE in your
   output — not in description, not in notes, not in any string field.
   Handles are internal references used only by the orchestrator. If a goal
   needs an artifact's bytes, set artifact_index to the 0-based position of
   that artifact in the ATTACHED ARTIFACT INDEX list below. Leave
   artifact_index null otherwise. When you need to talk about an artifact
   in notes, refer to it by what it IS ("the fetched page", "the AccuWeather
   markdown") rather than by its handle.
8. The "notes" field MUST be a single short sentence that:
   (a) names what just changed since the last iteration (or "first pass"
       on iter 1), and
   (b) names the next action that should follow.
   Bad: "Starting the process". Good: "Search done; weather snippet shows
   patchy rain — next step is to synthesize an indoor activity choice."

FALLBACKS (when things go wrong)
--------------------------------
- If the SAME tool has failed for the SAME goal TWICE in HISTORY, rewrite
  the goal description to suggest a DIFFERENT route (e.g. switch from
  fetch_url to web_search, or accept the snippets as sufficient and pivot
  the goal to a synthesis). Note the pivot in `notes`.
- If a goal becomes impossible (e.g. user asked for a website that 403s
  and there's no alternate source), keep it open but rewrite its
  description to say "answer with the apology / explanation that X is
  unavailable" — and set reasoning_type to "synthesis". The loop should
  not get stuck retrying the same failing call indefinitely.
- If Perception cannot decide between two decompositions, prefer FEWER
  goals over more. Coarser is safer than over-planning.

OUTPUT — exact schema (this is what the response_format validator enforces)
---------------------------------------------------------------------------
Return a JSON object with EXACTLY these two top-level keys:

{
  "goals": [ <Goal>, <Goal>, ... ],           // 1-5 items
  "notes": "<one short sentence>"             // see rule 8
}

Each <Goal> is an object with EXACTLY these five keys:

{
  "id":              "g1",                    // string, stable across iterations
  "description":    "Fetch the Wikipedia page for Claude Shannon",  // string
  "status":         "open",                   // "open" | "done"
  "artifact_index": null,                     // integer index into the
                                              //   ATTACHED ARTIFACT INDEX list,
                                              //   or null when not needed
  "reasoning_type": "fetch"                   // MANDATORY. exactly one of:
                                              //   lookup | search | fetch
                                              //   | extraction | synthesis
                                              //   | comparison | tool_action
                                              //   | memory_recall
}

All five keys are MANDATORY on every goal — never omit reasoning_type or
artifact_index (use null for artifact_index when not needed). Status must be
exactly "open" or "done", never any other string.

WORKED EXAMPLE (single iteration of Query A, after the page has been fetched):

{
  "goals": [
    {
      "id": "g1",
      "description": "Fetch the Wikipedia page for Claude Shannon",
      "status": "done",
      "artifact_index": 0,
      "reasoning_type": "fetch"
    },
    {
      "id": "g2",
      "description": "Extract birth date, death date, and three key contributions",
      "status": "open",
      "artifact_index": 0,
      "reasoning_type": "extraction"
    }
  ],
  "notes": "The Wikipedia page has been fetched and is available as an attached artifact; next step is to extract the three facts from it."
}

Reply with the JSON object alone — no surrounding prose, no Markdown fences.
"""


def _render_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(none)"
    lines = []
    for ev in history[-10:]:
        if ev.get("kind") == "action":
            lines.append(
                f"  iter {ev.get('iter')}: action goal={ev.get('goal_id')} "
                f"tool={ev.get('tool')} args={ev.get('arguments')} "
                f"result={ev.get('result_descriptor')}"
            )
        else:
            lines.append(
                f"  iter {ev.get('iter')}: answer goal={ev.get('goal_id')} "
                f"text={(ev.get('text') or '')[:200]!r}"
            )
    return "\n".join(lines)


def _render_memory(hits: list[MemoryItem]) -> tuple[str, list[str]]:
    """Render memory hits and return (text, artifact_index -> handle list)."""
    if not hits:
        return "(none)", []
    art_handles: list[str] = []
    lines = []
    for h in hits:
        lines.append(f"  - [{h.kind}] {h.text}  (keywords={h.keywords})")
        if h.artifact_id:
            art_handles.append(h.artifact_id)
    return "\n".join(lines), art_handles


def _render_prior_goals(prior: list[Goal]) -> str:
    if not prior:
        return "(none — first iteration)"
    return "\n".join(
        f"  {g.id} [{g.status}] {g.description}" for g in prior
    )


def _render_artifact_index(handles: list[str]) -> str:
    if not handles:
        return "(no artifacts available)"
    return "\n".join(f"  [{i}] {h}" for i, h in enumerate(handles))


_SYNTH_KEYWORDS = (
    "synthes", "extract", "list", "compare", "choose", "decide",
    "select", "recommend", "summari", "consolidat", "tell me",
)


def _is_synthesis(goal: Goal) -> bool:
    d = goal.description.lower()
    return any(k in d for k in _SYNTH_KEYWORDS)


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def observe(
    query: str,
    hits: list[MemoryItem],
    history: list[dict[str, Any]],
    prior_goals: list[Goal],
    run_id: str,
) -> PerceptionOutput:
    mem_text, art_handles = _render_memory(hits)
    user_prompt = (
        f"USER QUERY:\n{query}\n\n"
        f"PRIOR GOALS:\n{_render_prior_goals(prior_goals)}\n\n"
        f"HISTORY (most recent last):\n{_render_history(history)}\n\n"
        f"MEMORY HITS:\n{mem_text}\n\n"
        f"ATTACHED ARTIFACT INDEX (refer to these by artifact_index):\n"
        f"{_render_artifact_index(art_handles)}\n\n"
        "Return the updated goals list now."
    )

    llm = LLM()
    schema = _WirePerception.model_json_schema()
    reply = llm.chat(
        prompt=user_prompt,
        system=PERCEPTION_SYSTEM,
        cache_system=True,
        response_format={
            "type": "json_schema",
            "schema": schema,
            "name": "Perception",
            "strict": True,
        },
        provider="g",
        auto_route="perception",
        temperature=1.0,            # Gemini 3.x loops at temp=0 on structured out
        max_tokens=1024,
    )

    parsed = reply.get("parsed") or {}
    try:
        wire = _WirePerception.model_validate(parsed)
    except ValidationError:
        # Fallback: reuse prior goals as-is, or seed one open goal from the query.
        if prior_goals:
            return PerceptionOutput(goals=prior_goals, notes="(perception parse failed, kept prior)")
        return PerceptionOutput(
            goals=[Goal(id="g1", description=query, status="open")],
            notes="(perception parse failed, seeded one goal)",
        )

    # Map wire goals -> typed Goals with real artifact handles
    goals: list[Goal] = []
    prior_status = {g.id: g.status for g in prior_goals}
    for wg in wire.goals:
        status = wg.status if wg.status in ("open", "done") else "open"
        # Sticky-done: once Perception marked something done in a prior turn,
        # don't let a later iteration silently re-open it.
        if prior_status.get(wg.id) == "done":
            status = "done"
        attach: Optional[str] = None
        if wg.artifact_index is not None and 0 <= wg.artifact_index < len(art_handles):
            attach = art_handles[wg.artifact_index]
        goals.append(Goal(
            id=wg.id,
            description=wg.description,
            status=status,
            attach_artifact_id=attach,
            reasoning_type=wg.reasoning_type,
        ))

    out = PerceptionOutput(goals=goals, notes=wire.notes)

    # Force-attach safety net for synthesis goals.
    nxt = out.next_unfinished()
    if nxt is not None and nxt.attach_artifact_id is None and _is_synthesis(nxt) and art_handles:
        # Pick the most-recent artifact handle from memory hits.
        nxt.attach_artifact_id = art_handles[0]

    return out
