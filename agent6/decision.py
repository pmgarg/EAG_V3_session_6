"""
decision.py — pick the next step for one goal.

Decision is ONE LLM call. It receives the current goal, the memory hits, the
optional raw bytes of an attached artifact, the recent history, and the list
of MCP tools. It returns a DecisionOutput containing EITHER:

  - a final-answer string (when the goal can be answered from what is already
    in the prompt — no further tool calls needed), OR
  - a single typed ToolCall (the next MCP dispatch to perform).

It never returns both. It never picks more than one tool.

Routing
───────
Decision goes through ``auto_route="decision"``: the gateway's router pool
classifies the prompt into TINY/LARGE and picks a worker accordingly. There
is no provider pin for Decision.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

from schemas import DecisionOutput, Goal, MemoryItem, ToolCall

_GATEWAY = Path(__file__).resolve().parent.parent / "llm_gatewayV3"
if str(_GATEWAY) not in sys.path:
    sys.path.insert(0, str(_GATEWAY))
from client import LLM  # noqa: E402


DECISION_SYSTEM = """\
You are the Decision layer of an agent loop. For ONE goal, decide the NEXT
single step.

OUTPUT (the contract is strict)
-------------------------------
Reply with EXACTLY ONE of the following:
  (A) a plain-text answer to the goal, OR
  (B) exactly one tool call from the provided tool list.

Never reply with both. Never call more than one tool in a single turn. Do not
narrate your choice — your thinking happens silently before the reply, never
inside it.

HOW TO REASON (do this privately before replying)
-------------------------------------------------
Walk through these 4 steps in your head. Do NOT include the walkthrough in
the response — only the final answer or tool call ships.

  1. CLASSIFY the goal. Pick exactly one reasoning_type from this closed set:
       - lookup        : answer a single fact from MEMORY HITS or ATTACHED ARTIFACTS
       - extraction    : pull facts from already-known text (attached bytes, snippets)
       - synthesis     : combine multiple sources into a substantive list/comparison
       - comparison    : pick the best option from a known set against a criterion
       - tool_action   : a state-changing tool call (create_file, make_dir, ...)
       - fetch         : retrieve the bytes of one specifically named URL
       - search        : open-ended web_search
       - memory_recall : the answer is already a fact in MEMORY HITS

  2. CHECK what you already have. Scan ATTACHED ARTIFACTS, MEMORY HITS, and
     RECENT HISTORY. List (silently) what's available, what's missing, and
     whether what's available is sufficient to answer the goal substantively.

  3. DECIDE answer-or-tool:
     - If reasoning_type is lookup / memory_recall / synthesis / comparison /
       extraction AND the needed material is already in the prompt → ANSWER.
     - If reasoning_type is tool_action / fetch / search OR the prompt is
       missing what's needed → call exactly ONE tool.
     If you're 50/50, prefer ANSWER (fewer wasted iterations).

  4. SELF-CHECK before emitting:
     (a) Does any argument start with "art:"? If yes, fix it — those are
         internal handles, not paths/URLs.
     (b) If you're about to web_search, is the SAME query already in RECENT
         HISTORY? If yes, switch to ANSWER instead.
     (c) If you're about to fetch_url, are 3+ snippets already in MEMORY
         HITS that would let you answer? If yes, switch to ANSWER.
     (d) If you're about to ANSWER a list/comparison/synthesis/extraction
         goal, is your draft at least 3 sentences OR an explicit list with
         ≥3 items? If no, expand before sending.
     (e) If you're about to call create_file with a path containing "/",
         have you already seen a successful make_dir for that directory in
         HISTORY? If no, call make_dir FIRST.

RULES
-----
1. Strings beginning with "art:" are INTERNAL artifact handles, not file
   paths and not URLs. NEVER pass an "art:" string to read_file, fetch_url,
   or any other tool. When the bytes of an artifact are required to answer a
   goal, those bytes are already pasted under "ATTACHED ARTIFACTS" below —
   read them there.
2. If the goal asks for an EXTRACTION, LIST, COMPARISON, SELECTION, or
   SYNTHESIS, your answer must be SUBSTANTIVE — at least three sentences or
   a list of items that actually does the work. Never reply with a
   meta-answer like "the page has been fetched, how would you like to
   proceed?" or "I can do this, just let me know".
3. If the goal is IMPERATIVE — its verb is create / write / update / fetch
   / search / read / save / schedule / send / make / set up — you MUST
   call a tool IF a tool that can perform that action is in the provided
   tool list. Never satisfy an imperative goal with prose like "follow
   these steps in your calendar app" or "to set this up, open …" when
   the tool actually exists. The agent's job is to DO the action via the
   available MCP tools (e.g. create_file, update_file, fetch_url,
   web_search), not to instruct the user how to do it themselves.

   EXCEPTION — tool unavailable. If NO tool in the provided list can
   perform the requested action (e.g. user asks to send_email but the
   list contains no email-sending tool), do NOT invent a tool and do NOT
   blindly retry the closest tool. Instead apply Fallback A below:
   answer with a clear explanation and, when possible, use the closest
   available tool to leave a usable artifact (e.g. write the email body
   to a file with create_file). This exception overrides rule 3's "MUST
   call a tool" — the imperative-tool requirement assumes the tool
   exists.
4. Prefer answering directly only when the goal is a QUESTION or SYNTHESIS
   and the prompt already contains what's needed (bytes pasted under
   ATTACHED ARTIFACTS, or a fact in MEMORY HITS). Otherwise call a tool.
5. Search snippets in tool_outcome rows of MEMORY HITS and RECENT HISTORY
   already contain the title, URL, and a one-sentence summary of each
   result. For SYNTHESIS / SELECTION / LIST goals (e.g. "find 3 things
   to do in Tokyo and pick the best"), these snippets are USUALLY ENOUGH
   to answer — do NOT call fetch_url just to confirm details that are
   already in the snippet. Only call fetch_url when:
     - the goal explicitly names a single URL to fetch (e.g. "fetch
       https://en.wikipedia.org/wiki/Claude_Shannon and tell me ..."), OR
     - the snippets are genuinely insufficient (truncated to one phrase
       with no facts, or all snippets point to the same useless landing
       page).
   If you already called web_search and got 3+ snippets back, the NEXT
   step is almost always to ANSWER, not to fetch any of the result URLs.
   NEVER call web_search twice with the same or near-identical query in
   one run — if you see your previous web_search query in RECENT HISTORY,
   move on to the synthesis answer instead of re-searching.
6. When you call a tool, fill its arguments fully and correctly per its
   schema. For create_file, always pass a sensible relative path under the
   sandbox (e.g. "reminders/mom_birthday_2026.txt") AND non-empty content.
   create_file requires the parent directory to exist — if the path contains
   a "/", call make_dir on the directory FIRST, then create_file on the file.
   make_dir is idempotent (no error if the directory already exists).

FALLBACKS (when things go wrong or get blocked)
-----------------------------------------------
A. Tool unavailable (this OVERRIDES rule 3's "MUST call a tool" requirement).
   If the goal needs a tool that isn't in the provided tool list (e.g.
   user asks to send_email but no send_email tool exists, or to "open
   a calendar app" when the agent has no calendar tool), do NOT invent
   a tool and do NOT pick an obviously wrong substitute. Choose ONE of
   these two responses, in this order of preference:

     A.1 If a CLOSEST AVAILABLE TOOL can leave a usable artifact for the
         user (e.g. write the email body to a file with create_file,
         draft the calendar event as a text reminder), call THAT tool
         this turn. On the next iteration, answer with prose explaining
         what was done and what the user must do manually.

     A.2 If no available tool can produce a usable artifact, ANSWER
         immediately with a clear, one-paragraph explanation: name the
         missing capability ("I don't have an email-sending tool"),
         explain what you CAN do instead, and suggest what the user
         should do ("you can copy the body below into your mail client").

   Either way, the agent must MAKE PROGRESS — never block indefinitely on
   a missing tool.

B. Goal impossible / repeatedly failing. If RECENT HISTORY shows the
   same goal has been attempted TWICE with errors (403, 404, timeout,
   parse failure, etc.), do NOT retry the same tool with the same
   arguments a third time. Pivot:
     - If it was fetch_url that failed, try web_search to get a usable
       snippet, OR answer from snippets already in MEMORY HITS.
     - If it was a content tool (create_file, update_file) that failed
       because of a path issue, try the corrected path once.
     - If pivoting is impossible, ANSWER with an honest "X is unavailable
       because Y" explanation. Never loop indefinitely.

C. Arguments missing or ambiguous. If you cannot fill a required tool
   argument from the prompt (e.g. you need a URL but only have a topic),
   prefer the broader tool that can discover it (e.g. web_search) over
   guessing a URL. NEVER fabricate a URL, file path, or identifier.

D. Insufficient information for synthesis. If a SYNTHESIS / COMPARISON
   / SELECTION goal genuinely lacks evidence — fewer than 2 sources, or
   snippets that all say the same one-line headline — call ONE more
   search-or-fetch tool to enrich the picture. After at most ONE such
   enrichment attempt, you must answer with whatever you have, marking
   uncertainty explicitly in the answer (e.g. "Based on the single
   source available ...").

E. Uncertainty. If you are unsure between two answers (e.g. two sources
   disagree), present BOTH in the answer with a one-line note on which
   is more credible and why. Do NOT pick silently.
"""


def _render_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(none)"
    lines = []
    for ev in history[-6:]:
        if ev.get("kind") == "action":
            lines.append(
                f"  iter {ev.get('iter')} action: tool={ev.get('tool')} "
                f"args={ev.get('arguments')} -> {ev.get('result_descriptor')}"
            )
        else:
            lines.append(
                f"  iter {ev.get('iter')} answer: {(ev.get('text') or '')[:240]!r}"
            )
    return "\n".join(lines)


def _render_memory(hits: list[MemoryItem]) -> str:
    if not hits:
        return "(none)"
    return "\n".join(f"  - [{h.kind}] {h.text}" for h in hits)


def _render_attached(attached: list[tuple[str, bytes]]) -> str:
    if not attached:
        return ""
    parts = []
    for handle, raw in attached:
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = "(undecodable bytes)"
        # Cap each artifact to keep the prompt well below context limits.
        if len(text) > 30000:
            text = text[:30000] + "\n...[truncated]"
        parts.append(f"=== {handle} ({len(raw)} bytes) ===\n{text}")
    return "\n\nATTACHED ARTIFACTS:\n" + "\n\n".join(parts)


def next_step(
    goal: Goal,
    hits: list[MemoryItem],
    attached: list[tuple[str, bytes]],
    history: list[dict[str, Any]],
    mcp_tools: list[dict],
) -> DecisionOutput:
    user_prompt = (
        f"GOAL ({goal.id}): {goal.description}\n\n"
        f"MEMORY HITS:\n{_render_memory(hits)}\n\n"
        f"RECENT HISTORY:\n{_render_history(history)}"
        f"{_render_attached(attached)}\n\n"
        "Decide the next step now: either answer in plain text, or call ONE tool."
    )

    llm = LLM()
    # Decision is the canonical auto-routed call: the gateway's router pool
    # classifies prompt size + density into TINY/LARGE/HUGE and picks the
    # worker accordingly. This matches the Session 6 architecture diagram.
    #
    # Environment requirement: at least one of GROQ_API_KEY, CEREBRAS_API_KEY,
    # GITHUB_ACCESS_TOKEN must be set so the router pool has real routers,
    # AND at least two workers must be wired (Gemini + Groq, or Gemini +
    # NVIDIA + Cerebras, ...) so the tier-to-order failover actually has
    # somewhere to land. See README "Enabling auto_route".
    reply = llm.chat(
        prompt=user_prompt,
        system=DECISION_SYSTEM,
        cache_system=True,
        tools=mcp_tools,
        tool_choice="auto",
        auto_route="decision",
        reasoning="off",
        temperature=0.3,
        max_tokens=2048,
    )

    tool_calls = reply.get("tool_calls") or []
    if tool_calls:
        first = tool_calls[0]
        args = first.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        return DecisionOutput(
            tool_call=ToolCall(
                id=first.get("id") or uuid.uuid4().hex[:12],
                name=first["name"],
                arguments=args,
            )
        )

    text = (reply.get("text") or "").strip()
    return DecisionOutput(answer=text or "(no answer produced)")
