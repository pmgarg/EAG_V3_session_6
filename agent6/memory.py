"""
memory.py — durable, file-backed memory service.

Public API
──────────
    remember(text, *, source, run_id) -> Optional[MemoryItem]
        Classify a user query / tool outcome through the gateway and persist
        the *memorable* parts. Returns the stored item (or None if it was
        classified as non-memorable).

    record_outcome(*, tool_call, result_text, artifact_id, run_id, goal_id)
        Persist a tool_outcome item. Always saved (no LLM classification —
        these are infrastructure facts, not user content).

    read(query, history) -> list[MemoryItem]
        Keyword-search the store for items relevant to the current iteration.
        Combines tokens from the current query and the most recent history
        events. Returns at most MEMORY_TOP_K items, newest first.

Storage
───────
A single JSON file at state/memory.json, written atomically. Items are
Pydantic-validated on read; corrupt rows are silently dropped.

LLM routing
───────────
The classifier call uses ``auto_route="memory"`` with ``provider="g"`` so it
lands on Gemini (per the Session 6 design: pin Memory to Gemini for
reliability, but still surface the auto_route label for the dashboard).
"""
from __future__ import annotations

import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from schemas import MemoryClassification, MemoryItem, ToolCall

# Add gateway client to path. The gateway lives one directory up from agent6/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GATEWAY = _PROJECT_ROOT / "llm_gatewayV3"
if str(_GATEWAY) not in sys.path:
    sys.path.insert(0, str(_GATEWAY))
from client import LLM  # noqa: E402


# Durable runtime data lives at the project root so `rm -rf state/` from
# Session_6/ cleanly resets the agent between attempts.
STATE_DIR = _PROJECT_ROOT / "state"
MEMORY_PATH = STATE_DIR / "memory.json"
MEMORY_TOP_K = 6

# Words we never use as keywords: too common to discriminate between items.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "is", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "have", "has", "had", "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their", "this", "that", "these",
    "those", "what", "when", "where", "who", "why", "how", "which", "tell",
    "give", "find", "show", "please", "me", "us", "them", "from", "into",
    "about", "as", "by", "if", "so", "than", "then", "there", "here",
}


def _ensure() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_PATH.exists():
        MEMORY_PATH.write_text("[]", encoding="utf-8")


def _tokens(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", (text or "").lower())
    return [t for t in toks if len(t) >= 2 and t not in _STOPWORDS]


def _load_all() -> list[MemoryItem]:
    _ensure()
    try:
        raw = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    items: list[MemoryItem] = []
    for row in raw:
        try:
            items.append(MemoryItem.model_validate(row))
        except ValidationError:
            continue
    return items


def _save_all(items: list[MemoryItem]) -> None:
    _ensure()
    tmp = MEMORY_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps([i.model_dump() for i in items], indent=2),
        encoding="utf-8",
    )
    tmp.replace(MEMORY_PATH)


def _append(item: MemoryItem) -> None:
    items = _load_all()
    items.append(item)
    _save_all(items)


# ──────────────────────────────────────────────────────────────────────────
# Classifier prompt — typed structured output
# ──────────────────────────────────────────────────────────────────────────

_CLASSIFIER_SYSTEM = (
    "You decide whether a single user message contains durable information "
    "worth carrying into FUTURE conversations with this user. Return a "
    "MemoryClassification.\n\n"
    "IMPORTANT: a message that mixes a DECLARATIVE FACT with a request is "
    "STILL memorable. Extract the fact and ignore the request.\n"
    "  - \"My mom's birthday is 15 May 2026. Remember that and remind me.\"\n"
    "    → is_memorable=true, kind=fact, the trailing 'remind me' does NOT\n"
    "      disqualify it. The agent stores the fact; another layer handles\n"
    "      the reminder.\n"
    "  - \"I live in Tokyo. Find me 3 things to do this weekend.\"\n"
    "    → is_memorable=true (fact: lives in Tokyo).\n\n"
    "Memorable examples (is_memorable=true):\n"
    "  - facts about the user or people they mention (\"my mom's birthday is 15 May 2026\")\n"
    "  - durable preferences (\"I prefer metric units\", \"I live in Tokyo\")\n"
    "  - durable goals/projects (\"I'm learning Rust\")\n"
    "NOT memorable (is_memorable=false):\n"
    "  - pure one-off tasks with no embedded fact (\"summarise this article\", \"compute 2+2\")\n"
    "  - questions with no embedded fact (\"when is mom's birthday?\", \"what time is it?\")\n"
    "  - research / lookup queries with no personal fact (\"find 3 things to do in Tokyo\")\n\n"
    "If memorable, set kind to one of: fact, preference. Provide a one-line "
    "summary, 2-6 lowercased keywords (entities, dates, topics — drop "
    "stopwords), and a structured value dict. For dates always normalise to "
    "ISO 8601 in the value dict, e.g. "
    "{\"entity\": \"mom\", \"event\": \"birthday\", \"date\": \"2026-05-15\"}. "
    "If not memorable, leave the other fields empty."
)


# ──────────────────────────────────────────────────────────────────────────
# Deterministic fallback — if the LLM classifier says "not memorable" but the
# message clearly contains a personal date fact (relationship keyword + a date
# the dateutil parser can read), persist it anyway. This is a defence in
# depth against TINY-tier classifier hallucinations, exactly the kind of
# safety net the class doc recommends.
# ──────────────────────────────────────────────────────────────────────────

_RELATIONSHIP_RE = re.compile(
    r"\b(mom|mum|mother|dad|father|sister|brother|wife|husband|"
    r"son|daughter|friend|partner|anniversary|birthday|wedding)\b",
    re.IGNORECASE,
)

_DATE_PATTERNS = [
    # 15 May 2026 / 15th May 2026
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
               r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
               r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
               r"dec(?:ember)?)\s+(\d{4})\b", re.IGNORECASE),
    # May 15 2026 / May 15, 2026
    re.compile(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
               r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
               r"dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
               re.IGNORECASE),
    # 2026-05-15
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
]

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def _extract_iso_date(text: str) -> Optional[str]:
    for pat in _DATE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        groups = m.groups()
        try:
            if pat is _DATE_PATTERNS[0]:        # 15 May 2026
                d, mo, y = int(groups[0]), _MONTHS[groups[1].lower()], int(groups[2])
            elif pat is _DATE_PATTERNS[1]:      # May 15 2026
                mo, d, y = _MONTHS[groups[0].lower()], int(groups[1]), int(groups[2])
            else:                                # 2026-05-15
                y, mo, d = int(groups[0]), int(groups[1]), int(groups[2])
            return f"{y:04d}-{mo:02d}-{d:02d}"
        except (KeyError, ValueError):
            continue
    return None


def _heuristic_fact(text: str) -> Optional[MemoryClassification]:
    """If the message contains a relationship word AND a parseable date,
    treat it as a fact even if the LLM classifier disagreed."""
    rel_match = _RELATIONSHIP_RE.search(text)
    iso = _extract_iso_date(text)
    if not (rel_match and iso):
        return None
    entity = rel_match.group(0).lower()
    return MemoryClassification(
        is_memorable=True,
        kind="fact",
        summary=f"{entity}'s date: {iso} (from: {text[:120]!r})",
        keywords=[entity, "birthday" if "birthday" in text.lower() else "date",
                  iso[:4], iso[5:7]],
        value={"entity": entity, "date": iso},
    )


def _classify(text: str) -> MemoryClassification:
    llm = LLM()
    schema = MemoryClassification.model_json_schema()
    reply = llm.chat(
        prompt=f"User message:\n{text}\n\nReturn a MemoryClassification.",
        system=_CLASSIFIER_SYSTEM,
        cache_system=True,
        response_format={
            "type": "json_schema",
            "schema": schema,
            "name": "MemoryClassification",
            "strict": True,
        },
        provider="g",                 # pin to Gemini for reliability
        auto_route="memory",
        temperature=0.3,
        max_tokens=512,
    )
    parsed = reply.get("parsed")
    if parsed:
        try:
            return MemoryClassification.model_validate(parsed)
        except ValidationError:
            pass
    # Fallback: not memorable.
    return MemoryClassification(is_memorable=False)


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def remember(text: str, *, source: str, run_id: str) -> Optional[MemoryItem]:
    """Classify *text* and persist it if memorable. Returns the stored item."""
    if not text or not text.strip():
        return None
    try:
        cls = _classify(text)
    except Exception:
        cls = MemoryClassification(is_memorable=False)
    # Deterministic fallback: rescue personal-date facts the LLM mis-classified.
    if not cls.is_memorable:
        forced = _heuristic_fact(text)
        if forced is not None:
            cls = forced
    if not cls.is_memorable:
        return None
    kws = [k.lower() for k in (cls.keywords or [])]
    # Always include a few raw tokens from the text as a safety net.
    kws = list(dict.fromkeys(kws + _tokens(text)[:6]))
    item = MemoryItem(
        id=uuid.uuid4().hex[:12],
        kind=cls.kind or "fact",
        text=cls.summary or text[:200],
        keywords=kws,
        value=cls.value or {},
        run_id=run_id,
        created_at=time.time(),
    )
    _append(item)
    return item


def record_outcome(
    *,
    tool_call: ToolCall,
    result_text: str,
    artifact_id: Optional[str],
    run_id: str,
    goal_id: str,
) -> MemoryItem:
    """Persist a tool_outcome (or artifact pointer) without LLM classification."""
    summary = result_text[:300].replace("\n", " ")
    kws = list(dict.fromkeys(_tokens(tool_call.name) + _tokens(summary)))[:8]
    kind = "artifact" if artifact_id else "tool_outcome"
    item = MemoryItem(
        id=uuid.uuid4().hex[:12],
        kind=kind,
        text=f"{tool_call.name} -> {summary}",
        keywords=kws,
        value={"tool": tool_call.name, "arguments": tool_call.arguments},
        run_id=run_id,
        created_at=time.time(),
        artifact_id=artifact_id,
    )
    _append(item)
    return item


def read(query: str, history: list[dict[str, Any]]) -> list[MemoryItem]:
    """Keyword search across all memory rows.

    Score = number of distinct query tokens that hit an item's keyword list.
    Within a tie, newer items rank first. The top MEMORY_TOP_K are returned.
    """
    items = _load_all()
    if not items:
        return []
    # Build the search bag: current query + last few history events.
    bag = _tokens(query)
    for ev in history[-3:]:
        for field in ("text", "tool", "result_descriptor"):
            v = ev.get(field) if isinstance(ev, dict) else None
            if isinstance(v, str):
                bag.extend(_tokens(v))
    bag_set = set(bag)
    if not bag_set:
        return []

    def score(item: MemoryItem) -> tuple[int, float]:
        hits = sum(1 for k in item.keywords if k in bag_set)
        return (hits, item.created_at)

    ranked = sorted(items, key=score, reverse=True)
    out = [it for it in ranked if score(it)[0] > 0]
    return out[:MEMORY_TOP_K]
