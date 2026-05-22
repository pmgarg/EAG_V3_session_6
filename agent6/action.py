"""
action.py — pure MCP dispatch. Zero LLM calls.

Responsibilities (all of them):

1. Refuse to dispatch a tool call whose ``path`` / ``url`` / ``file`` /
   ``source`` argument starts with ``art:``. Artifact handles are internal
   and must not be passed to MCP tools — they are not paths and not URLs.
   This guard catches the TINY-tier hallucination of "fetch this artifact".

2. Await ``session.call_tool(name, arguments=...)``, collapse the result's
   content blocks into a single text string.

3. If the collapsed text is larger than ARTIFACT_THRESHOLD_BYTES, persist it
   via the artifact store and return a short descriptor of the form
   ``[artifact art:..., NNN bytes] preview: ...`` plus the new handle.
   Otherwise return the text directly and ``None`` for the handle.

Public API
──────────
    async execute(session, tool_call) -> tuple[str, str | None]
"""
from __future__ import annotations

from typing import Optional

from mcp import ClientSession

import artifacts
from schemas import ToolCall


_ARTIFACT_HANDLE_ARG_KEYS = ("path", "url", "file", "source", "src", "uri")


def _refuse_artifact_handle(tool_call: ToolCall) -> Optional[str]:
    for k in _ARTIFACT_HANDLE_ARG_KEYS:
        v = tool_call.arguments.get(k)
        if isinstance(v, str) and v.startswith("art:"):
            return (
                f"ERROR: argument {k}={v!r} is an internal artifact handle, "
                "not a real path or URL. Artifact bytes are pasted into the "
                "prompt under ATTACHED ARTIFACTS — read them there. Pick a "
                "different action."
            )
    return None


def _collapse_result(result) -> str:
    """Flatten MCP CallToolResult.content blocks into one string."""
    if not getattr(result, "content", None):
        return ""
    chunks: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text is not None:
            chunks.append(text)
            continue
        # Tool may also return structured data on some blocks.
        data = getattr(block, "data", None)
        if data is not None:
            chunks.append(str(data))
    return "\n".join(chunks)


async def execute(
    session: ClientSession,
    tool_call: ToolCall,
    *,
    run_id: str = "",
) -> tuple[str, Optional[str]]:
    """Dispatch one tool call. Return ``(descriptor, artifact_id_or_None)``."""
    refusal = _refuse_artifact_handle(tool_call)
    if refusal is not None:
        return refusal, None

    try:
        result = await session.call_tool(tool_call.name, arguments=tool_call.arguments)
    except Exception as e:
        return f"ERROR: MCP dispatch failed for {tool_call.name}: {e}", None

    text = _collapse_result(result)
    if not text:
        return "(empty tool result)", None

    if len(text.encode("utf-8")) >= artifacts.ARTIFACT_THRESHOLD_BYTES:
        handle = artifacts.put(text, source=tool_call.name, run_id=run_id)
        return artifacts.descriptor(handle), handle

    return text, None
