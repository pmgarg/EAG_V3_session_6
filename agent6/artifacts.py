"""
artifacts.py — file-backed artifact store for large tool payloads.

When a tool returns more than ARTIFACT_THRESHOLD_BYTES (4 KB) of text, Action
calls put() to persist the bytes and receives a short handle of the form
``art:<N>`` — where N is a monotonically increasing integer (art:1, art:2,
art:3, ...). The counter survives process restarts: on startup the next id
is computed as ``max(existing_ids) + 1`` from the index file, so Query C's
run-2 will continue from where run-1 left off.

The handle (not the bytes) is what flows through history and memory.
Perception decides whether to re-attach the bytes to a later goal via
Goal.attach_artifact_id.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Optional

ARTIFACT_THRESHOLD_BYTES = 4096

# State lives at the project root (one level above agent6/) so that
# `rm -rf state/` from Session_6/ resets the agent cleanly.
_STATE = Path(__file__).resolve().parent.parent / "state"
_ROOT = _STATE / "artifacts"
_INDEX = _STATE / "artifacts_index.json"

_counter_lock = threading.Lock()
_HANDLE_RE = re.compile(r"^art:(\d+)$")


def _ensure() -> None:
    _ROOT.mkdir(parents=True, exist_ok=True)
    if not _INDEX.exists():
        _INDEX.write_text("{}", encoding="utf-8")


def _load_index() -> dict:
    _ensure()
    try:
        return json.loads(_INDEX.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_index(idx: dict) -> None:
    _INDEX.write_text(json.dumps(idx, indent=2), encoding="utf-8")


def _next_id(idx: dict) -> int:
    """Return the next integer id, one greater than the maximum already used."""
    highest = 0
    for handle in idx:
        m = _HANDLE_RE.match(handle)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def put(text: str, *, source: str = "", run_id: str = "") -> str:
    """Persist *text* and return an ``art:<N>`` handle where N is a monotonic
    integer (1, 2, 3, ...) computed from the existing index."""
    raw = text.encode("utf-8")
    with _counter_lock:
        idx = _load_index()
        n = _next_id(idx)
        handle = f"art:{n}"
        (_ROOT / f"{n}.bin").write_bytes(raw)
        idx[handle] = {
            "id": n,
            "size_bytes": len(raw),
            "source": source,
            "run_id": run_id,
            "preview": text[:200],
        }
        _save_index(idx)
    return handle


def exists(handle: str) -> bool:
    if not handle:
        return False
    m = _HANDLE_RE.match(handle)
    if not m:
        return False
    return (_ROOT / f"{m.group(1)}.bin").exists()


def get_bytes(handle: str) -> bytes:
    m = _HANDLE_RE.match(handle)
    if not m:
        raise ValueError(f"not a valid artifact handle: {handle!r}")
    return (_ROOT / f"{m.group(1)}.bin").read_bytes()


def get_text(handle: str) -> str:
    return get_bytes(handle).decode("utf-8", errors="replace")


def info(handle: str) -> Optional[dict]:
    return _load_index().get(handle)


def descriptor(handle: str) -> str:
    """Short human-readable descriptor used in history and decision prompts."""
    meta = info(handle) or {}
    size = meta.get("size_bytes", 0)
    preview = (meta.get("preview", "") or "").replace("\n", " ")[:160]
    return f"[artifact {handle}, {size} bytes] preview: {preview}"
