# Sample outputs (reference evidence for graders)

This folder is a **snapshot of the live `state/` and `sandbox/` directories**
after running the four target queries on a clean repo. It exists so the
grader can inspect what the agent produces *without* having to re-run it
locally and without having access to the same API keys.

The live runtime directories at the project root (`Session_6/state/`,
`Session_6/sandbox/`, `Session_6/usage.json`) are gitignored — that's the
"cleanable between attempts" requirement from the class doc. The files
here are *copies*, taken once for the assignment submission.

```
docs/sample_outputs/
├── README.md                                                 ← you are here
├── state/
│   ├── memory.json                                           ← durable memory
│   └── artifacts_index.json                                  ← artifact registry
└── sandbox/
    └── reminders/
        ├── mom_birthday_2weeks_2026-05-01.txt                ← from C1 iter 2
        └── mom_birthday_2026-05-15.txt                       ← from C1 iter 3
```

## `state/memory.json` — durable memory across runs

The Memory layer's only on-disk storage. Each entry is a `MemoryItem`
(Pydantic v2; full schema in [`agent6/schemas.py`](../../agent6/schemas.py)).

**The fact that demonstrates the durable-memory contract for Query C** is the
single `kind: "fact"` row in this file. It was written by `memory.remember()`
during the *first* run of Query C1:

```json
{
  "kind": "fact",
  "text": "mom's date: 2026-05-15 (from: \"My mom's birthday is 15 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day.\")",
  "keywords": ["mom", "birthday", "2026", "05", "15", "may", "remember"],
  "value": {"entity": "mom", "date": "2026-05-15"},
  ...
}
```

The Query C2 trace in the top-level README shows Perception finding this
row via the keyword search at the start of iteration 1 (`[memory.read]   1 hits`),
and Decision answering "Mom's birthday is on May 15, 2026" without any
tool call.

The other rows in `memory.json` are `kind: "tool_outcome"` and
`kind: "artifact"` — by-products of the loop recording every tool call's
result. These are not used to answer C2; they're there for the keyword
search to find later if the same tool is run again on a related topic.

## `state/artifacts_index.json` — artifact handle registry

The map from artifact handle (`art:1`, `art:2`, ...) to metadata
(`size_bytes`, `source`, `run_id`, `preview`). The handles themselves are
monotonic integers; the actual artifact bytes live in `state/artifacts/<N>.bin`
files (not included here because they are large — the Shannon Wikipedia
fetch is ~260 KB).

Two entries are visible in this snapshot:
- `art:1` — Wikipedia page on Claude Shannon, ~260 KB (Query A's `fetch_url`)
- `art:2` — Tokyo weather forecast page or asyncio discussion thread,
  varies by query order (Query B or D's second `fetch_url`)

## `sandbox/reminders/*.txt` — files produced by C1

Two real files written by Decision via `create_file` calls in Query C1
iterations 2 and 3:

- **`mom_birthday_2weeks_2026-05-01.txt`** — the "two weeks before" reminder,
  written at iter 2 after `make_dir("reminders")` in iter 1.
- **`mom_birthday_2026-05-15.txt`** — the on-the-day reminder, written at
  iter 3.

The MCP server's `_safe()` guard ensures these paths can only be written
inside `Session_6/sandbox/` — no escape outside the project tree.

## Re-creating this folder from a clean repo

A grader who wants to verify these are reproducible should run, from
`Session_6/`:

```bash
rm -rf state/ sandbox/ usage.json
uv run python agent6/agent6.py --query A
uv run python agent6/agent6.py --query B
uv run python agent6/agent6.py --query C1
uv run python agent6/agent6.py --query C2
uv run python agent6/agent6.py --query D

# Then compare:
diff -r state/    docs/sample_outputs/state/
diff -r sandbox/  docs/sample_outputs/sandbox/
```

The `state/memory.json` will differ in `id` (UUIDs), `created_at`
(timestamps), and `run_id` (UUIDs) — but the structure and the `fact`
row's `value` field should match. The sandbox files should be byte-identical
when Decision produces the same reminder content.
