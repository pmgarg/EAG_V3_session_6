# EAG V3 — Session 6 Agent (`agent6.py`)

A four-role agent loop (**Memory · Perception · Decision · Action**) that runs
on top of `llm_gatewayV3` and a local MCP tool server. Every LLM call goes
through the gateway. Every layer boundary is typed by Pydantic v2.

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│   Memory   │ →  │ Perception │ →  │  Decision  │ →  │   Action   │
│ keyword    │    │ goal       │    │ answer OR  │    │  MCP       │
│ recall +   │    │ decomp.    │    │ one tool   │    │  dispatch  │
│ durable    │    │ (Gemini)   │    │ (router)   │    │  + art:    │
│ classifier │    │            │    │            │    │  store     │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
       ↑                                                     │
       └─────────────────── tool outcomes / artifacts ◄──────┘
```

## Layout

```
Session_6/
├── agent6/                       ← the agent code (the assignment)
│   ├── agent6.py                 ← orchestrator (the loop)
│   ├── memory.py                 ← durable JSON memory + LLM classifier
│   ├── perception.py             ← goal decomposition (structured output)
│   ├── decision.py               ← answer-or-tool, one LLM call
│   ├── action.py                 ← pure MCP dispatch + art: guard
│   ├── artifacts.py              ← file-backed store for >4 KB payloads
│   ├── schemas.py                ← Pydantic v2 contracts
│   ├── mcp_server.py             ← 10-tool MCP server (stdio)
│   └── __init__.py
├── llm_gatewayV3/                ← the gateway substrate (port 8101)
├── .env                          ← API keys, gitignored
├── README.md
├── pyproject.toml                ← uv-managed deps
└── state/, sandbox/, usage.json  ← runtime state (gitignored, cleanable)
```

| Module | Role | LLM? |
| --- | --- | --- |
| `agent6/schemas.py`     | Pydantic v2 contracts for every boundary             | — |
| `agent6/artifacts.py`   | File-backed store for >4 KB tool payloads (`art:N`)  | — |
| `agent6/memory.py`      | Durable JSON memory + LLM classifier                 | yes (Gemini, `auto_route="memory"`) |
| `agent6/perception.py`  | Decompose query into stable goals (`g1`, `g2`, ...)  | yes (Gemini, `auto_route="perception"`) |
| `agent6/decision.py`    | One LLM call → answer OR one `ToolCall`              | yes (`auto_route="decision"`, router picks tier) |
| `agent6/action.py`      | Pure MCP dispatch + `art:` guard + 4 KB threshold    | no |
| `agent6/agent6.py`      | The orchestrator loop                                | — |
| `agent6/mcp_server.py`  | 10-tool MCP server (stdio)                           | — |
| `llm_gatewayV3/`        | The substrate (port 8101)                            | — |

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency
management. No manual `venv activate` step is required.

```bash
cd Session_6

# Python deps for the agent + MCP server
uv sync                                    # reads pyproject.toml

# .env at Session_6/.env must contain:
#   GEMINI_API_KEY=<your gemini key>       # gateway expects this exact name
#   NVIDIA_API_KEY=<your nvidia key>       # router pool + optional worker
#   TAVILY_API_KEY=<optional>              # web_search primary; DDG fallback
#
# NOTE: The gateway code in llm_gatewayV3/providers.py reads GEMINI_API_KEY
#       (not GOOGLE_API_KEY). If you already have a Google AI Studio key
#       under GOOGLE_API_KEY, just add a second line:
#           GEMINI_API_KEY=$GOOGLE_API_KEY   # or paste the value directly

# Start the gateway (separate terminal — keep it running)
cd llm_gatewayV3 && ./run.sh
# Dashboard:  http://localhost:8101
```

## Running the four target queries

```bash
# Query A — Shannon Wikipedia (artifact attach test)
uv run python agent6/agent6.py --query A

# Query B — Tokyo activities + Saturday weather
uv run python agent6/agent6.py --query B

# Query C — Mom's birthday (durable memory, TWO RUNS)
uv run python agent6/agent6.py --query C1     # records the fact + creates reminders
uv run python agent6/agent6.py --query C2     # answers from state/memory.json

# Query D — Python asyncio best practices (multi-source synthesis)
uv run python agent6/agent6.py --query D

# Or pass any query as a single argument:
uv run python agent6/agent6.py "What time is it in Tokyo right now?"
```

## Resetting state between attempts

```bash
rm -rf state/ sandbox/ usage.json llm_gatewayV3/gateway_v3.db
```

(The `state/` directory holds `memory.json` and the artifact store;
`sandbox/` is the MCP file-tool sandbox.)

## The Decision contract

`decision.py` makes ONE LLM call per goal with `tool_choice="auto"`. It
returns a `DecisionOutput` containing **either** an `answer` string **or**
exactly one `ToolCall`. Never both. The system prompt enforces three rules:

1. **Pick one or the other** — answer in plain text, or call exactly one
   tool. Never narrate.
2. **`art:` strings are internal** — Decision must never pass them to
   `read_file`, `fetch_url`, etc. The bytes are pasted under
   `ATTACHED ARTIFACTS:` when needed. (Action enforces this at dispatch
   time too.)
3. **Substantive answers** — when the goal is an extraction / list /
   comparison / selection, the answer must be ≥3 sentences or a real list.
   No meta-answers like "the page has been fetched, how would you like to
   proceed?".

The full Decision system prompt and the structured-output schema for
Perception are in `decision.py` and `perception.py` respectively.

## The four "honest gateway answers" to expect

| Field on response          | Meaning when `false` / `0`                                                                                  |
| -------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `reasoning_applied: false` | The worker silently ignored the `reasoning="medium"` knob — common on Gemini free tier.                     |
| `cache_read_input_tokens: 0` & `cache_creation_input_tokens: 0` | Cache API quota is 0 on Gemini free tier; gateway falls back to no-cache.                                   |
| `router_decision.fallback_used: true`                          | Router pool was exhausted; tier picked deterministically by token-count rule. The call still succeeded.     |
| Perception temperature = 1.0                                   | Gemini 3.x flash-lite loops at temperature=0 on structured-output requests. Raising T to 1.0 fixes it.      |

The dashboard at `http://localhost:8101` shows all of the above per-call.

---

## Expected terminal output

The traces below are what a clean run should look like. Paste the actual
output from **your** machine into this section once you've run each query.
(Iteration counts must be within 2× of the documented number, per the
assignment.)

### Query A — Shannon Wikipedia

Expected iteration count: **3**

```
(eagv3-session6-agent) (base) prateekgarg@Prateeks-MacBook-Pro Session_6 % uv run python agent6/agent6.py --query A
══════════════════════════════════════════════════════════════════════════════
agent6.py — Session 6 four-role agent loop
run_id   : 58bc2c30
query    : Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory.
══════════════════════════════════════════════════════════════════════════════
[memory.remember]  query not classified as memorable
[05/19/26 05:25:02] INFO     Processing request of type ListToolsRequest                                                                                                                                                                                                                                                                                                   server.py:727
[mcp] tools loaded: ['web_search', 'fetch_url', 'get_time', 'currency_convert', 'read_file', 'list_dir', 'make_dir', 'create_file', 'update_file', 'edit_file']

─── iter 1 ────────────────────────────────────────────────────────────────
[memory.read]   0 hits
[perception]    [open] g1 Fetch the Wikipedia page for Claude Shannon
[perception]    [open] g2 Extract birth date, death date, and three key contributions to information theory from the Wikipedia page
[perception]    notes: Starting the process of fetching the information and extracting the requested details.
[decision]      TOOL_CALL: fetch_url({"url": "https://en.wikipedia.org/wiki/Claude_Shannon"})
[05/19/26 05:25:17] INFO     Processing request of type CallToolRequest                                                                                                                                                                                                                                                                                                    server.py:727
[INIT].... → Crawl4AI 0.8.6 
[FETCH]... ↓ https://en.wikipedia.org/wiki/Claude_Shannon                                                         | ✓ | ⏱: 1.71s 
[SCRAPE].. ◆ https://en.wikipedia.org/wiki/Claude_Shannon                                                         | ✓ | ⏱: 0.16s 
[COMPLETE] ● https://en.wikipedia.org/wiki/Claude_Shannon                                                         | ✓ | ⏱: 1.88s 
[action]        -> [artifact art:1, 260080 bytes] preview: {   "status": 200,   "content_type": "text/markdown",   "length_bytes": 254229,   "text": "[Jump to content](https://en.wikipedia.org/wiki/Claude_Shannon#bodyCo

─── iter 2 ────────────────────────────────────────────────────────────────
[memory.read]   1 hits
[perception]    [done] g1 Fetch the Wikipedia page for Claude Shannon  attach=art:1
[perception]    [open] g2 Extract birth date, death date, and three key contributions to information theory from the Wikipedia page  attach=art:1
[perception]    notes: The page has been successfully fetched and is ready for extraction.
[attach]        art:1 (260080 bytes)
[decision]      ANSWER: Claude Shannon's birth and death dates are as follows:
*   **Birth Date:** April 30, 1916
*   **Death Date:** February 24, 2001

Three key contributions to information theory include:
1.  **Founding the field:** He is widely recognized as t ...

─── iter 3 ────────────────────────────────────────────────────────────────
[memory.read]   1 hits
[perception]    [done] g1 Fetch the Wikipedia page for Claude Shannon  attach=art:1
[perception]    [done] g2 Extract birth date, death date, and three key contributions to information theory from the Wikipedia page
[perception]    notes: All requested information has been successfully extracted and provided.

[done] all goals satisfied

══════════════════════════════════════════════════════════════════════════════
FINAL ANSWER:
Claude Shannon's birth and death dates are as follows:
*   **Birth Date:** April 30, 1916
*   **Death Date:** February 24, 2001

Three key contributions to information theory include:
1.  **Founding the field:** He is widely recognized as the "father of information theory" for establishing the mathematical foundations of how information is measured and transmitted.
2.  **Introduction of Entropy:** He introduced the concept of entropy in information theory, which provides a measure of the uncertainty or information content associated with a random variable.
3.  **The Mathematical Theory of Communication:** He authored this seminal work, which defined the fundamental limits on signal processing and data communication, including the concept of channel capacity (the Shannon limit).
══════════════════════════════════════════════════════════════════════════════
(eagv3-session6-agent) (base) prateekgarg@Prateeks-MacBook-Pro Session_6 % uv run python agent6/agent6.py --query B
══════════════════════════════════════════════════════════════════════════════
agent6.py — Session 6 four-role agent loop
run_id   : 3ac72ae9
query    : Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate.
══════════════════════════════════════════════════════════════════════════════
[memory.remember]  query not classified as memorable
[05/19/26 05:27:39] INFO     Processing request of type ListToolsRequest                                                                                                                                                                                                                                                                                                   server.py:727
[mcp] tools loaded: ['web_search', 'fetch_url', 'get_time', 'currency_convert', 'read_file', 'list_dir', 'make_dir', 'create_file', 'update_file', 'edit_file']

─── iter 1 ────────────────────────────────────────────────────────────────
[memory.read]   0 hits
[perception]    [open] g1 Search for 3 family-friendly activities in Tokyo suitable for this weekend
[perception]    [open] g2 Check the weather forecast for Tokyo this Saturday
[perception]    [open] g3 Synthesize the activity list with the weather information to recommend the most appropriate option
[perception]    notes: Starting the process of identifying Tokyo activities and checking weather to make a recommendation.
[decision]      TOOL_CALL: web_search({"max_results": 5, "query": "family-friendly activities in Tokyo this weekend"})
[05/19/26 05:27:59] INFO     Processing request of type CallToolRequest                                                                                                                                                                                                                                                                                                    server.py:727
[05/19/26 05:28:00] INFO     response: https://grokipedia.com/api/typeahead?query=family-friendly+activities+in+Tokyo+this+weekend&limit=1 200                                                                                                                                                                                                                                lib.rs:495
                    INFO     response: https://en.wikipedia.org/w/api.php?action=opensearch&profile=fuzzy&limit=1&search=family-friendly%20activities%20in%20Tokyo%20this%20weekend 200                                                                                                                                                                                       lib.rs:495
[05/19/26 05:28:03] INFO     response: https://www.mojeek.com/search?q=family-friendly+activities+in+Tokyo+this+weekend 200                                                                                                                                                                                                                                                   lib.rs:495
[action]        -> {
  "title": "Fun Things To Do in Tokyo With Kids: 25 Family-Friendly",
  "url": "https://tokyoparents.com/fun-things-to-do-in-tokyo-with-kids-25-family-friendly-activities-indoor-outdoor/",
  "snippe ...

─── iter 2 ────────────────────────────────────────────────────────────────
[memory.read]   1 hits
[perception]    [done] g1 Search for 3 family-friendly activities in Tokyo suitable for this weekend
[perception]    [open] g2 Check the weather forecast for Tokyo this Saturday
[perception]    [open] g3 Synthesize the activity list with the weather information to recommend the most appropriate option
[perception]    notes: The search for activities is complete, now proceeding to check the weather forecast for Saturday.
[decision]      TOOL_CALL: web_search({"max_results": 5, "query": "Tokyo weather forecast Saturday May 23 2026"})
[05/19/26 05:28:17] INFO     Processing request of type CallToolRequest                                                                                                                                                                                                                                                                                                    server.py:727
[05/19/26 05:28:18] INFO     response: https://en.wikipedia.org/w/api.php?action=opensearch&profile=fuzzy&limit=1&search=Tokyo%20weather%20forecast%20Saturday%20May%2023%202026 200                                                                                                                                                                                          lib.rs:495
                    INFO     response: https://grokipedia.com/api/typeahead?query=Tokyo+weather+forecast+Saturday+May+23+2026&limit=1 200                                                                                                                                                                                                                                     lib.rs:495
[05/19/26 05:28:19] INFO     response: https://search.brave.com/search?q=Tokyo+weather+forecast+Saturday+May+23+2026&source=web 200                                                                                                                                                                                                                                           lib.rs:495
[action]        -> {
  "title": "Tokyo, Tokyo, Japan Monthly Weather | AccuWeather",
  "url": "https://www.accuweather.com/en/jp/tokyo/226396/may-weather/226396",
  "snippet": "Lo · Actual Hi · Actual Lo · Forecast Hi · ...

─── iter 3 ────────────────────────────────────────────────────────────────
[memory.read]   2 hits
[perception]    [done] g1 Search for 3 family-friendly activities in Tokyo suitable for this weekend
[perception]    [done] g2 Check the weather forecast for Tokyo this Saturday
[perception]    [open] g3 Synthesize the activity list with the weather information to recommend the most appropriate option
[perception]    notes: Both research tasks are complete; proceeding to final synthesis.
[decision]      ANSWER: Based on the weather forecast for Tokyo in late May 2026, which indicates pleasant, warm temperatures ranging from 71°F to 86°F, you have excellent flexibility for both indoor and outdoor activities.

Given these mild and comfortable condit ...

─── iter 4 ────────────────────────────────────────────────────────────────
[memory.read]   3 hits
[perception]    [done] g1 Search for 3 family-friendly activities in Tokyo suitable for this weekend
[perception]    [done] g2 Check the weather forecast for Tokyo this Saturday
[perception]    [done] g3 Synthesize the activity list with the weather information to recommend the most appropriate option
[perception]    notes: All goals have been completed and the user request has been fulfilled.

[done] all goals satisfied

══════════════════════════════════════════════════════════════════════════════
FINAL ANSWER:
Based on the weather forecast for Tokyo in late May 2026, which indicates pleasant, warm temperatures ranging from 71°F to 86°F, you have excellent flexibility for both indoor and outdoor activities.

Given these mild and comfortable conditions, I recommend prioritizing outdoor activities that take advantage of the spring weather, such as visiting **Ueno Park** or the **Shinjuku Gyoen National Garden**. These locations offer expansive green spaces perfect for a family picnic or a leisurely walk. If the temperature leans toward the higher end of the forecast (near 86°F), you might consider balancing your day with indoor options like the **teamLab Borderless** digital art museum or the **Sunshine Aquarium**, which provide climate-controlled environments to keep everyone comfortable during the peak heat of the afternoon.
══════════════════════════════════════════════════════════════════════════════
(eagv3-session6-agent) (base) prateekgarg@Prateeks-MacBook-Pro Session_6 % uv run python agent6/agent6.py --query C1
══════════════════════════════════════════════════════════════════════════════
agent6.py — Session 6 four-role agent loop
run_id   : 3b7d8372
query    : My mom's birthday is 15 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day.
══════════════════════════════════════════════════════════════════════════════
[memory.remember]  saved fact: 'mom\'s date: 2026-05-15 (from: "My mom\'s birthday is 15 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day.")'
[05/19/26 05:30:16] INFO     Processing request of type ListToolsRequest                                                                                                                                                                                                                                                                                                   server.py:727
[mcp] tools loaded: ['web_search', 'fetch_url', 'get_time', 'currency_convert', 'read_file', 'list_dir', 'make_dir', 'create_file', 'update_file', 'edit_file']

─── iter 1 ────────────────────────────────────────────────────────────────
[memory.read]   1 hits
[perception]    [open] g1 Create a calendar reminder for 1 May 2026 (two weeks before mom's birthday)
[perception]    [open] g2 Create a calendar reminder for 15 May 2026 (mom's birthday)
[perception]    notes: The memory store already holds the birthday; scheduling the two required reminder events.
[decision]      TOOL_CALL: make_dir({"path": "reminders"})
[05/19/26 05:30:34] INFO     Processing request of type CallToolRequest                                                                                                                                                                                                                                                                                                    server.py:727
[action]        -> {
  "ok": true,
  "path": "reminders",
  "created": true
}

─── iter 2 ────────────────────────────────────────────────────────────────
[memory.read]   2 hits
[perception]    [open] g1 Create a calendar reminder file for 1 May 2026
[perception]    [open] g2 Create a calendar reminder file for 15 May 2026
[perception]    notes: The directory for reminders was created, now I need to generate the individual reminder files.
[decision]      TOOL_CALL: create_file({"path": "reminders/reminder_2026_05_01.txt", "content": "Reminder: 1 May 2026 - Calendar event."})
[05/19/26 05:31:11] INFO     Processing request of type CallToolRequest                                                                                                                                                                                                                                                                                                    server.py:727
[action]        -> {
  "ok": true,
  "path": "reminders/reminder_2026_05_01.txt",
  "size_bytes": 38
}

─── iter 3 ────────────────────────────────────────────────────────────────
[memory.read]   3 hits
[perception]    [done] g1 Create a calendar reminder file for 1 May 2026
[perception]    [open] g2 Create a calendar reminder file for 15 May 2026
[perception]    notes: The reminder for 1 May 2026 has been successfully created, and the reminder for 15 May 2026 remains to be created.
[decision]      TOOL_CALL: create_file({"content": "Reminder: 15 May 2026 - Mom's birthday.", "path": "reminders/reminder_2026_05_15.txt"})
[05/19/26 05:31:31] INFO     Processing request of type CallToolRequest                                                                                                                                                                                                                                                                                                    server.py:727
[action]        -> {
  "ok": true,
  "path": "reminders/reminder_2026_05_15.txt",
  "size_bytes": 39
}

─── iter 4 ────────────────────────────────────────────────────────────────
[memory.read]   4 hits
[perception]    [done] g1 Create a calendar reminder file for 1 May 2026
[perception]    [done] g2 Create a calendar reminder file for 15 May 2026
[perception]    notes: All requested calendar reminders have been successfully created.

[done] all goals satisfied

══════════════════════════════════════════════════════════════════════════════
FINAL ANSWER:
(no final answer produced)
══════════════════════════════════════════════════════════════════════════════
(eagv3-session6-agent) (base) prateekgarg@Prateeks-MacBook-Pro Session_6 % uv run python agent6/agent6.py --query C2
══════════════════════════════════════════════════════════════════════════════
agent6.py — Session 6 four-role agent loop
run_id   : e7c8de8f
query    : When is mom's birthday?
══════════════════════════════════════════════════════════════════════════════
[memory.remember]  query not classified as memorable
[05/19/26 05:43:09] INFO     Processing request of type ListToolsRequest                                                                                                                                                                                                                                                                                                   server.py:727
[mcp] tools loaded: ['web_search', 'fetch_url', 'get_time', 'currency_convert', 'read_file', 'list_dir', 'make_dir', 'create_file', 'update_file', 'edit_file']

─── iter 1 ────────────────────────────────────────────────────────────────
[memory.read]   1 hits
[perception]    [open] g1 Answer when mom's birthday is, using memory hits as the source of truth
[perception]    notes: The birthday is stored in memory and can be retrieved directly.
[decision]      ANSWER: Mom's birthday is on May 15, 2026.

─── iter 2 ────────────────────────────────────────────────────────────────
[memory.read]   1 hits
[perception]    [done] g1 Answer when mom's birthday is, using memory hits as the source of truth
[perception]    notes: The query has been fully answered using the information retrieved from memory.

[done] all goals satisfied

══════════════════════════════════════════════════════════════════════════════
FINAL ANSWER:
Mom's birthday is on May 15, 2026.
══════════════════════════════════════════════════════════════════════════════
(eagv3-session6-agent) (base) prateekgarg@Prateeks-MacBook-Pro Session_6 % uv run python agent6/agent6.py --query D 
══════════════════════════════════════════════════════════════════════════════
agent6.py — Session 6 four-role agent loop
run_id   : 34efff3e
query    : Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on.
══════════════════════════════════════════════════════════════════════════════
[memory.remember]  query not classified as memorable
[05/19/26 05:47:17] INFO     Processing request of type ListToolsRequest                                                                                                                                                                                                                                                                                                   server.py:727
[mcp] tools loaded: ['web_search', 'fetch_url', 'get_time', 'currency_convert', 'read_file', 'list_dir', 'make_dir', 'create_file', 'update_file', 'edit_file']

─── iter 1 ────────────────────────────────────────────────────────────────
[memory.read]   0 hits
[perception]    [open] g1 Search for 'Python asyncio best practices' and obtain summaries of the top results via search snippets
[perception]    [open] g2 Synthesize a numbered list of advice the top sources agree on, using the search results
[perception]    notes: Starting search to identify best practices for Python asyncio.
[decision]      TOOL_CALL: web_search({"max_results": 5, "query": "Python asyncio best practices"})
[05/19/26 05:47:28] INFO     Processing request of type CallToolRequest                                                                                                                                                                                                                                                                                                    server.py:727
                    INFO     response: https://en.wikipedia.org/w/api.php?action=opensearch&profile=fuzzy&limit=1&search=Python%20asyncio%20best%20practices 200                                                                                                                                                                                                              lib.rs:495
[05/19/26 05:47:29] INFO     response: https://grokipedia.com/api/typeahead?query=Python+asyncio+best+practices&limit=1 200                                                                                                                                                                                                                                                   lib.rs:495
[05/19/26 05:47:33] INFO     response: https://www.mojeek.com/search?q=Python+asyncio+best+practices 200                                                                                                                                                                                                                                                                      lib.rs:495
[action]        -> {
  "title": "Asyncio best practices - Async-SIG - Discussions on Python.org",
  "url": "https://discuss.python.org/t/asyncio-best-practices/12576",
  "snippet": "So I thought it’d be great to catalog ...

─── iter 2 ────────────────────────────────────────────────────────────────
[memory.read]   3 hits
[perception]    [done] g1 Search for 'Python asyncio best practices' and obtain summaries of the top results via search snippets
[perception]    [open] g2 Synthesize a numbered list of advice the top sources agree on, using the search results
[perception]    notes: The search was successful; proceeding to synthesize the results into a numbered list of best practices.
[decision]      TOOL_CALL: fetch_url({"url": "https://discuss.python.org/t/asyncio-best-practices/12576"})
[05/19/26 05:47:51] INFO     Processing request of type CallToolRequest                                                                                                                                                                                                                                                                                                    server.py:727
[INIT].... → Crawl4AI 0.8.6 
[FETCH]... ↓ https://discuss.python.org/t/asyncio-best-practices/12576                                            | ✓ | ⏱: 2.58s 
[SCRAPE].. ◆ https://discuss.python.org/t/asyncio-best-practices/12576                                            | ✓ | ⏱: 0.02s 
[COMPLETE] ● https://discuss.python.org/t/asyncio-best-practices/12576                                            | ✓ | ⏱: 2.62s 
[action]        -> [artifact art:2, 6605 bytes] preview: {   "status": 200,   "content_type": "text/markdown",   "length_bytes": 6418,   "text": "[ Skip to main content ](https://discuss.python.org/t/asyncio-best-prac

─── iter 3 ────────────────────────────────────────────────────────────────
[memory.read]   5 hits
[perception]    [done] g1 Search for 'Python asyncio best practices' and obtain summaries of the top results via search snippets
[perception]    [open] g2 Synthesize a numbered list of advice the top sources agree on, using the search results  attach=art:2
[perception]    notes: I have successfully fetched the first result and am now ready to synthesize the advice into a numbered list.
[attach]        art:2 (6605 bytes)
[decision]      ANSWER: Based on the provided discussion from the Python community, here are the key best practices for writing asynchronous code in Python:

1.  **Understand `async/await` execution:** Simply using `async` and `await` keywords does not automatical ...

─── iter 4 ────────────────────────────────────────────────────────────────
[memory.read]   5 hits
[perception]    [done] g1 Search for 'Python asyncio best practices' and obtain summaries of the top results via search snippets
[perception]    [done] g2 Synthesize a numbered list of advice the top sources agree on, using the search results
[perception]    notes: All requested goals have been successfully completed.

[done] all goals satisfied

══════════════════════════════════════════════════════════════════════════════
FINAL ANSWER:
Based on the provided discussion from the Python community, here are the key best practices for writing asynchronous code in Python:

1.  **Understand `async/await` execution:** Simply using `async` and `await` keywords does not automatically make code asynchronous. If you await coroutines sequentially (e.g., `await task1(); await task2()`), they will still execute in a synchronous, top-to-bottom order. You must actively use the `asyncio` library to schedule tasks concurrently.
2.  **Avoid blocking the event loop with long-running loops:** Functions containing long-running loops can halt the entire event loop, causing significant performance lag. Instead of blocking, you should schedule iterations of your loop onto the event loop using mechanisms like `loop.call_soon` to allow other tasks to run in the meantime.
3.  **Utilize `asyncio.Task` for concurrency:** Use tasks to track and manage concurrent operations. Creating tasks is the standard way to schedule functions on the event loop, allowing the loop to track their progress and execute them concurrently rather than waiting for each one to finish before starting the next.
══════════════════════════════════════════════════════════════════════════════
```

### Query B — Tokyo activities

Expected iteration count: **~6**

```
TODO: paste your terminal output here
```

### Query C — Mom's birthday

Expected iteration count: **4** (Run 1) + **2** (Run 2)

```
TODO: paste your terminal output here for BOTH runs
```

### Query D — Python asyncio

Expected iteration count: **5–7**

```
TODO: paste your terminal output here
```

## Demo video

YouTube link: _TODO_

---

## Security note

A real API key was previously committed to `.env` in this repo. **Rotate the
Google AI Studio and NVIDIA NIM keys** before publishing. `.env` is now in
`.gitignore`.
