# Web search (Exa Snapshot)

Setting `web_snapshot_as_of` gives the agent two more tools backed by
[Exa Snapshot](https://exa.ai/docs/search/snapshot). Every page they return is the version Exa
stored at or before that date:

```sh
EXA_API_KEY=... harbor run ... --agent openworld_harness:OpenWorldHarness \
  --ak web_snapshot_as_of=2026-06-01
```

The model is never told the date. No tool description, result, error or note names it.
Exa's error text (which can mention the snapshot) goes only to the logs. Published dates are
hidden unless `web_show_dates=true`, because they cluster just below the cutoff.

The search loop is modeled as a Context Gathering Decision Process
([Kausik et al., arXiv:2605.07042](https://arxiv.org/abs/2605.07042)). The harness, not the
model, keeps the belief state and decides when a search line has gone stale:

| CGDP | Here |
|---|---|
| Hidden state c | The web as Exa stored it at `web_snapshot_as_of` |
| Actions | `web_search(query, goal)`, `web_fetch(url, offset)`, plus the harness's own tools |
| Observation function F | Exa `/search` (highlights) and `/contents`, both with `snapshotAsOf` |
| Per-action cost | `web_max_searches` and `web_max_fetches`. Each result shows the remaining budget. |
| Belief state b_t | Research notes. After each observation that shows something new (a search with new results, a page window not read before), one extractor call (the agent's own model) reasons about it, then adds new freeform notes with their sources after a `NOTES:` line. Past `web_notes_items + 4` notes, a second call curates them back to `web_notes_items`. The notes come back with a web result whenever they have changed (and with every halt), and are added to the compaction handoff. |
| Exhaustion gate | A search is stagnant when its token Jaccard against one of the last 3 queries is at least `web_gate_jaccard` and at most `web_gate_novelty` of its results are new pages. After `web_gate_patience` stagnant searches in a row, the agent gets the notes instead of the stale results, which stay unseen and out of the notes. The gate keeps halting until a search that is not stagnant ends the line, so re-sending a halted query does not reveal it. |
| Terminal action | `submit`. Search is a sub-loop, so the gate ends only the stagnant search line, not the run. |

Results the agent has already seen are marked `(seen)` and listed without excerpts. The
belief state and the gate follow the paper's choices: freeform notes, which beat structured
JSON in 10 of its 12 settings, with K=6 (reorganizing past 10), and the programmatic gate's
best configuration J≥0.6, UPR≤0.3, p=2, which beat both LLM-judged gates.

| Option | Default | |
|---|---|---|
| `web_snapshot_as_of` | unset (no web tools) | Turns the web tools on; the other options need it. ISO 8601 date or date-time (UTC), not in the future. Exa documents a rolling window of about five months; the harness passes older dates through rather than enforcing it. |
| `web_max_searches`, `web_max_fetches` | 50, 100 | Exa requests per trial. Repeating an identical query, or reading further into a fetched page, makes no request. |
| `web_fetch_chars` | 10000 | Page text per `web_fetch`. `offset` reads further. |
| `web_show_dates` | false | Show published dates. |
| `web_notes` | true | Turn the belief state on or off. |
| `web_notes_items` | 6 | Notes kept after a reorganization. |
| `web_extractor_reasoning_effort` | provider default | Reasoning effort for the extractor calls. The `reasoning_effort`, `reasoning`, `thinking` and `output_config` keys in `llm_call_kwargs` (also inside `extra_body`) do not apply to them. Other provider-specific thinking settings, such as Gemini's `thinking_config`, still do. |
| `web_gate` | true | When off, the gate still records where it would have fired. |
| `web_gate_jaccard`, `web_gate_novelty`, `web_gate_patience` | 0.6, 0.3, 2 | The gate's thresholds. |

For ablations, `web_notes=false` and `web_gate=false` keep the tool schemas identical, so
those arms differ only in the intervention. `disallowed_tools=["web_fetch"]` gives a
search-only arm. Each web action, with its query or URL, goal, Jaccard, novelty, gate
decision and the notes after it, is a line of `agent/web_log.jsonl`. The trial's summary is
in `result.json` under `agent_result.metadata.web`, with the extractor's tokens and cost, and
`gate.triggered_at_action` lists the actions (`i` in the log) where the gate fired. The
extractor's cost is also included in the trial's totals. Each extractor call is capped at 180 s and is skipped
once the trial is within the harness's end-of-budget margin. Its prompt starts with the part
that never changes in a run (instructions and task), which other providers cache on their
own and Anthropic gets a 5-minute cache breakpoint for; the harness's 1-hour breakpoints are
not used.

Exa returns at most 100,000 characters of a page. A page that comes back within 1,000
characters of that limit (Exa may count UTF-16 units or stop at a word) may have been cut, so
its last window says only that much is available rather than "end of page text".

The client retries network errors, 5xx, 408, 425 and 429 five times, waiting as Exa's
`Retry-After` says or backing off (up to 60 s). A request that still fails is one of three
kinds:

- A rejection: Exa refused this one request (a 4xx other than 401/402/403/408/425/429). A
  rejected fetch reads as a page that is not available, and a rejected search asks the agent
  to rephrase. Both are charged, counted under `rejected`, and remembered, so repeating them
  is free.
- Lasting: the account (401/402/403), or the cutoff or key. After 3 in a row, the tools answer
  "unavailable" for the rest of the trial, and `disabled_reason` records why.
- Transient: network errors, 5xx, 408/425/429, an unexpected response. After 3 in a row, the
  tools answer "unavailable right now" for 5 minutes and then try again (`outages`).

A refused request can also mean a cutoff or key Exa does not take. Until an answer with
content (search results or a page) has shown that they work, every refusal is checked with
one plain search under the same cutoff (`probes`): if Exa answers it, the request was the
problem; if Exa refuses it, the failure is lasting; if it fails otherwise, transient. A
fetch answered with "no version of this page" shows nothing, because a cutoff outside Exa's
window looks the same. After 3 refusals in a row the check runs again, since Exa's window
moves. `not_available` counts the pages Exa has no version of; a page it could not read
right now is charged but not remembered, so the agent can try again. Pages are cached by
URL without the fragment. Exa requests stop where the harness's end-of-budget margin starts,
and after that the tools refuse new requests, so a slow Exa cannot eat the time kept for the
final submit.

`EXA_API_KEY` is read like the model keys: from Harbor's agent env (`--ae`) first, then
the environment of the process that runs the agent. A run with `web_snapshot_as_of` and no
key fails at the start instead of running without the web tools. The tools run in that
process, so the key never reaches the sandbox: sandbox commands see it blanked, like the
model keys, with or without the web tools. `OpenWorldHarness.web_host_key_envs` names it, so
a runner can pass it to the host process only when the web tools are on.

Known limits:

- Exa ranks results with current signals. Only content is bounded by the date, so ranking
  can still reflect which pages became prominent later.
- In a task with public network access, the agent can reach the live web from the sandbox.
  `sandbox_network_mode` is recorded in the summary. Use no-network tasks to keep the web
  tools the only window onto the web.
- The sandbox clock still shows the real date.
- Web options are run options (`--ak` or overrides). A task's `[agent.openworld]` or
  `[agent.proximus]` table rejects them, and the other `web_*` options are rejected without
  `web_snapshot_as_of`.
- `agent_result.model_usage`, which Harbor sums from the trajectory's agent steps, leaves
  out the extractor calls, as it already leaves out the compaction-summary calls.
  `cost_usd` and the token totals include both.
- Re-injecting the notes at compaction looks for the core's handoff header and footer text
  (`prompts.HANDOFF_HEADER` and `HANDOFF_FOOTER`), and the extractor's usage goes through the
  core's `_accumulate_usage`. A change there needs a matching change in `web/agent.py` (tests
  cover both).
- The extractor runs inside each web call, as in the paper's loop, so its time is part of the
  web call's.
