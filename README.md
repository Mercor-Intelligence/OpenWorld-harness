# OpenWorld-harness

OpenWorld-harness is a fork of [Proximus](https://github.com/Proximal-Labs/proximus)
by Proximal AI, modified by Mercor; it is the agent harness of the RWE eval
platform. It is a coding agent for long-horizon tasks, packaged as an external agent
for [Harbor](https://pypi.org/project/harbor/). It runs a LiteLLM tool loop on the
orchestrator host and acts in the sandbox only through Harbor's environment
interface, so it works with any Harbor environment (Docker, Modal, ...).

Status: `0.2.0`, for Harbor `0.23.0`. "Proximus" below names the upstream design
this harness keeps; the model-facing prompts still call the agent Proximus.

## Changes from Proximus

- Renamed: package `openworld_harness`, agent `openworld_harness:OpenWorldHarness`
  (Harbor name `openworld-harness`), task table `[agent.openworld]` (a task's
  `[agent.proximus]` table is still read).
- Git in any image (opt-in): `static_git_path` copies a static `git` into a sandbox
  that has none, and `large_input_exclude_mb` keeps large files already in a fresh
  workspace out of the baseline commit (see [Setup options](#setup-options)).
- Web tools (opt-in): `web_search` and `web_fetch` over Exa Snapshot when
  `web_snapshot_as_of` is set (see [Web search](#web-search-exa-snapshot)).
- A failed `submit` never ends the run. Upstream armed the end-run confirmation
  before recording, so a submit that failed and any later submit ended the run.
  Only a recorded submission arms it now, and a failed one says that nothing was
  recorded and the run continues.
- A dead sandbox ends the run with `EnvironmentGoneError`: Harbor's
  `SandboxLikelyOutOfMemoryError`, an exec result that is only Modal's
  stopped-container message, and 5 exec failures in a row that a trivial probe
  command confirms count, besides upstream's transport errors. A death during
  the exit submit or restore raises once the artifacts are saved, so Harbor records
  an error and runs no verifier instead of grading a stale tree.
- The model's own submission wins. The exit snapshot is still recorded, but it is
  graded only when the model recorded no submission itself. It is committed on a
  private index with `git commit-tree`, so HEAD and the index stay as the model left
  them: restoring an older submission keeps the files the model left untracked.
- Snapshot staging is time-limited: `git add -A` gets `submission_timeout_sec`
  minus 60 s (at least 30 s). On a timeout the index lock and partial objects are
  removed and the model is told how to shrink the commit; a leftover `index.lock`
  is reported instead of waited on.
- OpenAI gpt-5-series requests with tools ask for reasoning summaries at the
  default effort too, and every reasoning item of a turn is kept (LiteLLM's bridge
  kept only the last of consecutive ones, where gpt-6 models do not summarize), so
  summaries reach the trajectory and all encrypted reasoning is replayed.
- The harness's own messages (continue nudges, error notes) are user steps in
  `trajectory.json`, tagged `extra.harness_message`.
- The compaction request reads as a routine checkpoint instead of opening with
  "Pause the task.", which some summaries passed on as a user pause.
- LiteLLM is locked to 1.106.2, with the `mcp` extra its Responses path imports.

## Install and run

```sh
uv pip install "openworld-harness @ git+https://github.com/mercor-code-envs/OpenWorld-harness"   # next to Harbor
harbor run ... --agent openworld_harness:OpenWorldHarness --model anthropic/claude-opus-5 \
  --ak time_budget_sec=7200
```

[examples/agent.yaml](examples/agent.yaml) shows a job-config fragment.
`harbor agent schema openworld_harness:OpenWorldHarness` lists every option.

## How it works

- **Tools:** `bash`, `submit`, `list_submissions`, and `view_image` for vision models.
- **Submissions** are Git commits in the workspace. The first `submit` records a
  commit; a `submit` on a later turn confirms it and ends the run, and any other
  tool call in between cancels the confirmation. The model's newest recorded
  submission wins.
- **At exit** Proximus records unsaved work as a final submission, graded only if
  the model recorded none itself, and checks out the winning commit, so Harbor
  captures the submitted tree for the verifier. If the sandbox has died, the run
  raises `EnvironmentGoneError` instead and is not graded.
- **Time:** `time_budget_sec` paces the run. Set Harbor's hard agent timeout above
  it (see `hard_timeout_sec` below) so the exit submit and restore can finish. With
  `task_provides_timer = true`, Proximus re-anchors the task's own timer instead of
  adding a second clock.
- **Context:** near the context limit the conversation is compacted into a handoff
  summary; the model's notes in `/tmp/proximus/PROGRESS.md` survive it.
- **Sandbox needs:** `git` (or `static_git_path`) and `bash`.

## Models and credentials

Any LiteLLM model works (`anthropic/…`, `openai/…`, `openrouter/…`, `gemini/…`,
or `openai/<model>` with `api_base` for OpenAI-compatible endpoints). There are
no built-in model presets: pass `reasoning_effort`, `llm_call_kwargs` or
`model_info` as needed. The context limit comes only from LiteLLM's
`max_input_tokens`. If it is missing or is not a positive integer, construction
fails: set `model_info={"max_input_tokens": 200000}` to the model's actual limit.
There is no context fallback. Unknown models default to no vision and $0 cost.

Credentials follow Harbor's own agents: the provider's usual variable
(`ANTHROPIC_API_KEY`, `OPENAI_BASE_URL`, ...) from Harbor's agent env
(`--ae 'KEY=${VAR}'`), else from your shell. `api_key_env` names a different
variable for one model. Proximus blanks the variables it uses inside the
sandbox, so the model cannot read them; Bedrock, SageMaker and Vertex keep
LiteLLM's own lookup. Never put keys in `--ak` or `llm_call_kwargs`: Harbor saves
agent options verbatim in job files.

## Options and customization

Options follow Harbor's Terminus 2 names where they exist: `max_turns`,
`enable_summarize`, `temperature`, `api_base`, `max_thinking_tokens`,
`model_info`, `llm_call_kwargs` and `prompt_template_path` (a Jinja template
around `{{ instruction }}`). Proximus adds `compaction_threshold` (default 0.85)
and `disallowed_tools` (`list_submissions`, `view_image`). Unset options keep the
default behavior, and `llm_call_kwargs` wins over the other request settings.
`max_thinking_tokens` and `reasoning_effort` are mutually exclusive, and
`max_turns` counts every loop iteration, including recovery retries.

For deeper changes, subclass the agent and select it with
`--agent mypkg.module:MyProximus`:

```python
from openworld_harness import OpenWorldHarness, tools

class MyProximus(OpenWorldHarness):
    def system_prompt(self) -> str:
        return super().system_prompt() + "\n\nPrefer small commits."

    def build_tools(self) -> dict[str, tools.Tool]:
        count = tools.Tool("count_lines", "...", Args, self._count)
        return {**super().build_tools(), count.name: count}

    async def _count(self, args) -> tools.ToolResult:
        result = await self.environment.exec(f"wc -l {args.path}", timeout_sec=30)
        return tools.ToolResult(str(result.stdout))
```

The hooks are `system_prompt()`, `initial_message()`, `summary_request()` and
`build_tools()`. The first and last run inside `__init__`, so set attributes they
need before `super().__init__()`; `self.environment` exists only during a run.
After a compaction the conversation restarts from a handoff that repeats the
instruction, not `initial_message()`.

## Setup options

Both are off by default and run once, as root, before the loop:

- `static_git_path`: host path of a statically linked `git`. When the sandbox has no
  `git`, it is copied to `/usr/local/bin/git` (with the templates git needs), so
  submissions work in images that ship none. Without it such an image fails at
  setup. A host path is a run option (`--ak` or overrides): task tables reject it.
- `large_input_exclude_mb`: in a fresh workspace (not yet a git repository), files
  larger than this many MB are added to `workspace_git_excludes`, so the baseline
  commit never hashes multi-GB inputs. The files stay in place.

`result.json` records what setup did under `agent_result.metadata`: `platform_git`
(`present` or `installed`) and `platform_git_excludes` (how many files were
excluded).

## Web search (Exa Snapshot)

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
  `cost_usd`, the token totals and the trial row include both.
- Re-injecting the notes at compaction looks for the core's handoff header and footer text
  (`prompts.HANDOFF_HEADER` and `HANDOFF_FOOTER`), and the extractor's usage goes through the
  core's `_accumulate_usage`. A change there needs a matching change in `web/agent.py` (tests
  cover both).
- The extractor runs inside each web call, as in the paper's loop, so its time is part of the
  web call's.

## Task configuration

Tasks can declare options in `task.toml`. Harbor ignores these tables, so a runner
reads them with `openworld_harness.task_agent_kwargs(task_dir, model, overrides,
defaults)` and sets the hard timeout to `openworld_harness.hard_timeout_sec(kwargs)`:

```toml
[agent.openworld]
task_provides_timer = true

[agent.openworld.models."anthropic/claude-opus-5"]
reasoning_effort = "max"
```

Later layers win: defaults, `[agent.openworld]` (or `[agent.proximus]`), the run's
model table, overrides.
`time_budget_sec` defaults to `[agent].timeout_sec`, relative template paths are
task-relative, and `workspace_git_excludes` defaults to the task's `/app` artifact
excludes. Invalid options raise `ValueError` before any sandbox starts.

## Limits

- Requires Harbor `0.23.0` and Python 3.12+. Upstream was tested on Python
  3.12–3.14 with LiteLLM 1.97.0 and 1.105.1; the fork's CI runs Python 3.12 with
  the locked LiteLLM 1.106.2.
- Installs a small process-wide LiteLLM shim at LLM construction: it accepts
  reasoning-only Responses turns and keeps every reasoning item of a turn. Harbor
  itself is not patched.
- No MCP servers, Harbor skills, session resume or handoff; tasks configuring MCP
  servers or skills fail at setup.
- Reproducing a published score needs the original model configuration.

## Development

```sh
uv sync --locked
uv run pytest -q          # offline: fake providers, temporary Git repos
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
uv build && uv run python scripts/verify_wheel.py dist/openworld_harness-0.2.0-py3-none-any.whl
```

`verify_wheel.py` reruns the suite against the installed wheel outside the
checkout; `--litellm-version 1.97.0` checks the lower bound.

### Formatting and lint
We use ruff.

```sh
uv run --locked ruff check --fix src tests scripts
uv run --locked ruff format src tests scripts
```

Licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
