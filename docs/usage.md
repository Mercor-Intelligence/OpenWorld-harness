# Using OpenWorld-harness

Reference for running and configuring the harness; the [README](../README.md) has the
quick start. "Proximus" below names the upstream design the harness keeps: the
model-facing prompts and the paths in the sandbox (`/tmp/proximus/`,
`refs/proximus/`) still use that name.

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

## Failure handling

- **Submissions.** A `submit` that fails to record never ends the run: only a recorded
  submission arms the end-run confirmation, and a failed one tells the model that
  nothing was recorded. The exit snapshot is recorded too, but graded only when the
  model recorded no submission itself. It is committed on a private index with
  `git commit-tree`, so HEAD and the index stay as the model left them, and restoring
  an older submission keeps the files the model left untracked.
- **Snapshot staging is time-limited.** `git add -A` gets `submission_timeout_sec` minus
  60 s (at least 30 s). On a timeout the index lock and partial objects are removed and
  the model is told how to shrink the commit; a leftover `index.lock` is reported
  instead of waited on.
- **A dead sandbox ends the run** with `EnvironmentGoneError`: Harbor's
  `SandboxLikelyOutOfMemoryError`, transport errors, an exec result that is only Modal's
  stopped-container message, and 5 exec failures in a row that a trivial probe command
  confirms. A death during the exit submit or restore raises once the artifacts are
  saved, so Harbor records an error and runs no verifier instead of grading a stale
  tree.
- **A bash command that times out is killed with every process it started.** Each
  command runs in its own session (`setsid`), and on a timeout the harness kills that
  whole session from `/proc`, background jobs and `timeout N ...` wrappers included.
  `result.json` counts `bash_timeouts`, `bash_timeout_killed` and
  `bash_timeout_survivors`.
- **`sandbox_limits.memory_mb` is enforced** (`memory_watchdog`, on by default). Modal
  takes the memory size as a request, not a limit, so a run that goes far over it is
  killed along with the whole sandbox. A root loop in the sandbox
  (`scripts/memwatch.sh`) checks memory use once a second (cgroup v2 or v1, or
  `/proc/meminfo` under gVisor). Above `memory_watchdog_fraction` (0.92) of the limit it
  kills the process with the largest resident set, and the model is told in its next
  bash result. Every 15 s a host-side poller also collects the loop's events and a
  memory sample (the counter and its parts, the peak, the largest processes, `/dev/shm`
  and `/tmp`, and GPU memory on GPU sandboxes), kept in `memwatch.jsonl` among the agent
  logs and in `result.json` under `memory_watchdog`. A GPU sandbox that exits 137 was
  most likely preempted: Modal preempts GPU sandboxes, and Harbor reports any exit 137
  as "likely out of memory".
- **Filtered responses.** A response the provider's output filter blanked (finish
  reason `content_filter`, no text, no tool call; Gemini's SAFETY, RECITATION and so on)
  is not a model turn. The request is sent again (`content_filter_attempts`, 4), then
  the newest tool output is withheld and it is tried as often again, and only then does
  the run end, as a graded provider failure. `result.json` counts
  `content_filter_hits`.
- **Prose-only turns** get the neutral "Continue." only up to `toolless_nudge_after`
  (20) in a row, or until the same reply comes back three times. Later ones name the
  ways on (a tool call, or submit), and `toolless_turn_limit` (50) turns in a row
  without a tool call end the run like a time-out (`ended_by: prose_loop`).
- **Streamed model calls** (`llm_call_kwargs.stream`) request usage, retry a stream cut
  off without a finish reason, and are bounded as a whole by the request timeout.
- **Reasoning.** OpenAI gpt-5-series requests with tools ask for reasoning summaries at
  the default effort too, and every reasoning item of a turn is kept, so summaries
  reach the trajectory and all encrypted reasoning is replayed.
- **Trajectory.** The harness's own messages (continue nudges, error notes) are user
  steps in `trajectory.json`, tagged `extra.harness_message`.

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

`llm_call_kwargs={"stream": true}` streams each model call, which keeps long
turns alive at gateways that cut silent requests. Some gateways also limit how
long a stream may run: Vercel's AI Gateway allows longer ones with `extra_headers:
{"x-vercel-gateway-extended-time": "true"}`, but still cut streams of about 20
minutes in our runs. Usage is
requested with `stream_options.include_usage` unless you set it, a stream that
ends without a finish reason is retried rather than accepted, and the request
`timeout` bounds the whole stream. `stream_idle_timeout_sec` also retries a
stream that sends nothing for that long; leave it unset for providers that stay
silent while reasoning.

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

All are off by default and run once, as root, before the loop:

- `static_git_path`: host path of a statically linked `git`. When the sandbox has no
  `git`, it is copied to `/usr/local/bin/git` (with the templates git needs), so
  submissions work in images that ship none. Without it such an image fails at
  setup. A host path is a run option (`--ak` or overrides): task tables reject it.
  `scripts/build-static-git.sh OUT` builds one for x86-64 Linux (git 2.56.0 and zlib
  1.3.2, checksum-pinned sources, musl libc; it needs curl, xz-utils, make, gcc and
  musl-tools). CI builds it and runs `scripts/git-smoke.sh` with it in images that
  ship no git. The binary is git, licensed GPL-2.0.
- `large_input_exclude_mb`: in a fresh workspace (not yet a git repository), files
  larger than this many MB are added to `workspace_git_excludes`, so the baseline
  commit never hashes multi-GB inputs. The files stay in place.
- `large_input_total_mb`: with `large_input_exclude_mb`, the largest remaining files
  of 1 MB or more are also excluded until the tracked ones total at most this many MB.

`result.json` records what setup did under `agent_result.metadata`: `platform_git`
(`present` or `installed`) and `platform_git_excludes` (how many files were
excluded).

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
- Reproducing a published score needs the original model configuration (see
  [Reproducing OpenWorldEngineer runs](reproducing-openworldengineer.md)).
