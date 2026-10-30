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
  tool call in between cancels the confirmation. The latest submission wins.
- **At exit** Proximus auto-submits unsaved work and checks out the winning
  commit, so Harbor captures the submitted tree for the verifier.
- **Time:** `time_budget_sec` paces the run. Set Harbor's hard agent timeout above
  it (see `hard_timeout_sec` below) so the exit submit and restore can finish. With
  `task_provides_timer = true`, Proximus re-anchors the task's own timer instead of
  adding a second clock.
- **Context:** near the context limit the conversation is compacted into a handoff
  summary; the model's notes in `/tmp/proximus/PROGRESS.md` survive it.
- **Sandbox needs:** `git` and `bash`.

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

- Requires Harbor `0.23.0` and Python 3.12+; tested on Python 3.12–3.14 with
  LiteLLM 1.97.0 and 1.105.1.
- Installs a small process-wide LiteLLM shim for reasoning-only Responses turns
  at LLM construction; Harbor itself is not patched.
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
