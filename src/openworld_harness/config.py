from pathlib import Path
from typing import Any, Literal, Self

from harbor.agents.options import AgentOptions
from harbor.utils.templating import render_prompt_template
from pydantic import ConfigDict, Field, field_validator, model_validator

# Seconds kept free at the end of the time budget: bash commands time out this
# long before the budget runs out, and a model request always gets at least
# this long, so the run can still finish cleanly.
BUDGET_MARGIN_SEC = 120
# Options of the web tools (OpenWorldConfig.web_*).
WEB_OPTION_PREFIX = "web_"
# Host paths, which a task table may not set (nor the web options).
RUN_ONLY_OPTIONS = frozenset({"static_git_path"})


class ProximusConfig(AgentOptions):
    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    time_budget_sec: float | None = None
    """Total time budget in seconds. The run stops when it is used up, whatever
    any task timer says. Set Harbor's hard agent timeout higher (see
    ``hard_timeout_sec``) so the exit submit and restore can finish."""

    submission_timeout_sec: int = 600
    """Seconds allowed for one submission's git work in the sandbox. Large
    workspaces can take minutes to commit."""

    submission_export: bool = True
    """After each submission the model records, copy the files it changed off
    the sandbox, into the agent logs' ``winner/`` (``sub.tar``, ``deleted.z``,
    ``manifest.json``). If the sandbox later dies, a regrade can rebuild the
    latest recorded submission in a fresh copy of the task image. Runs beside
    the agent and never blocks a turn; a newer submission replaces the export."""
    submission_export_max_mb: int | None = Field(default=4096, ge=1)
    """Largest export, in MB of changed files. A bigger submission is not
    exported; its manifest says why. None: no limit."""
    auto_submit_on_exit: bool = True
    """When the run ends, save any unsubmitted work as a final submission, so a
    run that times out without submitting is still graded on its last state.
    It is recorded but not graded when the model submitted anything itself.
    It never moves HEAD or the index, so untracked files survive the restore.
    Skipped if nothing changed since the last submission."""

    restore_on_exit: bool = True
    """When the run ends, check out the winning submission so the verifier sees
    it. Turn off only if grading reads the submitted commits directly; the
    workspace is then left as it was when the run stopped."""

    finalize_only_on_submit: bool = False
    """Do the exit submit and restore only if the model confirmed a submission.
    Runs that end any other way (time out, stall, error) are left as they are."""

    task_provides_timer: bool = False
    """Set when the task instruction already tells the model how to check the
    time (for example FrontierSWE's ``sandbox-timer``). Only changes what the
    model is told: Proximus skips its own deadline file and points the model to
    the task's timer. The run still ends at ``time_budget_sec``, and the task's
    timer is reset to show the same remaining time either way."""

    reasoning_effort: str | None = None
    """Reasoning effort passed to the model. A value the model does not
    support is rejected by the provider."""

    api_key_env: str | None = None
    """Name of the environment variable that holds this model's API key (never
    the key itself). Use it when two models share a provider route but need
    different keys. Read from Harbor's agent env first, then the host. Without
    it, the provider's usual variable is used."""

    workspace_git_excludes: list[str] = Field(default_factory=list)
    """Paths that submissions never commit, in ``.gitignore`` syntax (for
    example caches and checkpoints that grading ignores anyway). Written to
    ``.git/info/exclude``, so the visible workspace is unchanged."""

    litellm_model_info: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Extra LiteLLM model entries, keyed by model name, for models LiteLLM
    does not know. They set the context limit, vision support (needed for
    ``view_image``) and prices. A positive integer ``max_input_tokens`` is
    required; there is no context fallback. Unknown models get no vision and
    zero cost. For an ``openai/`` route, add both ``openai/<model>`` and
    ``<model>``. ``model_info`` is the shorthand for the run's model."""

    llm_call_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra arguments for every model request (for example ``max_tokens``,
    ``timeout`` or ``extra_body``). They override all other request settings.
    With ``stream: true`` the response is streamed and rebuilt; usage is
    requested (``stream_options.include_usage``) unless set here."""

    content_filter_attempts: int = Field(default=4, ge=1)
    """Times one request is sent when the provider's output filter blanks the
    response (finish reason ``content_filter``, no text, no tool call). Such a
    response is not a model turn and never counts toward the empty-turn stall.
    After the last attempt the run takes the refusal path: the newest tool
    output is withheld and the request tried again as often; if that is
    filtered too, the run ends as a provider failure and the winner is
    graded."""

    stream_idle_timeout_sec: float | None = Field(default=None, gt=0)
    """With a streamed request, give up on an attempt (and retry it) when no
    chunk arrives for this long. Leave unset for providers that stay silent
    while they reason (OpenAI Responses sends nothing for minutes)."""

    max_turns: int | None = Field(default=None, ge=1)
    """Maximum number of loop iterations, including retries. Reaching it ends
    the run like a time-out, with the usual exit submit and restore."""

    toolless_nudge_after: int = Field(default=20, ge=1)
    """Turns in a row without a tool call (prose or empty) that are answered
    with the neutral "Continue.". Each later one is answered with a message
    naming the ways on (a tool call, or submit) and the turn limit, as is any
    turn that repeats the previous two replies word for word."""

    toolless_turn_limit: int = Field(default=50, ge=0)
    """End the run after this many turns in a row without a tool call, like a
    time-out: the exit submit and restore run and the winner is graded
    (``ended_by`` is ``prose_loop``). 0: no limit. The longest legitimate run
    seen across 1,288 pass@8 trials was 32."""

    enable_summarize: bool = True
    """Summarize the conversation when it nears the context limit. When off, a
    full context window ends the run with an error."""

    compaction_threshold: float = Field(default=0.85, gt=0, le=1)
    """Fraction of the context limit at which the conversation is summarized."""

    disallowed_tools: list[str] = Field(default_factory=list)
    """Tools to remove: ``list_submissions``, ``view_image``, ``web_search`` or
    ``web_fetch``. ``bash`` and ``submit`` are always available."""

    prompt_template_path: str | None = None
    """Jinja2 template around the task instruction. It must contain
    ``{{ instruction }}``."""

    temperature: float | None = None
    """Sampling temperature. Omit to use the provider's default."""

    api_base: str | None = None
    """Base URL of the model API."""

    max_thinking_tokens: int | None = Field(default=None, ge=1024)
    """Thinking budget for Anthropic models (at least 1024). Cannot be combined
    with ``reasoning_effort``."""

    model_info: dict[str, Any] | None = None
    """LiteLLM model info for the run's model (context limit, vision, prices).
    Set ``max_input_tokens`` to a positive integer if the registry has no valid
    input limit. This is the only field used for the context limit."""

    @field_validator("prompt_template_path")
    @classmethod
    def _template_renders(cls, value: str | None) -> str | None:
        if value is not None:
            if not Path(value).is_file():
                raise ValueError(f"prompt_template_path not found: {value}")
            # Raises unless the template uses {{ instruction }}.
            render_prompt_template(Path(value), "instruction")
        return value

    @model_validator(mode="after")
    def _one_thinking_control(self) -> Self:
        if self.max_thinking_tokens is not None and self.reasoning_effort is not None:
            raise ValueError(
                "set either max_thinking_tokens or reasoning_effort, not both: "
                "LiteLLM maps both onto the provider's thinking setting"
            )
        return self


class OpenWorldConfig(ProximusConfig):
    """Options of OpenWorldHarness: the Proximus core's, plus the fork's."""

    static_git_path: str | None = None
    """Host path of a statically linked ``git``. When the sandbox has no git,
    setup copies it to /usr/local/bin/git, so submissions work in images that
    ship none. Unset, such an image fails at setup."""

    large_input_exclude_mb: int | None = Field(default=None, ge=1)
    """Files already in a fresh workspace (not a git repository) that are larger
    than this many MB are added to ``workspace_git_excludes`` at setup, so the
    baseline commit never hashes multi-GB inputs. The files stay in place."""

    large_input_total_mb: int | None = Field(default=None, ge=1)
    """With ``large_input_exclude_mb``, also exclude the largest remaining files
    of 1 MB or more until the ones still tracked total at most this many MB.
    Many medium inputs (sentinel2: 1.33 GB of 5-33 MB arrays) are each under the
    per-file limit, yet together overrun the baseline commit."""

    sandbox_limits: dict[str, Any] | None = None
    """The sandbox's resources, told to the model in the first message and
    written to ``/tmp/proximus/limits.json``: ``cpus``, ``memory_mb``, ``gpus``,
    ``gpu_type``. Under gVisor ``free``, ``/proc/meminfo`` and ``nproc`` show the
    host, and going over the memory can kill the whole sandbox, so a model sizing
    parallel jobs by what it sees can end its own run."""

    @field_validator("sandbox_limits")
    @classmethod
    def _limits(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        allowed = {"cpus", "memory_mb", "gpus", "gpu_type"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"sandbox_limits: unknown keys {unknown}")
        for key in ("cpus", "memory_mb", "gpus"):
            item = value.get(key)
            if item is not None and (
                isinstance(item, bool) or not isinstance(item, int | float) or item < 0
            ):
                raise ValueError(f"sandbox_limits.{key} must be a non-negative number")
        gpu_type = value.get("gpu_type")
        if gpu_type is not None and not isinstance(gpu_type, str):
            raise ValueError("sandbox_limits.gpu_type must be a string")
        return {k: v for k, v in value.items() if v is not None}

    memory_watchdog: bool = True
    """With ``sandbox_limits.memory_mb``, enforce it: a root loop in the sandbox
    kills the process with the largest resident set when the sandbox's memory
    use passes ``memory_watchdog_fraction`` of the limit, and the model is told.
    Modal treats the limit as a request only, so without this a run that goes
    over it is killed with the whole sandbox by host pressure. The loop is
    stopped before grading."""

    memory_watchdog_fraction: float = Field(default=0.92, gt=0, le=1)
    """Share of ``memory_mb`` at which the watchdog kills."""

    memory_watchdog_source: Literal["auto", "cgroup2", "cgroup1", "meminfo"] = "auto"
    """How the watchdog measures memory use: cgroup v2 ``memory.current`` or v1
    ``memory.usage_in_bytes`` (both less inactive page cache), or ``MemTotal -
    MemAvailable`` (only under gVisor, where /proc/meminfo is the sandbox's).
    ``auto`` takes the first readable one; with none the watchdog stays off."""

    web_snapshot_as_of: str | None = None
    """Turns on ``web_search`` and ``web_fetch`` over Exa Snapshot, pinned to
    this cutoff: an ISO 8601 date or date-time (UTC); pages are seen as Exa
    stored them at or before it. Never shown to the model. Exa documents a
    rolling window of about five months. Needs ``EXA_API_KEY`` in Harbor's agent
    env or the runner's environment; the tools run in the runner, so the key
    never enters the sandbox. The other ``web_*`` options need it set."""

    web_max_searches: int = Field(default=50, ge=1)
    """Exa search requests allowed per run. An identical repeated query is served
    again without a request."""

    web_max_fetches: int = Field(default=100, ge=1)
    """Exa page requests allowed per run. Further windows of a fetched page are
    free."""

    web_fetch_chars: int = Field(default=10_000, ge=1_000)
    """Characters of page text per ``web_fetch`` call; ``offset`` reads further."""

    web_show_dates: bool = False
    """Show pages' published dates. Off by default: in a snapshot they cluster
    just below the cutoff, so they would tell the model roughly when it is."""

    web_notes: bool = True
    """Keep the belief state: an extractor call (the agent's model) after each
    observation that has something new, the notes returned with web results when
    they change and re-injected at compaction."""

    web_notes_items: int = Field(default=6, ge=1)
    """Notes kept by a reorganization, which runs once there are more than this
    plus 4."""

    web_extractor_reasoning_effort: str | None = None
    """Reasoning effort for the extractor calls. None uses the provider's
    default; effort and thinking settings in ``llm_call_kwargs`` do not apply to
    them."""

    web_gate: bool = True
    """Halt stagnant searches (the exhaustion gate). When off, the gate still
    records where it would have fired."""

    web_gate_jaccard: float = Field(default=0.6, ge=0, le=1)
    """Gate: a search is a near-repeat when its token Jaccard against one of the
    last three queries is at least this."""

    web_gate_novelty: float = Field(default=0.3, ge=0, le=1)
    """Gate: and stagnant when at most this share of its results are pages not
    seen before."""

    web_gate_patience: int = Field(default=2, ge=1)
    """Gate: stagnant searches in a row that fire it."""

    @field_validator("web_snapshot_as_of")
    @classmethod
    def _as_of(cls, value: str | None) -> str | None:
        from openworld_harness.web.exa import normalize_as_of

        return None if value is None else normalize_as_of(value)

    @model_validator(mode="after")
    def _web_options_need_a_cutoff(self) -> Self:
        if self.web_snapshot_as_of is None:
            # Changed from the default, not merely given: a dumped no-web config
            # (every field set) validates again.
            fields = type(self).model_fields
            stray = sorted(
                f
                for f in self.model_fields_set
                if f.startswith(WEB_OPTION_PREFIX)
                and f != "web_snapshot_as_of"
                and getattr(self, f) != fields[f].get_default()
            )
            if stray:
                raise ValueError(
                    f"{', '.join(stray)} need web_snapshot_as_of, which turns the "
                    "web tools on"
                )
        return self
