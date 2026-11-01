import asyncio
import contextlib
import itertools
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any, Literal, override

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import ContextLengthExceededError, OutputLengthExceededError
from harbor.models import trajectories
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig
from harbor.utils.templating import render_prompt_template
from harbor.utils.trajectory_utils import format_trajectory_json

import openworld_harness.images as images
import openworld_harness.llm as llm
import openworld_harness.prompts as prompts
import openworld_harness.submissions as submissions
import openworld_harness.timer as timer
import openworld_harness.tools as tools
import openworld_harness.utils as utils
from openworld_harness.config import BUDGET_MARGIN_SEC, ProximusConfig

EMPTY_TURN_STALL_LIMIT = 3
# Fraction of the context window at which proactive compaction runs
# Most-recent messages preserved verbatim across a compaction
KEEP_RECENT_MESSAGES = 6
# Byte cap on the PROGRESS.md read folded into each compaction handoff
PROGRESS_REINJECT_MAX_BYTES = 25_000
# A budget that runs out while a provider error is being retried ends the run
# normally (the restored winner is graded) unless provider failures cost more
# than this fraction of the budget; then the run fails, to be re-run.
PROVIDER_LOSS_GRADED_FRACTION = 0.10
# Characters of a discarded over-long response kept in its trajectory step.
DISCARD_EXCERPT_CHARS = 2000
# How long the end of the run waits for an in-flight submission export.
EXPORT_FINALIZE_WAIT_SEC = 300
# Agent-logs directory of the latest submission export.
EXPORT_DIR_NAME = "winner"
# export_submission.sh: the changed files exceed the export limit.
EXPORT_TOO_LARGE_EXIT = 5


@dataclass
class RunState:
    messages: list[llm.Message] = field(default_factory=list)
    session_id: str = ""
    start_monotonic: float = field(default_factory=time.monotonic)
    n_turns: int = 0
    # Turn of the submit awaiting confirmation; see _submit.
    submit_armed_turn: int | None = None
    ended_by_submit: bool = False
    env_lost: bool = False
    consecutive_empty_turns: int = 0
    stalled: bool = False
    last_context_tokens: int | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: float = 0.0
    fallback_attempted_turns: int = 0
    fallback_turns: int = 0
    fallback_models: dict[str, int] = field(default_factory=dict)
    n_images: int = 0  # viewed images saved under <logs_dir>/images/
    # How the run ended: submit, budget, max_turns, stall, provider_failure,
    # sandbox_death, cancelled (Harbor's agent timeout) or error.
    ended_by: str | None = None
    # Set when the provider ended the run early and the winner is still graded.
    provider_failure: dict[str, Any] | None = None
    consecutive_refusals: int = 0


class Proximus(BaseAgent):
    """Long-horizon, native-function-calling agent harness."""

    capabilities = AgentCapabilities(atif=True)
    options_model = ProximusConfig

    _environment: BaseEnvironment
    _executor: tools.ToolExecutor
    _submissions: submissions.Submissions
    _instruction: str
    _tools: dict[str, tools.Tool]
    _tool_schemas: list[dict[str, Any]]

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        logger: logging.Logger | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        skills_dir: str | None = None,
        extra_env: dict[str, str] | None = None,
        load_trajectory: str | Path | None = None,
        environment_logs_dir: PurePosixPath | None = None,
        override_setup_timeout_sec: float | None = None,
        **kwargs,
    ):
        if model_name is None:
            raise ValueError("Proximus requires model_name to be set")
        super().__init__(
            logs_dir,
            model_name,
            logger=logger,
            mcp_servers=mcp_servers,
            skills_dir=skills_dir,
            extra_env=extra_env,
            load_trajectory=load_trajectory,
            environment_logs_dir=environment_logs_dir,
            override_setup_timeout_sec=override_setup_timeout_sec,
            **kwargs,
        )
        if not isinstance(self.options, ProximusConfig):
            raise TypeError("Proximus.options must be ProximusConfig")
        self.config = self.options
        self._llm = llm.LLM(
            model_name,
            self.config,
            self._remaining_budget_sec,
            self.logger,
            provider_health_path=self.logs_dir / "provider_health.json",
            extra_env=self.extra_env,
            on_provider_event=self._record_provider_event,
        )
        self._context_limit = self._llm.resolve_context_limit()
        self._system_prompt = self.system_prompt()
        self._tools = self._filter_tools(self.build_tools())
        self._tool_schemas = [tool.schema() for tool in self._tools.values()]
        self._run = RunState()
        self._steps: list[trajectories.Step] = []
        # Background copy of the newest submission off the sandbox.
        self._export_task: asyncio.Task | None = None
        self._export_pending: tuple[int, str] | None = None
        # Told to the model with its next tool result: exports that failed.
        self._export_notes: list[str] = []

    @staticmethod
    @override
    def name() -> str:
        return "proximus"

    @override
    def version(self) -> str | None:
        return version("openworld-harness")

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        if self.mcp_servers or self.skills_dir:
            raise ValueError(
                "Proximus supports neither MCP servers nor skills; "
                "remove them from the task/agent config please."
            )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        with self._sandbox_credential_scope(environment):
            await self._run_task(instruction, environment, context)

    def _sandbox_hidden_env_keys(self) -> set[str]:
        """Host-side credentials that sandbox commands must not see."""
        return set(self._llm.consumed_env_keys)

    def _sandbox_credential_scope(
        self, environment: BaseEnvironment
    ) -> contextlib.AbstractContextManager:
        """Hide host-side credentials from sandbox commands."""
        hidden = {name: "" for name in sorted(self._sandbox_hidden_env_keys())}
        scope = getattr(environment, "scoped_exec_env", None)
        return scope(hidden) if hidden and scope else contextlib.nullcontext()

    async def _run_task(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._run = RunState()
        self._steps = []
        if self.config.prompt_template_path:
            instruction = render_prompt_template(
                Path(self.config.prompt_template_path), instruction
            )
        self._instruction = instruction
        self._environment = environment
        workdir = await self._resolve_workdir(environment)

        self._run.session_id = self.session_id or str(uuid.uuid4())
        self._llm.session_id = self._run.session_id

        await self._ensure_progress_dir(environment)
        self._executor = tools.ToolExecutor(
            environment,
            default_workdir=workdir,
            remaining_sec=self._remaining_budget_sec,
            foreground_sessions=self._foreground_sessions,
        )
        await self._init_submissions(environment, workdir)
        await self._arm_timer(environment)
        self._seed_conversation(instruction, workdir, environment)

        try:
            await self._agent_loop()
        except tools.EnvironmentGoneError:
            self._run.ended_by = "sandbox_death"
            raise
        except asyncio.CancelledError:
            # Harbor's agent timeout: the live sandbox is graded as-is.
            self._run.ended_by = self._run.ended_by or "cancelled"
            raise
        except BaseException:
            self._run.ended_by = self._run.ended_by or "error"
            raise
        finally:
            await self._finish_exports()
            try:
                should_finalize = (
                    not self.config.finalize_only_on_submit or self._run.ended_by_submit
                )
                if self._run.env_lost:
                    self.logger.info("End-of-run restore skipped: the sandbox is gone.")
                elif not should_finalize:
                    self.logger.info(
                        "End-of-run restore skipped: the model did not "
                        "end with a confirmed submit."
                    )
                else:
                    if (
                        self.config.auto_submit_on_exit
                        and not self._run.ended_by_submit
                    ):
                        await self._auto_final_submission()
                    if not self._run.env_lost and self.config.restore_on_exit:
                        await self._restore_winner()
            finally:
                self._populate_context(context)
                self._dump_artifacts()
        if self._run.env_lost:
            self._run.ended_by = "sandbox_death"
            # Reached only when the loop itself ended normally: the sandbox died
            # during the end-of-run submission or restore, so the workspace the
            # verifier would grade is gone too. Fail the trial, as a death
            # during the loop does, rather than grade whatever remains.
            raise tools.EnvironmentGoneError(
                "the sandbox was lost while finalizing the run"
            )

    async def _restore_winner(self) -> None:
        try:
            await self._submissions.restore_winner()
        except tools.EnvironmentGoneError as e:
            self.logger.warning(f"Winner restore failed: the sandbox is gone ({e}).")
            self._run.env_lost = True

    async def _agent_loop(self) -> None:
        for turn in itertools.count(1):
            self._run.n_turns = turn

            if self._budget_spent():
                self.logger.debug("Time budget exhausted; ending run.")
                self._run.ended_by = "budget"
                return
            if self.config.max_turns and turn > self.config.max_turns:
                self.logger.debug("Turn limit reached; ending run.")
                self._run.n_turns = turn - 1
                self._run.ended_by = "max_turns"
                return

            tokens = self._run.last_context_tokens
            remaining = self._remaining_budget_sec()
            if (
                self.config.enable_summarize
                and tokens
                and tokens >= self.config.compaction_threshold * self._context_limit
                # Too close to the deadline for a summary call and a turn; the
                # next request still fits (the threshold is below the limit).
                and (remaining is None or remaining > BUDGET_MARGIN_SEC)
            ):
                await self._compact()
                if self._budget_spent():
                    self.logger.debug("Time budget exhausted during compaction.")
                    self._run.ended_by = "budget"
                    return

            try:
                parsed = await self._llm.query(
                    self._run.messages, tools=self._tool_schemas
                )
            except ContextLengthExceededError:
                if not self.config.enable_summarize:
                    raise
                self.logger.debug("Context window exceeded; compacting.")
                await self._compact()
                continue
            except OutputLengthExceededError as e:
                self.logger.debug("Output length exceeded; asking for smaller output.")
                self._record_discard(e)
                self._inject_user(prompts.OUTPUT_LENGTH_MESSAGE, "output_length")
                continue
            except llm.ProviderRefusalError as e:
                self._run.consecutive_refusals += 1
                if (
                    self._run.consecutive_refusals == 1
                    and self._withhold_newest_output()
                ):
                    self.logger.warning(
                        f"Provider refused the request ({e.code}); withheld the "
                        "newest tool output and retrying."
                    )
                    self._inject_user(
                        prompts.PROVIDER_REFUSAL_MESSAGE, "provider_refusal"
                    )
                    continue
                self.logger.warning(
                    f"Provider refused the request again ({e.code}); ending the "
                    "run so the restored winner is graded."
                )
                self._end_by_provider(
                    {"kind": "refusal", "code": e.code, "message": str(e)[:300]}
                )
                return
            except llm.ImageLimitExceededError:
                # Clear all images to avoid repeatedly hitting the image-count cap.
                evicted = images.evict_all(self._run.messages)
                if not evicted:
                    raise
                self.logger.debug(
                    "Provider image-count limit exceeded; removed %d images "
                    "and compacting context.",
                    evicted,
                )
                if self.config.enable_summarize:
                    await self._compact()
                continue
            except llm.InvalidImageError as e:
                # Remove the latest image so the model can repair and reload it.
                if not images.evict_newest(
                    self._run.messages, images.INVALID_IMAGE_NOTE
                ):
                    raise
                self.logger.debug(
                    "Provider rejected the newest attached image; removed it "
                    "and informed the model."
                )
                self._inject_user(
                    prompts.INVALID_IMAGE_MESSAGE.format(error=e), "invalid_image"
                )
                continue
            except llm.RequestTooLargeError as e:
                # Remove newest images first to preserve more of the prompt cache.
                if not images.evict_newest(self._run.messages):
                    raise
                self.logger.debug(
                    "Request exceeded the provider body cap; evicted the "
                    "newest attached image and informed the model."
                )
                # Reuse the recovery note across consecutive failures.
                self._inject_user(
                    prompts.REQUEST_TOO_LARGE_MESSAGE.format(error=e),
                    "request_too_large",
                    replace_same_kind=True,
                )
                continue
            except Exception as e:
                budget = self.config.time_budget_sec
                if budget and self._elapsed() >= budget and llm._is_retryable(e):
                    lost = self._llm.provider_lost_sec
                    if lost <= PROVIDER_LOSS_GRADED_FRACTION * budget:
                        # A run that lost little to the provider is graded on
                        # its restored winner, as if the budget ran out.
                        self.logger.warning(
                            "Budget expired during provider retries; ending the "
                            f"run ({lost:.0f}s lost to provider errors)."
                        )
                        self._end_by_provider(
                            {"kind": "retry_budget", "error_type": type(e).__name__}
                        )
                        return
                    # Starved by the provider: fail, so the trial is re-run.
                    self.logger.debug(
                        "Budget expired during provider retries; failing run."
                    )
                    raise llm.ProviderRetryBudgetError(
                        "Budget expired during provider retries "
                        f"({lost:.0f}s lost to provider errors)"
                    ) from e
                raise

            self._run.consecutive_refusals = 0
            self._accumulate_usage(parsed)
            self._run.messages.extend(self._llm.assistant_messages(parsed))
            stalled = self._model_stalled(parsed)

            if not parsed.tool_calls:
                self._add_agent_step(parsed, [], [])
                self._dump_artifacts()
                if stalled:
                    self.logger.debug(
                        f"No output for {EMPTY_TURN_STALL_LIMIT} consecutive "
                        "turns; ending run."
                    )
                    self._run.stalled = True
                    self._run.ended_by = "stall"
                    return

                if self._run.submit_armed_turn is not None:
                    self._inject_user(
                        prompts.CONTINUE_OR_CONFIRM_MESSAGE, "continue_or_confirm"
                    )
                elif self._run.consecutive_empty_turns > 0:
                    self._inject_user(prompts.EMPTY_RESPONSE_MESSAGE, "empty_response")
                else:
                    self._inject_user(prompts.CONTINUE_MESSAGE, "continue")
                continue

            end_run = await self._execute_tool_calls(parsed)
            self._dump_artifacts()
            if end_run:
                if self._run.ended_by_submit:
                    self._run.ended_by = "submit"
                return

    def _model_stalled(self, parsed: llm.ParsedResponse) -> bool:
        """Track zero-output turns; True once EMPTY_TURN_STALL_LIMIT land in
        a row."""
        self._run.consecutive_empty_turns = (
            self._run.consecutive_empty_turns + 1 if parsed.is_empty else 0
        )
        return self._run.consecutive_empty_turns >= EMPTY_TURN_STALL_LIMIT

    async def _execute_tool_calls(self, parsed: llm.ParsedResponse) -> bool:
        """Run the turn's tool calls; True when a confirmed final submission
        (or a dead sandbox) ends the run"""
        end_run = False
        env_gone: tools.EnvironmentGoneError | None = None
        observation_results: list[trajectories.ObservationResult] = []
        atif_tool_calls: list[trajectories.ToolCall] = []
        pending_images: list[images.Attachment] = []
        saved_images: list[str] = []

        for call in parsed.tool_calls:
            disarmed = False
            if call.name != "submit" and self._run.submit_armed_turn is not None:
                self._run.submit_armed_turn = None
                disarmed = True
            tool = self._tools.get(call.name)
            if env_gone is not None:
                result = tools.ToolResult(
                    "Not executed: the sandbox became unreachable earlier in "
                    "this turn.",
                    is_error=True,
                )
            elif end_run:
                # Calls after a confirming submit must not execute, but each
                # still needs a result to keep the conversation valid.
                result = tools.ToolResult(
                    "Not executed: an earlier submit in this turn ended the run.",
                    is_error=True,
                )
            elif tool is None:
                result = tools.ToolResult(
                    f"tools.Tool '{call.name}' is not available in this session. "
                    f"Available tools: {', '.join(sorted(self._tools))}.",
                    is_error=True,
                )
            else:
                try:
                    result = await tool.call(call.raw_arguments)
                except tools.EnvironmentGoneError as e:
                    # Ends the run, but only after the batch drains so every
                    # tool_call still gets a result.
                    env_gone = e
                    result = tools.ToolResult(
                        f"tools.Tool call failed: {e}", is_error=True
                    )

            if disarmed:
                result.content += (
                    "\n\n(Note: this call cancelled your pending submission "
                    "confirmation — the recorded submission itself is kept. To "
                    "end the run, call submit and then submit again with no "
                    "other tool calls in between.)"
                )
            if self._export_notes:
                result.content += "".join(f"\n\n{n}" for n in self._export_notes)
                self._export_notes.clear()

            if call.name == "submit" and result.ends_run:
                self._run.ended_by_submit = True
            end_run = end_run or result.ends_run
            if result.image is not None:
                # Deferred: an image message must not split the contiguous
                # tool results.
                pending_images.append(result.image)
                image_index = self._run.n_images
                self._run.n_images += 1
                path = images.save(
                    result.image,
                    self.logs_dir,
                    turn=self._run.n_turns,
                    index=image_index,
                    logger=self.logger,
                )
                if path:
                    saved_images.append(path)

            # Appended per call so a mid-batch cancellation keeps the
            # completed results in the dumped state.
            self._run.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": result.content,
                }
            )
            observation_results.append(
                trajectories.ObservationResult(
                    source_call_id=call.call_id, content=result.content
                )
            )
            atif_tool_calls.append(
                trajectories.ToolCall(
                    tool_call_id=call.call_id,
                    function_name=call.name or "unknown",
                    arguments=tools.args_for_trajectory(call.raw_arguments),
                )
            )

        # end_run is set only after the drain: setting it in the except would
        # mislabel the remaining stubs as "ended by an earlier submit".
        if env_gone is not None:
            self.logger.error(
                "Environment is gone: %s. Ending the run rather than burning "
                "budget against a dead sandbox.",
                env_gone,
            )
            self._run.env_lost = True
            self._run.ended_by = "sandbox_death"
            end_run = True

        if pending_images and not end_run:
            self._run.messages.append(images.user_message(pending_images))

        self._add_agent_step(
            parsed, atif_tool_calls, observation_results, images=saved_images
        )
        if env_gone is not None:
            # The trajectory and metadata are complete. Raise after recording
            # them so Harbor marks the trial as failed.
            raise env_gone
        return end_run

    # ----- tools -----
    # Handlers delegate through self: the tool surface exists from
    # construction, while the executor and submissions bind in run().

    # ----- customization hooks -----
    # Override these in a subclass and select it with --agent module:Class,
    # the way Harbor agents are customized.

    def system_prompt(self) -> str:
        return prompts.system_prompt(self.config.task_provides_timer)

    def initial_message(
        self, instruction: str, workdir: str, environment: BaseEnvironment
    ) -> str:
        """The first user message, wrapping the (templated) task instruction."""
        return prompts.initial_user_message(
            instruction,
            workdir,
            self.config.time_budget_sec,
            environment.network_policy.network_mode,
        )

    def summary_request(self) -> str:
        """The request for a handoff summary when the conversation is compacted."""
        return prompts.SUMMARY_REQUEST

    def build_tools(self) -> dict[str, tools.Tool]:
        """The model's tools, keyed by name. Called once, at construction;
        ``disallowed_tools`` then removes tools by name. A subclass may add or
        replace tools; their handlers can run commands through
        ``self.environment`` during the run."""
        surface = [
            tools.Tool("bash", tools.BASH_DESCRIPTION, tools.BashArgs, self._bash),
            tools.Tool(
                "view_image",
                tools.VIEW_IMAGE_DESCRIPTION,
                tools.ViewImageArgs,
                self._view_image,
            ),
            tools.Tool(
                "list_submissions",
                tools.LIST_SUBMISSIONS_DESCRIPTION,
                tools.NoArgs,
                self._list_submissions,
            ),
            tools.Tool(
                "submit", tools.SUBMIT_DESCRIPTION, tools.SubmitArgs, self._submit
            ),
        ]
        if not self._llm.supports_vision():
            surface = [tool for tool in surface if tool.name != "view_image"]
        return {tool.name: tool for tool in surface}

    @property
    def environment(self) -> BaseEnvironment:
        """The task sandbox of the current run."""
        return self._environment

    # Tools a run may lack, so disallowing one is never an error: view_image is
    # dropped for non-vision models.
    optional_tools: frozenset[str] = frozenset({"view_image"})

    def _filter_tools(self, by_name: dict[str, tools.Tool]) -> dict[str, tools.Tool]:
        removed = set(self.config.disallowed_tools)
        if removed & {"bash", "submit"}:
            raise ValueError("disallowed_tools cannot remove bash or submit")
        unknown = removed - set(by_name) - self.optional_tools
        if unknown:
            raise ValueError(f"disallowed_tools names unknown tools: {sorted(unknown)}")
        return {name: tool for name, tool in by_name.items() if name not in removed}

    async def _bash(self, args: tools.BashArgs) -> tools.ToolResult:
        return await self._executor.bash(args)

    async def _view_image(self, args: tools.ViewImageArgs) -> tools.ToolResult:
        return await self._executor.view_image(args)

    async def _list_submissions(self, _: tools.NoArgs) -> tools.ToolResult:
        return tools.ToolResult(
            prompts.submissions_list(
                self._submissions.summary(), self._submissions.winner_index()
            )
        )

    async def _auto_final_submission(self) -> None:
        """Pin the end-of-run workspace as a final submission — the exact
        argless ``submit()`` path — so a run that ends without submitting
        (budget timeout, stall) still leaves its last state on the
        submissions rail: recorded in ``submissions.jsonl`` (flagged
        ``auto``), and graded only when the model recorded no submission of
        its own (``Submissions.winner_hash``). Skipped when the live tree
        already matches the winner — a clean double-submit ending must not
        grow a duplicate. Best-effort: nothing here may block the restore."""
        try:
            winner = self._submissions.winner_hash()
            if winner is not None and await self._submissions.matches_winner(
                "", winner
            ):
                self.logger.debug("End state already pinned; no auto-submission.")
                return
            submission = await self._submissions.submit("", auto=True)
            self.logger.info(
                "Auto-recorded end-of-run state as submission "
                f"#{submission.index} ({submission.hash[:12]})."
            )
        except tools.EnvironmentGoneError as e:
            self.logger.warning(f"Auto-submission skipped: sandbox is gone ({e}).")
            self._run.env_lost = True
        except submissions.SubmissionError as e:
            # Already recorded as an ok:false row by submissions.submit.
            self.logger.warning(f"Auto-submission failed: {e}")
        except Exception as e:  # noqa: BLE001 — never let it block the restore
            self.logger.warning(f"Auto-submission errored: {e}")

    async def _submit(self, args: tools.SubmitArgs) -> tools.ToolResult:
        """A submit that records arms; a second on a LATER turn confirms
        and ends the run (two submits in one turn must not self-confirm).
        A submit that fails to record neither arms nor confirms, and cancels a
        pending confirmation: nothing new was recorded, so the model must not
        end the run on it. A confirm naming the already-recorded commit REUSES
        that record: re-recording would pin the just-emptied staging dir as the
        newest winner, which is what the exit restore hands the verifier. A
        confirm naming a different commit is new work: it records and ends the
        run."""
        run = self._run
        confirmed = (
            run.submit_armed_turn is not None and run.submit_armed_turn < run.n_turns
        )
        if confirmed:
            winner = self._submissions.winner_hash()
            if winner is not None and await self._submissions.matches_winner(
                args.commit, winner
            ):
                index = self._submissions.winner_index()
                self.logger.info(f"Confirm reuses submission #{index}; run ends.")
                return tools.ToolResult(prompts.SUBMIT_CONFIRMED, ends_run=True)
        try:
            submission = await self._submissions.submit(args.commit)
        except submissions.SubmissionError as e:
            self.logger.warning(f"Submission recording failed: {e}")
            run.submit_armed_turn = None
            return tools.ToolResult(
                prompts.submission_retry(
                    str(e),
                    self._submissions.winner_index() or 0,
                    self._elapsed(),
                    self.config.time_budget_sec,
                )
            )
        self._schedule_export(submission.index, submission.hash)
        # Numbered like list_submissions' rows, failed calls included.
        note = prompts.submission_note(
            submission.index, submission.hash, submission.description
        )
        if confirmed:
            return tools.ToolResult(f"{note} {prompts.SUBMIT_CONFIRMED}", ends_run=True)
        run.submit_armed_turn = run.n_turns
        guidance = prompts.submit_guidance(self._elapsed(), self.config.time_budget_sec)
        return tools.ToolResult(f"{note} {guidance}")

    # ----- submission export -----

    def _schedule_export(self, index: int, commit: str) -> None:
        """Export submission #index off the sandbox in the background; while
        one export runs, only the newest waiting submission follows it."""
        if not self.config.submission_export or not commit:
            return
        self._export_pending = (index, commit)
        if self._export_task is None or self._export_task.done():
            self._export_task = asyncio.create_task(self._export_loop())

    async def _export_loop(self) -> None:
        while self._export_pending is not None:
            index, commit = self._export_pending
            self._export_pending = None
            try:
                manifest = await self._submissions.export(
                    index,
                    commit,
                    self.logs_dir / EXPORT_DIR_NAME,
                    self.config.submission_export_max_mb,
                )
            except tools.EnvironmentGoneError:
                return
            except Exception as e:  # noqa: BLE001 — an export never fails the run
                self.logger.warning(f"Export of submission #{index} errored: {e}")
                self._export_notes.append(
                    prompts.export_failed_note(index, f"{type(e).__name__}: {e}")
                )
                continue
            if manifest.get("ok"):
                self.logger.info(
                    f"Exported submission #{index}: {manifest.get('files')} file(s), "
                    f"{manifest.get('bytes')} bytes."
                )
            else:
                self.logger.warning(
                    f"Submission #{index} not exported: {manifest.get('reason')}"
                )
                self._export_notes.append(
                    prompts.export_failed_note(
                        index,
                        str(manifest.get("reason") or "unknown error"),
                        too_large=manifest.get("exit_code") == EXPORT_TOO_LARGE_EXIT,
                    )
                )

    async def _finish_exports(self) -> None:
        """Let an in-flight export finish (bounded), unless the sandbox is gone
        or Harbor is cancelling the run (its timeout must not wait on it)."""
        task = self._export_task
        if task is None or task.done():
            return
        if self._run.env_lost or self._run.ended_by in ("sandbox_death", "cancelled"):
            # Cancelled exports clean up locally only; give that a moment.
            task.cancel()
            await asyncio.wait({task}, timeout=5)
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), EXPORT_FINALIZE_WAIT_SEC)
        except TimeoutError:
            self.logger.warning("Submission export did not finish; abandoned.")
            task.cancel()
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Submission export failed at the end: {e}")

    # ----- context management -----
    # The summary request must look byte-identical to a normal turn (same
    # tools, no tool_choice) or the provider's cached prefix is invalidated.
    # History rewrites happen only where that prefix is rebuilt anyway.

    async def _compact(self) -> None:
        """Replace older history with a handoff message, keeping the recent
        tail verbatim (minus any tool results the cut stranded)."""
        summary = await self._handoff_summary()
        progress = await self._read_progress()
        recorded = (
            prompts.submissions_list(
                self._submissions.summary(), self._submissions.winner_index()
            )
            if hasattr(self, "_submissions")
            else None
        )
        handoff = prompts.handoff_message(
            self._instruction,
            summary,
            progress,
            self._elapsed(),
            self.config.time_budget_sec,
            submissions=recorded,
        )

        messages = self._run.messages
        tail = messages[max(2, len(messages) - KEEP_RECENT_MESSAGES) :]
        messages[:] = [
            messages[0],
            {"role": "user", "content": handoff},
            *llm.drop_orphan_tool_results(tail),
        ]
        self._run.last_context_tokens = None

        self.logger.debug("Compaction.")
        self._add_step(
            "system",
            "Context compacted; conversation history replaced with a handoff summary.",
            extra={"context_management": {"type": "compaction"}},
        )
        self._add_step("user", handoff)
        self._dump_artifacts()

    async def _handoff_summary(self) -> str | None:
        """Ask the model for a handoff summary. The prompt says not to call
        tools; if the model calls one anyway, ask again. On overflow, halve
        the history and retry (the last halving still gets its attempt)."""
        request: llm.Message = {"role": "user", "content": self.summary_request()}
        for attempt in range(3):
            try:
                parsed = await self._llm.query(
                    self._run.messages + [request], tools=self._tool_schemas
                )
                self._accumulate_usage(parsed)
                if parsed.content.strip():
                    return parsed.content
                if not parsed.tool_calls:
                    return None
            except ContextLengthExceededError:
                if attempt == 2:
                    return None
                self._drop_middle()
            except Exception as e:
                self.logger.warning(f"Handoff summary generation failed: {e}")
                return None
        return None

    async def _read_progress(self) -> str | None:
        """Tail of the model's append-only durable notes: an overgrown file
        needs its latest state, not its opening plan."""
        try:
            result = await self._environment.exec(
                f"tail -c {PROGRESS_REINJECT_MAX_BYTES} {prompts.PROGRESS_FILE} 2>/dev/null",
                timeout_sec=30,
            )
        except Exception as e:
            self.logger.debug(f"PROGRESS.md re-injection read failed: {e}")
            return None
        return str(result.stdout or "").strip() or None

    def _drop_middle(self) -> None:
        """Halve the history between the head (system prompt + task message)
        and the recent tail. Runs only on overflow, where the cached prefix
        is already lost."""
        messages = self._run.messages
        if len(messages) <= 4:
            return
        tail = messages[(len(messages) + 2) // 2 :]
        self._run.messages[:] = [
            messages[0],
            messages[1],
            {
                "role": "user",
                "content": "[Earlier context omitted to fit the context window.]",
            },
            *llm.drop_orphan_tool_results(tail),
        ]

    # ----- run setup helpers -----
    # Sandbox-side setup lives in run(), not setup(): each agent phase runs
    # under its own user (dir ownership must match the user the model's bash
    # runs as) and failures here are best-effort, never trial-fatal.

    async def _resolve_workdir(self, environment: BaseEnvironment) -> str:
        """Resolve the task working directory. A run without one can never
        submit, so failing here — at setup, clearly attributed — beats
        burning the budget on a run with no voluntary exit."""
        workdir = environment.task_env_config.workdir
        if workdir:
            return workdir
        result = await environment.exec("pwd", timeout_sec=10)
        workdir = str(result.stdout or "").strip()
        if not workdir:
            raise RuntimeError(
                "could not resolve the task working directory: task.toml "
                "declares no [environment].workdir and `pwd` in the sandbox "
                "returned nothing"
            )
        return workdir

    async def _ensure_progress_dir(self, environment: BaseEnvironment) -> None:
        """Pre-create the durable-notes directory so the model's first
        append to PROGRESS.md cannot fail on a missing parent, and find out
        whether bash calls can run in their own session (so a timeout can
        kill everything the command started; see tools.FG_SESSION_PROBE)."""
        self._foreground_sessions = False
        try:
            result = await environment.exec(
                f"mkdir -p {prompts.PROGRESS_FILE.parent}; {tools.FG_SESSION_PROBE}",
                timeout_sec=30,
            )
            self._foreground_sessions = "fg-sessions-ok" in str(result.stdout or "")
        except Exception as e:  # noqa: BLE001 - backends raise varied types
            self.logger.debug(
                f"Could not pre-create {prompts.PROGRESS_FILE.parent}: {e}"
            )
        if not self._foreground_sessions:
            self.logger.warning(
                "bash calls run without their own session (no setsid?): a "
                "timed-out command's child processes cannot be killed"
            )

    async def _init_submissions(
        self, environment: BaseEnvironment, workdir: str
    ) -> None:
        """Bind the submission store and make sure the workspace is a git
        repository (initializing one when the task image ships none)."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._submissions = submissions.Submissions(
            self.logs_dir,
            environment,
            workdir=workdir,
            elapsed_sec=self._elapsed,
            logger=self.logger,
            git_excludes=self.config.workspace_git_excludes,
            submission_timeout_sec=self.config.submission_timeout_sec,
        )
        await self._submissions.ensure_git()

    async def _arm_timer(self, environment: BaseEnvironment) -> None:
        """Write the sandbox deadline file and re-anchor any task-side
        timers. With ``task_provides_timer`` the task's clock is the ONLY
        model-facing one — no deadline file, no prompt bullet, so the model
        hears a single timer story — but the re-anchor still runs: the task
        image's countdown measures the sandbox's hard timeout from container
        start, and only the reseed makes it mean "seconds until this run
        ends". A run without a time budget arms nothing."""
        remaining = self._remaining_budget_sec()
        if remaining is None:
            return
        if not self.config.task_provides_timer:
            await timer.write_deadline(environment, int(remaining), self.logger)
        # Seed with REMAINING, not the full budget: on any arm after time has
        # already passed, the full budget would over-promise.
        await timer.reanchor(environment, int(remaining), self.logger)

    def _seed_conversation(
        self,
        instruction: str,
        workdir: str,
        environment: BaseEnvironment,
    ) -> None:
        """Open the conversation: system prompt plus the initial task
        message, mirrored into the trajectory."""
        initial = self.initial_message(instruction, workdir, environment)
        self._run.messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": initial},
        ]
        self._add_step("system", self._system_prompt)
        self._add_step("user", initial)

    # ----- bookkeeping -----

    def _elapsed(self) -> float:
        return time.monotonic() - self._run.start_monotonic

    def _remaining_budget_sec(self) -> float | None:
        if not self.config.time_budget_sec:
            return None
        return max(0.0, self.config.time_budget_sec - self._elapsed())

    def _budget_spent(self) -> bool:
        budget = self.config.time_budget_sec
        return bool(budget) and self._elapsed() >= budget

    def _end_by_provider(self, failure: dict[str, Any]) -> None:
        """End the loop normally after a provider failure: finalize then
        restores the winner and Harbor grades it."""
        self._run.provider_failure = {
            **failure,
            "at_sec": round(self._elapsed(), 1),
            "provider_lost_sec": round(self._llm.provider_lost_sec, 1),
        }
        self._run.ended_by = "provider_failure"
        self._add_step(
            "system",
            f"Run ended by a provider failure ({failure.get('kind')}); the "
            "restored winner is graded.",
            extra={"provider_failure": self._run.provider_failure},
        )
        self._dump_artifacts()

    def _record_provider_event(self, event: dict[str, Any]) -> None:
        """A failed provider attempt, as a system step (the model never sees
        it): its error, how long it took and the budget left."""
        if not hasattr(self, "_steps"):
            return
        self._add_step(
            "system",
            f"Provider call {event.get('kind')}: {event.get('error_type')} after "
            f"{event.get('attempt_sec')}s; next attempt in {event.get('sleep_sec')}s.",
            extra={"provider_event": event},
        )
        self._dump_artifacts()

    def _record_discard(self, error: OutputLengthExceededError) -> None:
        """Count a discarded over-long response's usage and log what was
        discarded (the model is told only that it was)."""
        parsed = getattr(error, "parsed", None)
        if isinstance(parsed, llm.ParsedResponse):
            self._accumulate_usage(parsed, kept=False)
        text = getattr(error, "truncated_response", None) or ""
        half = DISCARD_EXCERPT_CHARS // 2
        excerpt = (
            text
            if len(text) <= DISCARD_EXCERPT_CHARS
            else (
                f"{text[:half]}\n[... {len(text) - DISCARD_EXCERPT_CHARS} chars ...]\n"
                f"{text[-half:]}"
            )
        )
        self._add_step(
            "system",
            "Response discarded: it hit the maximum output length.",
            extra={
                "discarded_output": {
                    "chars": len(text),
                    "completion_tokens": getattr(parsed, "completion_tokens", None),
                    "cost_usd": getattr(parsed, "cost_usd", None),
                    "excerpt": excerpt,
                }
            },
        )

    def _withhold_newest_output(self) -> bool:
        """Replace the newest turn's tool results with a note, after a provider
        refusal (the latest output most likely tripped its classifier). False
        when there is nothing to withhold."""
        messages = self._run.messages
        last_call = next(
            (
                i
                for i in range(len(messages) - 1, -1, -1)
                if messages[i].get("role") == "assistant"
                and messages[i].get("tool_calls")
            ),
            None,
        )
        if last_call is None:
            return False
        withheld = 0
        for message in messages[last_call + 1 :]:
            if (
                message.get("role") == "tool"
                and message.get("content") != prompts.WITHHELD_OUTPUT
            ):
                message["content"] = prompts.WITHHELD_OUTPUT
                withheld += 1
        if withheld:
            self._add_step(
                "system",
                f"Withheld {withheld} tool output(s) after a provider refusal.",
                extra={"withheld_outputs": withheld},
            )
        return withheld > 0

    def _accumulate_usage(
        self, parsed: llm.ParsedResponse, *, kept: bool = True
    ) -> None:
        """Add a response's tokens and cost. ``kept=False`` for a discarded
        response: billed, but its completion is not in the context."""
        run = self._run
        run.total_prompt_tokens += parsed.prompt_tokens
        run.total_completion_tokens += parsed.completion_tokens
        run.total_cached_tokens += parsed.cached_tokens
        run.total_cost_usd += parsed.cost_usd
        if parsed.fallback_attempted:
            run.fallback_attempted_turns += 1
        if parsed.fallback_used:
            run.fallback_turns += 1
            model = parsed.response_model or "unknown"
            run.fallback_models[model] = run.fallback_models.get(model, 0) + 1
        if parsed.prompt_tokens:
            self._run.last_context_tokens = parsed.prompt_tokens + (
                parsed.completion_tokens if kept else 0
            )

    def _populate_context(self, context: AgentContext) -> None:
        run = self._run
        context.n_input_tokens = run.total_prompt_tokens
        context.n_output_tokens = run.total_completion_tokens
        context.n_cache_tokens = run.total_cached_tokens
        context.cost_usd = run.total_cost_usd if run.total_cost_usd > 0 else None
        has_store = hasattr(self, "_submissions")
        context.metadata = {
            "n_turns": run.n_turns,
            "elapsed_sec": round(self._elapsed(), 1),
            "n_submissions": self._submissions.valid_hash_count() if has_store else 0,
            "submissions": self._submissions.summary() if has_store else [],
            # The commit grading checks out; None = score the workspace as-is.
            "submitted_commit": self._submissions.winner_hash() if has_store else None,
            "ended_by_submit": run.ended_by_submit,
            "ended_by": run.ended_by,
            "provider_failure": run.provider_failure,
            "provider_lost_sec": round(self._llm.provider_lost_sec, 1),
            "env_lost": run.env_lost,  # cut short by a dead sandbox
            "stalled": run.stalled,  # ended by the empty-turn breaker
            "fallback_attempted_turns": run.fallback_attempted_turns,
            "fallback_turns": run.fallback_turns,
            "fallback_models": dict(run.fallback_models),
            **self._bash_timeout_metadata(),
        }

    def _bash_timeout_metadata(self) -> dict[str, Any]:
        executor = getattr(self, "_executor", None)
        if executor is None:
            return {}
        return {
            "foreground_sessions": getattr(self, "_foreground_sessions", False),
            "bash_timeouts": executor.n_timeouts,
            "bash_timeout_killed": executor.n_timeout_killed,
            "bash_timeout_survivors": executor.n_timeout_survivors,
        }

    def _inject_user(
        self, content: str, kind: str, *, replace_same_kind: bool = False
    ) -> None:
        """Send the model a harness message and record it as a user step, so
        the trajectory shows everything the model was told. With
        ``replace_same_kind``, a message of the same kind that is still the
        last one is overwritten, with its step, instead of appending one; any
        other note (an invalid image's guidance) stays."""
        last_step = self._steps[-1] if self._steps else None
        last = self._run.messages[-1] if self._run.messages else {}
        if (
            replace_same_kind
            and last_step is not None
            and (last_step.extra or {}).get("harness_message") == kind
            and last.get("role") == "user"
            and last.get("content") == last_step.message
        ):
            last["content"] = content
            self._steps[-1] = last_step.model_copy(
                update={"message": content, "timestamp": datetime.now(UTC).isoformat()}
            )
            return
        self._run.messages.append({"role": "user", "content": content})
        self._add_step("user", content, extra={"harness_message": kind})

    def _add_step(
        self, source: Literal["system", "user", "agent"], message: str, **kwargs: Any
    ) -> None:
        self._steps.append(
            trajectories.Step(
                step_id=len(self._steps) + 1,
                timestamp=datetime.now(UTC).isoformat(),
                source=source,
                message=message,
                **kwargs,
            )
        )

    def _add_agent_step(
        self,
        parsed: llm.ParsedResponse,
        tool_calls: list[trajectories.ToolCall],
        observation_results: list[trajectories.ObservationResult],
        images: list[str] | None = None,
    ) -> None:
        metrics = None
        if parsed.prompt_tokens or parsed.completion_tokens or parsed.reasoning_tokens:
            metrics = trajectories.Metrics(
                prompt_tokens=parsed.prompt_tokens or None,
                completion_tokens=parsed.completion_tokens or None,
                cached_tokens=parsed.cached_tokens or None,
                cost_usd=parsed.cost_usd or None,
                extra=(
                    {"thinking_tokens": parsed.reasoning_tokens}
                    if parsed.reasoning_tokens
                    else None
                ),
            )
        extra: dict[str, Any] = {
            "thinking": {
                "summary": parsed.reasoning_content,
                "tokens": parsed.reasoning_tokens,
                "display": (parsed.request_thinking or {}).get("display"),
            },
            "model_routing": {
                "requested_model": self.model_name,
                "response_model": parsed.response_model,
                "fallback_attempted": parsed.fallback_attempted,
                "fallback_used": parsed.fallback_used,
                "iterations": parsed.fallback_iterations,
                "stop_reason": parsed.stop_reason,
                "stop_details": parsed.stop_details,
                "request_thinking": parsed.request_thinking,
                "request_output_config": parsed.request_output_config,
                "request_fallbacks": parsed.request_fallbacks,
            },
        }
        if images:
            extra["images"] = images
        self._add_step(
            "agent",
            parsed.content or "",
            model_name=parsed.response_model or self.model_name,
            reasoning_effort=self.config.reasoning_effort,
            reasoning_content=parsed.reasoning_content or None,
            tool_calls=tool_calls or None,
            observation=(
                trajectories.Observation(results=observation_results)
                if observation_results
                else None
            ),
            metrics=metrics,
            llm_call_count=1,
            extra=extra,
        )

    def _dump_artifacts(self) -> None:
        run = self._run
        if self._steps:
            trajectory = trajectories.Trajectory(
                session_id=run.session_id or None,
                agent=trajectories.Agent(
                    name=self.name(),
                    version=self.version() or "unknown",
                    model_name=self.model_name,
                    tool_definitions=self._tool_schemas,
                ),
                steps=list(self._steps),
                final_metrics=trajectories.FinalMetrics(
                    total_prompt_tokens=run.total_prompt_tokens or None,
                    total_completion_tokens=run.total_completion_tokens or None,
                    total_cached_tokens=run.total_cached_tokens or None,
                    total_cost_usd=(
                        run.total_cost_usd if run.total_cost_usd > 0 else None
                    ),
                    total_steps=len(self._steps),
                ),
            )
            utils.write_artifact(
                self.logs_dir,
                "trajectory.json",
                format_trajectory_json(trajectory.to_json_dict()),
                self.logger,
            )
        utils.write_artifact(
            self.logs_dir,
            "messages.json",
            json.dumps(run.messages, indent=2, default=str),
            self.logger,
        )
