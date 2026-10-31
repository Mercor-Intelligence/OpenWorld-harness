"""The agent Harbor runs: the Proximus core (proximus.py) plus the fork's additions.

Fixes to the core's behaviour go into proximus.py; options and tools that upstream
Proximus does not have go here, so the core stays easy to merge with upstream.
"""

import json
import shlex
from pathlib import Path
from typing import Any, override

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

import openworld_harness.sandbox_git as sandbox_git
from openworld_harness.web.agent import WebTools

LIMITS_PATH = "/tmp/proximus/limits.json"


def sandbox_limits_note(limits: dict[str, Any]) -> str | None:
    """The first message's line on the sandbox's resources, or None."""
    parts = []
    if limits.get("cpus"):
        parts.append(f"{limits['cpus']:g} CPUs")
    memory_mb = limits.get("memory_mb")
    if memory_mb:
        parts.append(f"{memory_mb / 1024:g} GB of memory")
    if limits.get("gpus"):
        gpu = f" {limits['gpu_type']}" if limits.get("gpu_type") else ""
        parts.append(f"{limits['gpus']}{gpu} GPU(s)")
    if not parts:
        return None
    note = f"Sandbox limits: {', '.join(parts)} (also in {LIMITS_PATH})."
    if memory_mb:
        note += (
            " Going over the memory limit can kill the whole sandbox, which ends "
            "the run. `free`, `/proc/meminfo` and `nproc` may show the host's "
            "resources, not these limits, so size parallel jobs by the limits."
        )
    if limits.get("gpus"):
        note += (
            " GPU tools here report GPU processes as PID 1: never kill PID 1, "
            "which ends the sandbox."
        )
    return note


class OpenWorldHarness(WebTools):
    """Long-horizon, native-function-calling agent harness: the Proximus core,
    the web tools when ``web_snapshot_as_of`` is set (web.agent), and the setup
    steps of sandbox_git."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Set by setup(); see sandbox_git.
        self._platform_git: str | None = None
        self._platform_git_excludes: list[str] | None = None

    @staticmethod
    @override
    def name() -> str:
        return "openworld-harness"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        # As root, unlike the core's run-time setup: these install into the image
        # and read the whole workspace.
        if self.config.static_git_path:
            self._platform_git = await sandbox_git.ensure_git(
                environment, Path(self.config.static_git_path)
            )
            self.logger.info(f"git in the sandbox: {self._platform_git}")
        if self.config.large_input_exclude_mb:
            self._platform_git_excludes = await sandbox_git.large_inputs(
                environment,
                await self._resolve_workdir(environment),
                self.config.large_input_exclude_mb,
                self.config.large_input_total_mb,
            )
            # One agent instance runs one trial, so the config can carry them.
            self.config.workspace_git_excludes = [
                *self.config.workspace_git_excludes,
                *(self._platform_git_excludes or []),
            ]
            if self._platform_git_excludes:
                self.logger.info(
                    f"submissions ignore {len(self._platform_git_excludes)} "
                    "large input file(s)"
                )

    @override
    def initial_message(
        self, instruction: str, workdir: str, environment: BaseEnvironment
    ) -> str:
        message = super().initial_message(instruction, workdir, environment)
        note = sandbox_limits_note(self.config.sandbox_limits or {})
        return f"{message}\n\n{note}" if note else message

    async def _write_limits(self, environment: BaseEnvironment) -> None:
        """Best effort: the note in the first message is what counts."""
        limits = self.config.sandbox_limits
        if not limits:
            return
        payload = shlex.quote(json.dumps(limits, sort_keys=True))
        try:
            result = await environment.exec(
                f"mkdir -p /tmp/proximus && printf '%s\\n' {payload} > {LIMITS_PATH}",
                timeout_sec=30,
            )
            if result.return_code != 0:
                raise RuntimeError(f"exit {result.return_code}: {result.stderr}")
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"{LIMITS_PATH} not written: {e}")

    @override
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        try:
            await self._write_limits(environment)
            await super().run(instruction, environment, context)
        finally:
            # Also when the run fails before the core populates the context.
            excludes = self._platform_git_excludes
            context.metadata = {
                **(context.metadata or {}),
                "platform_git": self._platform_git,
                "platform_git_excludes": None if excludes is None else len(excludes),
            }
