"""The agent Harbor runs: the Proximus core (proximus.py) plus the fork's additions.

Fixes to the core's behaviour go into proximus.py; options and tools that upstream
Proximus does not have go here, so the core stays easy to merge with upstream.
"""

from pathlib import Path
from typing import Any, override

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

import openworld_harness.sandbox_git as sandbox_git
from openworld_harness.config import OpenWorldConfig
from openworld_harness.proximus import Proximus


class OpenWorldHarness(Proximus):
    """Long-horizon, native-function-calling agent harness."""

    options_model = OpenWorldConfig
    config: OpenWorldConfig

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
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        try:
            await super().run(instruction, environment, context)
        finally:
            # Also when the run fails before the core populates the context.
            excludes = self._platform_git_excludes
            context.metadata = {
                **(context.metadata or {}),
                "platform_git": self._platform_git,
                "platform_git_excludes": None if excludes is None else len(excludes),
            }
