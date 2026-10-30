"""Setup steps for git-less images and large workspace inputs (sandbox_git)."""

import subprocess
from pathlib import Path
from typing import Any

import pytest
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext

from openworld_harness import OpenWorldHarness
from openworld_harness.sandbox_git import (
    SANDBOX_GIT,
    SANDBOX_TEMPLATES,
    ensure_git,
    gitignore_literal,
    large_inputs,
)

pytestmark = pytest.mark.unit


class FakeEnvironment:
    """Records exec/upload calls; `git` exists once installed (or from the start)."""

    def __init__(
        self, has_git: bool, version_rc: int = 0, large: str = "", repo: bool = False
    ) -> None:
        self.has_git = has_git
        self.version_rc = version_rc
        self.large = large  # `find` output
        self.repo = repo  # the workspace is already a git repository
        self.calls: list[tuple[str, Any]] = []

    async def exec(self, command: str, timeout_sec: int | None = None, user=None):
        self.calls.append(("exec", (command, user)))
        if command == "command -v git":
            return ExecResult(return_code=0 if self.has_git else 1)
        if command.endswith("git --version"):
            return ExecResult(stderr="exec format error", return_code=self.version_rc)
        if " find " in command:
            return ExecResult(stdout=self.large, return_code=3 if self.repo else 0)
        return ExecResult(return_code=0)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.calls.append(("upload", (Path(source_path), target_path)))
        self.has_git = True


@pytest.fixture
def static_git(tmp_path: Path) -> Path:
    path = tmp_path / "git"
    path.write_bytes(b"\x7fELF")
    return path


async def test_image_git_is_left_alone(static_git: Path) -> None:
    env = FakeEnvironment(has_git=True)
    assert await ensure_git(env, static_git) == "present"
    assert env.calls == [("exec", ("command -v git", "root"))]


async def test_static_git_is_installed_when_missing(static_git: Path) -> None:
    env = FakeEnvironment(has_git=False)
    assert await ensure_git(env, static_git) == "installed"
    assert env.calls == [
        ("exec", ("command -v git", "root")),
        ("exec", (f"mkdir -p /usr/local/bin {SANDBOX_TEMPLATES}", "root")),
        ("upload", (static_git, SANDBOX_GIT)),
        ("exec", (f"chmod 0755 {SANDBOX_GIT} && git --version", "root")),
    ]


async def test_install_failures_are_explicit(tmp_path: Path, static_git: Path) -> None:
    with pytest.raises(RuntimeError, match="is missing"):
        await ensure_git(FakeEnvironment(has_git=False), tmp_path / "absent")
    with pytest.raises(RuntimeError, match="exec format error"):
        await ensure_git(FakeEnvironment(has_git=False, version_rc=126), static_git)


def test_gitignore_literal_escapes_wildmatch() -> None:
    assert gitignore_literal("base-model/model.safetensors") == (
        "/base-model/model.safetensors"
    )
    assert gitignore_literal("data/#1 [a]*?.bin") == "/data/\\#1 \\[a\\]\\*\\?.bin"
    assert gitignore_literal("!x") == "/\\!x"


async def test_large_inputs() -> None:
    env = FakeEnvironment(
        has_git=True, large="./base-model/model.safetensors\n./data/a b.arrow\n"
    )
    assert await large_inputs(env, "/app", 100) == [
        "/base-model/model.safetensors",
        "/data/a b.arrow",
    ]
    command = env.calls[-1][1][0]
    assert "-size +102400k" in command
    repo = FakeEnvironment(has_git=True, repo=True)
    assert await large_inputs(repo, "/app", 100) is None


class ShellEnvironment:
    """Runs the command for real with POSIX sh, as a sandbox would."""

    def __init__(self, timeout_sec: int | None = None) -> None:
        self.timeout_sec = timeout_sec  # overrides the caller's, to fail fast

    async def exec(self, command: str, timeout_sec: int | None = None, user=None):
        done = subprocess.run(
            ["sh", "-c", command],
            capture_output=True,
            text=True,
            timeout=self.timeout_sec or timeout_sec,
        )
        return ExecResult(
            stdout=done.stdout, stderr=done.stderr, return_code=done.returncode
        )


async def test_large_inputs_runs_in_a_real_shell(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    for name, size in [("big.bin", 2 << 20), ("data/a b.arrow", 3 << 20)]:
        with open(tmp_path / name, "wb") as f:
            f.truncate(size)  # sparse: the size is what find tests
    (tmp_path / "small.txt").write_text("x")
    found = await large_inputs(ShellEnvironment(), str(tmp_path), 1)
    assert sorted(found or []) == ["/big.bin", "/data/a b.arrow"]
    (tmp_path / ".git").mkdir()
    assert await large_inputs(ShellEnvironment(), str(tmp_path), 1) is None
    # A workdir inside an enclosing repository is not fresh either.
    assert await large_inputs(ShellEnvironment(), str(tmp_path / "data"), 1) is None


async def test_large_inputs_walks_up_from_a_relative_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "w").mkdir()
    with open(tmp_path / "w" / "big.bin", "wb") as f:
        f.truncate(2 << 20)
    monkeypatch.chdir(tmp_path)
    assert await large_inputs(ShellEnvironment(timeout_sec=10), "w", 1) == ["/big.bin"]
    (tmp_path / ".git").mkdir()
    assert await large_inputs(ShellEnvironment(timeout_sec=10), "w", 1) is None
    assert await large_inputs(ShellEnvironment(timeout_sec=10), ".", 1) is None


async def test_a_listing_that_times_out_excludes_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Slow(FakeEnvironment):
        async def exec(self, command: str, timeout_sec=None, user=None):
            raise TimeoutError()

    assert await large_inputs(Slow(has_git=True), "/app", 100) == []
    assert "timed out" in caplog.text


async def test_a_failing_listing_keeps_what_it_found(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Partial(FakeEnvironment):
        async def exec(self, command: str, timeout_sec=None, user=None):
            return ExecResult(
                stdout="./w.bin\n", stderr="find: ./gone: No such file", return_code=1
            )

    found = await large_inputs(Partial(has_git=True), "/app", 100)
    assert found == ["/w.bin"]
    assert "No such file" in caplog.text


def make_agent(tmp_path: Path, **kwargs) -> OpenWorldHarness:
    return OpenWorldHarness(
        logs_dir=tmp_path / "logs", model_name="openai/gpt-test", **kwargs
    )


async def workdir(self: Any, environment: Any) -> str:
    return "/app"


async def test_setup_does_nothing_by_default(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    env = FakeEnvironment(has_git=False, large="w.bin\n")
    await agent.setup(env)  # type: ignore[arg-type]
    assert env.calls == []
    assert (agent._platform_git, agent._platform_git_excludes) == (None, None)


async def test_setup_installs_git_and_excludes_large_inputs(
    tmp_path: Path, static_git: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(OpenWorldHarness, "_resolve_workdir", workdir)
    agent = make_agent(
        tmp_path,
        static_git_path=str(static_git),
        large_input_exclude_mb=50,
        workspace_git_excludes=["/cache/"],
    )
    env = FakeEnvironment(has_git=False, large="./base-model/w.bin\n")
    await agent.setup(env)  # type: ignore[arg-type]
    assert agent._platform_git == "installed"
    assert agent._platform_git_excludes == ["/base-model/w.bin"]
    assert "-size +51200k" in env.calls[-1][1][0]
    assert agent.config.workspace_git_excludes == ["/cache/", "/base-model/w.bin"]


async def test_run_reports_platform_git_even_when_the_run_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_run(self: Any, instruction: str, environment: Any, context: Any):
        context.metadata = {"n_turns": 3}
        raise RuntimeError("model down")

    monkeypatch.setattr(OpenWorldHarness, "_run_task", failing_run)
    agent = make_agent(tmp_path)
    agent._platform_git = "installed"
    agent._platform_git_excludes = ["/a", "/b"]
    context = AgentContext()
    with pytest.raises(RuntimeError, match="model down"):
        await agent.run("task", FakeEnvironment(has_git=True), context)  # type: ignore[arg-type]
    assert context.metadata == {
        "n_turns": 3,
        "platform_git": "installed",
        "platform_git_excludes": 2,
    }


async def test_a_stopped_container_is_not_a_partial_listing() -> None:
    from openworld_harness.tools import EnvironmentGoneError

    class Stopped(FakeEnvironment):
        async def exec(self, command: str, timeout_sec=None, user=None):
            return ExecResult(
                stderr="executing processes for container: cannot execute in "
                'container "ta-1" in state stopped',
                return_code=137,
            )

    with pytest.raises(EnvironmentGoneError):
        await large_inputs(Stopped(has_git=True), "/app", 100)
