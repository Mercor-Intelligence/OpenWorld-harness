"""The bash tool's foreground sessions, run for real against local bash.

A backend timeout (Modal, Harbor's Docker exec) kills only the exec's top
process. ModalLikeEnv does the same, so these tests show what the wrapper and
kill_session.sh add: every process the timed-out command started is gone.
Each test uses its own fg directory and unique sleep durations, and kills
only the sessions its own wrapper recorded.
"""

import asyncio
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from harbor.environments.base import ExecResult

from openworld_harness.tools import (
    FG_SESSION_PROBE,
    BashArgs,
    ToolExecutor,
    foreground_command,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        shutil.which("setsid") is None or not Path("/proc/self/stat").exists(),
        reason="needs util-linux setsid and /proc",
    ),
]


class ModalLikeEnv:
    """Runs commands with local bash. On a timeout it kills only the top
    process and returns -1 without waiting for the output pipes, as Modal's
    exec does; stdin is a pipe that is never written or closed."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.calls.append(
            {"command": command, "user": user, "timeout_sec": timeout_sec}
        )
        return await asyncio.to_thread(self._run, command, cwd, env, timeout_sec)

    @staticmethod
    def _run(command, cwd, env, timeout_sec) -> ExecResult:
        proc = subprocess.Popen(
            ["bash", "-c", command],
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            out, err = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()  # the top process only, like Modal
            proc.wait()
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe is not None:
                    pipe.close()
            return ExecResult(stdout="", stderr="", return_code=-1)
        return ExecResult(
            stdout=out.decode(), stderr=err.decode(), return_code=proc.returncode
        )


def processes_with(marker: str) -> list[str]:
    """PIDs whose command line contains ``marker`` (a unique sleep length)."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            state = (entry / "stat").read_text().rsplit(") ", 1)[1].split()[0]
        except (OSError, IndexError):
            continue
        if marker in cmdline and state not in ("Z", "X"):
            found.append(entry.name)
    return found


def kill_marked(marker: str) -> None:
    for pid in processes_with(marker):
        try:
            os.kill(int(pid), 9)
        except ProcessLookupError:
            pass


@pytest.fixture
def fg_dir(tmp_path):
    return str(tmp_path / "fg")


def marker() -> str:
    # A sleep length no other process uses, e.g. 987.1234567.
    return f"9{uuid.uuid4().int % 10**7:07d}.{uuid.uuid4().int % 10**6:06d}"


async def run_both(command: str, fg_dir: str) -> tuple[ExecResult, ExecResult]:
    env = ModalLikeEnv()
    plain = await env.exec(command, timeout_sec=30)
    wrapped = await env.exec(
        foreground_command(command, uuid.uuid4().hex[:16], None, fg_dir), timeout_sec=30
    )
    return plain, wrapped


@pytest.mark.parametrize(
    "command",
    [
        "echo hello",
        "printf '%s|' \"a b\" 'c\"d' `echo e` $((1+2)) '$HOME' \\\\; echo",
        "cat <<'EOF'\nline 1 $HOME\n  'quoted' \"dq\" `bt`\nEOF",
        "cat <<EOF\nexpanded $((6*7))\nEOF",
        "exit 7",
        "set -e; false; echo not reached",
        "echo out; echo err >&2; exit 3",
        "pwd",
        "f() { echo fn $1; }; f arg",
        "echo ~ | grep -c /",
        "x=$(cat); echo read:[$x]",  # stdin is /dev/null: no blocking
        # No extra descriptor (the supervisor's copy of stderr) leaks in.
        "for fd in 3 4 5; do [ -e /proc/$$/fd/$fd ] && echo leak $fd; done; echo done",
    ],
)
async def test_wrapper_matches_plain_bash(command, fg_dir):
    plain, wrapped = await run_both(command, fg_dir)
    if "cat)" in command:
        # Plain bash would block on the never-closed stdin pipe; the wrapper
        # gives the command /dev/null instead.
        assert wrapped.stdout == "read:[]\n"
        assert wrapped.return_code == 0
        return
    assert (wrapped.stdout, wrapped.stderr, wrapped.return_code) == (
        plain.stdout,
        plain.stderr,
        plain.return_code,
    )


async def test_signal_death_reads_as_137_without_noise(fg_dir):
    env = ModalLikeEnv()
    result = await env.exec(
        foreground_command("kill -9 $$", "t1", None, fg_dir), timeout_sec=30
    )
    assert result.return_code == 137
    assert result.stderr == ""


async def test_command_text_is_not_in_any_argv(fg_dir):
    mark = marker()
    env = ModalLikeEnv()
    command = f"tr '\\0' ' ' < /proc/$PPID/cmdline; echo; echo {mark}"
    result = await env.exec(
        foreground_command(command, "t2", None, fg_dir), timeout_sec=30
    )
    # The parent of the command's shell is the session leader `sh -c ...`
    # (exec'd into bash): its argv must not carry the command text, or a
    # `pkill -f <pattern>` in the command would kill its own wrapper.
    first_line = result.stdout.splitlines()[0]
    assert mark not in first_line


async def test_probe_reports_working_sessions():
    env = ModalLikeEnv()
    result = await env.exec(FG_SESSION_PROBE, timeout_sec=30)
    assert "fg-sessions-ok" in result.stdout


async def test_timeout_kills_the_whole_session(fg_dir):
    a, b, c = marker(), marker(), marker()
    env = ModalLikeEnv()
    executor = ToolExecutor(env, foreground_sessions=True, fg_dir=fg_dir)  # type: ignore[arg-type]
    try:
        result = await executor.bash(
            BashArgs(
                command=(
                    f"timeout 300 sleep {a} & "
                    f"nohup sleep {b} >/dev/null 2>&1 & "
                    f"sleep {c}"
                ),
                timeout_sec=2,
            )
        )
        assert result.is_error
        assert "timed out" in result.content
        assert "every process it started" in result.content
        # Modal-like timeout killed only the top process: without the session
        # kill, all three sleeps (and `timeout`) would still be running.
        for m in (a, b, c):
            assert processes_with(m) == [], m
        killer = env.calls[-1]
        assert killer["user"] == "root"
        assert executor.n_timeouts == 1
        assert executor.n_timeout_killed >= 4
        assert executor.n_timeout_survivors == 0
    finally:
        for m in (a, b, c):
            kill_marked(m)


async def test_background_job_holding_output_is_explained(fg_dir):
    mark = marker()
    env = ModalLikeEnv()
    executor = ToolExecutor(env, foreground_sessions=True, fg_dir=fg_dir)  # type: ignore[arg-type]
    try:
        result = await executor.bash(
            BashArgs(command=f"sleep {mark} & echo started", timeout_sec=2)
        )
        assert "had finished (exit 0)" in result.content
        assert "nohup cmd > /tmp/cmd.log 2>&1 &" in result.content
        assert processes_with(mark) == []
    finally:
        kill_marked(mark)


async def test_redirected_background_job_survives_a_normal_finish(fg_dir):
    mark = marker()
    env = ModalLikeEnv()
    executor = ToolExecutor(env, foreground_sessions=True, fg_dir=fg_dir)  # type: ignore[arg-type]
    try:
        # A short timeout: a leaked copy of the exec's pipes would make the
        # call wait for the background job and time out here.
        result = await executor.bash(
            BashArgs(
                command=f"nohup sleep {mark} >/dev/null 2>&1 & echo started",
                timeout_sec=20,
            )
        )
        assert result.content.startswith("exit code: 0")
        assert result.exit_code == 0
        assert len(processes_with(mark)) == 1
        # The next call removes the previous call's session files.
        await executor.bash(BashArgs(command="true"))
        assert sorted(os.listdir(fg_dir)) == sorted(
            f"{executor._previous_token}.{ext}" for ext in ("sid", "rc")
        )
    finally:
        kill_marked(mark)


async def test_a_symlinked_session_file_is_refused(fg_dir, tmp_path):
    # kill_session.sh runs as root on files the command's user can write: a
    # .sid that is a symlink (or has no usable owner) kills nothing.
    mark = marker()
    victim = subprocess.Popen(["sleep", mark], start_new_session=True)
    try:
        os.makedirs(fg_dir, exist_ok=True)
        target = tmp_path / "elsewhere"
        target.write_text(f"{victim.pid}\n")
        os.symlink(target, os.path.join(fg_dir, "tok.sid"))
        env = ModalLikeEnv()
        executor = ToolExecutor(env, foreground_sessions=True, fg_dir=fg_dir)  # type: ignore[arg-type]
        cleanup = await executor._kill_session("tok")
        assert cleanup.sid is None and cleanup.killed == 0
        assert victim.poll() is None
    finally:
        victim.kill()
        victim.wait()


async def test_only_the_session_file_owners_processes_are_killed(fg_dir):
    # The owner is the current user here, so its own session is killed; the
    # uid filter is what keeps a forged sid from reaching other users.
    mark = marker()
    victim = subprocess.Popen(["sleep", mark], start_new_session=True)
    try:
        os.makedirs(fg_dir, exist_ok=True)
        Path(fg_dir, "tok.sid").write_text(f"{victim.pid}\n")
        env = ModalLikeEnv()
        executor = ToolExecutor(env, foreground_sessions=True, fg_dir=fg_dir)  # type: ignore[arg-type]
        cleanup = await executor._kill_session("tok")
        assert cleanup.sid == str(victim.pid) and cleanup.killed == 1
        assert victim.wait(timeout=10) == -9
    finally:
        if victim.poll() is None:
            victim.kill()
            victim.wait()
