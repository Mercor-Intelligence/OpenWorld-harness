"""The memory watchdog (memwatch.py, scripts/memwatch.sh).

The script runs for real under dash, against a fake usage file and limited to
a session of victim processes this test starts (SCOPE_SID), so it can never
pick a developer's process. The Python side runs against fake environments.
"""

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import litellm
import pytest
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext

from openworld_harness import memwatch, tools, utils
from openworld_harness.agent import sandbox_limits_note
from openworld_harness.memwatch import MemoryWatchdog, WatchdogEvent, watchdog_note
from tests.test_proximus_loop import (
    FakeCompletion,
    FakeEnv,
    bash_response,
    finish_responses,
    make_agent,
    read_messages,
)

pytestmark = pytest.mark.unit

SHELL = shutil.which("dash") or shutil.which("sh")
GIB = 1024**3
MIB = 1024**2

needs_proc = pytest.mark.skipif(
    SHELL is None or not Path("/proc/self/statm").exists(),
    reason="needs a POSIX sh and /proc",
)


# ----- the script, for real -----


class Victims:
    """Two python processes, each in its own session (so its session id is
    its pid): ``big`` holds ``big_mb`` of memory, ``small`` almost none. The
    tests scope the script to ``big``'s session only."""

    def __init__(self, big_mb: int) -> None:
        self.big = self._spawn(
            f"b = bytearray({big_mb} * 1024 * 1024)\n"
            "for i in range(0, len(b), 4096): b[i] = 1\n"
        )
        self.small = self._spawn("")

    @staticmethod
    def _spawn(setup: str) -> subprocess.Popen:
        code = f"import time\n{setup}print('ready', flush=True)\ntime.sleep(600)\n"
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            start_new_session=True,
        )
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == b"ready"
        return proc

    def kill(self) -> None:
        for proc in (self.big, self.small):
            if proc.poll() is None:
                proc.kill()
                proc.wait()


@pytest.fixture
def victims():
    made: list[Victims] = []

    def make(big_mb: int) -> Victims:
        v = Victims(big_mb)
        made.append(v)
        return v

    yield make
    for v in made:
        v.kill()


def run_script(
    state: Path,
    usage: Path,
    *,
    threshold: int,
    limit: int,
    ticks: int,
    scope: int,
    source: str | None = None,
) -> subprocess.Popen:
    script = state.parent / "memwatch.sh"
    script.write_text(utils.read_script("memwatch.sh"))
    return subprocess.Popen(
        [
            SHELL,
            str(script),
            str(state),
            str(threshold),
            str(limit),
            source or f"file:{usage}",
            str(ticks),
            str(scope),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def events(state: Path) -> list[list[str]]:
    path = state / "events.tsv"
    if not path.exists():
        return []
    return [line.split("\t") for line in path.read_text().splitlines() if line]


def wait_for(predicate, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


@needs_proc
def test_kills_the_largest_process_then_reports_nokill(tmp_path, victims):
    # Only the big victim's session is in scope; it holds ~200 MB.
    v = victims(200)
    state = tmp_path / "state"
    usage = tmp_path / "usage"
    usage.write_text(str(3 * GIB))
    limit = 2 * GIB  # min victim = limit/50 ≈ 41 MB
    proc = run_script(
        state,
        usage,
        threshold=int(limit * 0.92),
        limit=limit,
        ticks=60,
        scope=v.big.pid,
    )
    try:
        assert wait_for(lambda: v.big.poll() is not None), "big victim not killed"
        assert v.big.returncode == -signal.SIGKILL
        assert wait_for(lambda: [e[2] for e in events(state)] == ["kill", "nokill"])
        kill, nokill = events(state)
        seq, epoch, kind, used, lim, pid, rss, command = kill
        assert (seq, kind, used, lim, pid) == (
            "1",
            "kill",
            str(3 * GIB),
            str(limit),
            str(v.big.pid),
        )
        assert 190 * MIB < int(rss) < 260 * MIB
        assert "bytearray(200" in command
        assert abs(int(epoch) - time.time()) < 60
        # Nothing left in scope above the floor: reported, not killed.
        assert nokill[2] == "nokill" and nokill[0] == "2"
        assert (state / "status").read_text().startswith(f"source=file:{usage} used=")
        assert v.small.poll() is None  # out of scope, untouched
    finally:
        proc.kill()
        proc.wait()


@needs_proc
def test_below_the_threshold_nothing_happens_and_it_expires(tmp_path, victims):
    v = victims(50)
    state = tmp_path / "state"
    usage = tmp_path / "usage"
    usage.write_text(str(GIB))
    proc = run_script(
        state, usage, threshold=int(1.8 * GIB), limit=2 * GIB, ticks=2, scope=v.big.pid
    )
    assert proc.wait(timeout=15) == 0
    assert v.big.poll() is None
    assert [e[2] for e in events(state)] == ["expire"]
    assert not (state / "pid").exists()


@needs_proc
def test_one_instance_per_state_directory(tmp_path, victims):
    v = victims(10)
    state = tmp_path / "state"
    usage = tmp_path / "usage"
    usage.write_text("0")
    first = run_script(
        state, usage, threshold=GIB, limit=2 * GIB, ticks=30, scope=v.big.pid
    )
    try:
        assert wait_for(lambda: (state / "status").exists())
        second = run_script(
            state, usage, threshold=GIB, limit=2 * GIB, ticks=30, scope=v.big.pid
        )
        assert second.wait(timeout=10) == 0
        assert first.poll() is None
        assert (state / "pid").read_text().strip() == str(first.pid)
    finally:
        first.kill()
        first.wait()


@needs_proc
def test_unreadable_source_reports_none(tmp_path):
    state = tmp_path / "state"
    proc = run_script(
        state,
        tmp_path / "missing",
        threshold=GIB,
        limit=2 * GIB,
        ticks=30,
        scope=1,
    )
    assert proc.wait(timeout=10) == 0
    assert (state / "status").read_text().strip() == "source=none"
    assert not (state / "pid").exists()


@needs_proc
def test_auto_source_is_one_of_the_known_ones(tmp_path):
    state = tmp_path / "state"
    proc = run_script(
        state, tmp_path / "x", threshold=1, limit=1, ticks=0, scope=1, source="auto"
    )
    assert proc.wait(timeout=10) == 0
    source = (state / "status").read_text().split()[0]
    assert source in {
        "source=cgroup2",
        "source=cgroup1",
        "source=meminfo",
        "source=none",
    }
    if source != "source=none":
        # A real reading: positive and below a petabyte.
        used = int((state / "status").read_text().split()[1].removeprefix("used="))
        assert 0 < used < 2**50


@needs_proc
def test_cgroup2_reading_leaves_out_reclaimable_page_cache(tmp_path):
    current = Path("/sys/fs/cgroup/memory.current")
    stat = Path("/sys/fs/cgroup/memory.stat")
    if not (os.access(current, os.R_OK) and os.access(stat, os.R_OK)):
        pytest.skip("no readable cgroup v2 memory files here")
    state = tmp_path / "state"
    before = int(current.read_text())
    proc = run_script(
        state, tmp_path / "x", threshold=1, limit=1, ticks=0, scope=1, source="cgroup2"
    )
    assert proc.wait(timeout=10) == 0
    fields = dict(line.split() for line in stat.read_text().splitlines())
    used = int((state / "status").read_text().split()[1].removeprefix("used="))
    expected = before - int(fields["file"]) + int(fields["shmem"])
    # Both readings move a little between the two moments.
    assert abs(used - expected) < 256 * MIB
    assert used <= int(current.read_text()) + 256 * MIB


class LocalEnv:
    """Runs the watchdog's commands in local bash (user= is ignored)."""

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout_sec)
        return ExecResult(
            stdout=out.decode(), stderr=err.decode(), return_code=proc.returncode
        )


def loops(state: Path) -> list[int]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if f" {state}/memwatch.sh " in cmdline:
            found.append(int(entry.name))
    return found


@needs_proc
async def test_real_loop_starts_restarts_and_stops(tmp_path, victims):
    v = victims(1)
    state = tmp_path / "state"
    usage = tmp_path / "usage"
    usage.write_text("0\n")
    watchdog = make_watchdog(
        LocalEnv(),
        memory_mb=4096,
        source=f"file:{usage}",
        state_dir=str(state),
        scope_sid=v.big.pid,
    )
    try:
        assert await watchdog.start()
        (first,) = loops(state)
        assert await watchdog.check() == [] and watchdog.restarts == 0
        # A model command kills the loop and deletes its pid file.
        os.kill(first, signal.SIGKILL)
        (state / "pid").unlink(missing_ok=True)
        assert wait_for(lambda: first not in loops(state), 5)
        await watchdog.check()
        assert watchdog.restarts == 1 and watchdog.active
        (second,) = loops(state)
        assert second != first
        # A second start never leaves two loops.
        assert await watchdog.start()
        assert len(loops(state)) == 1
        await watchdog.stop()
        assert wait_for(lambda: loops(state) == [], 5)
    finally:
        for pid in loops(state):
            os.kill(pid, signal.SIGKILL)


# ----- MemoryWatchdog against a fake environment -----


class WatchEnv:
    """Answers the watchdog's three commands from scripted state."""

    def __init__(self, status: str = "source=cgroup1 used=123 page=4096") -> None:
        self.status = status
        self.alive = True
        self.lines: list[str] = []
        self.calls: list[dict] = []
        self.fail: Exception | None = None

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.calls.append({"command": command, "user": user})
        if self.fail is not None:
            raise self.fail
        if "memwatch.sh\" <<'" in command:
            self.alive = True
            return ExecResult(stdout=self.status + "\n", stderr="", return_code=0)
        since = int(command.split("since=", 1)[1].split("\n", 1)[0])
        out = [f"alive={1 if self.alive else 0}"]
        out += [
            f"event\t{line}" for line in self.lines if int(line.split("\t")[0]) > since
        ]
        if 'kill "$p"' in command:
            self.alive = False
        return ExecResult(stdout="\n".join(out) + "\n", stderr="", return_code=0)


def kill_line(
    seq: int, pid: int = 4242, rss: int = 50 * GIB, used: int = 59 * GIB
) -> str:
    return f"{seq}\t1790000000\tkill\t{used}\t{64 * GIB}\t{pid}\t{rss}\tpython3 train.py --workers 32"


def make_watchdog(env, **kwargs) -> MemoryWatchdog:
    kwargs.setdefault("memory_mb", 65536)
    kwargs.setdefault("fraction", 0.92)
    kwargs.setdefault("source", "auto")
    kwargs.setdefault("lifetime_sec", 7200)
    return MemoryWatchdog(env, logger=logging.getLogger("test"), **kwargs)


async def test_start_runs_the_script_detached_as_root_with_a_clean_env():
    env = WatchEnv()
    watchdog = make_watchdog(env, memory_mb=4096, source="cgroup1")
    assert await watchdog.start()
    (call,) = env.calls
    assert call["user"] == "root"
    command = call["command"]
    assert "env -i PATH=" in command and "$detach sh" in command
    assert "detach=setsid" in command
    threshold = int(4096 * MIB * 0.92)
    assert f"/tmp/proximus-memwatch {threshold} {4096 * MIB} cgroup1 7200" in command
    # The script travels inline in a quoted heredoc.
    assert utils.read_script("memwatch.sh").strip() in command
    assert watchdog.metadata()["source"] == "cgroup1"
    assert watchdog.metadata()["threshold_mb"] == round(threshold / MIB)


async def test_start_without_a_memory_source_leaves_it_off():
    env = WatchEnv(status="source=none")
    watchdog = make_watchdog(env)
    assert not await watchdog.start()
    assert not watchdog.active
    assert watchdog.metadata()["error"] == "no memory source"
    assert await watchdog.check() == []
    assert len(env.calls) == 1


async def test_start_failure_is_not_fatal_but_a_gone_sandbox_is():
    env = WatchEnv()
    env.fail = RuntimeError("exec failed")
    watchdog = make_watchdog(env)
    assert not await watchdog.start()
    assert "exec failed" in watchdog.metadata()["error"]
    env.fail = type("SandboxTerminatedError", (Exception,), {})("gone")
    with pytest.raises(tools.EnvironmentGoneError):
        await make_watchdog(env).start()


async def test_check_reports_each_event_once():
    env = WatchEnv()
    watchdog = make_watchdog(env)
    await watchdog.start()
    env.lines = [kill_line(1)]
    (first,) = await watchdog.check()
    assert first.pid == 4242 and first.kind == "kill"
    assert await watchdog.check() == []
    env.lines.append(f"2\t1790000060\tnokill\t{60 * GIB}\t{64 * GIB}\t0\t0\t")
    (second,) = await watchdog.check()
    assert second.kind == "nokill"
    assert watchdog.metadata()["kills"] == 1
    assert watchdog.metadata()["nokill"] == 1
    assert [e["seq"] for e in watchdog.metadata()["events"]] == [1, 2]


async def test_due_is_rate_limited():
    env = WatchEnv()
    watchdog = make_watchdog(env, check_interval_sec=3600)
    await watchdog.start()
    assert not watchdog.due()
    watchdog._last_check -= 3601
    assert watchdog.due()
    await watchdog.check()
    assert not watchdog.due()


async def test_a_killed_watchdog_is_restarted_up_to_the_cap():
    env = WatchEnv()
    watchdog = make_watchdog(env, max_restarts=2)
    await watchdog.start()
    for expected in (1, 2):
        env.alive = False
        await watchdog.check()
        assert watchdog.restarts == expected
        assert watchdog.active
    env.alive = False
    await watchdog.check()
    assert watchdog.restarts == 2
    assert not watchdog.active
    starts = [c for c in env.calls if "<<'PROXIMUS_MEMWATCH_EOF'" in c["command"]]
    assert len(starts) == 3


async def test_an_expired_watchdog_is_not_restarted():
    env = WatchEnv()
    watchdog = make_watchdog(env)
    await watchdog.start()
    env.alive = False
    env.lines = [f"1\t1790000000\texpire\t0\t{64 * GIB}\t0\t0\t"]
    await watchdog.check()
    assert watchdog.restarts == 0


async def test_stop_kills_it_and_collects_the_last_events():
    env = WatchEnv()
    watchdog = make_watchdog(env)
    await watchdog.start()
    env.lines = [kill_line(1)]
    (event,) = await watchdog.stop()
    assert event.kind == "kill"
    assert not env.alive
    assert await watchdog.stop() == []
    assert await watchdog.check() == []
    assert 'kill "$p"' in env.calls[1]["command"]
    assert len(env.calls) == 2


def test_note_wording():
    kill = WatchdogEvent.parse(kill_line(1).split("\t"))
    assert kill is not None
    note = watchdog_note([kill], 65536)
    assert note.startswith("\n\n(Memory watchdog: killed PID 4242 (`python3 train.py")
    assert (
        "50.0 GB) when the sandbox was using 59.0 GB of its 64 GB memory limit" in note
    )
    assert "fewer or smaller jobs" in note
    nokill = WatchdogEvent(2, 0, "nokill", 60 * GIB, 64 * GIB, 0, 0, "")
    assert "no single process is large enough" in watchdog_note([nokill], 65536)
    expire = WatchdogEvent(3, 0, "expire", 0, 0, 0, 0, "")
    assert watchdog_note([expire], 65536) == ""
    assert watchdog_note([], 65536) == ""


def test_limits_note_says_when_the_limit_is_enforced():
    enforced = sandbox_limits_note({"memory_mb": 8192}, watchdog=True)
    assert enforced is not None
    assert "enforced" in enforced and "largest process is killed" in enforced
    assert "kill the whole sandbox" not in enforced
    plain = sandbox_limits_note({"memory_mb": 8192})
    assert plain is not None and "kill the whole sandbox" in plain


# ----- wiring in the agent -----


class WatchedFakeEnv(FakeEnv):
    """FakeEnv plus a running watchdog that kills the model's first command."""

    def __init__(self) -> None:
        super().__init__()
        self.watch = WatchEnv()

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if "proximus-memwatch" in command:
            self.all_commands.append(command)
            return await self.watch.exec(command, cwd, env, timeout_sec, user)
        if "python3 big.py" in command:
            self.all_commands.append(command)
            self.watch.lines = [kill_line(1, pid=777)]
            return ExecResult(stdout="", stderr="Killed", return_code=137)
        return await super().exec(command, cwd, env, timeout_sec, user)


async def test_agent_enforces_the_limit_and_tells_the_model(tmp_path, monkeypatch):
    fake = FakeCompletion([bash_response("python3 big.py"), *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)
    agent = make_agent(tmp_path, sandbox_limits={"memory_mb": 65536})
    env = WatchedFakeEnv()
    context = AgentContext()
    await agent.run("Task", env, context)  # type: ignore[arg-type]

    messages = read_messages(agent)
    assert "The memory limit is enforced" in messages[1]["content"]
    tool_result = next(m for m in messages if m["role"] == "tool")
    assert "exit code: 137" in tool_result["content"]
    assert "Memory watchdog: killed PID 777" in tool_result["content"]
    assert "59.0 GB of its 64 GB memory limit" in tool_result["content"]
    # Started first, stopped last (before grading), always as root.
    watch_commands = [c["command"] for c in env.watch.calls]
    assert "<<'PROXIMUS_MEMWATCH_EOF'" in watch_commands[0]
    assert 'kill "$p"' in watch_commands[-1]
    assert all(c["user"] == "root" for c in env.watch.calls)
    metadata = (context.metadata or {})["memory_watchdog"]
    assert metadata["kills"] == 1 and metadata["active"] is True
    assert metadata["events"][0]["pid"] == 777


async def test_no_watchdog_without_a_memory_limit_or_when_disabled(
    tmp_path, monkeypatch
):
    for kwargs in (
        {"sandbox_limits": {"cpus": 4}},
        {"sandbox_limits": {"memory_mb": 4096}, "memory_watchdog": False},
    ):
        fake = FakeCompletion(finish_responses())
        monkeypatch.setattr(litellm, "acompletion", fake)
        agent = make_agent(tmp_path / str(len(kwargs)), **kwargs)
        env = FakeEnv()
        context = AgentContext()
        await agent.run("Task", env, context)  # type: ignore[arg-type]
        assert not any("proximus-memwatch" in c for c in env.all_commands)
        assert (context.metadata or {})["memory_watchdog"] is None


def test_scripts_parse_under_dash():
    if shutil.which("dash") is None:
        pytest.skip("no dash")
    for name in ("memwatch.sh", "memsample.sh"):
        result = subprocess.run(
            ["dash", "-n", str(Path(memwatch.__file__).parent / "scripts" / name)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


# ----- pass@8: host-side poller, samples, peak, GPU wording -----

SAMPLE_OUT = "\n".join(
    [
        "mem\tcg=1 usage=%d max=%d epoch=1790000100 src=cgroup1 sused="
        % (40 * GIB, 41 * GIB),
        "stat\ttotal_cache=%d" % (6 * GIB),
        "stat\ttotal_shmem=%d" % (2 * GIB),
        "stat\ttotal_rss=%d" % (30 * GIB),
        "peak\t%d" % (39 * GIB),
        # rss pages (4 KiB): three DataLoader workers in one group
        "procs\t%d\t4" % (14 * GIB // 4096),
        "grp\t100\t%d\t3" % (13 * GIB // 4096),
        "grp\t200\t%d\t1" % (1 * GIB // 4096),
        "top\t101\t100\t%d\tpython3\tpython3 train.py --workers 2" % (5 * GIB // 4096),
        "top\t200\t200\t%d\tnvcc with space\t" % (1 * GIB // 4096),
        "df\t/dev/shm\t2097152\t33554432",
        "gpu\t70000\t81559",
    ]
)


def test_parse_sample_reads_the_counter_processes_and_groups():
    sample = memwatch.parse_sample(SAMPLE_OUT, page=4096)
    # used = usage - (cache - shmem), as the loop computes it
    assert sample["used_mb"] == 36 * 1024
    assert sample["source"] == "cgroup1"
    assert sample["usage_mb"] == 40 * 1024
    assert sample["max_usage_mb"] == 41 * 1024
    assert sample["peak_mb"] == 39 * 1024
    assert sample["stat_mb"]["total_rss"] == 30 * 1024
    assert sample["top"][0] == [101, 100, 5 * 1024, "python3 train.py --workers 2"]
    assert sample["top"][-1] == [200, 200, 1024, "nvcc with space"]
    assert sample["groups"][0] == [100, 13 * 1024, 3]
    assert sample["df_mb"]["/dev/shm"] == [2048, 32768]
    assert sample["gpu_mb"] == [[70000, 81559]]
    assert sample["n_procs"] == 4
    assert sample["rss_total_mb"] == 14 * 1024


def test_a_sample_uses_the_loops_reading_for_non_cgroup_sources():
    # gVisor without cgroup files: the loop reads MemTotal - MemAvailable.
    out = "mem\tcg= usage= max= epoch=1790000100 src=meminfo sused=%d\n" % (20 * GIB)
    sample = memwatch.parse_sample(out, page=4096)
    assert sample["used_mb"] == 20 * 1024 and sample["usage_mb"] is None
    # A cgroup counter is not mistaken for the reading of another source.
    out = "mem\tcg=1 usage=%d max= epoch=1 src=meminfo sused=\n" % (40 * GIB)
    assert memwatch.parse_sample(out, page=4096)["used_mb"] is None


class PollEnv(WatchEnv):
    """WatchEnv that also answers the poller's sample section."""

    def __init__(self) -> None:
        super().__init__()
        self.samples = 0

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        result = await super().exec(command, cwd, env, timeout_sec, user)
        if "procs=$(" in command:
            self.samples += 1
            result.stdout = (result.stdout or "") + SAMPLE_OUT + "\n"
        return result


async def test_poller_collects_events_and_samples_without_tool_calls(tmp_path):
    env = PollEnv()
    samples_file = tmp_path / "memwatch.jsonl"
    watchdog = make_watchdog(
        env,
        memory_mb=40960,
        poll_interval_sec=0.01,
        samples_path=samples_file,
        gpus=True,
    )
    assert await watchdog.start()
    env.lines = [kill_line(1)]
    watchdog.start_polling()
    assert watchdog.polling and not watchdog.due()
    for _ in range(200):
        if env.samples >= 5:
            break
        await asyncio.sleep(0.01)
    (event,) = watchdog.take_pending()
    assert event.kind == "kill"
    assert watchdog.take_pending() == []  # told once
    await watchdog.stop()
    assert not watchdog.polling
    meta = watchdog.metadata()
    assert meta["n_samples"] >= 5 and meta["lost_with_sandbox"] is False
    assert len(meta["last_samples"]) == memwatch.KEPT_SAMPLES_ALIVE
    # The loop's peak; the cgroup's raw max (41 GiB) counts page cache.
    assert meta["peak_used_mb"] == 39 * 1024
    # 36 GiB of a 40 GiB limit is above the high-water mark: every sample kept.
    assert len(samples_file.read_text().splitlines()) == meta["n_samples"]
    # GPU memory is read on the first poll and every GPU_EVERY after.
    sample_cmds = [c["command"] for c in env.calls if "procs=$(" in c["command"]]
    assert "gpu=1" in sample_cmds[0] and "gpu=0" in sample_cmds[1]


async def test_only_every_few_samples_are_persisted_below_high_water(tmp_path):
    env = PollEnv()
    samples_file = tmp_path / "memwatch.jsonl"
    # 36 GiB used of a 128 GiB limit: below 75%.
    watchdog = make_watchdog(env, memory_mb=131072, samples_path=samples_file)
    await watchdog.start()
    for _ in range(8):
        await watchdog.check(sample=True)
    assert watchdog.n_samples == 8
    assert len(samples_file.read_text().splitlines()) == 8 // memwatch.PERSIST_EVERY


async def test_a_sandbox_lost_while_polling_keeps_every_sample(tmp_path):
    env = PollEnv()
    watchdog = make_watchdog(env, poll_interval_sec=0.01)
    await watchdog.start()
    watchdog.start_polling()
    for _ in range(200):
        if env.samples >= memwatch.RING_SAMPLES + 5:
            break
        await asyncio.sleep(0.01)
    env.fail = type("SandboxTerminatedError", (Exception,), {})("gone")
    await asyncio.sleep(0.05)
    # A failed poll decides nothing: the kill loop may be alive and must still
    # be stopped before grading. The agent loop calls lost() on env_lost.
    assert watchdog.polling and not watchdog.lost_with_sandbox
    await watchdog.lost()
    assert not watchdog.polling
    meta = watchdog.metadata()
    assert meta["lost_with_sandbox"] is True
    assert len(meta["last_samples"]) == memwatch.RING_SAMPLES
    assert await watchdog.stop() == []  # nothing is sent to a dead sandbox


@needs_proc
def test_the_loop_records_its_peak(tmp_path, victims):
    v = victims(1)
    state = tmp_path / "state"
    usage = tmp_path / "usage"
    usage.write_text(f"{1 * GIB}\n")
    proc = run_script(
        state, usage, threshold=60 * GIB, limit=64 * GIB, ticks=4, scope=v.big.pid
    )
    try:
        assert wait_for(lambda: (state / "peak").exists(), 5)
        assert int((state / "peak").read_text()) == 1 * GIB
        usage.write_text(f"{3 * GIB}\n")
        assert wait_for(lambda: int((state / "peak").read_text()) == 3 * GIB, 5)
        # A small rise is not written; a fall never is.
        usage.write_text(f"{3 * GIB + MIB}\n")
        time.sleep(1.5)
        assert int((state / "peak").read_text()) == 3 * GIB
    finally:
        proc.kill()
        proc.wait()


@needs_proc
async def test_real_poll_reads_the_loop_and_the_processes(tmp_path, victims):
    v = victims(64)
    state = tmp_path / "state"
    usage = tmp_path / "usage"
    usage.write_text(f"{2 * GIB}\n")
    watchdog = make_watchdog(
        LocalEnv(),
        memory_mb=4096,
        source=f"file:{usage}",
        state_dir=str(state),
        scope_sid=v.big.pid,
    )
    try:
        assert await watchdog.start()
        await watchdog.check(sample=True)
        (sample,) = watchdog.samples
        assert sample["peak_mb"] == 2048
        assert sample["source"] == "file" and sample["used_mb"] == 2048
        assert sample["n_procs"] >= 2  # the victims, at least
        assert sample["rss_total_mb"] >= sum(row[2] for row in sample["top"])
        assert 1 <= len(sample["top"]) <= memwatch.TOP_PROCESSES
        assert all(isinstance(row[0], int) for row in sample["top"])
        assert 1 <= len(sample["groups"]) <= memwatch.TOP_GROUPS
        # Groups are the largest by resident memory, and within the total.
        sizes = [g[1] for g in sample["groups"]]
        assert sizes == sorted(sizes, reverse=True)
        assert sizes[0] <= sample["rss_total_mb"]
        await watchdog.stop()
    finally:
        for pid in loops(state):
            os.kill(pid, signal.SIGKILL)


def test_gpu_sandboxes_get_the_same_memory_wording():
    # The pass@8 probe: on Modal the counter sees pinned memory and CUDA
    # contexts, and the GPU "OOM" deaths were preemptions.
    gpu = sandbox_limits_note({"memory_mb": 65536, "gpus": 1}, watchdog=True)
    cpu = sandbox_limits_note({"memory_mb": 65536}, watchdog=True)
    assert gpu is not None and cpu is not None
    assert (
        "The memory limit is enforced" in gpu and "The memory limit is enforced" in cpu
    )


class PolledFakeEnv(FakeEnv):
    """FakeEnv whose watchdog reports a kill from the background, while the
    model's command exits 0."""

    def __init__(self) -> None:
        super().__init__()
        self.watch = PollEnv()

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if "proximus-memwatch" in command:
            self.all_commands.append(command)
            return await self.watch.exec(command, cwd, env, timeout_sec, user)
        if "sleep 1" in command:
            self.all_commands.append(command)
            self.watch.lines = [kill_line(1, pid=888)]
            await asyncio.sleep(0.2)  # the poller runs meanwhile
            return ExecResult(stdout="", stderr="", return_code=0)
        return await super().exec(command, cwd, env, timeout_sec, user)


async def test_agent_tells_the_model_about_a_background_kill(tmp_path, monkeypatch):
    monkeypatch.setattr(memwatch, "POLL_INTERVAL_SEC", 0.02)
    fake = FakeCompletion([bash_response("sleep 1"), *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)
    agent = make_agent(tmp_path, sandbox_limits={"memory_mb": 65536})
    env = PolledFakeEnv()
    context = AgentContext()
    await agent.run("Task", env, context)  # type: ignore[arg-type]

    tool_result = next(m for m in read_messages(agent) if m["role"] == "tool")
    assert "exit code: 0" in tool_result["content"]
    assert "Memory watchdog: killed PID 888" in tool_result["content"]
    metadata = (context.metadata or {})["memory_watchdog"]
    assert metadata["n_samples"] > 0 and metadata["lost_with_sandbox"] is False
    assert (agent.logs_dir / "memwatch.jsonl").exists()
