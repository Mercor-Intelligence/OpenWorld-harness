"""The memory watchdog: enforce the sandbox's disclosed memory limit.

Harbor's Modal backend sends ``memory_mb`` as a request with no limit, so a run
can use far more; past some point host pressure kills the whole sandbox and
the run ends with nothing graded. ``scripts/memwatch.sh`` runs as a detached
root loop in the sandbox and kills the largest process near the limit instead,
as the kernel's OOM killer would. This module starts it, collects its events
for the model, restarts it when an agent command kills it, and stops it before
grading.

A host-side poller (``start_polling``) also reads the loop's events and a
memory sample every ``POLL_INTERVAL_SEC``, whatever the model is doing: the
counter and its parts, the largest processes and process groups, tmpfs use and
(on GPU sandboxes) GPU memory. Samples live on the host, so when the sandbox
dies the last minutes before the death are kept (pass@8: 15 GPU sandboxes died
of OOM with the watchdog silent, and nothing told why).
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import shlex
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment

from openworld_harness import tools, utils

STATE_DIR = "/tmp/proximus-memwatch"
SCRIPT_NAME = "memwatch.sh"
SAMPLE_SCRIPT_NAME = "memsample.sh"
# The daemon runs with an empty environment: Harbor overlays the agent env
# (credentials included, until scoped out) onto exec'd commands.
_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_EOF = "PROXIMUS_MEMWATCH_EOF"
CHECK_INTERVAL_SEC = 60.0
MAX_RESTARTS = 5
EXEC_TIMEOUT_SEC = 30
# Lives this much past the time budget if the harness never stops it (the
# end-of-run submission and restore fit in it; grading should not).
LIFETIME_MARGIN_SEC = 900
DEFAULT_LIFETIME_SEC = 24 * 3600
_KEPT_EVENTS = 20
POLL_INTERVAL_SEC = 15.0
# Samples kept on the host (10 minutes at the poll interval); all of them go
# into the metadata when the sandbox dies, the last few otherwise.
RING_SAMPLES = 40
KEPT_SAMPLES_ALIVE = 3
# Every this many samples one is appended to the samples file, and every one
# taken above HIGH_WATER of the limit.
PERSIST_EVERY = 4
HIGH_WATER = 0.75
# GPU memory is read on every this many polls.
GPU_EVERY = 4
TOP_PROCESSES = 5
TOP_GROUPS = 3


@dataclass(frozen=True)
class WatchdogEvent:
    seq: int
    epoch: int
    kind: str  # kill | nokill | expire
    used_bytes: int
    limit_bytes: int
    pid: int
    rss_bytes: int
    command: str

    @classmethod
    def parse(cls, fields: list[str]) -> WatchdogEvent | None:
        if len(fields) < 7:
            return None
        try:
            return cls(
                seq=int(fields[0]),
                epoch=int(fields[1]),
                kind=fields[2],
                used_bytes=int(fields[3]),
                limit_bytes=int(fields[4]),
                pid=int(fields[5]),
                rss_bytes=int(fields[6]),
                command=fields[7] if len(fields) > 7 else "",
            )
        except ValueError:
            return None


def _gb(n: int) -> str:
    return f"{n / 1024**3:.1f}"


def watchdog_note(events: list[WatchdogEvent], memory_mb: float) -> str:
    """What the model is told, appended to the tool result; "" if nothing."""
    limit_gb = f"{memory_mb / 1024:g}"
    lines = []
    for event in events:
        if event.kind == "kill":
            command = event.command.strip() or "?"
            if len(command) > 120:
                command = command[:117] + "..."
            lines.append(
                f"(Memory watchdog: killed PID {event.pid} (`{command}`, "
                f"{_gb(event.rss_bytes)} GB) when the sandbox was using "
                f"{_gb(event.used_bytes)} GB of its {limit_gb} GB memory limit. "
                "Run fewer or smaller jobs at once, or the next one will be "
                "killed too.)"
            )
        elif event.kind == "nokill":
            lines.append(
                f"(Memory watchdog: the sandbox is using {_gb(event.used_bytes)} GB "
                f"of its {limit_gb} GB memory limit, but no single process is large "
                "enough to kill. Free memory now (fewer parallel processes, files "
                "in /dev/shm), or the whole sandbox may be killed.)"
            )
    return "\n\n" + "\n".join(lines) if lines else ""


def _find_instances(state: str) -> str:
    """Shell that sets $pids to every running loop of this state directory,
    found by command line (a model command can delete the pid file), and
    $alive to 1 when there is one."""
    return (
        f"state={shlex.quote(state)}\n"
        "pids=\n"
        "for d in /proc/[0-9]*; do\n"
        "    c=$(tr '\\0' ' ' < \"$d/cmdline\" 2>/dev/null) || continue\n"
        f'    case "$c" in *" $state/{SCRIPT_NAME} "*) pids="$pids ${{d#/proc/}}" ;; esac\n'
        "done\n"
        'alive=0; [ -n "$pids" ] && alive=1\n'
    )


def _events_since(since: int) -> str:
    return (
        f"since={int(since)}\n"
        'if [ -r "$state/events.tsv" ]; then\n'
        "    while IFS= read -r line; do\n"
        '        s=${line%%"\t"*}\n'
        "        case \"$s\" in ''|*[!0-9]*) continue ;; esac\n"
        '        [ "$s" -gt "$since" ] && printf \'event\\t%s\\n\' "$line"\n'
        '    done < "$state/events.tsv"\n'
        "fi\n"
    )


def _sample_snippet(gpu: bool) -> str:
    """The poll's sample section: ``scripts/memsample.sh`` with its settings."""
    return (
        f"top_n={TOP_PROCESSES}; top_groups={TOP_GROUPS}; gpu={1 if gpu else 0}\n"
        + utils.read_script(SAMPLE_SCRIPT_NAME)
    )


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None


def _mb(n: int | None) -> int | None:
    return None if n is None else round(n / 1024**2)


def parse_sample(out: str, *, page: int) -> dict[str, Any]:
    """One sample from the poll output (scripts/memsample.sh). ``used_mb`` is
    the loop's own figure: the counter less reclaimable cache for the cgroup
    sources (computed here as memwatch.sh does), else the loop source's
    reading. ``page`` converts the resident sizes, given in pages."""
    mem: dict[str, str] = {}
    stat: dict[str, int] = {}
    top: list[list[Any]] = []
    groups: list[list[int | None]] = []
    procs: list[int] = []
    df: dict[str, list[int]] = {}
    gpu: list[list[int]] = []
    peak: int | None = None
    for line in out.splitlines():
        tag, _, rest = line.partition("\t")
        fields = rest.split("\t")
        if tag == "mem":
            mem = dict(part.split("=", 1) for part in rest.split() if "=" in part)
        elif tag == "stat":
            key, _, value = rest.partition("=")
            if (number := _int(value)) is not None:
                stat[key] = number
        elif tag == "peak":
            peak = _int(rest)
        elif tag == "procs":
            if len(fields) == 2 and all(f.isdigit() for f in fields):
                procs = [int(fields[0]) * page, int(fields[1])]
        elif tag == "grp":
            if len(fields) == 3 and all(f.isdigit() for f in fields):
                groups.append(
                    [int(fields[0]), _mb(int(fields[1]) * page), int(fields[2])]
                )
        elif tag == "top":
            fields = rest.split("\t", 4)
            if len(fields) == 5 and all(f.isdigit() for f in fields[:3]):
                pid, pgid, rss, comm, command = fields
                top.append(
                    [int(pid), int(pgid), _mb(int(rss) * page), command.strip() or comm]
                )
        elif tag == "df":
            if len(fields) == 3 and fields[1].isdigit() and fields[2].isdigit():
                df[fields[0]] = [
                    round(int(fields[1]) / 1024),
                    round(int(fields[2]) / 1024),
                ]
        elif tag == "gpu":
            if len(fields) == 2 and all(f.strip().isdigit() for f in fields):
                gpu.append([int(fields[0]), int(fields[1])])
    usage = _int(mem.get("usage"))
    used = _int(mem.get("sused"))
    source = mem.get("src") or ""
    if used is None and usage is not None and source in ("", "cgroup1", "cgroup2"):
        cache = stat.get("total_cache", stat.get("file", 0))
        shmem = stat.get("total_shmem", stat.get("shmem", 0))
        used = usage - max(0, cache - shmem)
    sample: dict[str, Any] = {
        "epoch": _int(mem.get("epoch")) or int(time.time()),
        "source": source or None,
        "cgroup": mem.get("cg") or None,
        "used_mb": _mb(used),
        "usage_mb": _mb(usage),
        "max_usage_mb": _mb(_int(mem.get("max"))),
        "peak_mb": _mb(peak),
        "stat_mb": {k: _mb(v) for k, v in stat.items()},
        "rss_total_mb": _mb(procs[0]) if procs else None,
        "n_procs": procs[1] if procs else None,
        "top": top,
        "groups": groups,
        "df_mb": df,
    }
    if gpu:
        sample["gpu_mb"] = gpu
    return sample


class MemoryWatchdog:
    def __init__(
        self,
        environment: BaseEnvironment,
        *,
        memory_mb: float,
        fraction: float,
        source: str,
        lifetime_sec: int,
        logger: logging.Logger,
        state_dir: str = STATE_DIR,
        check_interval_sec: float = CHECK_INTERVAL_SEC,
        max_restarts: int = MAX_RESTARTS,
        scope_sid: int | None = None,
        gpus: bool = False,
        samples_path: Path | None = None,
        poll_interval_sec: float | None = None,
    ) -> None:
        self._environment = environment
        self.memory_mb = memory_mb
        self.limit_bytes = int(memory_mb * 1024 * 1024)
        self.threshold_bytes = int(self.limit_bytes * fraction)
        self.source = source
        self.lifetime_sec = max(60, int(lifetime_sec))
        self._logger = logger
        self._state = state_dir
        self._check_interval = check_interval_sec
        self._max_restarts = max_restarts
        self._scope_sid = scope_sid
        self.active = False
        self.stopped = False
        self.status: dict[str, str] = {}
        self.restarts = 0
        self.n_kills = 0
        self.n_nokill = 0
        self.events: list[WatchdogEvent] = []
        self._last_seq = 0
        self._last_check = time.monotonic()
        self._started_at = time.monotonic()
        self.error: str | None = None
        self._gpus = gpus
        self._samples_path = samples_path
        self._poll_interval = (
            POLL_INTERVAL_SEC if poll_interval_sec is None else poll_interval_sec
        )
        self._poll_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._n_polls = 0
        self.samples: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=RING_SAMPLES
        )
        self.n_samples = 0
        self.peak_bytes = 0
        self.lost_with_sandbox = False
        # Events not yet told to the model (see take_pending).
        self._pending: list[WatchdogEvent] = []

    def start_command(self) -> str:
        state = shlex.quote(self._state)
        script = utils.read_script(SCRIPT_NAME)
        args = [
            self._state,
            str(self.threshold_bytes),
            str(self.limit_bytes),
            self.source,
            str(self.lifetime_sec),
        ]
        if self._scope_sid is not None:
            args.append(str(self._scope_sid))
        quoted = " ".join(shlex.quote(a) for a in args)
        return (
            _find_instances(self._state)
            # One loop per state directory: end any earlier one first.
            + 'for p in $pids; do kill "$p" 2>/dev/null; done\n'
            + f"state={state}\n"
            + 'umask 077; mkdir -p "$state" && chmod 700 "$state" || exit 3\n'
            + 'rm -f "$state/pid"\n'
            + f"cat > \"$state/{SCRIPT_NAME}\" <<'{_EOF}'\n{script}\n{_EOF}\n"
            'rm -f "$state/status"\n'
            "if command -v setsid >/dev/null 2>&1; then detach=setsid; else detach=nohup; fi\n"
            f'env -i PATH={_PATH} $detach sh "$state/{SCRIPT_NAME}" {quoted} '
            "</dev/null >/dev/null 2>&1 &\n"
            "i=0\n"
            'while [ "$i" -lt 50 ] && [ ! -s "$state/status" ]; do\n'
            "    sleep 0.1 2>/dev/null || sleep 1\n"
            "    i=$((i + 1))\n"
            "done\n"
            'cat "$state/status" 2>/dev/null\n'
            "true\n"
        )

    async def _exec(self, command: str) -> str:
        try:
            result = await self._environment.exec(
                command, timeout_sec=EXEC_TIMEOUT_SEC, user="root"
            )
        except Exception as e:
            tools.raise_if_environment_gone(e)
            raise
        tools.raise_if_exec_result_gone(result)
        return str(result.stdout or "")

    async def start(self) -> bool:
        """Launch the loop; True when it measures memory and is running."""
        try:
            out = await self._exec(self.start_command())
        except tools.EnvironmentGoneError:
            raise
        except Exception as e:  # noqa: BLE001 — the watchdog never fails a run
            self.error = f"start failed: {e}"[:300]
            self._logger.warning(f"Memory watchdog not started: {e}")
            return False
        status: dict[str, str] = {}
        for line in out.splitlines():
            if line.startswith("source="):
                status = dict(
                    part.split("=", 1) for part in line.split() if "=" in part
                )
        self.status = status
        source = status.get("source")
        self.active = bool(source) and source != "none"
        self._last_check = time.monotonic()
        if self.active:
            self._logger.info(
                f"Memory watchdog: source={source}, kills above "
                f"{self.threshold_bytes / 1024**3:.1f} GB "
                f"(limit {self.memory_mb / 1024:g} GB), now {status.get('used')} B"
            )
        else:
            self.error = self.error or (
                "no memory source" if source == "none" else "no status"
            )
            self._logger.warning(
                f"Memory watchdog not running ({self.error}): the sandbox's memory "
                "limit is not enforced"
            )
        return self.active

    def due(self) -> bool:
        """A check is due after a tool call: only when the poller is not
        running (it collects the events itself)."""
        return (
            self.active
            and not self.stopped
            and not self.polling
            and time.monotonic() - self._last_check >= self._check_interval
        )

    @property
    def polling(self) -> bool:
        return self._poll_task is not None and not self._poll_task.done()

    def take_pending(self) -> list[WatchdogEvent]:
        """Events the model has not been told about yet, once each."""
        pending, self._pending = self._pending, []
        return pending

    def _take(self, out: str) -> tuple[bool, list[WatchdogEvent]]:
        alive = "alive=1" in out.splitlines()
        new: list[WatchdogEvent] = []
        for line in out.splitlines():
            if not line.startswith("event\t"):
                continue
            event = WatchdogEvent.parse(line.split("\t")[1:])
            if event is None or event.seq <= self._last_seq:
                continue
            self._last_seq = event.seq
            new.append(event)
            self._pending.append(event)
            if event.kind == "kill":
                self.n_kills += 1
            elif event.kind == "nokill":
                self.n_nokill += 1
        self.events = (self.events + new)[-_KEPT_EVENTS:]
        return alive, new

    async def check(self, *, sample: bool = False) -> list[WatchdogEvent]:
        """New events since the last check (also queued for take_pending);
        restarts a killed loop (capped). With ``sample``, also takes a memory
        sample. Never raises except for a gone sandbox."""
        async with self._lock:
            return await self._check(sample)

    async def _check(self, sample: bool) -> list[WatchdogEvent]:
        if not self.active or self.stopped:
            return []
        self._last_check = time.monotonic()
        command = (
            _find_instances(self._state)
            + 'echo "alive=$alive"\n'
            + _events_since(self._last_seq)
        )
        if sample:
            self._n_polls += 1
            command += _sample_snippet(self._gpus and self._n_polls % GPU_EVERY == 1)
        try:
            out = await self._exec(command)
        except tools.EnvironmentGoneError:
            raise
        except Exception as e:  # noqa: BLE001
            self._logger.debug(f"Memory watchdog check failed: {e}")
            return []
        alive, new = self._take(out)
        if sample:
            self._record_sample(out)
        expired = any(e.kind == "expire" for e in new)
        if not alive and not expired:
            if self.restarts < self._max_restarts:
                self.restarts += 1
                self._logger.warning(
                    f"Memory watchdog was not running; restarting "
                    f"({self.restarts}/{self._max_restarts})"
                )
                remaining = self.lifetime_sec - (time.monotonic() - self._started_at)
                self.lifetime_sec = max(60, int(remaining))
                self._started_at = time.monotonic()
                await self.start()
            else:
                self._logger.warning(
                    "Memory watchdog stopped for good: restart cap hit"
                )
                self.active = False
        return new

    def start_polling(self) -> None:
        """Poll events and memory samples in the background until stop()."""
        if self.active and not self.stopped and not self.polling:
            self._poll_task = asyncio.create_task(self._poll_loop())

    async def _poll_loop(self) -> None:
        while self.active and not self.stopped:
            await asyncio.sleep(self._poll_interval)
            if not self.active or self.stopped:
                return
            try:
                await self.check(sample=True)
            except Exception as e:  # noqa: BLE001 — telemetry never fails a run
                # Even a "sandbox gone" error only skips this tick: the agent
                # loop decides that (env_lost), and the kill loop may well be
                # alive, so the watchdog must still be stopped before grading.
                self._logger.debug(f"Memory watchdog poll failed: {e}")

    async def _stop_polling(self) -> None:
        task, self._poll_task = self._poll_task, None
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    def _mark_lost(self) -> None:
        if self.lost_with_sandbox:
            return
        self.lost_with_sandbox = True
        self.stopped = True
        if self.samples:
            last = self.samples[-1]
            self._logger.warning(
                "Memory watchdog: sandbox lost; last sample "
                f"{int(time.time()) - int(last.get('epoch') or 0)} s before, "
                f"used {last.get('used_mb')} MB of {round(self.memory_mb)} MB"
            )

    async def lost(self) -> None:
        """The sandbox died: stop polling (no command is sent) and keep every
        sample for the metadata."""
        await self._stop_polling()
        self._mark_lost()

    def _record_sample(self, out: str) -> None:
        page = _int(self.status.get("page")) or 4096
        try:
            sample = parse_sample(out, page=page)
        except Exception as e:  # noqa: BLE001
            self._logger.debug(f"Memory sample unreadable: {e}")
            return
        # The loop's own figures only: max_usage_mb (the raw cgroup peak)
        # includes page cache the watchdog does not count.
        for key in ("used_mb", "peak_mb"):
            if sample.get(key) is not None:
                self.peak_bytes = max(self.peak_bytes, sample[key] * 1024**2)
        self.samples.append(sample)
        self.n_samples += 1
        used = sample.get("used_mb") or 0
        if self._samples_path is not None and (
            self.n_samples % PERSIST_EVERY == 1
            or used * 1024**2 >= HIGH_WATER * self.limit_bytes
        ):
            try:
                with self._samples_path.open("a") as f:
                    f.write(json.dumps(sample, separators=(",", ":")) + "\n")
            except OSError as e:
                self._logger.debug(f"Memory samples not written: {e}")

    async def stop(self) -> list[WatchdogEvent]:
        """Collect the last events and end the loop before grading."""
        await self._stop_polling()
        if self.stopped:
            return []
        self.stopped = True
        command = (
            _find_instances(self._state)
            + 'for p in $pids; do kill "$p" 2>/dev/null; done\n'
            + 'rm -f "$state/pid"\n'
            + 'echo "alive=$alive"\n'
            + _events_since(self._last_seq)
        )
        for attempt in (1, 2):
            try:
                async with self._lock:
                    out = await self._exec(command)
            except tools.EnvironmentGoneError:
                raise
            except Exception as e:  # noqa: BLE001
                self._logger.warning(f"Memory watchdog stop failed ({attempt}/2): {e}")
                continue
            _, new = self._take(out)
            return new
        return []

    def metadata(self) -> dict[str, Any]:
        kept = RING_SAMPLES if self.lost_with_sandbox else KEPT_SAMPLES_ALIVE
        return {
            "active": self.active,
            "source": self.status.get("source"),
            "limit_mb": self.memory_mb,
            "threshold_mb": round(self.threshold_bytes / 1024**2),
            "kills": self.n_kills,
            "nokill": self.n_nokill,
            "restarts": self.restarts,
            "error": self.error,
            "events": [asdict(e) for e in self.events],
            "peak_used_mb": _mb(self.peak_bytes) if self.peak_bytes else None,
            "n_samples": self.n_samples,
            "lost_with_sandbox": self.lost_with_sandbox,
            "last_samples": list(self.samples)[-kept:],
        }
