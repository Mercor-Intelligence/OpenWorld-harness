"""The memory watchdog: enforce the sandbox's disclosed memory limit.

Harbor's Modal backend sends ``memory_mb`` as a request with no limit, so a run
can use far more; past some point host pressure kills the whole sandbox and
the run ends with nothing graded. ``scripts/memwatch.sh`` runs as a detached
root loop in the sandbox and kills the largest process near the limit instead,
as the kernel's OOM killer would. This module starts it, collects its events
for the model, restarts it when an agent command kills it, and stops it before
grading.
"""

from __future__ import annotations

import logging
import shlex
import time
from dataclasses import asdict, dataclass
from typing import Any

from harbor.environments.base import BaseEnvironment

from openworld_harness import tools, utils

STATE_DIR = "/tmp/proximus-memwatch"
SCRIPT_NAME = "memwatch.sh"
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
        return (
            self.active
            and not self.stopped
            and time.monotonic() - self._last_check >= self._check_interval
        )

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
            if event.kind == "kill":
                self.n_kills += 1
            elif event.kind == "nokill":
                self.n_nokill += 1
        self.events = (self.events + new)[-_KEPT_EVENTS:]
        return alive, new

    async def check(self) -> list[WatchdogEvent]:
        """New events since the last check; restarts a killed loop (capped).
        Never raises except for a gone sandbox."""
        if not self.active or self.stopped:
            return []
        self._last_check = time.monotonic()
        try:
            out = await self._exec(
                _find_instances(self._state)
                + 'echo "alive=$alive"\n'
                + _events_since(self._last_seq)
            )
        except tools.EnvironmentGoneError:
            raise
        except Exception as e:  # noqa: BLE001
            self._logger.debug(f"Memory watchdog check failed: {e}")
            return []
        alive, new = self._take(out)
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

    async def stop(self) -> list[WatchdogEvent]:
        """Collect the last events and end the loop before grading."""
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
        }
