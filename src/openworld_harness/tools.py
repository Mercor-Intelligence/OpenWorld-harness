import json
import re
import shlex
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.environments.base import BaseEnvironment
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

import openworld_harness.images as images
from openworld_harness import utils
from openworld_harness.config import BUDGET_MARGIN_SEC


class EnvironmentGoneError(RuntimeError):
    """The sandbox no longer exists or was terminated."""


# Provider SDK exceptions that mean the sandbox itself is gone, matched by
# name so this package needs no provider imports. "NotFoundError" is too
# generic to match alone, so it also requires the modal namespace.
_GONE_EXCEPTION_NAMES = frozenset(
    {
        "SandboxTerminatedError",  # modal: OOM, preemption, kill
        # harbor's modal environment: an operation failed and the sandbox
        # itself had exited with 137 (sandbox.poll()), i.e. an OOM kill
        "SandboxLikelyOutOfMemoryError",
        "SandboxTimeoutError",  # modal: sandbox lifetime expired
        "DaytonaNotFoundError",  # daytona: sandbox 404
        "SandboxNotFoundException",  # e2b: deleted or expired
    }
)


def _is_environment_transport_loss(error: Exception) -> bool:
    """Recognize control-plane failures raised by an environment operation.

    A task command's own network error returns inside ``ExecResult``. These
    exceptions instead mean Harbor could not reach the sandbox exec service.
    """
    message = str(error).lower()
    return (
        isinstance(error, ConnectionError)
        and (
            "[errno -2] name or service not known" in message
            or "[errno -5] no address associated with hostname" in message
            or "[errno 111] connect call failed" in message
        )
    ) or (
        "failed to read exec stdio stream" in message
        or "modal sandbox is shutting down" in message
        or ("containermanager.waitpid" in message and "failed: eof" in message)
        or (
            "connecting to control server" in message
            and "connection refused" in message
        )
        or _names_a_stopped_container(message)
    )


# Modal refusing an exec because the container has stopped.
_STOPPED_CONTAINER = re.compile(
    r"(?:executing processes for container: )?"
    r'cannot execute in container "[^"\n]*" in state stopped',
    re.IGNORECASE,
)


def _names_a_stopped_container(text: str) -> bool:
    return _STOPPED_CONTAINER.search(text) is not None


def exec_result_shows_environment_gone(result: Any) -> bool:
    """An exec that "completed" only because the container had already
    stopped: Modal reports that as a nonzero result, not an exception. A
    process killed inside a live sandbox exits 137 too, so the exit code alone
    never counts. The command's own output lands in the same fields, so the
    result must be nothing but Modal's message: no stdout, and a stderr that
    is that one line."""
    return (
        bool(result.return_code)
        and not str(result.stdout or "").strip()
        and _STOPPED_CONTAINER.fullmatch(str(result.stderr or "").strip()) is not None
    )


def raise_if_exec_result_gone(result: Any) -> None:
    """Raise EnvironmentGoneError for a result from a stopped container."""
    if exec_result_shows_environment_gone(result):
        detail = truncate_middle(str(result.stderr or "").strip(), 300)
        raise EnvironmentGoneError(f"exit code {result.return_code}: {detail}")


def raise_if_environment_gone(error: Exception) -> None:
    """Re-raise a dead-sandbox or lost-control-channel provider exception."""
    name = type(error).__name__
    module = type(error).__module__ or ""
    gone = (
        name in _GONE_EXCEPTION_NAMES
        or (name == "NotFoundError" and module.startswith("modal"))
        or _is_environment_transport_loss(error)
    )
    if gone:
        raise EnvironmentGoneError(f"{name}: {error}") from error


# Sandbox commands in a row that fail without running (the exec call itself
# raises, unclassified) before a trivial probe command decides whether the
# sandbox is gone or only those commands were refused.
MAX_CONSECUTIVE_EXEC_FAILURES = 5
LIVENESS_PROBE_TIMEOUT_SEC = 60
KILL_SESSION_TIMEOUT_SEC = 60

# Fixed executor limits. Command timeouts are additionally capped at the
# remaining time budget minus BUDGET_MARGIN_SEC, so the loop regains control
# before the orchestrator's hard kill.
DEFAULT_TIMEOUT_SEC = 600
MAX_TIMEOUT_SEC = 1800
OUTPUT_LIMIT_CHARS = 30_000

# Non-interactive defaults applied to every bash exec
EXEC_ENV: dict[str, str] = {
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
    "DEBIAN_FRONTEND": "noninteractive",
}


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    # (media_type, base64_data) — attached to the conversation as an image
    # content block by the agent loop (kept out of the ATIF trajectory).
    image: images.Attachment | None = None
    # Set only by a confirmed final submission.
    ends_run: bool = False
    # The bash command's own exit code; None for timeouts and other tools.
    exit_code: int | None = None


BASH_DESCRIPTION = (
    "Run a shell command in the task environment and return its "
    "output. Each call runs in a FRESH shell from the task "
    "working directory: environment variables and shell state "
    "do not persist between calls, so prefer absolute paths. "
    "Long output is truncated in the middle. Default timeout "
    f"{DEFAULT_TIMEOUT_SEC}s, max "
    f"{MAX_TIMEOUT_SEC}s. Anything that may run longer "
    "(builds, training runs, benchmarks, servers, parameter "
    "sweeps) must be launched in the background rather than "
    "waited on in the foreground: redirect output to a log "
    "file and append ` &` (e.g. `nohup ./run.sh > /tmp/run.log "
    "2>&1 &`), checkpoint best-so-far results to a file, then "
    "poll the log with later calls. A foreground command that "
    "hits the timeout is killed, together with every process it "
    "started (background jobs included), and its work is lost."
)

VIEW_IMAGE_DESCRIPTION = (
    "View an image file (png/jpg/gif/webp; format detected "
    "from content, any extension works) from the task "
    "environment. The image is attached to the conversation "
    "so you can see it. Images consume significant context — "
    "view only what you need."
)

LIST_SUBMISSIONS_DESCRIPTION = (
    "List your recorded submissions (when, id, description) and "
    "which one is currently kept as your result."
)

SUBMIT_DESCRIPTION = (
    "Submit the current workspace as your result. A commit is "
    "optional: omit it and the harness commits the live files for "
    "you (uncommitted work included); or pass a commit hash that "
    "already exists in the workspace repository."
)


class BashArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(description="The shell command to execute.")
    timeout_sec: int = Field(
        default=DEFAULT_TIMEOUT_SEC,
        strict=True,
        ge=1,
        description=(
            "Seconds before the command is killed (default "
            f"{DEFAULT_TIMEOUT_SEC}, max {MAX_TIMEOUT_SEC})."
        ),
    )


class ViewImageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Absolute path to the image file.")
    # Accepted and ignored: models habitually pass one, and the call must
    # never fail on it.
    query: str = Field(
        default="",
        description=(
            "Optional: what to look for in the image. The image is attached "
            "to the conversation for you to view directly."
        ),
    )


class SubmitArgs(BaseModel):
    # Stray extras are tolerated: failing a submission over one is worse.
    model_config = ConfigDict(extra="ignore")

    commit: str = Field(
        default="",
        description=(
            "Optional commit hash or ref. Omit (or pass HEAD) and the "
            "harness commits the current workspace state for you, "
            "uncommitted changes included. An explicit hash must already "
            "exist in the workspace repository."
        ),
    )

    @field_validator("commit")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    handler: Callable[[Any], Awaitable[ToolResult]]

    def schema(self) -> dict[str, Any]:
        parameters = self.args.model_json_schema()
        parameters.pop("title", None)
        for prop in parameters.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    async def call(self, raw_arguments: str) -> ToolResult:
        try:
            args = self.args.model_validate_json((raw_arguments or "").strip() or "{}")
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(str(loc) for loc in err['loc']) or 'arguments'}: "
                f"{err['msg']}"
                for err in e.errors()
            )
            return ToolResult(
                f"Invalid arguments for '{self.name}': {problems}.", is_error=True
            )
        try:
            return await self.handler(args)
        except EnvironmentGoneError:
            raise
        except Exception as e:
            raise_if_environment_gone(e)
            return ToolResult(
                f"Tool '{self.name}' failed: {type(e).__name__}: {e}", is_error=True
            )


def args_for_trajectory(raw_arguments: str) -> dict[str, Any]:
    """The call's raw arguments as a dict for the ATIF record, valid or not."""
    try:
        value = json.loads((raw_arguments or "").strip() or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw_arguments}
    return value if isinstance(value, dict) else {"_raw": raw_arguments}


def truncate_middle(text: str, limit_chars: int) -> str:
    """Truncate the middle of long text, keeping the head and tail."""
    if len(text) <= limit_chars:
        return text
    head = int(limit_chars * 0.6)
    tail = limit_chars - head
    omitted = len(text) - head - tail
    return (
        text[:head]
        + f"\n[... output truncated: {omitted} characters omitted ...]\n"
        + text[-tail:]
    )


def format_exec_output(stdout: str | None, stderr: str | None) -> str:
    """Join stdout + stderr into one model-readable block, truncated in
    the middle."""
    parts = []
    if stdout:
        parts.append(stdout.rstrip("\n"))
    if stderr:
        parts.append(f"[stderr]\n{stderr.rstrip()}")
    body = "\n".join(parts).strip()
    return truncate_middle(body, OUTPUT_LIMIT_CHARS) if body else "(no output)"


# ----- foreground command sessions -----
#
# A backend exec timeout kills only the exec's top process (Modal and Harbor's
# Docker exec alike), so a timed-out command's children, `timeout N ...`
# wrappers and background jobs kept running while the model was told the
# command "was killed". Each bash call therefore runs in its own session: the
# wrapper records the session id, and on a timeout kill_session.sh (as root)
# kills every process in that session. Sessions survive setpgid, which
# `timeout` and job control use, so those are found too.

FG_DIR = "/tmp/proximus/fg"
_FG_ENV = "PROXIMUS_FG_CMD"


# Runs as `bash --posix -c SUPERVISOR proximus-fg TOKEN PREVIOUS`. The model's
# command travels in $PROXIMUS_FG_CMD (an env var, so the wrapper never shows
# it in an argv that `pkill -f` might match). setsid is not the last command,
# so bash runs it as a child that is no group leader: setsid then makes the
# session without forking, and a signal death still reads as 128+n (-w only
# matters if setsid ever has to fork: the call must still wait for it). The
# supervisor's own diagnostics (mkdir, bash's "Killed" notices) go to
# /dev/null; the session's shell gets the exec's stderr back from fd 3 and
# closes fd 3, so a background job that redirects its output holds no copy
# of the exec's pipes (the call would otherwise wait for it). stdin is
# /dev/null: an accidental read must not block until the timeout.
def _fg_supervisor(fg_dir: str) -> str:
    return (
        f"D={fg_dir}\n"
        '[ -d "$D" ] || mkdir -p "$D" 2>/dev/null\n'
        '[ -n "$2" ] && rm -f "$D/$2.sid" "$D/$2.rc" 2>/dev/null\n'
        "{ setsid -w sh -c "
        '\'{ echo $$ >"$1"; } 2>/dev/null; c=$' + _FG_ENV + "; unset " + _FG_ENV + "; "
        'exec 2>&3 3>&-; exec bash -c "$c"\' fg "$D/$1.sid" </dev/null; } 3>&2 2>/dev/null\n'
        "rc=$?\n"
        '{ echo "$rc" >"$D/$1.rc"; } 2>/dev/null\n'
        'exit "$rc"'
    )


def _double_quote(text: str) -> str:
    """Quote for a POSIX shell inside double quotes. Unlike shlex.quote, a
    single quote costs nothing here; the backend's own quoting of the whole
    command (su -c '...') would otherwise grow each one about 25-fold."""
    return '"' + re.sub(r'([\\"$`])', r"\\\1", text) + '"'


def foreground_command(
    command: str, token: str, previous: str | None, fg_dir: str = FG_DIR
) -> str:
    """The exec string that runs ``command`` in its own session (see
    _fg_supervisor), recording the session id under ``token`` and dropping
    ``previous``'s leftover files."""
    return (
        f"{_FG_ENV}={_double_quote(command)} exec bash --posix -c "
        f"{shlex.quote(_fg_supervisor(fg_dir))} proximus-fg {token} {previous or ''}"
    ).rstrip()


# Prints fg-sessions-ok when setsid exists and a process it starts leads its
# own session (stat field 6 == its pid), i.e. when the timeout kill can work.
FG_SESSION_PROBE = (
    f"mkdir -p {FG_DIR} 2>/dev/null; "
    "setsid -w sh -c 'read -r l </proc/$$/stat && r=${l##*) } && set -- $r && "
    '[ "$4" = "$$" ] && echo fg-sessions-ok\' </dev/null 2>/dev/null; true'
)


@dataclass
class TimeoutCleanup:
    """What kill_session.sh found after a timeout."""

    attempted: bool = False
    sid: str | None = None
    killed: int = 0
    rc: str | None = None
    survivors: list[str] | None = None
    error: str | None = None


def parse_kill_session(stdout: str) -> TimeoutCleanup:
    cleanup = TimeoutCleanup(attempted=True, survivors=[])
    for line in (stdout or "").splitlines():
        key, _, value = line.strip().partition("=")
        if key == "sid":
            cleanup.sid = None if value == "none" else value
        elif key == "killed" and value.isdigit():
            cleanup.killed = int(value)
        elif key == "rc":
            cleanup.rc = value or None
        elif key == "survivor" and value:
            assert cleanup.survivors is not None
            cleanup.survivors.append(value)
    return cleanup


def _is_backend_timeout(error: Exception) -> bool:
    """Harbor's Docker exec reports a command timeout as this RuntimeError."""
    return isinstance(error, RuntimeError) and str(error).startswith(
        "Command timed out after"
    )


class ToolExecutor:
    """Executes bash and view_image calls against a task environment."""

    def __init__(
        self,
        environment: BaseEnvironment,
        *,
        default_workdir: str | None = None,
        remaining_sec: Callable[[], float | None] | None = None,
        foreground_sessions: bool = False,
        fg_dir: str = FG_DIR,
    ):
        self._environment = environment
        # Whether bash calls run in their own session (see FG_SESSION_PROBE).
        self._foreground_sessions = foreground_sessions
        self._fg_dir = fg_dir
        self._previous_token: str | None = None
        # Telemetry for the run's metadata.
        self.n_timeouts = 0
        self.n_timeout_killed = 0
        self.n_timeout_survivors = 0
        # Caps command timeouts so a foreground command can't outlive the
        # time budget — the loop must regain control before the hard kill.
        self._remaining_sec = remaining_sec
        self._default_workdir = default_workdir
        self._exec_failures = 0  # in a row; see MAX_CONSECUTIVE_EXEC_FAILURES

    async def bash(self, args: BashArgs) -> ToolResult:
        if args.timeout_sec > MAX_TIMEOUT_SEC:
            # Teach instead of silently clamping.
            return ToolResult(
                f"timeout_sec={args.timeout_sec} exceeds the "
                f"{MAX_TIMEOUT_SEC}s foreground limit. Long jobs (builds, "
                "sweeps, training, benchmarks) should run in the background: "
                "redirect output to a log and append ` &`, then poll the log "
                "with later commands — or bound the work to fit the limit.",
                is_error=True,
            )
        timeout = self._clamp_timeout(args.timeout_sec)
        command, token = args.command, None
        if self._foreground_sessions:
            token = uuid.uuid4().hex[:16]
            command = foreground_command(
                args.command, token, self._previous_token, self._fg_dir
            )
            self._previous_token = token
        started = time.monotonic()
        try:
            result = await self._environment.exec(
                command,
                cwd=self._default_workdir,
                env=dict(EXEC_ENV),
                timeout_sec=timeout,
            )
        except UnicodeDecodeError as e:
            # Binary bytes on stdout/stderr break the transport's text
            # decoding; the whole output is lost.
            self._exec_failures = 0
            return ToolResult(
                f"Command output could not be decoded as text ({e}). Its "
                "output was discarded. Binary data must not reach "
                "stdout/stderr directly: redirect it to a file and inspect "
                "with od/xxd/cat -v/strings, or pipe through such a filter.",
                is_error=True,
            )
        except TimeoutError as e:
            # Modal's SandboxTimeoutError subclasses builtin TimeoutError but
            # means the sandbox died, not the command — classify first.
            raise_if_environment_gone(e)
            self._exec_failures = 0
            return await self._timed_out(token, timeout, f"{timeout}s", None)
        except EnvironmentGoneError:
            raise
        except Exception as e:
            raise_if_environment_gone(e)
            if _is_backend_timeout(e):
                self._exec_failures = 0
                return await self._timed_out(token, timeout, f"{timeout}s", None)
            self._exec_failures += 1
            if self._exec_failures >= MAX_CONSECUTIVE_EXEC_FAILURES:
                if not await self._sandbox_answers():
                    # Unrecognized, but the sandbox has stopped running commands.
                    raise EnvironmentGoneError(
                        f"{self._exec_failures} commands in a row could not run "
                        f"in the sandbox; the last: {type(e).__name__}: {e}"
                    ) from e
                # The sandbox is fine; the exec refused these commands.
                self._exec_failures = 0
            raise
        self._exec_failures = 0
        # The command's own output can look the same, so the sandbox decides.
        if exec_result_shows_environment_gone(result) and not (
            await self._sandbox_answers()
        ):
            raise_if_exec_result_gone(result)
        body = format_exec_output(result.stdout, result.stderr)
        elapsed = time.monotonic() - started
        if self._looks_like_timeout(result.return_code, elapsed, timeout):
            return await self._timed_out(token, timeout, f"~{int(elapsed)}s", body)
        return ToolResult(
            f"exit code: {result.return_code}\n{body}",
            is_error=result.return_code != 0,
            exit_code=result.return_code,
        )

    async def _timed_out(
        self, token: str | None, timeout: int, took: str, body: str | None
    ) -> ToolResult:
        """The result of a timed-out command, after killing what it left
        running (when its session is known)."""
        self.n_timeouts += 1
        cleanup = await self._kill_session(token)
        self.n_timeout_killed += cleanup.killed
        self.n_timeout_survivors += len(cleanup.survivors or [])
        head = f"Command timed out after {took} (timeout_sec={timeout})"
        if not cleanup.attempted or cleanup.error or cleanup.sid is None:
            fate = (
                "; it was stopped, but processes it started may still be "
                "running: check with `ps` (or ls /proc) before relaunching it"
            )
        elif cleanup.survivors:
            fate = (
                f"; {cleanup.killed} of its processes were killed, but these "
                "are still running: " + ", ".join(cleanup.survivors[:10])
            )
        elif cleanup.rc is not None and cleanup.killed:
            fate = (
                f". The command itself had finished (exit {cleanup.rc}), but "
                "background processes it started kept its output open until "
                f"the timeout; they ({cleanup.killed}) were killed. Redirect a "
                "background job's output so the call can return, e.g. "
                "`nohup cmd > /tmp/cmd.log 2>&1 &`"
            )
        elif cleanup.rc is not None:
            fate = (
                f"; the command finished (exit {cleanup.rc}) just as the "
                "timeout hit, and nothing it started is still running"
            )
        elif cleanup.killed:
            fate = (
                f" and was killed, together with every process it started "
                f"({cleanup.killed} in all)"
            )
        else:
            fate = " and was stopped; nothing it started is still running"
        guidance = self._timeout_guidance(timeout)
        if body is None:
            return ToolResult(f"{head}{fate}. {guidance}", is_error=True)
        return ToolResult(
            f"{head}{fate}. Output up to that point:\n{body}\n{guidance}",
            is_error=True,
        )

    async def _kill_session(self, token: str | None) -> TimeoutCleanup:
        if token is None:
            return TimeoutCleanup()
        command = (
            f"fg_dir={shlex.quote(self._fg_dir)}; token={token};\n"
            f"{utils.read_script('kill_session.sh')}"
        )
        try:
            result = await self._environment.exec(
                command, timeout_sec=KILL_SESSION_TIMEOUT_SEC, user="root"
            )
        except Exception as e:  # noqa: BLE001 - backends raise varied types
            raise_if_environment_gone(e)
            return TimeoutCleanup(attempted=True, error=f"{type(e).__name__}: {e}")
        raise_if_exec_result_gone(result)
        return parse_kill_session(str(result.stdout or ""))

    async def _sandbox_answers(self) -> bool:
        """Whether a trivial command still runs in the sandbox. A probe that
        only times out is inconclusive (an overloaded sandbox is slow, a dead
        one refuses at once), so it counts as alive."""
        try:
            result = await self._environment.exec(
                "true", timeout_sec=LIVENESS_PROBE_TIMEOUT_SEC
            )
        except TimeoutError as e:
            raise_if_environment_gone(e)
            return True
        except Exception as e:
            raise_if_environment_gone(e)
            return False
        return not exec_result_shows_environment_gone(result)

    async def view_image(self, args: ViewImageArgs) -> ToolResult:
        path = args.path
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = Path(tmp_dir) / PurePosixPath(path).name
            try:
                await self._environment.download_file(path, local_path)
                data = local_path.read_bytes()
            except EnvironmentGoneError:
                raise
            except Exception as e:
                raise_if_environment_gone(e)
                return ToolResult(f"Could not read {path}: {e}", is_error=True)

        try:
            content, image = images.prepare(data, path)
        except ValueError as e:
            return ToolResult(str(e), is_error=True)
        return ToolResult(content, image=image)

    def _clamp_timeout(self, timeout_sec: int) -> int:
        timeout = min(timeout_sec, MAX_TIMEOUT_SEC)
        if self._remaining_sec is not None:
            remaining = self._remaining_sec()
            if remaining is not None:
                # Return control before budget end so the graceful finalize
                # path wins the race against the hard kill.
                timeout = min(timeout, max(60, int(remaining) - BUDGET_MARGIN_SEC))
        return max(1, timeout)

    @staticmethod
    def _looks_like_timeout(return_code: int, elapsed_sec: float, timeout: int) -> bool:
        """Environments report exec timeouts as a bare -1 exit; infer a
        timeout when the command also consumed the whole window."""
        return return_code == -1 and elapsed_sec >= min(timeout - 1, timeout * 0.9)

    @staticmethod
    def _timeout_guidance(timeout: int) -> str:
        if timeout >= MAX_TIMEOUT_SEC:
            # At the ceiling, "raise timeout_sec" is impossible advice.
            return (
                f"This is the maximum foreground timeout ({MAX_TIMEOUT_SEC}s). "
                "Run this in the background instead: redirect to a log and "
                "append ` &`, then poll the log with later commands."
            )
        return (
            f"Raise timeout_sec (up to {MAX_TIMEOUT_SEC}s), or run it in "
            "the background (redirect to a log, append ` &`, poll the log)."
        )
