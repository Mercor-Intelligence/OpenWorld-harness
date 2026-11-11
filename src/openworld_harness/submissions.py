import contextlib
import json
import logging
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment

from openworld_harness import tools, utils

# The baseline commit hashes the whole fresh workspace; large-input excludes
# keep it small, and this bound is the backstop (sentinel2's 1.33 GB of inputs
# overran the old 60 s in every cell).
GIT_TIMEOUT_SEC = 600
EXPORT_TIMEOUT_SEC = 900
# Sandbox staging directory of an export (export_submission.sh).
EXPORT_SANDBOX_DIR = "/tmp/proximus/export"
# Harness git writes loose objects uncompressed: zlib was most of the time of
# staging large incompressible outputs (a 3.4 GB checkpoint took 4.7 min per
# submit). Halves `git add` time on such files; the model's own git and gc are
# unaffected.
_GIT_CONFIG = (
    ("core.compression", "0"),
    ("core.looseCompression", "0"),
)
DEFAULT_SUBMISSION_TIMEOUT_SEC = 600
# Of a submission's time, what staging the live workspace may not use: the
# commit, pin and chown still need it.
STAGE_RESERVE_SEC = 60
MIN_STAGE_TIMEOUT_SEC = 30
RESTORE_TIMEOUT_SEC = 300
_FULL_COMMIT_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


@dataclass
class Submission:
    """A recorded submission."""

    hash: str
    description: str
    duration_sec: float
    # True when the harness itself made the commit (a commitless submit on
    # a dirty tree); mirrored to the record row as ``harness_commit``.
    harness_committed: bool = False
    # The row's number, as list_submissions shows it (failed calls count).
    index: int = 0


class SubmissionError(Exception):
    """A submit call that could not be recorded."""


class Submissions:
    """Prepares the workspace repository, pins submitted commits, and
    records one ``submissions.jsonl`` row per submit call.

    All harness git runs as root. Empty/HEAD submit snapshots the live
    workspace so a model that cannot invoke git still has a pin; when
    that snapshot makes a commit, the row is flagged ``harness_commit``."""

    def __init__(
        self,
        logs_dir: Path,
        environment: BaseEnvironment,
        *,
        workdir: str,
        elapsed_sec: Callable[[], float],
        logger: logging.Logger,
        git_excludes: list[str] | None = None,
        submission_timeout_sec: int = DEFAULT_SUBMISSION_TIMEOUT_SEC,
    ):
        self._record_path = logs_dir / "submissions.jsonl"
        self._environment = environment
        self._workdir = workdir
        self._elapsed_sec = elapsed_sec
        self._logger = logger
        self._git_excludes = list(git_excludes or [])
        self._submission_timeout_sec = int(submission_timeout_sec)
        self._submissions: list[dict[str, Any]] = []

    async def _git_exec(self, script: str, env: dict[str, str], timeout_sec: int):
        """Run a harness git script as root, not as the agent user. A result
        from a stopped container raises EnvironmentGoneError.

        Root git refuses the agent-owned workspace repo ("dubious ownership")
        unless safe.directory covers it. Injected as GIT_CONFIG_* env — every
        git subcommand in every script inherits it, on any task image, without
        mutating the image's git config. Without it, every submit fails on
        an image that lacks the setting, and the trial is left ungradable."""
        config = (("safe.directory", self._workdir), *_GIT_CONFIG)
        git_env = {"GIT_CONFIG_COUNT": str(len(config))}
        for i, (key, value) in enumerate(config):
            git_env[f"GIT_CONFIG_KEY_{i}"] = key
            git_env[f"GIT_CONFIG_VALUE_{i}"] = value
        result = await self._environment.exec(
            script,
            env={**git_env, **env},
            timeout_sec=timeout_sec,
            user="root",
        )
        tools.raise_if_exec_result_gone(result)
        return result

    async def ensure_git(self) -> None:
        """Make the workspace a git repository."""
        result = await self._git_exec(
            utils.read_script("ensure_git.sh"),
            {
                "wd": self._workdir,
                # Newline-joined: gitwildmatch patterns cannot contain
                # newlines, so the join is unambiguous for the script.
                "exclude_patterns": "\n".join(self._git_excludes),
            },
            GIT_TIMEOUT_SEC,
        )
        if result.return_code != 0:
            detail = tools.format_exec_output(result.stdout, result.stderr)
            raise RuntimeError(
                f"workspace cannot host submissions: "
                f"{tools.truncate_middle(detail, 300)}"
            )
        for line in str(result.stdout or "").splitlines():
            if line.startswith("gitlinks:"):
                self._logger.warning(
                    "Workspace contains nested git repositories (%s): submissions "
                    "do not capture their files and the end-of-run restore does "
                    "not touch them.",
                    line.removeprefix("gitlinks:").strip(),
                )

    async def submit(self, ref: str, *, auto: bool = False) -> Submission:
        """Pin ``ref`` as the next submission and record the row. Raises
        ``SubmissionError`` after recording the failure; a dead sandbox
        propagates as ``EnvironmentGoneError`` before anything is recorded.
        ``auto`` marks the harness's end-of-run submission so analysis can
        tell it from ones the model chose to make; it is committed without
        moving HEAD or the index."""
        started = time.monotonic()
        try:
            # Pinned under the row's number, the one the model is shown.
            hash_, description, harness_committed = await self._resolve_and_pin(
                len(self._submissions) + 1, ref, keep_head=auto
            )
        except SubmissionError as e:
            failure: dict[str, Any] = {
                "event": "submission",
                "index": len(self._submissions) + 1,
                "ok": False,
                "hash": "",
                "record_sec": self._since(started),
                "elapsed_sec": round(self._elapsed_sec(), 1),
                "error": str(e),
            }
            if auto:
                failure["auto"] = True
            self._append(failure)
            raise
        submission = Submission(
            hash_,
            description,
            self._since(started),
            harness_committed,
            index=len(self._submissions) + 1,
        )
        entry: dict[str, Any] = {
            "event": "submission",
            "index": submission.index,
            "ok": True,
            "hash": submission.hash,
            "record_sec": submission.duration_sec,
            "elapsed_sec": round(self._elapsed_sec(), 1),
        }
        if auto:
            entry["auto"] = True
        if submission.harness_committed:
            # The harness committed the live workspace for this pin (the
            # model gave no commit); flagged like ``auto`` so analysis can
            # tell harness-made commits from ones the model created.
            entry["harness_commit"] = True
        if submission.description:
            entry["description"] = submission.description
        self._append(entry)
        return submission

    async def matches_winner(self, ref: str, winner: str) -> bool:
        """True when a confirming submit names the current ``winner``: a
        bare/HEAD ref must match the live tree (the agent often cannot
        commit, so HEAD may be stale); an explicit ref must resolve to it.
        A dead sandbox propagates; any other failure reads as "does not
        match" so confirm records a fresh snapshot."""
        try:
            result = await self._git_exec(
                utils.read_script("matches_winner.sh"),
                {"wd": self._workdir, "winner": winner, "commit": ref},
                self._submission_timeout_sec,
            )
        except tools.EnvironmentGoneError:
            raise
        except Exception as e:
            tools.raise_if_environment_gone(e)
            return False
        return result.return_code == 0

    async def _resolve_and_pin(
        self, index: int, ref: str, *, keep_head: bool = False
    ) -> tuple[str, str, bool]:
        """Run ``record_commit.sh``; returns (hash, subject,
        harness_committed). ``keep_head`` snapshots without moving HEAD or
        the index (the exit snapshot). Raises ``SubmissionError`` with
        model-readable guidance."""
        try:
            result = await self._git_exec(
                utils.read_script("record_commit.sh"),
                {
                    "wd": self._workdir,
                    "commit": ref,
                    "idx": str(int(index)),
                    "stage_timeout": self._stage_timeout_sec(),
                    "keep_head": "1" if keep_head else "",
                },
                self._submission_timeout_sec,
            )
        except TimeoutError as e:
            # Modal's SandboxTimeoutError subclasses TimeoutError but means
            # the sandbox died — classify before recording a failure.
            tools.raise_if_environment_gone(e)
            raise SubmissionError(
                f"recording timed out after {self._submission_timeout_sec:g}s"
            ) from None
        except tools.EnvironmentGoneError:
            raise
        except Exception as e:
            tools.raise_if_environment_gone(e)
            raise SubmissionError(f"recording failed: {e}") from e
        if result.return_code != 0:
            detail = tools.format_exec_output(result.stdout, result.stderr)
            raise SubmissionError(tools.truncate_middle(detail, 500))
        return self._parse_record_output(str(result.stdout or ""))

    def _stage_timeout_sec(self) -> str:
        """Seconds staging may take, or "" (unbounded) when the submission's
        time cannot fit the bound plus its cleanup: the exec's own timeout
        would then kill git first and skip the cleanup."""
        stage = int(self._submission_timeout_sec - STAGE_RESERVE_SEC)
        return str(stage) if stage >= MIN_STAGE_TIMEOUT_SEC else ""

    @staticmethod
    def _parse_record_output(stdout: str) -> tuple[str, str, bool]:
        """Lenient on purpose: a zero exit already means the ref was pinned,
        so a malformed line degrades to empty fields, never to a failure."""
        hash_ = ""
        description = ""
        harness_committed = False
        for line in stdout.strip().splitlines():
            if line.startswith("subject="):
                description = line.removeprefix("subject=").strip()
            elif line.startswith("hash="):
                hash_ = line.removeprefix("hash=").strip()
            elif line.startswith("committed="):
                harness_committed = line.removeprefix("committed=").strip() == "1"
        return hash_, description, harness_committed

    @staticmethod
    def _since(started: float) -> float:
        return round(time.monotonic() - started, 3)

    def _append(self, entry: dict[str, Any]) -> None:
        """In memory always, to the JSONL best-effort."""
        self._submissions.append(entry)
        try:
            self._record_path.parent.mkdir(parents=True, exist_ok=True)
            with self._record_path.open("a") as f:
                f.write(json.dumps({"ts": time.time(), **entry}) + "\n")
        except Exception as e:
            self._logger.warning(f"Could not write submissions.jsonl: {e}")

    def valid_hash_count(self) -> int:
        """Successful rows that carry a complete Git object id."""
        return len(self._restorable())

    def _restorable(self) -> list[dict[str, Any]]:
        return [
            entry
            for entry in self._submissions
            if entry.get("ok")
            and isinstance(entry.get("hash"), str)
            and _FULL_COMMIT_OID.fullmatch(entry["hash"]) is not None
        ]

    def summary(self) -> list[dict[str, Any]]:
        """Per-submission rows for run metadata and ``list_submissions``."""
        return [
            {k: v for k, v in entry.items() if k != "event"}
            for entry in self._submissions
        ]

    def _winner(self) -> dict[str, Any] | None:
        winners = self._restorable()
        chosen = [entry for entry in winners if not entry.get("auto")] or winners
        return chosen[-1] if chosen else None

    def winner_hash(self) -> str | None:
        """The commit grading restores: the newest successful submission the
        model made, else the newest end-of-run auto-submission. A run that
        times out after submitting is graded on the tracked files the model
        submitted, not on its later edits to them; files it created after that
        submission stay in place (restore_winner.sh), and the auto row stays
        on the record."""
        winner = self._winner()
        return None if winner is None else winner["hash"]

    def winner_index(self) -> int | None:
        """The ``index`` of the row ``winner_hash`` comes from."""
        winner = self._winner()
        return None if winner is None else winner["index"]

    async def export(
        self, index: int, commit: str, dest: Path, max_mb: int | None
    ) -> dict[str, Any]:
        """Copy submission #``index`` off the sandbox into ``dest`` for a
        regrade after a sandbox death: ``sub.tar`` (the files it adds or
        changes against the run's baseline, from the commit) and
        ``deleted.z``, plus ``manifest.json``. ``dest`` always describes the
        newest export attempt: a failed one leaves only its manifest
        (``ok: false``), never an older submission's files. A dead sandbox
        raises EnvironmentGoneError and leaves ``dest`` as it was."""
        manifest: dict[str, Any] = {
            "index": index,
            "commit": commit,
            # Where a replay unpacks sub.tar (paths in it are relative to it).
            "workdir": self._workdir,
            "ok": False,
            "started_elapsed_sec": round(self._elapsed_sec(), 1),
        }
        partial = dest.with_name(dest.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        started = time.monotonic()
        sandbox_alive = True
        try:
            result = await self._git_exec(
                utils.read_script("export_submission.sh"),
                {
                    "wd": self._workdir,
                    "commit": commit,
                    "out_dir": EXPORT_SANDBOX_DIR,
                    "max_kb": str(int(max_mb) * 1024) if max_mb else "",
                },
                EXPORT_TIMEOUT_SEC,
            )
            if result.return_code != 0:
                detail = tools.format_exec_output(result.stdout, result.stderr)
                manifest["reason"] = tools.truncate_middle(detail, 300)
                manifest["exit_code"] = result.return_code
            else:
                for line in str(result.stdout or "").splitlines():
                    key, _, value = line.partition("=")
                    if key in ("bytes", "files", "deleted"):
                        manifest[key] = int(value) if value.isdigit() else value
                    elif key == "base":
                        manifest["base"] = value
                for name in ("sub.tar", "deleted.z"):
                    await self._environment.download_file(
                        f"{EXPORT_SANDBOX_DIR}/{name}", partial / name
                    )
                manifest["ok"] = True
        except Exception as e:
            try:
                if isinstance(e, tools.EnvironmentGoneError):
                    raise
                tools.raise_if_environment_gone(e)
            except tools.EnvironmentGoneError:
                sandbox_alive = False
                shutil.rmtree(partial, ignore_errors=True)
                raise
            manifest["reason"] = f"{type(e).__name__}: {e}"[:300]
        except BaseException:
            # Cancelled (the run is ending): no partial copy, no sandbox calls.
            sandbox_alive = False
            shutil.rmtree(partial, ignore_errors=True)
            raise
        finally:
            if sandbox_alive:
                with contextlib.suppress(Exception):
                    await self._environment.exec(
                        f"rm -rf {EXPORT_SANDBOX_DIR}", timeout_sec=60, user="root"
                    )
        if not manifest["ok"]:
            for name in ("sub.tar", "deleted.z"):
                (partial / name).unlink(missing_ok=True)
        manifest["export_sec"] = self._since(started)
        (partial / "manifest.json").write_text(json.dumps(manifest, indent=2))
        shutil.rmtree(dest, ignore_errors=True)
        partial.replace(dest)
        return manifest

    async def restore_winner(self) -> bool:
        """Restore the winning commit; True once the workspace holds it. A
        dead sandbox raises EnvironmentGoneError; other failures are logged."""
        winner = self.winner_hash()
        if not winner:
            self._logger.warning("Winner restore skipped: no valid submission exists.")
            return False
        try:
            result = await self._git_exec(
                utils.read_script("restore_winner.sh"),
                {"wd": self._workdir, "winner": winner},
                RESTORE_TIMEOUT_SEC,
            )
        except tools.EnvironmentGoneError:
            raise
        except Exception as e:
            tools.raise_if_environment_gone(e)
            self._logger.warning(f"Winner restore failed: {e}")
            return False
        if result.return_code != 0:
            detail = tools.format_exec_output(result.stdout, result.stderr)
            self._logger.warning(
                f"Winner restore failed: {tools.truncate_middle(detail, 300)}"
            )
            return False
        self._logger.info(f"Restored winning submission {winner[:12]}.")
        return True
