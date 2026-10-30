"""The record script against a REAL repository (every other suite fakes the
git exec): a commitless submit must commit the live workspace and pin THAT
commit — never silently bind to a stale HEAD that misses uncommitted work.
A clean workspace reuses HEAD; an explicit ref is pinned as-is and leaves
the tree untouched."""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from harbor.environments.base import ExecResult

from openworld_harness.submissions import SubmissionError, Submissions

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        shutil.which("git") is None or shutil.which("bash") is None,
        reason="requires git and bash on PATH",
    ),
]

logger = logging.getLogger(__name__)


def _base_env(home: Path) -> dict[str, str]:
    """Hermetic git: no user/system config (hooks, signing) may leak in."""
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
    }


class LocalGitEnv:
    """Duck-typed environment that runs harness scripts FOR REAL with bash,
    the way a sandbox would: inputs as exec env vars, nothing spliced."""

    def __init__(self, base_env: dict[str, str]):
        self._base_env = base_env

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            env={**self._base_env, **(env or {})},
            cwd=cwd,
            timeout=timeout_sec,
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )


def git(repo: Path, home: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args],
        capture_output=True,
        text=True,
        check=True,
        env=_base_env(home),
    )
    return result.stdout.strip()


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path, Submissions]:
    """A real repo with a baseline commit, plus a store wired to run the
    scripts against it. Returns (repo, home, submissions)."""
    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, home, "init", "-q")
    git(repo, home, "config", "user.email", "test@localhost")
    git(repo, home, "config", "user.name", "Test")
    (repo / "main.py").write_text("print('v1')\n")
    git(repo, home, "add", "-A")
    git(repo, home, "commit", "-qm", "baseline", "--no-verify")
    subs = Submissions(
        tmp_path / "logs",
        LocalGitEnv(_base_env(home)),  # type: ignore[arg-type]
        workdir=str(repo),
        elapsed_sec=lambda: 0.0,
        logger=logger,
    )
    return repo, home, subs


async def test_commitless_submit_on_dirty_workspace_commits_the_live_tree(workspace):
    repo, home, subs = workspace
    baseline = git(repo, home, "rev-parse", "HEAD")
    (repo / "main.py").write_text("print('v2')\n")  # modified, uncommitted
    (repo / "extra.py").write_text("print('new')\n")  # untracked

    submission = await subs.submit("")

    # A NEW commit, never the stale HEAD, and it is what got pinned.
    assert submission.hash != baseline
    assert len(submission.hash) == 40
    assert submission.harness_committed is True
    assert submission.description == "Proximus submission snapshot"
    pinned = git(repo, home, "rev-parse", "refs/proximus/submissions/1")
    assert pinned == submission.hash
    # The live work — modified AND untracked — is inside that commit.
    tree = git(repo, home, "ls-tree", "-r", "--name-only", submission.hash)
    assert "extra.py" in tree.splitlines()
    assert "print('v2')" in git(repo, home, "show", f"{submission.hash}:main.py")
    assert git(repo, home, "status", "--porcelain") == ""
    # The row says the harness made the commit (like the ``auto`` flag).
    row = json.loads((subs._record_path).read_text())
    assert row["ok"] is True
    assert row["hash"] == submission.hash
    assert row["harness_commit"] is True


async def test_commitless_submit_on_clean_workspace_reuses_head(workspace):
    repo, home, subs = workspace
    baseline = git(repo, home, "rev-parse", "HEAD")

    submission = await subs.submit("")

    assert submission.hash == baseline
    assert submission.harness_committed is False
    assert git(repo, home, "rev-parse", "refs/proximus/submissions/1") == baseline
    row = json.loads((subs._record_path).read_text())
    assert "harness_commit" not in row


async def test_explicit_commit_submit_pins_the_named_commit_unchanged(workspace):
    repo, home, subs = workspace
    baseline = git(repo, home, "rev-parse", "HEAD")
    (repo / "main.py").write_text("print('wip, not part of the submission')\n")

    submission = await subs.submit(baseline)

    # Pinned exactly as named; the harness committed nothing and the
    # uncommitted work is still live in the tree.
    assert submission.hash == baseline
    assert submission.harness_committed is False
    assert git(repo, home, "status", "--porcelain") != ""
    row = json.loads((subs._record_path).read_text())
    assert "harness_commit" not in row


async def test_explicit_unknown_ref_fails_with_guidance(workspace):
    repo, home, subs = workspace
    with pytest.raises(SubmissionError) as excinfo:
        await subs.submit("deadbeef")
    assert "not a commit in the workspace repository" in str(excinfo.value)
    assert git(repo, home, "status", "--porcelain") == ""  # nothing touched


async def test_auto_submit_on_dirty_tree_flags_auto_and_harness_commit(workspace):
    # The end-of-run auto-submission goes through the same argless path;
    # when the tree moved, the row carries BOTH flags.
    repo, home, subs = workspace
    (repo / "late.py").write_text("print('kill-time work')\n")

    await subs.submit("", auto=True)

    row = json.loads((subs._record_path).read_text())
    assert row["auto"] is True
    assert row["harness_commit"] is True


async def test_matches_winner_tracks_the_live_tree(workspace):
    # Dedup semantics: a bare ref matches the winner only while the live
    # tree still equals it — new work must read as "does not match" so a
    # confirm (or the exit auto-submission) records a fresh snapshot.
    repo, home, subs = workspace
    submission = await subs.submit("")
    assert await subs.matches_winner("", submission.hash) is True
    (repo / "more.py").write_text("print('newer work')\n")
    assert await subs.matches_winner("", submission.hash) is False


# ----- bounded staging (H4) -----


def shim(bin_dir: Path, name: str, body: str) -> None:
    bin_dir.mkdir(exist_ok=True)
    path = bin_dir / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)


def shimmed(subs: Submissions, home: Path, bin_dir: Path) -> Submissions:
    env = _base_env(home)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    subs._environment = LocalGitEnv(env)  # type: ignore[assignment]
    return subs


@pytest.mark.parametrize("busybox", [False, True])
async def test_staging_that_outlasts_its_limit_is_cleaned_up(
    workspace, tmp_path, monkeypatch, busybox
):
    import openworld_harness.submissions as submissions_module

    repo, home, subs = workspace
    if busybox:
        # busybox's timeout execs the program, so a TERM-killed git add
        # exits 143 instead of GNU's 124.
        real_timeout = shutil.which("timeout")
        shim(
            tmp_path / "bin",
            "timeout",
            f'{real_timeout} "$@"; rc=$?; [ "$rc" -eq 124 ] && exit 143; exit "$rc"',
        )
    monkeypatch.setattr(submissions_module, "MIN_STAGE_TIMEOUT_SEC", 1)
    monkeypatch.setattr(submissions_module, "STAGE_RESERVE_SEC", 29)
    subs._submission_timeout_sec = 30  # staging gets 1 s
    real_git = shutil.which("git")
    # A `git add` that hangs after taking the lock and writing a temp pack.
    shim(
        tmp_path / "bin",
        "git",
        f'if [ "$1" = add ]; then\n'
        f"  d=$({real_git} rev-parse --git-dir)\n"
        f'  touch "$d/index.lock" "$d/objects/pack/tmp_pack_ours"\n'
        f"  exec sleep 30\n"
        f"fi\n"
        f'exec {real_git} "$@"',
    )
    old_pack = repo / ".git" / "objects" / "pack" / "tmp_pack_older"
    old_pack.write_text("not ours")
    os.utime(old_pack, (1_000_000_000, 1_000_000_000))
    (repo / "big.bin").write_bytes(b"x" * 4096)  # untracked: listed as a hint

    with pytest.raises(SubmissionError) as excinfo:
        await shimmed(subs, home, tmp_path / "bin").submit("")
    message = str(excinfo.value)
    assert "did not finish within 1s" in message
    assert "add them to .gitignore" in message
    assert "big.bin" in message
    assert not (repo / ".git" / "index.lock").exists()
    assert not (repo / ".git" / "objects" / "pack" / "tmp_pack_ours").exists()
    assert old_pack.exists()  # an older temp file is not ours to remove
    refs = git(repo, home, "for-each-ref", "refs/proximus")
    assert refs == ""
    assert [row["ok"] for row in subs.summary()] == [False]


async def test_a_leftover_index_lock_fails_with_guidance(workspace):
    repo, home, subs = workspace
    (repo / ".git" / "index.lock").write_text("")
    (repo / "main.py").write_text("print('v2')\n")
    with pytest.raises(SubmissionError, match="index.lock"):
        await subs.submit("")
    assert git(repo, home, "for-each-ref", "refs/proximus") == ""


async def test_a_timeout_without_kill_option_stages_unbounded(workspace, tmp_path):
    repo, home, subs = workspace
    # Old busybox: no -k; staging falls back to a plain `git add -A`.
    shim(
        tmp_path / "bin",
        "timeout",
        'case " $* " in *" -k "*) exit 125 ;; esac\nshift\nexec "$@"',
    )
    (repo / "main.py").write_text("print('v2')\n")
    submission = await shimmed(subs, home, tmp_path / "bin").submit("")
    assert submission.harness_committed is True
    assert "print('v2')" in git(repo, home, "show", f"{submission.hash}:main.py")
