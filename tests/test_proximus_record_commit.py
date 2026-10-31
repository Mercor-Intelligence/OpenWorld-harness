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
    baseline = git(repo, home, "rev-parse", "HEAD")
    (repo / "late.py").write_text("print('kill-time work')\n")

    submission = await subs.submit("", auto=True)

    row = json.loads((subs._record_path).read_text())
    assert row["auto"] is True
    assert row["harness_commit"] is True
    # Committed beside HEAD: HEAD and the index are as the model left them.
    assert git(repo, home, "rev-parse", "HEAD") == baseline
    assert git(repo, home, "status", "--porcelain") == "?? late.py"
    assert git(repo, home, "rev-parse", f"{submission.hash}^") == baseline
    assert "late.py" in git(repo, home, "ls-tree", "--name-only", submission.hash)
    assert not list((repo / ".git").glob("proximus-snapshot-index*"))
    # A clean tree reuses HEAD, as a bare submit does.
    (repo / "late.py").unlink()
    assert (await subs.submit("", auto=True)).hash == baseline


async def test_restoring_past_the_exit_snapshot_keeps_its_files(workspace):
    """The model commits A and leaves an output untracked; the exit snapshot B
    has it, but A is graded (H7). The restore must leave what checking out A
    would have left without B: the output stays, tracked edits revert."""
    repo, home, subs = workspace
    (repo / "main.py").write_text("print('v2')\n")
    git(repo, home, "commit", "-qam", "fix", "--no-verify")
    a = git(repo, home, "rev-parse", "HEAD")
    await subs.submit(a)
    (repo / "out.csv").write_text("1,2\n")
    (repo / "main.py").write_text("print('v3, unsubmitted')\n")
    b = (await subs.submit("", auto=True)).hash
    assert b != a and subs.winner_hash() == a

    assert await subs.restore_winner()

    assert git(repo, home, "rev-parse", "HEAD") == a
    assert git(repo, home, "rev-parse", "refs/proximus/submissions/2") == b
    assert (repo / "out.csv").read_text() == "1,2\n"
    assert (repo / "main.py").read_text() == "print('v2')\n"


async def test_a_graded_exit_snapshot_is_restored_as_is(workspace):
    repo, home, subs = workspace
    (repo / "out.csv").write_text("1,2\n")
    b = (await subs.submit("", auto=True)).hash

    assert await subs.restore_winner()

    assert git(repo, home, "rev-parse", "HEAD") == b
    assert "out.csv" in git(repo, home, "ls-files")


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


@pytest.mark.parametrize(
    ("busybox", "auto"), [(False, False), (True, False), (False, True)]
)
async def test_staging_that_outlasts_its_limit_is_cleaned_up(
    workspace, tmp_path, monkeypatch, busybox, auto
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
        f'  touch "${{GIT_INDEX_FILE:-$d/index}}.lock" "$d/objects/pack/tmp_pack_ours"\n'
        f"  exec sleep 30\n"
        f"fi\n"
        f'exec {real_git} "$@"',
    )
    old_pack = repo / ".git" / "objects" / "pack" / "tmp_pack_older"
    old_pack.write_text("not ours")
    os.utime(old_pack, (1_000_000_000, 1_000_000_000))
    (repo / "big.bin").write_bytes(b"x" * 4096)  # untracked: listed as a hint

    baseline = git(repo, home, "rev-parse", "HEAD")
    with pytest.raises(SubmissionError) as excinfo:
        await shimmed(subs, home, tmp_path / "bin").submit("", auto=auto)
    message = str(excinfo.value)
    assert "did not finish within 1s" in message
    assert "add them to .git/info/exclude" in message
    assert "big.bin" in message
    assert not list((repo / ".git").glob("*.lock"))
    assert not list((repo / ".git").glob("proximus-snapshot-index*"))
    assert git(repo, home, "rev-parse", "HEAD") == baseline
    assert not (repo / ".git" / "objects" / "pack" / "tmp_pack_ours").exists()
    assert old_pack.exists()  # an older temp file is not ours to remove
    refs = git(repo, home, "for-each-ref", "refs/proximus")
    assert refs == ""
    assert [row["ok"] for row in subs.summary()] == [False]


async def test_a_staging_git_killed_early_is_not_called_a_timeout(workspace, tmp_path):
    repo, home, subs = workspace
    real_git = shutil.which("git")
    # An OOM kill: git add dies of SIGKILL at once, long before the limit.
    shim(
        tmp_path / "bin",
        "git",
        f'if [ "$1" = add ]; then kill -9 $$; fi\nexec {real_git} "$@"',
    )
    (repo / "main.py").write_text("print('v2')\n")
    with pytest.raises(SubmissionError) as excinfo:
        await shimmed(subs, home, tmp_path / "bin").submit("")
    message = str(excinfo.value)
    assert "was killed after" in message and "exit 137" in message
    assert "did not finish" not in message
    # Nothing is untracked, so there is no list of large files.
    assert "Largest untracked files" not in message


async def test_an_exit_snapshot_cut_short_after_its_commit_moves_nothing(
    workspace, tmp_path
):
    """The exec can time out after the snapshot commit (the chown of .git on a
    huge repository): the failed row must not leave HEAD on the snapshot, or
    restoring the winner would delete what only the snapshot tracked."""
    repo, home, subs = workspace
    a = git(repo, home, "rev-parse", "HEAD")
    await subs.submit(a)
    (repo / "out.csv").write_text("1,2\n")
    shim(tmp_path / "bin", "chown", "exec sleep 3")
    subs = shimmed(subs, home, tmp_path / "bin")
    subs._submission_timeout_sec = 1

    with pytest.raises(SubmissionError):
        await subs.submit("", auto=True)

    assert git(repo, home, "rev-parse", "HEAD") == a
    assert await subs.restore_winner()
    assert (repo / "out.csv").read_text() == "1,2\n"


async def test_a_leftover_index_lock_fails_with_guidance(workspace):
    repo, home, subs = workspace
    (repo / ".git" / "index.lock").write_text("")
    (repo / "main.py").write_text("print('v2')\n")
    with pytest.raises(SubmissionError, match="index.lock"):
        await subs.submit("")
    assert git(repo, home, "for-each-ref", "refs/proximus") == ""
    # The exit snapshot uses its own index, so the model's lock does not stop
    # it, and the lock is left alone.
    baseline = git(repo, home, "rev-parse", "HEAD")
    snapshot = await subs.submit("", auto=True)
    assert "print('v2')" in git(repo, home, "show", f"{snapshot.hash}:main.py")
    assert git(repo, home, "rev-parse", "HEAD") == baseline
    assert (repo / ".git" / "index.lock").exists()


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
