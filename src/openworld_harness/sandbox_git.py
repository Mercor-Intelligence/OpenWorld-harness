"""Setup steps that let submissions work in task images built without git in mind.

Ported from the RWE eval platform's ``PlatformProximus``; both are off by default.

- Submissions are commits in a git repository inside the workspace, so a sandbox
  without ``git`` fails at setup. With ``static_git_path`` set, setup copies that
  statically linked binary to /usr/local/bin/git when the sandbox has none. The
  run's ``platform_git`` metadata records "present" (the image's own git) or
  "installed".
- Before the first turn the whole workspace is committed as a baseline, within
  ``submissions.GIT_TIMEOUT_SEC``. A workspace that ships multi-GB inputs (model
  weights, datasets) cannot be hashed in time. With ``large_input_exclude_mb`` set,
  files already in a fresh workspace and larger than that are added to the
  submission excludes; they stay in place, only git ignores them. The run's
  ``platform_git_excludes`` metadata counts them.
"""

import logging
import re
import shlex
from pathlib import Path, PurePosixPath
from typing import Any

from openworld_harness import tools

SANDBOX_GIT = "/usr/local/bin/git"
# Without a templates directory every `git init` warns; an empty one is enough
# (submissions create .git/info themselves).
SANDBOX_TEMPLATES = "/usr/local/share/git-core/templates"
NOT_A_FRESH_WORKSPACE = 3

logger = logging.getLogger(__name__)


async def ensure_git(environment: Any, static_git: Path) -> str:
    """Make ``git`` available in the sandbox. Returns "present" or "installed"."""
    found = await environment.exec("command -v git", timeout_sec=30, user="root")
    if found.return_code == 0:
        return "present"
    if not static_git.is_file():
        raise RuntimeError(f"the task image has no git, and {static_git} is missing")
    parent = PurePosixPath(SANDBOX_GIT).parent
    await environment.exec(
        f"mkdir -p {parent} {SANDBOX_TEMPLATES}", timeout_sec=30, user="root"
    )
    await environment.upload_file(static_git, SANDBOX_GIT)
    check = await environment.exec(
        f"chmod 0755 {SANDBOX_GIT} && git --version", timeout_sec=60, user="root"
    )
    if check.return_code != 0:
        detail = (check.stderr or check.stdout or "").strip()[:500]
        raise RuntimeError(f"installing git in the sandbox failed: {detail}")
    return "installed"


def gitignore_literal(path: str) -> str:
    """An anchored .gitignore pattern matching exactly this workspace-relative
    path. Trailing spaces are escaped too: git strips them otherwise."""
    escaped = re.sub(r"([\\*?\[\]!#])", r"\\\1", path)
    stripped = escaped.rstrip(" ")
    return "/" + stripped + "\\ " * (len(escaped) - len(stripped))


async def large_inputs(environment: Any, workdir: str, min_mb: int) -> list[str] | None:
    """Patterns for the files over ``min_mb`` MB in ``workdir``, or None when it is
    already in a git repository, its own or an enclosing one (a task's own
    repository is kept as it is, and setup writes no excludes into it).

    Mounted filesystems are searched too, since ``git add -A`` stages them. POSIX
    ``find`` only (busybox has no ``-printf``). Best effort, because the
    excludes only save time: a listing that exits nonzero (an error, or a
    timeout some environments report as -1) keeps what it found and logs the
    rest, and one whose exec raises (a timeout, a transient error) excludes
    nothing. A dead sandbox still raises."""
    wd = shlex.quote(workdir)
    try:
        result = await environment.exec(
            # $PWD is absolute, so the walk up always reaches /.
            f'cd {wd} || exit 1; d=$PWD; while :; do test -e "$d/.git" &&'
            f' exit {NOT_A_FRESH_WORKSPACE}; [ "$d" = / ] && break;'
            ' d=$(dirname "$d"); done;'
            f" find . -type f -size +{int(min_mb) * 1024}k",
            timeout_sec=60,
            user="root",
        )
    except Exception as e:
        tools.raise_if_environment_gone(e)
        logger.warning(
            "listing large workspace files %s; none excluded",
            "timed out" if isinstance(e, TimeoutError) else f"failed: {e!r}",
        )
        return []
    if result.return_code == NOT_A_FRESH_WORKSPACE:
        return None
    tools.raise_if_exec_result_gone(result)
    paths = [
        line[2:]
        for line in (result.stdout or "").splitlines()
        if line.startswith("./") and len(line) > 2
    ]
    if result.return_code != 0:
        logger.warning(
            "listing large workspace files exited %s; excluding the %d found: %s",
            result.return_code,
            len(paths),
            str(result.stderr or "").strip()[:500],
        )
    return [gitignore_literal(p) for p in paths]
