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

import re
import shlex
from pathlib import Path, PurePosixPath
from typing import Any

SANDBOX_GIT = "/usr/local/bin/git"
# Without a templates directory every `git init` warns; an empty one is enough
# (submissions create .git/info themselves).
SANDBOX_TEMPLATES = "/usr/local/share/git-core/templates"
NOT_A_FRESH_WORKSPACE = 3


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
    path."""
    return "/" + re.sub(r"([\\*?\[\]!#])", r"\\\1", path)


async def large_inputs(environment: Any, workdir: str, min_mb: int) -> list[str] | None:
    """Patterns for the files over ``min_mb`` MB in ``workdir``, or None when it is
    already a git repository (a task's own repository is kept as it is)."""
    wd = shlex.quote(workdir)
    result = await environment.exec(
        f"test -e {wd}/.git && exit {NOT_A_FRESH_WORKSPACE};"
        f" find {wd} -xdev -type f -size +{int(min_mb)}M -printf '%P\\n'",
        timeout_sec=60,
        user="root",
    )
    if result.return_code == NOT_A_FRESH_WORKSPACE:
        return None
    if result.return_code != 0:
        detail = f"listing large workspace files failed: {result.stderr or ''}"
        raise RuntimeError(detail[:500])
    return [gitignore_literal(p) for p in (result.stdout or "").splitlines() if p]
