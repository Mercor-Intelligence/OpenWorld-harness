"""Write Harbor job files that run OpenWorldEngineer tasks as the leaderboard ran them.

Usage, from the repository root inside its ``uv sync --locked`` environment::

    python examples/openworldengineer/make_jobs.py --model anthropic/claude-opus-5 \\
        --reasoning-effort max --static-git static-git/git --env modal \\
        --out owe-runs path/to/OpenWorldEngineer/tasks/*

It writes one job file per task to ``OUT/jobs/<task>.json`` and prints the
``harbor run`` command for each one. One job per task is needed because the time
budget and hard timeout differ between tasks, and a Harbor job gives every task the
same agent options. Each job file holds:

- the agent ``owe_agent:OpenWorldEngineerHarness`` (the harness plus the leaderboard's
  web-access section of the system prompt);
- the options the leaderboard's runner set (``leaderboard_defaults``), the task's own
  ``[agent.openworld]`` table if it has one, then your options;
- Harbor's hard agent timeout from ``openworld_harness.hard_timeout_sec``;
- ``n_attempts`` (8, as on the leaderboard);
- a copy of the task under ``OUT/tasks/<task>`` whose agent phase has no network
  (``[agent] network_mode = "no-network"``). The web tools, which run on the host, are
  then the agent's only way out. Files are hard-linked where possible (copied across
  filesystems); the copy's ``task.toml`` is rewritten, and a marker file records the
  source.

Every task is checked before anything is written. A task is skipped, with the reason,
when it needs a GPU and the environment is ``docker`` (Harbor's Docker environment
allocates none), or when it has no ``tests/`` (its grader is private; pass
``--allow-ungraded`` to run it with verification off).

Model credentials are read from your shell, as the harness reads them by default.
The web tools need ``EXA_API_KEY`` in the shell that runs Harbor.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import shutil
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.cli.utils import parse_kwargs
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import JobConfig
from harbor.models.task.config import TaskConfig

from openworld_harness import hard_timeout_sec, task_agent_kwargs

AGENT_IMPORT_PATH = "owe_agent:OpenWorldEngineerHarness"
# Every leaderboard run searched Exa's copy of the web as of this date.
WEB_SNAPSHOT_AS_OF = "2026-10-01"
ATTEMPTS = 8
NO_NETWORK = "no-network"
# Marks a task copy this script made, so it is the only directory it ever replaces.
COPY_MARKER = ".owe-offline-copy"

_TABLE_HEADER = re.compile(r"^\s*\[")
_AGENT_HEADER = re.compile(r"^\s*\[\s*agent\s*\]\s*(#.*)?$")
_AGENT_NETWORK_KEY = re.compile(r"^\s*(network_mode|allowed_hosts)\s*=")


def sandbox_limits(task_toml: str) -> dict[str, Any] | None:
    """The task's sandbox resources, which the harness tells the model and whose
    memory its watchdog enforces; None when the task declares none."""
    env = TaskConfig.model_validate_toml(task_toml).environment
    limits: dict[str, Any] = {}
    if env.cpus:
        limits["cpus"] = env.cpus
    if env.memory_mb:
        limits["memory_mb"] = env.memory_mb
    if env.gpus:
        limits["gpus"] = env.gpus
        if env.gpu_types and len(env.gpu_types) == 1:
            limits["gpu_type"] = env.gpu_types[0]
    return limits or None


def leaderboard_defaults(
    task_toml: str, static_git: str | None = None
) -> dict[str, Any]:
    """The lowest layer of agent options, as the leaderboard's runner set it; the
    task's tables and your options override it."""
    defaults: dict[str, Any] = {
        # Keep multi-GB inputs (weights, datasets) out of the baseline commit, which
        # would otherwise time out hashing them.
        "large_input_exclude_mb": 100,
        "large_input_total_mb": 256,
        # LiteLLM imports its MCP gateway handler on every call with tools; the
        # harness uses no MCP gateway.
        "llm_call_kwargs": {"_skip_mcp_handler": True},
    }
    limits = sandbox_limits(task_toml)
    if limits:
        defaults["sandbox_limits"] = limits
    if static_git:
        defaults["static_git_path"] = static_git
    return defaults


def offline_toml(text: str) -> str:
    """task.toml text with the agent phase set to no-network. Raises ValueError when
    the file can't be rewritten safely."""
    data = tomllib.loads(text)
    for i, step in enumerate(data.get("steps") or []):
        agent = (step or {}).get("agent") or {}
        if "network_mode" in agent or "allowed_hosts" in agent:
            # A step's own agent policy wins over the task's, so the override would
            # not hold.
            raise ValueError(f"steps[{i}] sets its own agent network policy")
    lines = text.splitlines(keepends=True)
    header = next(
        (i for i, line in enumerate(lines) if _AGENT_HEADER.match(line)), None
    )
    setting = f'network_mode = "{NO_NETWORK}"\n'
    if header is None:
        tail = "" if not lines or lines[-1].endswith("\n") else "\n"
        out = f"{text}{tail}\n[agent]\n{setting}"
    else:
        end = next(
            (j for j in range(header + 1, len(lines)) if _TABLE_HEADER.match(lines[j])),
            len(lines),
        )
        body = [
            line
            for line in lines[header + 1 : end]
            if not _AGENT_NETWORK_KEY.match(line)
        ]
        out = "".join([*lines[: header + 1], setting, *body, *lines[end:]])
    try:
        parsed = tomllib.loads(out)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"forcing the agent offline breaks task.toml: {e}") from e
    expected = copy.deepcopy(data)
    agent = expected.setdefault("agent", {})
    agent.pop("allowed_hosts", None)
    agent["network_mode"] = NO_NETWORK
    if parsed != expected:
        raise ValueError(
            "task.toml's [agent] network settings are not plain one-line keys of an "
            "[agent] table, so the agent cannot be forced offline"
        )
    return out


def offline_task_copy(task_dir: Path, dest: Path, text: str) -> Path:
    """A copy of ``task_dir`` at ``dest`` with ``text`` (from ``offline_toml``) as
    its task.toml. Files are hard-linked (copied across filesystems), so the
    original task is never modified. Replaces only an earlier copy made here."""
    if dest.exists():
        if not (dest / COPY_MARKER).is_file():
            raise ValueError(
                f"{dest} exists and is not a task copy made by this script"
            )
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def link(src: str, dst: str) -> None:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    shutil.copytree(task_dir, dest, symlinks=True, copy_function=link)
    (dest / "task.toml").unlink()  # a hard link: never write through it
    (dest / "task.toml").write_text(text)
    (dest / COPY_MARKER).write_text(f"{task_dir}\n")
    return dest


@dataclass
class Plan:
    """One task's job, checked and ready to write."""

    task_dir: Path  # resolved
    name: str
    toml: str  # the offline task.toml
    kwargs: dict[str, Any]
    verify: bool


def plan_task(
    task_dir: Path,
    out: Path,
    model: str,
    overrides: dict[str, Any],
    static_git: str | None,
    environment: str,
    allow_ungraded: bool,
) -> Plan | str:
    """The checked job for one task, or the reason it is skipped. Raises ValueError
    for a task that cannot be run as given. Writes nothing."""
    task_dir = task_dir.resolve()
    name = task_dir.name
    if not name or not (task_dir / "task.toml").is_file():
        raise ValueError(f"{task_dir}: not a task directory (no task.toml)")
    out = out.resolve()
    if out.is_relative_to(task_dir):
        raise ValueError(f"--out {out} is inside the task {task_dir}")
    if task_dir.is_relative_to(out):
        raise ValueError(f"{task_dir} is inside --out {out}; pass the original task")
    text = (task_dir / "task.toml").read_text()
    if (
        TaskConfig.model_validate_toml(text).environment.gpus
        and environment == "docker"
    ):
        return (
            "needs a GPU, and Harbor's docker environment allocates none (--env modal)"
        )
    verify = (task_dir / "tests").is_dir()
    if not verify and not allow_ungraded:
        return "no tests/ (private grader); --allow-ungraded runs it unscored"
    kwargs = task_agent_kwargs(
        task_dir,
        model,
        overrides=overrides,
        defaults=leaderboard_defaults(text, static_git),
    )
    return Plan(task_dir, name, offline_toml(text), kwargs, verify)


def job_config(
    plan: Plan, out: Path, model: str, arm: str, attempts: int, environment: str
) -> dict[str, Any]:
    """Write the task copy and return its Harbor job, validated against JobConfig."""
    copy_dir = offline_task_copy(
        plan.task_dir, out.resolve() / "tasks" / plan.name, plan.toml
    )
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")
    job: dict[str, Any] = {
        "job_name": f"{plan.name}--{slug}--{arm}",
        "jobs_dir": str(out.resolve() / "runs"),
        "n_attempts": attempts,
        "environment": {"type": environment},
        "agents": [
            {
                "import_path": AGENT_IMPORT_PATH,
                "model_name": model,
                "override_timeout_sec": hard_timeout_sec(plan.kwargs),
                "kwargs": plan.kwargs,
            }
        ],
        "tasks": [{"path": str(copy_dir)}],
    }
    if not plan.verify:
        job["verifier"] = {"disable": True}
    JobConfig.model_validate(job)
    return job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write Harbor job files that run OpenWorldEngineer tasks with "
        "the leaderboard's harness settings."
    )
    parser.add_argument("tasks", nargs="+", type=Path, help="task directories")
    parser.add_argument("--model", required=True, help="LiteLLM model name")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--reasoning-effort", help="the model's reasoning effort")
    parser.add_argument(
        "--static-git",
        type=Path,
        help="static git binary for images without git "
        "(build it with scripts/build-static-git.sh)",
    )
    parser.add_argument(
        "--web-as-of",
        default=WEB_SNAPSHOT_AS_OF,
        help=f"Exa snapshot date for the web tools (default {WEB_SNAPSHOT_AS_OF})",
    )
    parser.add_argument(
        "--no-web", action="store_true", help="run without the web tools"
    )
    parser.add_argument("--attempts", type=int, default=ATTEMPTS)
    parser.add_argument(
        "--env",
        default="docker",
        choices=[e.value for e in EnvironmentType],
        metavar="TYPE",
        help="Harbor environment type (default docker; the leaderboard ran on modal)",
    )
    parser.add_argument(
        "--allow-ungraded",
        action="store_true",
        help="also write jobs, with verification off, for tasks without tests/",
    )
    parser.add_argument(
        "--ak",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="more harness options, parsed like harbor run --ak; they override "
        "the leaderboard's",
    )
    args = parser.parse_args(argv)
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")

    overrides: dict[str, Any] = {}
    if args.reasoning_effort:
        overrides["reasoning_effort"] = args.reasoning_effort
    if not args.no_web:
        overrides["web_snapshot_as_of"] = args.web_as_of
    try:
        overrides.update(parse_kwargs(args.ak))
    except ValueError as e:
        parser.error(str(e))
    arm = (
        f"web-{overrides['web_snapshot_as_of']}"
        if overrides.get("web_snapshot_as_of")
        else "noweb"
    )
    static_git = None
    if args.static_git is not None:
        if not args.static_git.is_file():
            parser.error(f"--static-git: no file at {args.static_git}")
        static_git = str(args.static_git.resolve())

    # Check every task before writing anything.
    plans: list[Plan] = []
    names: dict[str, Path] = {}
    for task_dir in args.tasks:
        try:
            plan = plan_task(
                task_dir,
                args.out,
                args.model,
                overrides,
                static_git,
                args.env,
                args.allow_ungraded,
            )
        except ValueError as e:
            parser.error(str(e))
        if isinstance(plan, str):
            print(f"skipped {task_dir}: {plan}", file=sys.stderr)
            continue
        if plan.name in names:
            parser.error(
                f"two tasks are named {plan.name}: {names[plan.name]}, {task_dir}"
            )
        names[plan.name] = task_dir
        plans.append(plan)

    jobs_dir = args.out / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    here = shlex.quote(str(Path(__file__).resolve().parent))
    for plan in plans:
        try:
            job = job_config(plan, args.out, args.model, arm, args.attempts, args.env)
        except ValueError as e:
            parser.error(str(e))
        path = (jobs_dir / f"{plan.name}.json").resolve()
        path.write_text(json.dumps(job, indent=2) + "\n")
        note = "" if plan.verify else "  # unscored"
        print(
            f"PYTHONPATH={here}${{PYTHONPATH:+:$PYTHONPATH}} harbor run -c "
            f"{shlex.quote(str(path))}{note}"
        )
    if static_git is None and plans:
        print(
            "note: without --static-git, tasks whose image has no git fail at setup",
            file=sys.stderr,
        )
    if "web_snapshot_as_of" in overrides and not os.environ.get("EXA_API_KEY"):
        print("note: the web tools need EXA_API_KEY when Harbor runs", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
