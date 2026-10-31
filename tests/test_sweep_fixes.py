"""Fixes from the RWE 735c43a sweep (RWE-eval-platform HARNESS_ISSUES.md).

- H10: a provider refusal withholds the newest tool output once, then ends the
  run normally so the restored winner is graded.
- H3: a budget that runs out during provider retries is graded unless the
  provider ate more than 10% of it; failed attempts become trajectory steps.
- H7: no compaction in the last BUDGET_MARGIN_SEC, and no turn after a
  compaction that used up the budget.
- H6: the compaction handoff carries the harness's own submission list.
- H1: each recorded submission is exported off the sandbox (export_submission.sh
  against a real repository here), and the submit text states the guarantee.
- H8: the sandbox's limits are in the first message and limits.json.
- P2: large inputs are also excluded by their total size.
"""

import asyncio
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import litellm
import pytest
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext

import openworld_harness.submissions as submissions_mod
from openworld_harness import llm as llm_mod
from openworld_harness import prompts, sandbox_git, tools
from openworld_harness.agent import LIMITS_PATH, sandbox_limits_note
from openworld_harness.config import ProximusConfig
from openworld_harness.submissions import Submissions
from tests.test_proximus_loop import (
    FakeCompletion,
    FakeEnv,
    bash_response,
    finish_responses,
    make_agent,
    read_messages,
    read_trajectory,
    small_context,
    submit_response,
    text_response,
)
from tests.test_proximus_record_commit import LocalGitEnv, _base_env, git

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)


def cyber_refusal() -> litellm.exceptions.BadRequestError:
    return litellm.exceptions.BadRequestError(
        message=(
            'OpenAIException - {"error": {"message": "This request was flagged for '
            'possible cybersecurity risk.", "type": "invalid_request_error", '
            '"code": "cyber_policy"}}'
        ),
        model="gpt-test",
        llm_provider="openai",
    )


def refusals(n: int = llm_mod.REFUSAL_ATTEMPTS) -> list[Exception]:
    return [cyber_refusal() for _ in range(n)]


def harness_kinds(agent) -> list[str]:
    return [
        (s.get("extra") or {}).get("harness_message")
        for s in read_trajectory(agent)["steps"]
        if (s.get("extra") or {}).get("harness_message")
    ]


# ----- H10: provider refusals -----


async def test_refusal_withholds_the_newest_output_then_ends_graded(
    tmp_path, monkeypatch
):
    fake = FakeCompletion(
        [
            submit_response(),
            bash_response("gdb -batch -ex bt ./a.out"),
            *refusals(),  # the request carrying the backtrace
            *refusals(),  # still refused without it
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)
    agent = make_agent(tmp_path)
    env = FakeEnv()
    context = AgentContext()
    await agent.run("Task", env, context)  # type: ignore[arg-type]

    assert len(fake.calls) == 2 + 2 * llm_mod.REFUSAL_ATTEMPTS
    metadata = context.metadata or {}
    assert metadata["ended_by"] == "provider_failure"
    assert metadata["provider_failure"]["kind"] == "refusal"
    assert metadata["provider_failure"]["code"] == "cyber_policy"
    # The second request no longer carried the backtrace.
    second = fake.calls[2 + llm_mod.REFUSAL_ATTEMPTS]["messages"]
    tool_outputs = [m["content"] for m in second if m.get("role") == "tool"]
    assert tool_outputs[-1] == prompts.WITHHELD_OUTPUT
    assert "provider_refusal" in harness_kinds(agent)
    # Ended normally: the winner was restored for grading.
    assert any("checkout" in c for c in env.all_commands)


async def test_refusal_with_nothing_to_withhold_ends_at_once(tmp_path, monkeypatch):
    fake = FakeCompletion(refusals())
    monkeypatch.setattr(litellm, "acompletion", fake)
    agent = make_agent(tmp_path)
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]
    assert len(fake.calls) == llm_mod.REFUSAL_ATTEMPTS
    assert (context.metadata or {})["ended_by"] == "provider_failure"


async def test_a_passing_replay_resets_the_refusal_count(tmp_path, monkeypatch):
    fake = FakeCompletion(
        [
            bash_response("ls"),
            *refusals(),
            text_response("carrying on"),
            bash_response("ls"),
            *refusals(),
            *finish_responses(),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)
    agent = make_agent(tmp_path)
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]
    assert (context.metadata or {})["ended_by"] == "submit"
    assert harness_kinds(agent).count("provider_refusal") == 2


async def test_content_policy_flags_are_capped(monkeypatch):
    calls = {"n": 0}

    async def flagged(**kwargs):
        calls["n"] += 1
        raise litellm.exceptions.ContentPolicyViolationError(
            message="Invalid prompt: flagged as potentially violating our usage policy.",
            model="gpt-test",
            llm_provider="openai",
        )

    monkeypatch.setattr(litellm, "acompletion", flagged)
    model = llm_mod.LLM(
        "openai/gpt-test", ProximusConfig(), lambda: 3600.0, logging.getLogger("t")
    )
    with pytest.raises(llm_mod.ProviderRefusalError) as raised:
        await model.query([{"role": "user", "content": "hi"}], tools=None)
    assert raised.value.code == "content_policy"
    assert calls["n"] == llm_mod.REFUSAL_ATTEMPTS


def test_refusal_codes():
    assert llm_mod.provider_refusal_code(cyber_refusal()) == "cyber_policy"
    other = litellm.exceptions.BadRequestError(
        message="invalid tool schema", model="m", llm_provider="openai"
    )
    assert llm_mod.provider_refusal_code(other) is None
    assert not llm_mod._is_retryable(other)


# ----- H3: budget expiry during provider retries -----


def run_out_during_a_retry(agent, monkeypatch, *, lost_sec: float):
    """Turn 1 submits; turn 2's request times out just as the budget ends."""
    clock = {"t": 0.0}
    monkeypatch.setattr(agent, "_elapsed", lambda: clock["t"])
    queue: list[Any] = [submit_response()]

    async def acompletion(**kwargs):
        if queue:
            return queue.pop(0)
        clock["t"] = 1000.0  # past the 100 s budget
        agent._llm.provider_lost_sec += lost_sec
        raise litellm.exceptions.Timeout("provider hung", "openai", "gpt-test")

    monkeypatch.setattr(litellm, "acompletion", acompletion)


async def test_budget_spent_in_a_retry_is_graded_when_little_was_lost(
    tmp_path, monkeypatch
):
    agent = make_agent(tmp_path, time_budget_sec=100)
    run_out_during_a_retry(agent, monkeypatch, lost_sec=1.0)
    env = FakeEnv()
    context = AgentContext()
    await agent.run("Task", env, context)  # type: ignore[arg-type]
    metadata = context.metadata or {}
    assert metadata["ended_by"] == "provider_failure"
    assert metadata["provider_failure"]["kind"] == "retry_budget"
    assert metadata["provider_lost_sec"] >= 1.0
    assert any("checkout" in c for c in env.all_commands)


async def test_a_run_starved_by_the_provider_fails(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, time_budget_sec=100)
    run_out_during_a_retry(agent, monkeypatch, lost_sec=50.0)
    context = AgentContext()
    with pytest.raises(llm_mod.ProviderRetryBudgetError, match="lost to provider"):
        await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]
    assert (context.metadata or {})["ended_by"] == "error"


async def test_failed_attempts_are_trajectory_steps(tmp_path, monkeypatch):
    fake = FakeCompletion(
        [litellm.exceptions.Timeout("hung", "openai", "gpt-test"), *finish_responses()]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)
    agent = make_agent(tmp_path)
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]
    events = [
        s["extra"]["provider_event"]
        for s in read_trajectory(agent)["steps"]
        if (s.get("extra") or {}).get("provider_event")
    ]
    assert [e["kind"] for e in events] == ["retry"]
    assert events[0]["error_type"] == "Timeout" and events[0]["attempt"] == 1
    assert (context.metadata or {})["provider_lost_sec"] >= 0
    # The model never saw it.
    assert not any(
        "Provider call" in str(m.get("content")) for m in read_messages(agent)
    )


async def test_discarded_output_is_still_billed(tmp_path, monkeypatch):
    truncated = text_response("x" * 5000)
    truncated.choices[0].finish_reason = "length"
    fake = FakeCompletion([truncated, *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)
    agent = make_agent(tmp_path)
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]
    # Three billed responses of 10 completion tokens each, the discarded one too.
    assert context.n_output_tokens == 30
    step = next(
        s
        for s in read_trajectory(agent)["steps"]
        if (s.get("extra") or {}).get("discarded_output")
    )
    discard = step["extra"]["discarded_output"]
    assert discard["chars"] == 5000 and "chars ..." in discard["excerpt"]


# ----- H7 / H6: compaction -----


async def test_no_compaction_in_the_last_minutes(tmp_path, monkeypatch):
    fake = FakeCompletion([bash_response("ls", prompt_tokens=900), *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)
    small_context(monkeypatch)
    agent = make_agent(tmp_path, time_budget_sec=1000)
    monkeypatch.setattr(agent, "_elapsed", lambda: 950.0)  # 50 s left
    await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]
    assert len(fake.calls) == 3  # no summary call
    assert "[Context compacted]" not in read_messages(agent)[1]["content"]


async def test_no_turn_after_a_compaction_that_used_up_the_budget(
    tmp_path, monkeypatch
):
    clock = {"t": 0.0}
    queue: list[Any] = [bash_response("ls", prompt_tokens=900)]

    async def acompletion(**kwargs):
        if queue:
            return queue.pop(0)
        clock["t"] = 1100.0  # the summary call ran past the deadline
        return text_response("handoff summary")

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    small_context(monkeypatch)
    agent = make_agent(tmp_path, time_budget_sec=1000)
    monkeypatch.setattr(agent, "_elapsed", lambda: clock["t"])
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]
    steps = read_trajectory(agent)["steps"]
    assert (
        steps[-1]["source"] == "user" and "[Context compacted]" in steps[-1]["message"]
    )
    assert (context.metadata or {})["ended_by"] == "budget"


async def test_handoff_lists_the_recorded_submissions(tmp_path, monkeypatch):
    fake = FakeCompletion(
        [
            submit_response(),
            bash_response("ls", prompt_tokens=900),
            text_response("summary that misremembers the submissions"),
            *finish_responses(),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)
    small_context(monkeypatch)
    agent = make_agent(tmp_path)
    await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]
    handoff = read_messages(agent)[1]["content"]
    assert "Recorded submissions (the harness's record" in handoff
    assert "#1  " in handoff and "<- current result" in handoff
    assert handoff.rstrip().endswith(prompts.HANDOFF_FOOTER)


# ----- H1: the submit text and submission exports -----


def test_submit_text_states_the_real_guarantee():
    text = prompts.SUBMIT_CONTINUE_OR_CONFIRM
    assert "never lost" not in text
    assert (
        "sandbox itself dies" in text and "latest recorded submission is graded" in text
    )


class ExportingEnv(FakeEnv):
    """FakeEnv plus the export script: answers it and serves its files."""

    def __init__(self) -> None:
        super().__init__()
        self.exports: list[str] = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if "diff-tree -r --no-renames --diff-filter=d" in command:
            self.exports.append((env or {}).get("commit", ""))
            return ExecResult(
                stdout="bytes=10240\nfiles=2\ndeleted=1\nbase=abc",
                stderr=None,
                return_code=0,
            )
        if command.startswith("rm -rf /tmp/proximus/export"):
            return ExecResult(stdout="", stderr=None, return_code=0)
        return await super().exec(command, cwd, env, timeout_sec, user)

    async def download_file(self, source_path, target_path):
        Path(target_path).write_text(f"copy of {source_path}")


async def test_the_newest_recorded_submission_is_exported(tmp_path, monkeypatch):
    fake = FakeCompletion([submit_response(), bash_response("ls"), *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)
    agent = make_agent(tmp_path, submission_export=True)
    env = ExportingEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]
    winner = agent.logs_dir / "winner"
    manifest = json.loads((winner / "manifest.json").read_text())
    assert (
        manifest["ok"] is True and manifest["files"] == 2 and manifest["deleted"] == 1
    )
    # The newest recorded submission is the one on disk.
    assert manifest["index"] == 2 and env.exports[-1] == manifest["commit"]
    assert (winner / "sub.tar").read_text() == "copy of /tmp/proximus/export/sub.tar"
    assert not (tmp_path / "logs" / "winner.partial").exists()


async def test_a_dead_sandbox_does_not_wait_for_an_export(tmp_path, monkeypatch):
    fake = FakeCompletion([submit_response(), bash_response("echo hi")])
    monkeypatch.setattr(litellm, "acompletion", fake)
    agent = make_agent(tmp_path, submission_export=True)
    hung = asyncio.Event()

    class HungExportEnv(ExportingEnv):
        """The export never returns; the next bash call finds the sandbox gone."""

        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            if "diff-tree -r --no-renames --diff-filter=d" in command:
                hung.set()
                await asyncio.Event().wait()
            if command.startswith("echo "):
                await hung.wait()  # the export is in flight
                raise tools.EnvironmentGoneError("Modal sandbox is gone")
            return await super().exec(command, cwd, env, timeout_sec, user)

    context = AgentContext()
    with pytest.raises(tools.EnvironmentGoneError):
        await asyncio.wait_for(
            agent.run("Task", HungExportEnv(), context),  # type: ignore[arg-type]
            timeout=10,
        )
    assert (context.metadata or {})["ended_by"] == "sandbox_death"
    assert agent._export_task is not None and agent._export_task.cancelled()
    assert not (agent.logs_dir / "winner").exists()
    assert not (agent.logs_dir / "winner.partial").exists()


@pytest.fixture
def real_repo(tmp_path: Path, monkeypatch):
    if shutil.which("git") is None or shutil.which("awk") is None:
        pytest.skip("requires git and awk")
    monkeypatch.setattr(
        submissions_mod, "EXPORT_SANDBOX_DIR", str(tmp_path / "sandbox-export")
    )
    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, home, "init", "-q")
    git(repo, home, "config", "user.email", "t@localhost")
    git(repo, home, "config", "user.name", "T")
    (repo / "main.py").write_text("print('v1')\n")
    (repo / "old.txt").write_text("to be deleted\n")
    (repo / "keep.txt").write_text("unchanged\n")
    git(repo, home, "add", "-A")
    git(repo, home, "commit", "-qm", "image state", "--no-verify")

    class CopyEnv(LocalGitEnv):
        fail_downloads = False

        async def download_file(self, source_path, target_path):
            if self.fail_downloads:
                raise FileNotFoundError(source_path)
            shutil.copy(source_path, target_path)

    subs = Submissions(
        tmp_path / "logs",
        CopyEnv(_base_env(home)),  # type: ignore[arg-type]
        workdir=str(repo),
        elapsed_sec=lambda: 0.0,
        logger=logging.getLogger("t"),
    )
    return repo, home, subs


def rebuild(image_state: Path, export: Path, into: Path) -> dict[str, str]:
    """What the platform's replay does in a fresh copy of the task image."""
    shutil.copytree(image_state, into, ignore=shutil.ignore_patterns(".git"))
    subprocess.run(["tar", "-xf", str(export / "sub.tar"), "-C", str(into)], check=True)
    for path in (export / "deleted.z").read_bytes().split(b"\0"):
        if path:
            (into / path.decode()).unlink()
    return {
        str(p.relative_to(into)): p.read_text()
        for p in sorted(into.rglob("*"))
        if p.is_file()
    }


async def test_export_rebuilds_the_submission_in_a_fresh_image(real_repo, tmp_path):
    repo, home, subs = real_repo
    await subs.ensure_git()  # pins refs/proximus/baseline at the run's start
    image_state = tmp_path / "image"
    shutil.copytree(repo, image_state)

    (repo / "main.py").write_text("print('v2')\n")
    (repo / "old.txt").unlink()
    weird = repo / "dir with space" / "ñ\ttab.txt"
    weird.parent.mkdir()
    weird.write_text("odd name\n")
    (repo / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (repo / "run.sh").chmod(0o755)
    submission = await subs.submit("")
    (repo / "main.py").write_text("print('later, unsubmitted')\n")  # not exported

    manifest = await subs.export(1, submission.hash, tmp_path / "winner", None)
    assert manifest["ok"] is True, manifest
    assert manifest["files"] == 3 and manifest["deleted"] == 1
    assert manifest["workdir"] == str(repo)
    files = rebuild(image_state, tmp_path / "winner", tmp_path / "rebuilt")
    assert files == {
        "dir with space/ñ\ttab.txt": "odd name\n",
        "keep.txt": "unchanged\n",
        "main.py": "print('v2')\n",
        "run.sh": "#!/bin/sh\necho hi\n",
    }
    assert (tmp_path / "rebuilt" / "run.sh").stat().st_mode & 0o111
    assert not (tmp_path / "sandbox-export").exists()  # cleaned up in the sandbox


async def test_export_over_the_limit_is_refused(real_repo, tmp_path):
    repo, home, subs = real_repo
    await subs.ensure_git()
    (repo / "big.bin").write_bytes(b"\1" * (2 * 1024 * 1024))
    submission = await subs.submit("")
    manifest = await subs.export(1, submission.hash, tmp_path / "winner", 1)
    assert manifest["ok"] is False and "export limit" in manifest["reason"]
    assert sorted(p.name for p in (tmp_path / "winner").iterdir()) == ["manifest.json"]


async def test_export_refuses_a_gitlink(real_repo, tmp_path):
    repo, home, subs = real_repo
    await subs.ensure_git()
    head = git(repo, home, "rev-parse", "HEAD")
    git(repo, home, "update-index", "--add", "--cacheinfo", f"160000,{head},nested")
    git(repo, home, "commit", "-qm", "nested repo", "--no-verify")
    commit = git(repo, home, "rev-parse", "HEAD")
    manifest = await subs.export(1, commit, tmp_path / "winner", None)
    assert manifest["ok"] is False and "gitlink" in manifest["reason"]


async def test_a_failed_export_never_leaves_an_older_submission(real_repo, tmp_path):
    repo, home, subs = real_repo
    await subs.ensure_git()
    (repo / "main.py").write_text("print('v2')\n")
    first = await subs.submit("")
    winner = tmp_path / "winner"
    assert (await subs.export(1, first.hash, winner, None))["ok"] is True
    (repo / "main.py").write_text("print('v3')\n")
    second = await subs.submit("")
    subs._environment.fail_downloads = True
    manifest = await subs.export(2, second.hash, winner, None)
    assert manifest["index"] == 2 and manifest["ok"] is False
    assert "FileNotFoundError" in manifest["reason"]
    # Submission #1's files are gone: a regrade must not mistake them for #2.
    assert sorted(p.name for p in winner.iterdir()) == ["manifest.json"]
    assert json.loads((winner / "manifest.json").read_text()) == manifest


# ----- H8 / E1: sandbox limits -----


def test_limits_note():
    note = sandbox_limits_note(
        {"cpus": 16, "memory_mb": 65536, "gpus": 1, "gpu_type": "H100"}
    )
    assert note is not None
    assert "16 CPUs, 64 GB of memory, 1 H100 GPU(s)" in note
    assert "can kill the whole sandbox" in note and "never kill PID 1" in note
    cpu_only = sandbox_limits_note({"cpus": 2, "memory_mb": 8192})
    assert cpu_only is not None and "PID 1" not in cpu_only
    assert sandbox_limits_note({}) is None


async def test_limits_reach_the_model_and_the_sandbox(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)
    limits = {"cpus": 8, "memory_mb": 32768}
    agent = make_agent(tmp_path, sandbox_limits=limits)
    env = FakeEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]
    first = read_messages(agent)[1]["content"]
    assert "Sandbox limits: 8 CPUs, 32 GB of memory" in first
    written = [c for c in env.all_commands if LIMITS_PATH in c]
    assert len(written) == 1 and '"memory_mb": 32768' in written[0]


def test_sandbox_limits_are_validated():
    from openworld_harness.config import OpenWorldConfig

    with pytest.raises(ValueError, match="unknown keys"):
        OpenWorldConfig(sandbox_limits={"ram": 1})
    with pytest.raises(ValueError, match="non-negative number"):
        OpenWorldConfig(sandbox_limits={"memory_mb": "lots"})


# ----- P2: total-size large-input excludes; H2: errno -5 -----


def test_total_cap_excludes_the_largest_inputs_first():
    lines = [
        f"{kb}\t./{name}"
        for kb, name in [
            (200_000, "huge.bin"),  # over the 100 MB per-file limit
            (30_000, "a.npz"),
            (20_000, "b.npz"),
            (10_000, "c.npz"),
            (5_000, "d.npz"),
        ]
    ]
    # 65 MB stay under the per-file limit; dropping the largest (a, 30 MB)
    # leaves 35 MB, under the 40 MB cap.
    assert sandbox_git._over_total(lines, 100 * 1024, 40 * 1024) == [
        "huge.bin",
        "a.npz",
    ]
    assert sandbox_git._over_total(lines, 100 * 1024, 1024 * 1024) == ["huge.bin"]


async def test_large_inputs_with_a_total_cap_sizes_files():
    class ListingEnv:
        command = ""

        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            ListingEnv.command = command
            return ExecResult(
                stdout="30000\t./x/a.npz\n20000\t./b.npz\n", stderr=None, return_code=0
            )

    patterns = await sandbox_git.large_inputs(ListingEnv(), "/app", 100, 25)
    assert "-exec du -k {} +" in ListingEnv.command
    assert patterns == ["/x/a.npz"]


def test_errno_minus_5_is_a_lost_sandbox():
    error = ConnectionError("[Errno -5] No address associated with hostname")
    assert tools._is_environment_transport_loss(error)
