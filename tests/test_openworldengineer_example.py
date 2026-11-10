"""examples/openworldengineer: the leaderboard's prompt section and job files."""

import importlib
import json
import shutil
import tomllib
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest
from harbor.agents.factory import AgentFactory
from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import AgentConfig

from openworld_harness import OpenWorldHarness, hard_timeout_sec
from tests.test_proximus_loop import FakeCompletion, FakeEnv, finish_responses

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "openworldengineer"
if not EXAMPLE.is_dir():  # verify_wheel.py runs the suite without examples/
    pytest.skip("examples/ is not next to the tests", allow_module_level=True)

MODEL = "anthropic/claude-test"
WEB_LINE = (
    "You have a web search tool and a page fetch tool. "
    "Use them whenever they would help with the task."
)
TASK_TOML = """\
version = "1.0"

[metadata]
name = "demo"

[agent]
timeout_sec = 14400.0
network_mode = "allowlist"
allowed_hosts = ["example.com"]

[environment]
build_timeout_sec = 600.0
cpus = 4
memory_mb = 8192
gpus = 1
gpu_types = ["H100!"]

[verifier]
timeout_sec = 900.0

[[artifacts]]
source = "/app/"
exclude = ["data/**", "./build"]
"""


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-0123456789abcdef")
    return SimpleNamespace(
        agent=importlib.import_module("owe_agent"),
        jobs=importlib.import_module("make_jobs"),
    )


def make(tmp_path, **kwargs):
    config = AgentConfig(
        import_path="owe_agent:OpenWorldEngineerHarness",
        model_name=MODEL,
        kwargs={"time_budget_sec": 600, **kwargs},
    )
    return AgentFactory.create_agent_from_config(config, logs_dir=tmp_path / "logs")


def test_web_access_section_ends_the_system_prompt(tmp_path, example):
    plain = OpenWorldHarness(
        logs_dir=tmp_path / "plain", model_name=MODEL, time_budget_sec=600
    )
    base = plain.system_prompt().rstrip()

    web = make(tmp_path / "web", web_snapshot_as_of="2026-10-01")
    assert isinstance(web, example.agent.OpenWorldEngineerHarness)
    assert web.system_prompt() == f"{base}\n\n## Web access\n\n{WEB_LINE}\n"

    offline = make(tmp_path / "offline")
    assert offline.system_prompt() == (
        f"{base}\n\n## Web access\n\nYou do not have web access.\n"
    )
    both_off = make(
        tmp_path / "both",
        web_snapshot_as_of="2026-10-01",
        disallowed_tools=["web_search", "web_fetch"],
    )
    assert both_off.system_prompt().endswith("You do not have web access.\n")
    with pytest.raises(ValueError, match="only some of the web tools"):
        make(
            tmp_path / "one",
            web_snapshot_as_of="2026-10-01",
            disallowed_tools=["web_fetch"],
        )


async def test_the_model_receives_the_section(tmp_path, example, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)
    agent = make(tmp_path, task_provides_timer=True)
    env = FakeEnv()
    await agent.setup(env)
    await agent.run("Implement the task.", env, AgentContext())
    system = fake.calls[0]["messages"][0]
    assert system["role"] == "system"
    text = system["content"]
    if not isinstance(text, str):  # content blocks (prompt caching)
        text = "".join(block["text"] for block in text)
    assert text.endswith("## Web access\n\nYou do not have web access.\n")


def test_offline_toml(example):
    offline_toml = example.jobs.offline_toml
    data = tomllib.loads(offline_toml(TASK_TOML))
    assert data["agent"] == {"timeout_sec": 14400.0, "network_mode": "no-network"}
    assert data["environment"] == tomllib.loads(TASK_TOML)["environment"]

    no_agent = 'version = "1.0"\n\n[environment]\ncpus = 1'
    assert tomllib.loads(offline_toml(no_agent))["agent"] == {
        "network_mode": "no-network"
    }
    with pytest.raises(ValueError, match="steps\\[0\\]"):
        offline_toml('[[steps]]\nname = "a"\n[steps.agent]\nnetwork_mode = "public"\n')
    with pytest.raises(ValueError, match="cannot be forced offline|breaks task.toml"):
        offline_toml('agent = { timeout_sec = 60.0, network_mode = "public" }\n')


CPU_TASK_TOML = TASK_TOML.replace('gpus = 1\ngpu_types = ["H100!"]\n', "")


def write_task(root: Path, name="demo", toml=TASK_TOML, tests=True) -> Path:
    task = root / "tasks" / name
    (task / "environment").mkdir(parents=True)
    (task / "task.toml").write_text(toml)
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "instruction.md").write_text("Do the task.\n")
    if tests:
        (task / "tests").mkdir()
        (task / "tests" / "test.sh").write_text("#!/bin/sh\n")
    return task


def test_job_files(tmp_path, example, capsys):
    task = write_task(tmp_path / "src")
    git = tmp_path / "git"
    git.write_bytes(b"\x7fELF")
    out = tmp_path / "out"
    args = [str(task), "--model", MODEL, "--out", str(out), "--static-git", str(git)]
    args += ["--env", "modal"]
    assert example.jobs.main([*args, "--reasoning-effort", "high"]) == 0
    path = (out / "jobs" / "demo.json").resolve()
    assert capsys.readouterr().out.strip() == (
        f"PYTHONPATH={EXAMPLE}${{PYTHONPATH:+:$PYTHONPATH}} harbor run -c {path}"
    )

    job = json.loads(path.read_text())
    JobConfig.model_validate(job)
    assert job["job_name"] == "demo--anthropic-claude-test--web-2026-10-01"
    assert job["n_attempts"] == 8
    assert job["environment"] == {"type": "modal"}
    assert "verifier" not in job
    copy = out / "tasks" / "demo"
    assert job["tasks"] == [{"path": str(copy.resolve())}]
    (agent,) = job["agents"]
    assert agent["import_path"] == "owe_agent:OpenWorldEngineerHarness"
    kwargs = agent["kwargs"]
    assert kwargs == {
        "large_input_exclude_mb": 100,
        "large_input_total_mb": 256,
        "llm_call_kwargs": {"_skip_mcp_handler": True},
        "sandbox_limits": {
            "cpus": 4,
            "memory_mb": 8192,
            "gpus": 1,
            "gpu_type": "H100!",
        },
        "static_git_path": str(git.resolve()),
        "reasoning_effort": "high",
        "web_snapshot_as_of": "2026-10-01",
        "time_budget_sec": 14400.0,
        "workspace_git_excludes": ["data", "build"],
    }
    assert agent["override_timeout_sec"] == hard_timeout_sec(kwargs) > 14400

    # The copy runs offline; the original task is untouched.
    assert (task / "task.toml").read_text() == TASK_TOML
    copied = tomllib.loads((copy / "task.toml").read_text())
    assert copied["agent"]["network_mode"] == "no-network"
    assert "allowed_hosts" not in copied["agent"]
    source, linked = (
        task / "environment" / "Dockerfile",
        copy / "environment" / "Dockerfile",
    )
    assert linked.read_text() == source.read_text()
    if source.stat().st_dev == linked.stat().st_dev:
        assert linked.stat().st_ino == source.stat().st_ino

    # The job's agent constructs as Harbor would build it.
    built = AgentFactory.create_agent_from_config(
        AgentConfig.model_validate(agent), logs_dir=tmp_path / "logs"
    )
    assert built.config.time_budget_sec == 14400.0
    assert built.system_prompt().endswith(f"{WEB_LINE}\n")

    # A rerun replaces its own copy; --no-web drops the web tools and names the arm.
    assert example.jobs.main([*args, "--no-web"]) == 0
    job = json.loads(path.read_text())
    assert job["job_name"].endswith("--noweb")
    kwargs = job["agents"][0]["kwargs"]
    assert "web_snapshot_as_of" not in kwargs and "reasoning_effort" not in kwargs


def test_skips_gpu_tasks_on_docker_and_ungraded_tasks(tmp_path, example, capsys):
    gpu = write_task(tmp_path / "a", "gpu")
    ungraded = write_task(tmp_path / "b", "ungraded", CPU_TASK_TOML, tests=False)
    cpu = write_task(tmp_path / "c", "cpu", CPU_TASK_TOML)
    out = tmp_path / "out"
    tasks = [str(gpu), str(ungraded), str(cpu)]
    assert example.jobs.main([*tasks, "--model", MODEL, "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert f"skipped {gpu}: needs a GPU" in captured.err
    assert f"skipped {ungraded}: no tests/" in captured.err
    assert "without --static-git" in captured.err
    assert sorted(p.name for p in (out / "jobs").iterdir()) == ["cpu.json"]

    assert (
        example.jobs.main(
            [str(ungraded), "--model", MODEL, "--out", str(out), "--allow-ungraded"]
        )
        == 0
    )
    assert capsys.readouterr().out.rstrip().endswith("# unscored")
    job = json.loads((out / "jobs" / "ungraded.json").read_text())
    assert job["verifier"] == {"disable": True}
    JobConfig.model_validate(job)


def run_error(example, capsys, argv) -> str:
    with pytest.raises(SystemExit) as raised:
        example.jobs.main(argv)
    assert raised.value.code == 2
    return capsys.readouterr().err


def test_paths_are_checked_before_anything_is_written(
    tmp_path, example, capsys, monkeypatch
):
    task = write_task(tmp_path / "src", toml=CPU_TASK_TOML)
    out = tmp_path / "out"
    base = ["--model", MODEL, "--out", str(out)]

    # Only a copy this script made is ever replaced.
    (out / "tasks" / "demo").mkdir(parents=True)
    (out / "tasks" / "demo" / "keep.txt").write_text("mine")
    err = run_error(example, capsys, [str(task), *base])
    assert "not a task copy made by this script" in err
    assert (out / "tasks" / "demo" / "keep.txt").read_text() == "mine"
    shutil.rmtree(out)

    # Output inside the task, or a task taken from the output, is refused.
    err = run_error(
        example, capsys, [str(task), "--model", MODEL, "--out", str(task / "o")]
    )
    assert "is inside the task" in err and not (task / "o").exists()
    assert example.jobs.main([str(task), *base]) == 0
    capsys.readouterr()
    copy = out / "tasks" / "demo"
    assert "is inside --out" in run_error(example, capsys, [str(copy), *base])
    assert (copy / "task.toml").is_file()

    # Two tasks with one name, or a bad task, stop the run before it writes.
    other = write_task(tmp_path / "other", toml=CPU_TASK_TOML)
    fresh = tmp_path / "fresh"
    argv = [str(task), str(other), "--model", MODEL, "--out", str(fresh)]
    assert "two tasks are named demo" in run_error(example, capsys, argv)
    argv = [str(task), str(tmp_path), "--model", MODEL, "--out", str(fresh)]
    assert "not a task directory" in run_error(example, capsys, argv)
    assert not fresh.exists()

    # "." names the task by its directory.
    monkeypatch.chdir(task)
    assert example.jobs.main([".", "--model", MODEL, "--out", str(fresh)]) == 0
    assert (fresh / "jobs" / "demo.json").is_file()


def test_ak_is_parsed_like_harbor(tmp_path, example, capsys):
    task = write_task(tmp_path / "src", toml=CPU_TASK_TOML)
    out = tmp_path / "out"
    argv = [str(task), "--model", MODEL, "--out", str(out), "--no-web"]
    argv += ["--ak", "max_turns = 5", "--ak", "enable_summarize=False"]
    assert example.jobs.main(argv) == 0
    kwargs = json.loads((out / "jobs" / "demo.json").read_text())["agents"][0]["kwargs"]
    assert kwargs["max_turns"] == 5 and kwargs["enable_summarize"] is False
    assert "Expected key=value" in run_error(
        example, capsys, [*argv[:5], "--ak", "novalue"]
    )
    assert "--attempts" in run_error(example, capsys, [*argv[:5], "--attempts", "0"])
