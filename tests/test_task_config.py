from pathlib import Path

import pytest
from harbor.models.task.config import TaskConfig

from openworld_harness import task_agent_kwargs

TOML = """
[agent]
timeout_sec = 7200.0
user = "agent"

[agent.proximus]
task_provides_timer = true
reasoning_effort = "high"
llm_call_kwargs = { extra_body = { a = 1, b = 2 } }

[agent.proximus.models."anthropic/claude-opus-5"]
reasoning_effort = "max"
api_key_env = "ANTHROPIC_API_KEY"
llm_call_kwargs = { extra_body = { b = 3 } }
"""


def task(tmp_path: Path, text: str = TOML) -> Path:
    (tmp_path / "task.toml").write_text(text)
    return tmp_path


def test_harbor_still_parses_the_task(tmp_path):
    config = TaskConfig.model_validate_toml(
        task(tmp_path).joinpath("task.toml").read_text()
    )
    assert config.agent.timeout_sec == 7200.0


def test_task_table_and_budget_default(tmp_path):
    assert task_agent_kwargs(task(tmp_path), "openai/gpt-5.5") == {
        "task_provides_timer": True,
        "reasoning_effort": "high",
        "llm_call_kwargs": {"extra_body": {"a": 1, "b": 2}},
        "time_budget_sec": 7200.0,
    }


def test_openworld_table_reads_like_the_legacy_one(tmp_path):
    legacy = task_agent_kwargs(task(tmp_path), "anthropic/claude-opus-5")
    renamed = TOML.replace("[agent.proximus", "[agent.openworld")
    assert task_agent_kwargs(task(tmp_path, renamed), "anthropic/claude-opus-5") == (
        legacy
    )


def test_a_web_option_error_names_its_table(tmp_path):
    text = '[agent.openworld.models."m"]\nweb_max_searches = 3\n'
    with pytest.raises(ValueError, match=r'\[agent\.openworld\.models\."m"\] in'):
        task_agent_kwargs(task(tmp_path, text), "m")


def test_a_task_table_cannot_name_a_host_file(tmp_path):
    # The runner picks which host file is copied into the sandbox.
    text = '[agent.openworld]\nstatic_git_path = "/home/runner/.netrc"\n'
    with pytest.raises(ValueError, match="static_git_path: web options and host"):
        task_agent_kwargs(task(tmp_path, text), "m")
    kwargs = task_agent_kwargs(
        task(tmp_path), "m", defaults={"static_git_path": "/opt/git"}
    )
    assert kwargs["static_git_path"] == "/opt/git"


def test_a_dumped_no_web_config_validates_again():
    from openworld_harness.config import OpenWorldConfig

    dumped = OpenWorldConfig(time_budget_sec=60).model_dump()
    assert OpenWorldConfig.model_validate(dumped).web_snapshot_as_of is None
    with pytest.raises(ValueError, match="need web_snapshot_as_of"):
        OpenWorldConfig(web_max_searches=3)


def test_both_tables_are_rejected(tmp_path):
    text = TOML + '\n[agent.openworld]\nreasoning_effort = "low"\n'
    with pytest.raises(ValueError, match="not both"):
        task_agent_kwargs(task(tmp_path, text), "m")


def test_model_table_then_overrides_win_per_nested_key(tmp_path):
    kwargs = task_agent_kwargs(
        task(tmp_path),
        "anthropic/claude-opus-5",
        {"time_budget_sec": 600, "llm_call_kwargs": {"extra_body": {"a": 9}}},
    )
    assert kwargs["reasoning_effort"] == "max"
    assert kwargs["api_key_env"] == "ANTHROPIC_API_KEY"
    assert kwargs["llm_call_kwargs"] == {"extra_body": {"a": 9, "b": 3}}
    assert kwargs["time_budget_sec"] == 600


def test_task_without_table(tmp_path):
    assert task_agent_kwargs(task(tmp_path, "[agent]\n"), "m") == {}
    assert task_agent_kwargs(task(tmp_path, "[agent]\ntimeout_sec = 60\n"), "m") == {
        "time_budget_sec": 60.0
    }


def test_unknown_option_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="reasoning_efort"):
        task_agent_kwargs(
            task(tmp_path, '[agent.proximus]\nreasoning_efort = "max"\n'), "m"
        )


def test_prompt_template_path_is_relative_to_the_task(tmp_path):
    (tmp_path / "prompt.j2").write_text("{{ instruction }}")
    kwargs = task_agent_kwargs(
        task(tmp_path, '[agent.proximus]\nprompt_template_path = "prompt.j2"\n'), "m"
    )
    assert kwargs["prompt_template_path"] == str((tmp_path / "prompt.j2").resolve())


def test_workspace_git_excludes_follow_the_app_artifact(tmp_path):
    toml = """
[[artifacts]]
source = "/app/"
exclude = ["./data", "**/.venv/**", "**/runs/**", "**/*.pt", "*"]

[agent]
timeout_sec = 60
"""
    kwargs = task_agent_kwargs(task(tmp_path, toml), "m")
    assert kwargs["workspace_git_excludes"] == [
        "data",
        "**/.venv",
        "**/runs",
        "**/*.pt",
    ]
    explicit = task_agent_kwargs(
        task(tmp_path, toml), "m", {"workspace_git_excludes": ["x"]}
    )
    assert explicit["workspace_git_excludes"] == ["x"]


def test_defaults_are_the_lowest_layer(tmp_path):
    defaults = {"task_provides_timer": True, "reasoning_effort": "low"}
    kwargs = task_agent_kwargs(task(tmp_path), "openai/gpt-5.5", defaults=defaults)
    assert kwargs["task_provides_timer"] is True
    assert kwargs["reasoning_effort"] == "high"  # the task table wins
