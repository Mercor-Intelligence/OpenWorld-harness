"""Fixes for failures seen in long multi-model evaluation runs.

- A response the provider's output filter blanked (finish reason
  content_filter, no text, no tool call) is sent again, not counted as an
  empty model turn; persistent filtering takes the refusal path and ends the
  run as a graded provider failure. Counted as empty turns, they ended some
  Gemini runs as stalls.
"""

import logging

import litellm
import pytest
from harbor.models.agent.context import AgentContext

from openworld_harness import llm as llm_mod
from openworld_harness import prompts
from openworld_harness.config import ProximusConfig
from tests.test_proximus_loop import (
    FakeCompletion,
    FakeEnv,
    FakeMessage,
    FakeResponse,
    bash_response,
    finish_responses,
    make_agent,
    read_messages,
    read_trajectory,
)
from tests.test_sweep_fixes import harness_kinds

pytestmark = pytest.mark.unit

ATTEMPTS = ProximusConfig().content_filter_attempts


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)


def filtered_response(
    completion_tokens: int = 800, reasoning: str | None = None, **hidden
) -> FakeResponse:
    response = FakeResponse(
        FakeMessage(content=None, reasoning_content=reasoning),
        prompt_tokens=1000,
        completion_tokens=completion_tokens,
    )
    response.choices[0].finish_reason = "content_filter"
    response._hidden_params.update(hidden)
    return response


def filtered(n: int = ATTEMPTS) -> list[FakeResponse]:
    return [filtered_response() for _ in range(n)]


# ----- content filter -----


async def test_filtered_responses_are_resampled_not_counted_as_turns(
    tmp_path, monkeypatch
):
    fake = FakeCompletion([*filtered(3), bash_response("ls"), *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)
    agent = make_agent(tmp_path)
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]

    metadata = context.metadata or {}
    assert metadata["ended_by"] == "submit"
    assert metadata["stalled"] is False
    assert metadata["content_filter_hits"] == 3
    assert metadata["n_turns"] == 3  # the resamples are not turns
    # The resamples are the same request, and nothing was added for the model.
    assert fake.calls[0]["messages"] == fake.calls[3]["messages"]
    assert not harness_kinds(agent)
    # Each filtered response is billed: 3 x 800 tokens plus the 3 real turns.
    assert context.n_output_tokens == 3 * 800 + 3 * 10
    assert context.cost_usd == pytest.approx(6 * 0.01)
    events = [
        (s.get("extra") or {}).get("provider_event")
        for s in read_trajectory(agent)["steps"]
        if (s.get("extra") or {}).get("provider_event")
    ]
    assert [e["error_type"] for e in events] == ["ProviderContentFilterError"] * 3
    assert events[0]["content_filter"]["completion_tokens"] == 800


async def test_persistent_filter_withholds_then_ends_graded(tmp_path, monkeypatch):
    fake = FakeCompletion(
        [
            bash_response("cat /app/vendor/big.c"),
            *filtered(),  # the request carrying the file
            *filtered(),  # still filtered without it
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)
    agent = make_agent(tmp_path)
    env = FakeEnv()
    context = AgentContext()
    await agent.run("Task", env, context)  # type: ignore[arg-type]

    assert len(fake.calls) == 1 + 2 * ATTEMPTS
    metadata = context.metadata or {}
    assert metadata["ended_by"] == "provider_failure"
    assert metadata["provider_failure"]["kind"] == "refusal"
    assert metadata["provider_failure"]["code"] == llm_mod.CONTENT_FILTER_CODE
    assert metadata["stalled"] is False
    assert metadata["content_filter_hits"] == 2 * ATTEMPTS
    second = fake.calls[1 + ATTEMPTS]["messages"]
    assert [m["content"] for m in second if m.get("role") == "tool"][
        -1
    ] == prompts.WITHHELD_OUTPUT
    assert second[-1]["content"] == prompts.content_filter_message(withheld=True)
    assert harness_kinds(agent) == ["content_filter"]
    # Ended normally: the winner was restored for grading.
    assert any("checkout" in c for c in env.all_commands)


async def test_filter_with_nothing_to_withhold_still_gets_a_second_try(
    tmp_path, monkeypatch
):
    fake = FakeCompletion([*filtered(), bash_response("ls"), *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)
    agent = make_agent(tmp_path)
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]

    metadata = context.metadata or {}
    assert metadata["ended_by"] == "submit"
    closers = [m["content"] for m in read_messages(agent) if m["role"] == "user"]
    assert prompts.content_filter_message(withheld=False) in closers


async def test_a_filtered_turn_with_text_or_a_tool_call_is_a_turn(
    tmp_path, monkeypatch
):
    with_text = FakeResponse(FakeMessage(content="Partial answer"))
    with_text.choices[0].finish_reason = "content_filter"
    fake = FakeCompletion([with_text, bash_response("ls"), *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)
    agent = make_agent(tmp_path)
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]
    assert (context.metadata or {})["content_filter_hits"] == 0
    assert len(fake.calls) == 4


async def test_reasoning_alone_does_not_save_a_filtered_response(monkeypatch):
    calls = {"n": 0}
    responses = [
        filtered_response(reasoning="thinking about it"),
        FakeResponse(FakeMessage(content="ok")),
    ]

    async def completion(**kwargs):
        calls["n"] += 1
        return responses.pop(0)

    monkeypatch.setattr(litellm, "acompletion", completion)
    seen: list = []
    model = llm_mod.LLM(
        "openai/gpt-test",
        ProximusConfig(),
        lambda: 3600.0,
        logging.getLogger("t"),
        on_discarded_response=seen.append,
    )
    parsed = await model.query([{"role": "user", "content": "hi"}], tools=None)
    assert parsed.content == "ok"
    assert calls["n"] == 2
    assert len(seen) == 1 and seen[0].completion_tokens == 800


async def test_filter_attempts_are_capped_and_carry_the_safety_details(monkeypatch):
    calls = {"n": 0}
    ratings = [[{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "blocked": True}]]

    async def completion(**kwargs):
        calls["n"] += 1
        return filtered_response(vertex_ai_safety_ratings=ratings)

    monkeypatch.setattr(litellm, "acompletion", completion)
    events: list[dict] = []
    model = llm_mod.LLM(
        "openai/gpt-test",
        ProximusConfig(content_filter_attempts=2),
        lambda: 3600.0,
        logging.getLogger("t"),
        on_provider_event=events.append,
    )
    with pytest.raises(llm_mod.ProviderRefusalError) as raised:
        await model.query([{"role": "user", "content": "hi"}], tools=None)
    assert raised.value.code == llm_mod.CONTENT_FILTER_CODE
    assert calls["n"] == 2
    assert [e["kind"] for e in events] == ["retry", "failed"]
    assert "HARM_CATEGORY_DANGEROUS_CONTENT" in events[0]["content_filter"]["details"]
    assert model.provider_lost_sec >= 0


async def test_a_spent_budget_keeps_the_retryable_filter_error(monkeypatch):
    async def completion(**kwargs):
        return filtered_response()

    monkeypatch.setattr(litellm, "acompletion", completion)
    remaining = iter([3600.0, 3600.0, 0.0, 0.0, 0.0, 0.0])
    model = llm_mod.LLM(
        "openai/gpt-test",
        ProximusConfig(),
        lambda: next(remaining, 0.0),
        logging.getLogger("t"),
    )
    with pytest.raises(llm_mod.ProviderContentFilterError):
        await model.query([{"role": "user", "content": "hi"}], tools=None)


async def test_anthropic_refusals_are_not_resampled(monkeypatch):
    # LiteLLM maps Anthropic's "refusal" stop reason to content_filter too.
    calls = {"n": 0}

    async def completion(**kwargs):
        calls["n"] += 1
        return filtered_response()

    monkeypatch.setattr(litellm, "acompletion", completion)
    model = llm_mod.LLM(
        "anthropic/claude-test",
        ProximusConfig(),
        lambda: 3600.0,
        logging.getLogger("t"),
    )
    parsed = await model.query([{"role": "user", "content": "hi"}], tools=None)
    assert parsed.is_empty and calls["n"] == 1  # an empty turn, as before


async def test_filtered_time_is_not_provider_loss(monkeypatch, tmp_path):
    responses = [
        filtered_response(),
        filtered_response(),
        FakeResponse(FakeMessage(content="ok")),
    ]

    async def completion(**kwargs):
        return responses.pop(0)

    monkeypatch.setattr(litellm, "acompletion", completion)
    health = tmp_path / "provider_health.json"
    model = llm_mod.LLM(
        "openai/gpt-test",
        ProximusConfig(),
        lambda: 3600.0,
        logging.getLogger("t"),
        provider_health_path=health,
    )
    parsed = await model.query([{"role": "user", "content": "hi"}], tools=None)
    assert parsed.content == "ok"
    # A run the filter keeps blanking is graded at the budget, not re-run.
    assert model.provider_lost_sec == 0
    assert not health.exists() or '"consecutive_errors": 0' in health.read_text()


async def test_the_filter_streak_is_broken_by_other_failures(monkeypatch):
    def outage():
        return litellm.exceptions.ServiceUnavailableError("busy", "openai", "m")

    script = [
        filtered_response(),
        outage(),
        filtered_response(),
        FakeResponse(FakeMessage(content="ok")),
    ]

    async def completion(**kwargs):
        item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(litellm, "acompletion", completion)
    model = llm_mod.LLM(
        "openai/gpt-test",
        ProximusConfig(content_filter_attempts=2),
        lambda: 3600.0,
        logging.getLogger("t"),
    )
    parsed = await model.query([{"role": "user", "content": "hi"}], tools=None)
    assert parsed.content == "ok" and not script
