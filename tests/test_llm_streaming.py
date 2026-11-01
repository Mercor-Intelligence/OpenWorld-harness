"""Streamed model calls (``llm_call_kwargs.stream: true``), through real LiteLLM.

The gateway is an ``httpx.MockTransport`` behind the OpenAI SDK client that
LiteLLM's ``vercel_ai_gateway`` route uses, so the SSE parsing, the chunk
builder and the harness's own stream collection all run for real. The chunk
shapes are copied from a recorded GLM-5.3 (Baseten) stream: ``reasoning``
deltas, tool calls split across deltas, a final chunk that carries both the
finish reason and the usage, and the gateway's in-band timeout error chunk.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import openai
import pytest
from harbor.llms.base import OutputLengthExceededError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from openworld_harness.config import ProximusConfig
from openworld_harness.llm import (
    LLM,
    StreamIncompleteError,
    StreamStalledError,
)

pytestmark = pytest.mark.unit

MODEL = "vercel_ai_gateway/zai/glm-5.3"
MODEL_INFO = {
    "litellm_provider": "vercel_ai_gateway",
    "mode": "chat",
    "max_input_tokens": 1000000,
    "max_output_tokens": 128000,
    "input_cost_per_token": 1.4e-06,
    "output_cost_per_token": 4.4e-06,
    "cache_read_input_token_cost": 1.4e-07,
    "supports_function_calling": True,
    "supports_reasoning": True,
}
USAGE = {
    "prompt_tokens": 177,
    "completion_tokens": 21,
    "total_tokens": 198,
    "prompt_tokens_details": {"cached_tokens": 64},
    "completion_tokens_details": {"reasoning_tokens": 5},
}
EXPECTED_COST = (177 - 64) * 1.4e-06 + 64 * 1.4e-07 + 21 * 4.4e-06
MESSAGES: list[Any] = [{"role": "user", "content": "List /app and print the date."}]
# Requests carry no tool schemas: with tools, LiteLLM first loads its MCP
# handler (the litellm[mcp] extra, absent from some dev venvs). The scripted
# replies still contain tool calls, and parsing them does not depend on it.
NO_TOOLS = None


def chunk(delta: dict, finish: str | None = None, **extra: Any) -> dict:
    return {
        "id": "gen_test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "zai/glm-5.3",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        **extra,
    }


def sse(*events: dict | str) -> bytes:
    out = b""
    for event in events:
        data = event if isinstance(event, str) else json.dumps(event)
        out += f"data: {data}\n\n".encode()
    return out


def call_delta(
    index: int, *, id: str | None = None, name: str | None = None, args: str = ""
) -> dict:
    call: dict[str, Any] = {"index": index, "function": {"arguments": args}}
    if id is not None:
        call.update(id=id, type="function")
        call["function"]["name"] = name
    return {"tool_calls": [call]}


HEAD = [
    chunk({"role": "assistant"}),
    chunk({"reasoning": "Two commands, "}),
    chunk({"reasoning": "in parallel."}),
    chunk({"content": "Running both."}),
    chunk(call_delta(0, id="call_a", name="bash")),
    chunk(call_delta(0, args='{"command": ')),
    chunk(call_delta(0, args='"ls /app"}')),
    chunk(call_delta(1, id="call_b", name="bash")),
    chunk(call_delta(1, args='{"command": "date"}')),
]
TOOL_TURN = sse(*HEAD, chunk({}, "tool_calls", usage=USAGE), "[DONE]")
TEXT_TURN = sse(
    chunk({"role": "assistant"}),
    chunk({"content": "done"}),
    chunk({}, "stop", usage=USAGE),
    "[DONE]",
)
# The gateway's ~785 s cut, as recorded: an error chunk, then [DONE].
GATEWAY_CUT = sse(
    *HEAD[:4],
    chunk(
        {"content": ""},
        "error",
        error={
            "code": "gateway_stream_timeout",
            "message": "Stream exceeded maximum duration before function timeout",
            "origin": "gateway",
        },
    ),
    "[DONE]",
)
NON_STREAM_TOOL_TURN = {
    "id": "gen_test",
    "object": "chat.completion",
    "created": 1,
    "model": "zai/glm-5.3",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "Running both.",
                "reasoning": "Two commands, in parallel.",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "ls /app"}',
                        },
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "date"}',
                        },
                    },
                ],
            },
        }
    ],
    "usage": USAGE,
}


class Stalled:
    """A response body that sends ``data`` and then nothing, ever (or one
    chunk every ``trickle`` seconds); records whether it was closed."""

    def __init__(self, data: bytes, trickle: float | None = None) -> None:
        self.data = data
        self.trickle = trickle
        self.closed = False

    async def __aiter__(self):
        try:
            yield self.data
            while True:
                if self.trickle is None:
                    await asyncio.sleep(3600)
                await asyncio.sleep(self.trickle)
                yield sse(chunk({"content": "."}))
        finally:
            self.closed = True


class Gateway:
    """Answers each request with the next scripted body."""

    def __init__(self, *bodies: bytes | dict | Stalled) -> None:
        self.bodies = list(bodies)
        self.requests: list[dict] = []
        self.headers: list[httpx.Headers] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        self.headers.append(request.headers)
        body = self.bodies.pop(0)
        if isinstance(body, dict):
            return httpx.Response(200, json=body)
        content = body.__aiter__() if isinstance(body, Stalled) else body
        return httpx.Response(
            200, content=content, headers={"content-type": "text/event-stream"}
        )

    def client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key="test-key",
            base_url="https://ai-gateway.vercel.sh/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)),
        )


def make_llm(
    gateway: Gateway,
    monkeypatch: pytest.MonkeyPatch,
    *,
    idle: float | None = None,
    **call_kwargs: Any,
) -> tuple[LLM, list[dict]]:
    monkeypatch.setenv("VERCEL_AI_GATEWAY_API_KEY", "test-key")
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)
    events: list[dict] = []
    config = ProximusConfig(
        model_info=MODEL_INFO,
        llm_call_kwargs={"client": gateway.client(), **call_kwargs},
        stream_idle_timeout_sec=idle,
    )
    llm = LLM(
        model_name=MODEL,
        config=config,
        remaining_sec=lambda: None,
        logger=logging.getLogger("test"),
        on_provider_event=events.append,
    )
    return llm, events


def summary(parsed: Any) -> dict:
    return {
        "content": parsed.content,
        "reasoning": parsed.reasoning_content,
        "calls": [
            (c.call_id, c.name, json.loads(c.raw_arguments)) for c in parsed.tool_calls
        ],
        "tokens": (
            parsed.prompt_tokens,
            parsed.completion_tokens,
            parsed.reasoning_tokens,
            parsed.cached_tokens,
        ),
    }


async def test_streamed_turn_parses_like_the_json_one(monkeypatch):
    streamed_gateway = Gateway(TOOL_TURN)
    plain_gateway = Gateway(NON_STREAM_TOOL_TURN)
    streamed_llm, _ = make_llm(streamed_gateway, monkeypatch, stream=True)
    plain_llm, _ = make_llm(plain_gateway, monkeypatch)

    streamed = await streamed_llm.query(MESSAGES, NO_TOOLS)
    plain = await plain_llm.query(MESSAGES, NO_TOOLS)

    assert summary(streamed) == summary(plain)
    assert summary(streamed) == {
        "content": "Running both.",
        "reasoning": "Two commands, in parallel.",
        "calls": [
            ("call_a", "bash", {"command": "ls /app"}),
            ("call_b", "bash", {"command": "date"}),
        ],
        "tokens": (177, 21, 5, 64),
    }
    assert streamed.cost_usd == pytest.approx(EXPECTED_COST)
    assert plain.cost_usd == pytest.approx(EXPECTED_COST)
    assert streamed_gateway.requests[0]["stream"] is True
    assert "stream" not in plain_gateway.requests[0]


async def test_usage_is_requested_and_headers_pass_through(monkeypatch):
    gateway = Gateway(TEXT_TURN)
    llm, _ = make_llm(
        gateway,
        monkeypatch,
        stream=True,
        extra_headers={"x-vercel-gateway-extended-time": "true"},
    )
    await llm.query(MESSAGES, NO_TOOLS)
    assert gateway.requests[0]["stream_options"] == {"include_usage": True}
    assert gateway.headers[0]["x-vercel-gateway-extended-time"] == "true"


async def test_explicit_stream_options_are_kept(monkeypatch):
    gateway = Gateway(TEXT_TURN)
    llm, _ = make_llm(
        gateway, monkeypatch, stream=True, stream_options={"include_usage": False}
    )
    await llm.query(MESSAGES, NO_TOOLS)
    assert gateway.requests[0]["stream_options"] == {"include_usage": False}


async def test_non_streamed_requests_get_no_stream_options(monkeypatch):
    gateway = Gateway(NON_STREAM_TOOL_TURN)
    llm, _ = make_llm(gateway, monkeypatch)
    await llm.query(MESSAGES, NO_TOOLS)
    assert "stream_options" not in gateway.requests[0]


async def test_length_finish_raises_output_length(monkeypatch):
    gateway = Gateway(
        sse(*HEAD[:4], chunk({"content": " and more"}, "length", usage=USAGE), "[DONE]")
    )
    llm, _ = make_llm(gateway, monkeypatch, stream=True)
    with pytest.raises(OutputLengthExceededError) as raised:
        await llm.query(MESSAGES, NO_TOOLS)
    assert raised.value.truncated_response == "Running both. and more"
    assert len(gateway.requests) == 1


async def test_gateway_stream_timeout_chunk_is_retried(monkeypatch):
    gateway = Gateway(GATEWAY_CUT, TOOL_TURN)
    llm, events = make_llm(gateway, monkeypatch, stream=True)
    parsed = await llm.query(MESSAGES, NO_TOOLS)
    assert [c.call_id for c in parsed.tool_calls] == ["call_a", "call_b"]
    assert len(gateway.requests) == 2
    assert [e["kind"] for e in events] == ["retry"]
    assert events[0]["error_type"] == "MidStreamFallbackError"
    assert "Stream exceeded maximum duration" in events[0]["error_message"]


@pytest.mark.parametrize("done", [True, False], ids=["with-done", "eof"])
async def test_stream_cut_without_finish_reason_is_retried(monkeypatch, done):
    # Half a tool call and no finish reason: LiteLLM alone would build a
    # "stop" turn with truncated arguments.
    cut = sse(*HEAD[:6], *(["[DONE]"] if done else []))
    gateway = Gateway(cut, TOOL_TURN)
    llm, events = make_llm(gateway, monkeypatch, stream=True)
    parsed = await llm.query(MESSAGES, NO_TOOLS)
    assert json.loads(parsed.tool_calls[0].raw_arguments) == {"command": "ls /app"}
    assert len(gateway.requests) == 2
    (event,) = events
    assert event["error_type"] == "StreamIncompleteError"
    # LiteLLM re-chunks the deltas, so count what came through, not lines.
    assert event["stream"]["chunks"] >= 6
    assert set(event["stream"]) == {"chunks", "stream_sec", "max_gap_sec"}


async def test_stalled_stream_is_closed_and_retried(monkeypatch):
    stalled = Stalled(sse(*HEAD[:3]))
    gateway = Gateway(stalled, TOOL_TURN)
    llm, events = make_llm(gateway, monkeypatch, stream=True, idle=0.3)
    parsed = await asyncio.wait_for(llm.query(MESSAGES, NO_TOOLS), timeout=20)
    assert parsed.content == "Running both."
    assert stalled.closed
    (event,) = events
    assert event["error_type"] == "StreamStalledError"
    assert "no stream chunk for 0.3s" in event["error_message"]
    assert event["stream"]["chunks"] == 3


async def test_cancelled_stream_is_closed_and_not_retried(monkeypatch):
    stalled = Stalled(sse(*HEAD[:3]))
    gateway = Gateway(stalled, TOOL_TURN)
    llm, events = make_llm(gateway, monkeypatch, stream=True)
    task = asyncio.create_task(llm.query(MESSAGES, NO_TOOLS))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if gateway.requests:
            break
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(gateway.requests) == 1
    assert stalled.closed
    assert events == []


class FakeStream:
    """Stand-in for LiteLLM's CustomStreamWrapper."""

    def __init__(
        self,
        delays: list[float],
        *,
        error: BaseException | None = None,
        finish: str | None = "stop",
    ) -> None:
        self.delays = list(delays)
        self.error = error
        self.received_finish_reason: str | None = None
        self.intermittent_finish_reason: str | None = None
        self._finish = finish
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.error is not None:
            raise self.error
        if not self.delays:
            self.received_finish_reason = self._finish
            raise StopAsyncIteration
        await asyncio.sleep(self.delays.pop(0))
        return "chunk"

    async def aclose(self) -> None:
        self.closed = True


async def test_trickling_stream_hits_the_total_bound():
    stream = FakeStream([0.05] * 200)
    with pytest.raises(StreamStalledError, match="still open after 0.4s") as raised:
        await LLM._collect_stream(stream, total_sec=0.4, idle_sec=5)
    assert stream.closed
    stats = raised.value.stream_stats  # type: ignore[attr-defined]
    assert 3 <= stats["chunks"] <= 9
    assert stats["max_gap_sec"] < 0.4


async def test_transport_timeout_is_not_reported_as_a_stall():
    # aiohttp's read timeout is an asyncio.TimeoutError too: it must surface
    # as itself, not as our own deadline.
    stream = FakeStream([], error=TimeoutError("sock_read"))
    with pytest.raises(TimeoutError, match="sock_read") as raised:
        await LLM._collect_stream(stream, total_sec=30, idle_sec=10)
    assert not isinstance(raised.value, StreamStalledError)
    assert stream.closed


async def test_clean_stream_returns_every_chunk():
    stream = FakeStream([0, 0, 0])
    assert (
        await LLM._collect_stream(stream, total_sec=30, idle_sec=None) == ["chunk"] * 3
    )
    assert stream.closed


async def test_finish_reason_missing_raises_incomplete():
    stream = FakeStream([0, 0], finish=None)
    with pytest.raises(StreamIncompleteError):
        await LLM._collect_stream(stream, total_sec=30, idle_sec=None)


async def test_iterators_without_finish_tracking_are_trusted():
    async def chunks():
        yield "a"
        yield "b"

    assert await LLM._collect_stream(chunks(), total_sec=30, idle_sec=None) == [
        "a",
        "b",
    ]


def anthropic_events(*, complete: bool) -> bytes:
    def event(name: str, data: dict) -> bytes:
        return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()

    events = [
        event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            },
        ),
        event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
        ),
    ]
    if complete:
        events += [
            event("content_block_stop", {"type": "content_block_stop", "index": 0}),
            event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 5},
                },
            ),
            event("message_stop", {"type": "message_stop"}),
        ]
    return b"".join(events)


async def test_anthropic_stream_cut_is_detected_and_retried(monkeypatch):
    bodies = [anthropic_events(complete=False), anthropic_events(complete=True)]

    async def post(self, url, *args, **kwargs):
        return httpx.Response(
            200,
            content=bodies.pop(0),
            headers={"content-type": "text/event-stream"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(AsyncHTTPHandler, "post", post)
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)
    events: list[dict] = []
    llm = LLM(
        model_name="anthropic/claude-sonnet-4-5",
        config=ProximusConfig(
            api_key_env="TEST_ANTHROPIC_KEY", llm_call_kwargs={"stream": True}
        ),
        remaining_sec=lambda: None,
        logger=logging.getLogger("test"),
        extra_env={"TEST_ANTHROPIC_KEY": "test-key"},
        on_provider_event=events.append,
    )
    parsed = await llm.query(MESSAGES, None)
    assert parsed.content == "hello"
    assert parsed.completion_tokens == 5
    assert bodies == []
    assert [e["error_type"] for e in events] == ["StreamIncompleteError"]
