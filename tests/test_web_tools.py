"""The Exa Snapshot web tools: CGDP signals, the client, and the agent's web tools."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import types
import typing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import NetworkMode

import openworld_harness.prompts as prompts
from openworld_harness import OpenWorldHarness, llm
from openworld_harness.proximus import Proximus, RunState
from openworld_harness.web import agent as web_agent
from openworld_harness.web import cgdp, exa

pytestmark = pytest.mark.unit

# Registered by conftest; an Anthropic model, so the extractor's cache markers apply.
MODEL = "anthropic/claude-test"
AS_OF = "2026-06-01T00:00:00Z"
# Anything that would tell the model the date or that the web is frozen.
LEAKS = re.compile(r"2026-06|2026-05|snapshot|cut-?off|as of|cache|archiv|histor", re.I)


# ----- CGDP signals and belief state -----


def test_gate_fires_after_patience_stagnant_searches() -> None:
    gate = cgdp.Gate()
    assert not gate.observe(
        "parse json python error", 5, 5
    ).stagnant  # nothing to repeat
    first = gate.observe("python json parse error fix", 5, 1)  # J 0.8, novelty 0.2
    assert first.stagnant and not first.fire
    second = gate.observe("json parse error python", 5, 0)
    assert second.stagnant and second.fire
    # It keeps halting the line, the halted query included, until a search finds new pages.
    assert gate.observe("json parse error python", 5, 0).fire
    # Enough new pages breaks the streak, however similar the query.
    assert (
        not gate.observe("json parse error python", 5, 2).stagnant and gate.streak == 0
    )
    # An A-B-A oscillation counts against the query before last.
    gate = cgdp.Gate(window=3)
    gate.observe("alpha beta gamma", 5, 5)
    gate.observe("delta epsilon zeta", 5, 5)
    assert gate.observe("alpha beta gamma", 5, 0).jaccard == 1.0
    assert len(gate.queries) == 3  # only the window is kept


def test_no_results_count_as_no_novelty() -> None:
    gate = cgdp.Gate(patience=1)
    gate.observe("rare thing", 0, 0)
    assert gate.observe("rare thing", 0, 0).fire


def test_notes_parse_dedupe_and_stay_bounded() -> None:
    reply = "Reasoning first.\n- fact one (https://a)\n* fact two (https://b)\nNothing relevant."
    assert cgdp.parse_notes(reply) == ["fact one (https://a)", "fact two (https://b)"]
    assert cgdp.parse_notes(cgdp.NOTHING) == []
    notes = cgdp.Notes(target=2)
    assert notes.add(["a", "b", "a"]) == 2 and notes.items == ["a", "b"]
    notes.add(["c", "d", "e", "f"])
    assert not notes.needs_reorganizing()  # 6 = target + 4
    notes.add(["g"])
    assert notes.needs_reorganizing()
    notes.replace(["x", "y", "z"])
    assert notes.items == ["x", "y"]
    notes.items = [str(i) for i in range(9)]
    notes.keep_newest()
    assert notes.items == [str(i) for i in range(3, 9)]
    # An overlong note is shortened in the middle and keeps its source.
    long_note = "x" * 700 + " (https://docs.example/api)"
    (clipped,) = cgdp.parse_notes(f"- {long_note}")
    assert len(clipped) == cgdp.NOTE_MAX_CHARS and clipped.endswith(
        "… (https://docs.example/api)"
    )
    assert cgdp.clip_note("y" * 700) == "y" * (cgdp.NOTE_MAX_CHARS - 1) + "…"


def test_notes_marker_versions_and_sources() -> None:
    # Bullets in the reasoning before the NOTES: line are not notes.
    reply = "- The page states X.\n- That overlaps a note.\n**NOTES:**\n- X is 3 (https://a)"
    assert cgdp.parse_notes(reply) == ["X is 3 (https://a)"]
    assert cgdp.parse_notes(f"- reasoning\nNOTES: {cgdp.NOTHING}") == []
    # A closing "Notes: ..." remark is not the marker.
    closing = "Reasoning.\nNOTES:\n- X is 3 (https://a)\nNotes: all of the above are from the page."
    assert cgdp.parse_notes(closing) == ["X is 3 (https://a)"]
    notes = cgdp.Notes(target=1)
    assert notes.version == 0 and notes.add(["a"]) == 1 and notes.version == 1
    assert notes.add(["a"]) == 0 and notes.version == 1  # nothing new: same version
    notes.keep_newest()
    assert notes.version == 1  # nothing to drop
    notes.replace(["b"])
    assert notes.version == 2
    # A source URL holding parentheses survives clipping.
    wiki = "(https://en.wikipedia.org/wiki/Python_(programming_language))"
    assert cgdp.clip_note("z" * 700 + " " + wiki).endswith(f"… {wiki}")
    stable, varying = cgdp.extraction_prompt(
        "searched the web", "Fix it.", "g", notes, "OBS-17"
    )
    assert cgdp.NOTES_MARKER in cgdp.reorganization_prompt("Fix it.", notes)[0]
    assert stable.endswith("Fix it.") and "OBS-17" in varying and "OBS-17" not in stable


def test_normalize_url_and_as_of() -> None:
    assert cgdp.normalize_url("HTTPS://Docs.Python.org/3/library/#x") == (
        "https://docs.python.org/3/library"
    )
    assert (
        cgdp.normalize_url(" http://[::1/x ") == "http://[::1/x"
    )  # unparsable: as given
    assert exa.normalize_as_of("2026-06-01") == AS_OF
    assert exa.normalize_as_of("2026-06-01T02:00:00+02:00") == AS_OF
    with pytest.raises(ValueError, match="future"):
        exa.normalize_as_of("2999-01-01")
    with pytest.raises(ValueError, match="ISO 8601"):
        exa.normalize_as_of("June 1st")


# ----- the Exa client -----


async def test_exa_client_pins_every_request(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[str, dict[str, Any]]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        sent.append((req.url.path, body))
        assert req.headers["x-api-key"] == "k"
        if req.url.path == "/search":
            hit = {
                "id": "u",
                "url": "https://a.dev/x",
                "title": " A ",
                "highlights": ["h", " "],
            }
            return httpx.Response(
                200, json={"results": [hit | {"publishedDate": "2026-05-01"}]}
            )
        if body["ids"] == ["https://gone.dev"]:
            error = {"httpStatusCode": 404, "tag": "CONTENT_NOT_CACHED"}
            status = {"id": body["ids"][0], "status": "error", "error": error}
            return httpx.Response(200, json={"results": [], "statuses": [status]})
        return httpx.Response(
            200, json={"results": [{"url": "https://a.dev/x", "text": "body"}]}
        )

    client = exa.SnapshotClient("k", AS_OF, transport=httpx.MockTransport(handler))
    hits = await client.search("q", 3)
    assert hits == [exa.Hit("https://a.dev/x", "A", ["h"], "2026-05-01")]
    page = await client.contents("https://a.dev/x", 1000)
    assert page == exa.Page("https://a.dev/x", "", "body")
    assert await client.contents("https://gone.dev", 1000) is None
    await client.aclose()
    assert sent[0] == (
        "/search",
        {
            "query": "q",
            "type": "auto",
            "numResults": 3,
            "contents": {"highlights": True, "snapshotAsOf": AS_OF},
        },
    )
    assert sent[1] == (
        "/contents",
        {
            "ids": ["https://a.dev/x"],
            "text": {"maxCharacters": 1000},
            "snapshotAsOf": AS_OF,
        },
    )


async def test_exa_client_waits_cleans_and_reads_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []

    async def record_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(exa.asyncio, "sleep", record_sleep)
    replies = [
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(503),
        httpx.Response(
            200, content=b'{"results": [{"url": "https://a.dev", "text": "a\\ud83db"}]}'
        ),
    ]
    client = exa.SnapshotClient(
        "k", AS_OF, transport=httpx.MockTransport(lambda r: replies.pop(0))
    )
    page = await client.contents("https://a.dev", 10)
    assert waits[0] == 7 and 4 <= waits[1] < 5  # Exa's Retry-After, then backoff
    assert page is not None and page.text == "a?b"  # a lone surrogate, replaced

    def contents_reply(body: dict[str, Any]) -> exa.SnapshotClient:
        reply = httpx.MockTransport(lambda r: httpx.Response(200, json=body))
        return exa.SnapshotClient("k", AS_OF, transport=reply)

    # Only an explicit, lasting 4xx for the id means the page is not there.
    for body in (
        {"results": []},
        {
            "results": [],
            "statuses": [{"id": "u", "status": "error", "error": "timeout"}],
        },
        {
            "results": [],
            "statuses": [{"id": "u", "status": "error", "error": {"tag": "X"}}],
        },
    ):
        with pytest.raises(exa.SnapshotError) as transient:
            await contents_reply(body).contents("https://a.dev", 10)
        assert transient.value.page_level


async def test_exa_client_retries_then_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(exa.asyncio, "sleep", no_sleep)
    replies = [
        httpx.Response(429),
        httpx.Response(408),
        httpx.Response(200, json={"results": []}),
    ]
    client = exa.SnapshotClient(
        "k", AS_OF, transport=httpx.MockTransport(lambda r: replies.pop(0))
    )
    assert await client.search("q", 1) == []

    # Unexpected shapes: a non-object body fails as a SnapshotError, junk entries are skipped.
    junk = {
        "results": [
            "x",
            {"url": "https://a.dev", "highlights": "h", "publishedDate": 5},
        ]
    }
    for body, expected in (([], None), (junk, [exa.Hit("https://a.dev", "", [], "5")])):
        client = exa.SnapshotClient(
            "k",
            AS_OF,
            transport=httpx.MockTransport(
                lambda r, b=body: httpx.Response(200, json=b)
            ),
        )
        if expected is None:
            with pytest.raises(exa.SnapshotError, match="not a JSON object"):
                await client.search("q", 1)
        else:
            assert await client.search("q", 1) == expected

    # Per-id statuses: a missing version is None, a transient failure is page-level.
    def statuses(code: int, tag: str) -> httpx.MockTransport:
        error = {"httpStatusCode": code, "tag": tag}
        body = {
            "results": [],
            "statuses": [{"id": "u", "status": "error", "error": error}],
        }
        return httpx.MockTransport(lambda r: httpx.Response(200, json=body))

    client = exa.SnapshotClient(
        "k", AS_OF, transport=statuses(404, "CONTENT_NOT_CACHED")
    )
    assert await client.contents("https://a.dev", 10) is None
    client = exa.SnapshotClient(
        "k", AS_OF, transport=statuses(502, "CRAWL_UNKNOWN_ERROR")
    )
    with pytest.raises(exa.SnapshotError) as page_error:
        await client.contents("https://a.dev", 10)
    assert page_error.value.page_level and page_error.value.status == 502

    refusal = {
        "error": "Historical content is served from cache only.",
        "tag": "INVALID_REQUEST",
    }
    calls: list[int] = []

    def bad(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json=refusal)

    client = exa.SnapshotClient("k", AS_OF, transport=httpx.MockTransport(bad))
    with pytest.raises(exa.SnapshotError) as info:
        await client.search("q", 1)
    assert (
        info.value.status == 400
        and info.value.tag == "INVALID_REQUEST"
        and len(calls) == 1
    )


# ----- the web tools -----


class FakeExa:
    def __init__(
        self,
        results: dict[str, list[exa.Hit]] | None = None,
        pages: dict[str, exa.Page | None] | None = None,
        fail: bool = False,
        errors: dict[str, Exception] | None = None,  # by query or URL
    ):
        self.results = results or {}
        self.pages = pages or {}
        self.fail = fail
        self.errors = errors or {}
        self.calls: list[tuple[str, str]] = []

    async def search(self, query: str, num_results: int) -> list[exa.Hit]:
        self.calls.append(("search", query))
        if self.fail:
            raise exa.SnapshotError(
                f"/search: HTTP 400 INVALID_REQUEST {AS_OF} cache only", 400
            )
        if query in self.errors:
            raise self.errors[query]
        return self.results.get(query, [])[:num_results]

    async def contents(self, url: str, max_chars: int) -> exa.Page | None:
        self.calls.append(("contents", url))
        if url in self.errors:
            raise self.errors[url]
        return self.pages.get(url)

    async def aclose(self) -> None:
        return None


class FakeLLM:
    def __init__(self, replies: list[str] | None = None):
        self.replies = list(replies or [])
        self.prompts: list[str] = []

    def is_anthropic(self) -> bool:
        return False

    async def query(
        self, messages: list[dict[str, Any]], tools: Any
    ) -> llm.ParsedResponse:
        assert tools is None
        self.prompts.append(messages[0]["content"])
        content = self.replies.pop(0) if self.replies else cgdp.NOTHING
        return llm.ParsedResponse(
            content=content, prompt_tokens=100, completion_tokens=20, cost_usd=0.01
        )


JSON_HIT = exa.Hit(
    "https://a.dev/json",
    "JSON errors",
    ["json.loads raises JSONDecodeError"],
    "2026-05-28",
)
STAGNANT = (
    "parse json python error",
    "python json parse error fix",
    "json parse error python",
)
HITS = {
    "parse json python error": [
        JSON_HIT,
        exa.Hit("https://b.dev/py", "Python docs", ["msg, doc, pos"]),
    ],
}
HITS["python json parse error fix"] = HITS["parse json python error"]
HITS["json parse error python"] = HITS["parse json python error"][::-1]


def make_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeExa | None = None,
    **options: Any,
) -> OpenWorldHarness:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-0123456789abcdef")
    agent = OpenWorldHarness(
        logs_dir=tmp_path / "agent",
        model_name=MODEL,
        web_snapshot_as_of="2026-06-01",
        time_budget_sec=3600,
        **options,
    )
    agent._instruction = "Make tests/test_parse.py pass."
    agent._exa = fake if fake is not None else FakeExa(HITS)  # type: ignore[assignment]
    agent._extractor = FakeLLM()  # type: ignore[assignment]
    return agent


async def call(agent: OpenWorldHarness, tool: str, **arguments: Any) -> Any:
    return await agent._tools[tool].call(json.dumps(arguments))


async def test_search_marks_seen_results_and_halts_a_stagnant_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = make_agent(tmp_path, monkeypatch)
    agent._extractor = FakeLLM(
        ["- json.loads raises JSONDecodeError (https://a.dev/json)"]
    )
    goal = "Which exception does json.loads raise?"

    first = await call(agent, "web_search", query="parse json python error", goal=goal)
    assert (
        "2 results (2 new)" in first.content and "> json.loads raises" in first.content
    )
    assert "- json.loads raises JSONDecodeError (https://a.dev/json)" in first.content
    assert (
        "Web budget left: 49 of 50 searches, 100 of 100 page fetches." in first.content
    )

    second = await call(
        agent, "web_search", query="python json parse error fix", goal=goal
    )
    assert "(0 new)" in second.content and "[1] (seen) JSON errors" in second.content
    assert "> json.loads" not in second.content  # seen results come without excerpts

    third = await call(agent, "web_search", query="json parse error python", goal=goal)
    assert (
        third.content.startswith("Search halted:")
        and "(0 new of 2 this time)" in third.content
    )
    assert (
        "JSONDecodeError (https://a.dev/json)" in third.content
    )  # b_t comes with the halt

    # The line stays halted, even for the halted query re-sent verbatim. Identical queries are
    # served again without a request or a charge.
    for query in ("json parse error python", "parse json python error"):
        again = await call(agent, "web_search", query=query, goal=goal)
        assert (
            again.content.startswith("Search halted:")
            and "47 of 50 searches" in again.content
        )
    assert [c for c in agent._exa.calls if c[0] == "search"] == [  # type: ignore[union-attr]
        ("search", "parse json python error"),
        ("search", "python json parse error fix"),
        ("search", "json parse error python"),
    ]
    assert len(agent._extractor.prompts) == 1  # type: ignore[attr-defined]  # only new results
    log = [json.loads(line) for line in (tmp_path / "agent" / "web_log.jsonl").open()]
    assert [e["status"] for e in log] == ["ok", "ok", "halted", "halted", "halted"]
    assert log[1]["jaccard"] == 0.8 and log[1]["novelty"] == 0.0 and log[1]["stagnant"]
    assert log[0]["notes"] == ["json.loads raises JSONDecodeError (https://a.dev/json)"]


async def test_gate_off_still_records_where_it_would_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = make_agent(tmp_path, monkeypatch, web_gate=False, web_notes=False)
    for query in STAGNANT:
        result = await call(agent, "web_search", query=query, goal="g")
    assert (
        "Search halted" not in result.content and "Research notes" not in result.content
    )
    assert agent._extractor.prompts == []  # type: ignore[attr-defined]
    context = AgentContext()
    agent._populate_context(context)
    web = context.metadata["web"]
    assert web["gate"] == {"live": False, "triggered_at_action": [3]}
    assert web["searches"] == 3 and web["notes"] is None and web["pages_seen"] == 2


async def test_fetch_windows_a_page_and_reports_missing_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "".join(f"{i:05d}" for i in range(5000))  # 25,000 characters
    fake = FakeExa(
        pages={"https://a.dev/json": exa.Page("https://a.dev/json", "JSON", text)}
    )
    agent = make_agent(tmp_path, monkeypatch, fake, web_fetch_chars=10_000)

    first = await call(
        agent, "web_fetch", url="https://a.dev/json", goal="the exception"
    )
    assert (
        "[characters 0-10,000 of 25,000; call web_fetch with offset=10000 for more]"
        in (first.content)
    )
    last = await call(agent, "web_fetch", url="https://a.dev/json#top", offset=20_000)
    assert "end of page text" in last.content and text[20_000:] in last.content
    past = await call(agent, "web_fetch", url="https://a.dev/json", offset=30_000)
    assert past.is_error and "past the end" in past.content
    missing = await call(agent, "web_fetch", url="https://gone.dev/page")
    assert missing.is_error and missing.content.startswith("This page is not available")
    assert "98 of 100 page fetches" in missing.content
    assert fake.calls == [
        ("contents", "https://a.dev/json"),
        ("contents", "https://gone.dev/page"),
    ]
    assert (await call(agent, "web_fetch", url="file:///etc/passwd")).is_error
    assert (await call(agent, "web_fetch", url="http://[::1/x#a")).content.startswith(
        "web_fetch needs a valid"
    )
    # The fetch's goal defaults to the last search goal and reaches the extractor.
    assert "the exception" in agent._extractor.prompts[0]  # type: ignore[attr-defined]
    # A window already read is not sent to the extractor again.
    await call(agent, "web_fetch", url="https://a.dev/json")
    assert len(agent._extractor.prompts) == 2  # type: ignore[attr-defined]
    log = [json.loads(line) for line in (tmp_path / "agent" / "web_log.jsonl").open()]
    assert [e["status"] for e in log][2:6] == [
        "bad_offset",
        "not_available",
        "invalid",
        "invalid",
    ]


async def test_fetch_says_when_only_part_of_a_page_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = exa.Page("https://a.dev/long", "Long", "y" * 25_000, truncated=True)
    fake = FakeExa(pages={page.url: page})
    agent = make_agent(tmp_path, monkeypatch, fake, web_fetch_chars=10_000)

    first = await call(agent, "web_fetch", url="https://a.dev/long")
    assert (
        "[characters 0-10,000 of the first 25,000 available; call web_fetch"
        in first.content
    )
    last = await call(agent, "web_fetch", url="https://a.dev/long", offset=20_000)
    assert (
        "the page may continue, but only its first 25,000 characters are available"
        in (last.content)
    )
    assert "end of page text" not in last.content
    past = await call(agent, "web_fetch", url="https://a.dev/long", offset=25_000)
    assert (
        "past the end of the first 25,000 characters of this page, all that is"
        in past.content
    )

    fake.pages["https://a.dev/short"] = exa.Page(
        "https://a.dev/short", "S", "z" * 23_000
    )
    short = await call(agent, "web_fetch", url="https://a.dev/short", offset=20_000)
    assert "[characters 20,000-23,000 of 23,000; end of page text]" in short.content


async def test_exa_client_flags_pages_that_may_be_cut() -> None:
    def client(text: str) -> exa.SnapshotClient:
        body = {"results": [{"url": "https://a.dev", "text": text}]}
        return exa.SnapshotClient(
            "k",
            AS_OF,
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)),
        )

    # Exa may count UTF-16 units (an emoji is 2) or stop at a word: near the limit counts too.
    for text, cut in (
        ("😀" * 49_600, True),
        ("y" * 99_500, True),
        ("z" * 98_000, False),
    ):
        page = await client(text).contents("https://a.dev", 100_000)
        assert page is not None and page.truncated is cut


async def test_budgets_errors_and_missing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = make_agent(tmp_path, monkeypatch, web_max_searches=1)
    await call(agent, "web_search", query="parse json python error", goal="g")
    spent = await call(agent, "web_search", query="something else", goal="g")
    assert spent.is_error and "budget for this run is used up (1 of 1)" in spent.content
    blank = await call(agent, "web_search", query="  ", goal="g")
    assert blank.is_error and "non-empty query" in blank.content
    log = (tmp_path / "agent" / "web_log.jsonl").read_text().splitlines()
    assert [json.loads(line)["status"] for line in log] == ["ok", "budget", "invalid"]

    failing = FakeExa(fail=True)
    agent = make_agent(tmp_path / "b", monkeypatch, failing)
    for _ in range(web_agent.MAX_CONSECUTIVE_ERRORS):
        result = await call(agent, "web_search", query="q", goal="g")
        assert result.is_error and result.content.startswith("web_search failed")
        assert not LEAKS.search(result.content)
    result = await call(agent, "web_search", query="q", goal="g")
    assert result.content.startswith(web_agent.UNAVAILABLE)
    # Each refused search was checked with a probe, which Exa refused too: the cutoff.
    assert len(failing.calls) == 2 * web_agent.MAX_CONSECUTIVE_ERRORS
    log = (tmp_path / "b" / "agent" / "web_log.jsonl").read_text()
    assert "cache only" in log  # Exa's detail stays in the host-side log
    context = AgentContext()
    agent._populate_context(context)
    assert "cache only" in context.metadata["web"]["disabled_reason"]

    agent = make_agent(tmp_path / "c", monkeypatch)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="EXA_API_KEY is not set"):
        await agent.run("x", SimpleNamespace(), AgentContext())  # type: ignore[arg-type]


async def test_rejections_are_answers_and_are_remembered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refused = exa.SnapshotError(f"HTTP 400 INVALID_REQUEST {AS_OF}", status=400)
    errors = {"first": refused, "bad query": refused, "https://bad.dev": refused}
    fake = FakeExa(HITS, errors=errors)
    agent = make_agent(tmp_path, monkeypatch, fake)
    # Before anything is verified, a refused search is checked with a probe. Exa answers it,
    # so the query was the problem: the agent is asked to rephrase.
    early = await call(agent, "web_search", query="first", goal="g")
    assert early.content.startswith(web_agent.REJECTED) and not LEAKS.search(
        early.content
    )
    assert agent._web.probes == 1 and agent._web.verified
    # A refused query is remembered: re-sending it makes no request and costs nothing.
    for _ in range(3):
        again = await call(agent, "web_search", query="first", goal="g")
        assert again.content.startswith(web_agent.REJECTED)
    assert (
        "49 of 50 searches" in again.content
        and fake.calls.count(("search", "first")) == 1
    )
    gone = await call(agent, "web_fetch", url="https://bad.dev")
    assert gone.content.startswith("This page is not available")
    assert "99 of 100 page fetches" in gone.content
    await call(
        agent, "web_fetch", url="https://bad.dev"
    )  # remembered: no second request
    assert fake.calls.count(("contents", "https://bad.dev")) == 1

    # A refusal is an answer: transient errors on either side of it are not "in a row".
    fake.errors["down"] = exa.SnapshotError("HTTP 503", status=503)
    for query in ("down", "bad query", "down", "down"):
        await call(agent, "web_search", query=query, goal="g")
    assert agent._web.outages == 0 and agent._web.transient_streak == 2

    # Account errors end the tools for good, verified or not.
    for status in (401, 402, 403):
        fake.errors["account"] = exa.SnapshotError(f"HTTP {status}", status=status)
        await call(agent, "web_search", query="account", goal="g")
    assert agent._web.disabled_reason == "HTTP 403"
    context = AgentContext()
    agent._populate_context(context)
    web = context.metadata["web"]
    # A refused page is counted as rejected, not as missing, and only once.
    assert web["rejected"] == 3 and web["errors"] == 6 and web["not_available"] == 0


async def test_breaker_tells_the_cutoff_from_bad_requests_and_outages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refused = exa.SnapshotError(f"HTTP 400 {AS_OF} cache only", status=400)
    # A cutoff Exa refuses. A fetch answered with "no version" verifies nothing (it looks the
    # same), so each refused search is probed, the probe is refused too, and the tools end.
    errors: dict[str, Exception] = dict.fromkeys(
        ("q1", "q2", "q3", web_agent.PROBE_QUERY), refused
    )
    agent = make_agent(tmp_path, monkeypatch, FakeExa(errors=errors))
    await call(agent, "web_fetch", url="https://gone.dev")
    assert not agent._web.verified
    for query in ("q1", "q2", "q3"):
        result = await call(agent, "web_search", query=query, goal="g")
        assert result.content.startswith("web_search failed")
    assert agent._web.disabled_reason is not None and agent._web.probes == 3

    # Bad URLs during an outage: the probe fails transiently, so the tools pause, not end.
    down = exa.SnapshotError("HTTP 503", status=503)
    errors = {f"https://u{i}.dev": refused for i in range(3)} | {
        web_agent.PROBE_QUERY: down
    }
    agent = make_agent(tmp_path / "b", monkeypatch, FakeExa(errors=errors))
    for i in range(3):
        await call(agent, "web_fetch", url=f"https://u{i}.dev")
    assert agent._web.disabled_reason is None and agent._web.outages == 1

    # Refusal after refusal checks the cutoff again (Exa's window moves); here it now fails.
    fake = FakeExa(HITS, errors={f"bad{i}": refused for i in range(4)})
    agent = make_agent(tmp_path / "c", monkeypatch, fake)
    await call(agent, "web_search", query="parse json python error", goal="g")
    for i in range(3):
        result = await call(agent, "web_search", query=f"bad{i}", goal="g")
        assert result.content.startswith(web_agent.REJECTED)
    fake.errors[web_agent.PROBE_QUERY] = refused
    result = await call(agent, "web_search", query="bad3", goal="g")
    assert result.content.startswith("web_search failed")
    assert agent._web.probes == 1 and agent._web.lasting_streak == 1


async def test_halted_results_stay_unseen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def hit(name: str) -> exa.Hit:
        return exa.Hit(f"https://{name}.dev", name.upper(), [f"about {name}"])

    results = {
        "alpha beta gamma": [hit("a"), hit("b"), hit("c"), hit("d")],
        "alpha beta gamma delta": [
            hit("a"),
            hit("b"),
            hit("c"),
            hit("e"),
        ],  # J 0.75, 1 new
        "something else entirely": [hit("e")],
    }
    agent = make_agent(tmp_path, monkeypatch, FakeExa(results), web_gate_patience=1)
    await call(agent, "web_search", query="alpha beta gamma", goal="g")
    halted = await call(agent, "web_search", query="alpha beta gamma delta", goal="g")
    assert (
        halted.content.startswith("Search halted:") and "(1 new of 4" in halted.content
    )
    assert "https://e.dev" not in agent._web.seen
    assert len(agent._extractor.prompts) == 1  # type: ignore[attr-defined]
    later = await call(agent, "web_search", query="something else entirely", goal="g")
    assert "1 results (1 new)" in later.content and "> about e" in later.content


async def test_early_fetch_refusals_and_unexpected_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refused = exa.SnapshotError("HTTP 400 INVALID_REQUEST", status=400)
    flaky = exa.SnapshotError(
        "HTTP 502 CRAWL_UNKNOWN_ERROR", status=502, page_level=True
    )
    errors: dict[str, Exception] = {f"https://old{i}.dev": refused for i in range(4)}
    errors["https://flaky.dev"] = flaky
    errors["odd"] = AttributeError("'list' object has no attribute 'get'")
    fake = FakeExa(HITS, errors=errors)
    agent = make_agent(tmp_path, monkeypatch, fake)
    # A fetch refused before any success is checked with one plain search: Exa answers it,
    # so the refusals are about their URLs and never stop the tools.
    for i in range(4):
        gone = await call(agent, "web_fetch", url=f"https://old{i}.dev")
        assert gone.content.startswith("This page is not available")
    assert agent._web.lasting_streak == agent._web.transient_streak == 0
    # The fourth refusal in a row checks again (Exa's window moves): a second probe.
    assert agent._web.probes == 2 and agent._web.verified
    assert ("search", web_agent.PROBE_QUERY) in fake.calls
    ok = await call(agent, "web_search", query="parse json python error", goal="g")
    assert "2 results (2 new)" in ok.content

    # A page Exa could not read just now: charged, but not remembered as missing.
    for _ in range(2):
        flaky_result = await call(agent, "web_fetch", url="https://flaky.dev")
        assert flaky_result.is_error and flaky_result.content.startswith(
            "web_fetch could not"
        )
    assert fake.calls.count(("contents", "https://flaky.dev")) == 2
    assert "94 of 100 page fetches" in flaky_result.content

    # Any other exception is a logged service error with the generic message.
    odd = await call(agent, "web_search", query="odd", goal="g")
    assert odd.content.startswith(
        "web_search failed: the web service returned an error"
    )
    assert "AttributeError" not in odd.content and agent._web.transient_streak == 1
    log = [json.loads(line) for line in (tmp_path / "agent" / "web_log.jsonl").open()]
    assert log[-1]["status"] == "error" and "AttributeError" in log[-1]["error"]
    assert [e["status"] for e in log].count("page_error") == 2


async def test_outages_pause_the_tools_and_fetch_refusals_can_end_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(web_agent, "COOLDOWN_SEC", 0.05)
    down = exa.SnapshotError("/search: HTTP 503", status=503)
    agent = make_agent(tmp_path, monkeypatch, FakeExa(HITS, errors={"q": down}))
    for _ in range(web_agent.MAX_CONSECUTIVE_ERRORS):
        await call(agent, "web_search", query="q", goal="g")
    paused = await call(agent, "web_search", query="parse json python error", goal="g")
    assert paused.content.startswith(web_agent.UNAVAILABLE_NOW)
    await asyncio.sleep(0.06)
    back = await call(agent, "web_search", query="parse json python error", goal="g")
    assert "2 results (2 new)" in back.content
    context = AgentContext()
    agent._populate_context(context)
    web = context.metadata["web"]
    assert web["outages"] == 1 and web["disabled_reason"] is None and web["errors"] == 3

    # When the probe search is refused too, the fetch refusals are about the cutoff or key.
    refused = exa.SnapshotError(f"HTTP 400 {AS_OF}", status=400)
    errors: dict[str, Exception] = {f"https://u{i}.dev": refused for i in range(3)}
    errors[web_agent.PROBE_QUERY] = refused
    agent = make_agent(tmp_path / "b", monkeypatch, FakeExa(errors=errors))
    for i in range(3):
        result = await call(agent, "web_fetch", url=f"https://u{i}.dev")
        assert result.content.startswith("web_fetch failed")
    assert agent._web.disabled_reason is not None and agent._web.probes == 3


async def test_fetch_spellings_canonical_urls_and_notes_shown_on_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = exa.Page("https://x.dev/a", "A", "text of a")
    pages = {
        "http://x.dev/a": canonical,
        "https://docs.x.dev/api/": exa.Page("u", "API", "t"),
    }
    fake = FakeExa({"x": [exa.Hit("https://x.dev/a", "A", ["text"])]}, pages)
    agent = make_agent(tmp_path, monkeypatch, fake)
    agent._extractor = FakeLLM(["NOTES:\n- a says text (https://x.dev/a)"])

    first = await call(agent, "web_fetch", url="http://x.dev/a")
    assert "- a says text (https://x.dev/a)" in first.content
    # Exa's canonical URL counts as seen; the unchanged notes are not repeated.
    found = await call(agent, "web_search", query="x", goal="g")
    assert "(0 new)" in found.content and "[1] (seen) A" in found.content
    assert (
        web_agent.NOTES_UNCHANGED in found.content
        and "a says text" not in found.content
    )

    # Each spelling but the fragment is its own request: one missing does not hide another.
    missing = await call(agent, "web_fetch", url="https://docs.x.dev/api#run")
    assert missing.content.startswith("This page is not available")
    found_api = await call(agent, "web_fetch", url="https://docs.x.dev/api/#run")
    assert not found_api.is_error and "API" in found_api.content
    assert ("contents", "https://docs.x.dev/api") in fake.calls
    assert ("contents", "https://docs.x.dev/api/") in fake.calls


async def test_duplicate_results_and_a_fetch_only_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dupes = [
        exa.Hit("https://a.dev/x/", "X", ["one"]),
        exa.Hit("https://a.dev/x#part", "X again", ["two"]),
        exa.Hit("https://b.dev/y", "Y", ["three"]),
    ]
    agent = make_agent(tmp_path, monkeypatch, FakeExa({"q": dupes}))
    result = await call(agent, "web_search", query="q", goal="g")
    assert "2 results (2 new)" in result.content and "X again" not in result.content
    log = [json.loads(line) for line in (tmp_path / "agent" / "web_log.jsonl").open()]
    assert (
        log[0]["urls"] == ["https://a.dev/x/", "https://b.dev/y"]
        and log[0]["novelty"] == 1
    )

    fetch_only = make_agent(
        tmp_path / "f", monkeypatch, disallowed_tools=["web_search"]
    )
    assert "web_search" not in json.dumps(fetch_only._tool_schemas)
    page = await call(fetch_only, "web_fetch", url="https://gone.dev")
    assert "Web budget left: 99 of 100 page fetches." in page.content
    assert "searches" not in page.content


async def test_extractor_is_bounded_in_time_and_skipped_near_the_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class SlowLLM(FakeLLM):
        async def query(
            self, messages: list[dict[str, Any]], tools: Any
        ) -> llm.ParsedResponse:
            await asyncio.sleep(30)
            raise AssertionError("not reached")

    monkeypatch.setattr(web_agent, "EXTRACTOR_TIMEOUT_SEC", 0.05)
    agent = make_agent(tmp_path, monkeypatch)
    agent._extractor = SlowLLM()  # type: ignore[assignment]
    result = await call(agent, "web_search", query="parse json python error", goal="g")
    assert not result.is_error and "2 results (2 new)" in result.content
    assert agent._web.extractor["failures"] == 1

    agent = make_agent(tmp_path / "b", monkeypatch)
    agent._remaining_budget_sec = lambda: 60.0  # type: ignore[method-assign]  # < the margin
    late = await call(agent, "web_search", query="parse json python error", goal="g")
    assert late.is_error and late.content.startswith(web_agent.NO_TIME)
    assert agent._exa.calls == []  # type: ignore[union-attr]
    assert await agent._ask_extractor("stable", "varying") is None
    assert agent._extractor.prompts == []  # type: ignore[attr-defined]
    assert agent._web.extractor["skipped"] == 1 and agent._web.extractor["calls"] == 0

    # An Exa request is cut off where the end-of-budget margin starts.
    class SlowExa(FakeExa):
        async def search(self, query: str, num_results: int) -> list[exa.Hit]:
            await asyncio.sleep(30)
            raise AssertionError("not reached")

    agent = make_agent(tmp_path / "c", monkeypatch, SlowExa())
    agent._remaining_budget_sec = lambda: web_agent.BUDGET_MARGIN_SEC + 0.05  # type: ignore[method-assign]
    slow = await call(agent, "web_search", query="q", goal="g")
    assert slow.is_error and slow.content.startswith("web_search failed")
    log = (tmp_path / "c" / "agent" / "web_log.jsonl").read_text()
    assert "outlasted the run's time budget" in log


def test_extractor_has_no_effort_settings_or_cache_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-0123456789abcdef")
    call_kwargs = {
        "reasoning_effort": "max",
        "thinking": {"type": "enabled", "budget_tokens": 9999},
        "extra_body": {"reasoning": {"effort": "high"}, "store": False},
        "_skip_mcp_handler": True,
    }
    agent = OpenWorldHarness(
        logs_dir=tmp_path,
        model_name=MODEL,
        web_snapshot_as_of="2026-06-01",
        reasoning_effort="max",
        llm_call_kwargs=call_kwargs,
    )
    extractor = agent._extractor._config  # type: ignore[attr-defined]
    assert extractor.reasoning_effort is None and extractor.max_thinking_tokens is None
    assert extractor.llm_call_kwargs == {
        "extra_body": {"store": False},
        "_skip_mcp_handler": True,
    }
    assert (
        agent.config.llm_call_kwargs == call_kwargs
    )  # the agent's own calls are unchanged
    messages = [{"role": "user", "content": "x"}]
    assert agent._extractor._apply_prompt_caching(messages) == [
        {"role": "user", "content": "x"}
    ]
    assert (
        agent._llm._apply_prompt_caching(messages) != messages
    )  # Proximus's own: cached
    # The extractor marks only the stable start of its prompt for caching.
    (message,) = agent._extractor_messages("stable", "varying")
    assert message["content"] == [
        {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "varying"},
    ]
    agent._extractor.is_anthropic = lambda: False  # type: ignore[method-assign]
    assert agent._extractor_messages("stable", "varying") == [
        {"role": "user", "content": "stable\n\nvarying"}
    ]


def test_extractor_usage_matches_proximus_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_add_extractor_usage` counts every usage total `_accumulate_usage` does, and leaves
    the per-turn fields (context size, fallback counters) as they were."""
    agent = make_agent(tmp_path, monkeypatch)
    usage: dict[str, Any] = {}
    for i, (name, hint) in enumerate(
        typing.get_type_hints(llm.ParsedResponse).items(), 1
    ):
        options = (
            typing.get_args(hint) if isinstance(hint, types.UnionType) else (hint,)
        )
        if (
            int in options
        ):  # int and int | None alike; bool flags stay at their defaults
            usage[name] = 10 * i
        elif float in options:
            usage[name] = 0.5 * i
    parsed = llm.ParsedResponse(
        **usage, fallback_attempted=True, fallback_used=True, response_model="backup"
    )
    agent._run = RunState()
    agent._add_extractor_usage(parsed)
    ours = dataclasses.asdict(agent._run)
    assert ours["last_context_tokens"] is None and ours["fallback_models"] == {}
    assert ours["fallback_attempted_turns"] == ours["fallback_turns"] == 0
    agent._run = RunState()
    Proximus._accumulate_usage(agent, parsed)
    theirs = dataclasses.asdict(agent._run)
    skip = {
        "start_monotonic",
        "last_context_tokens",
        "fallback_attempted_turns",
        "fallback_turns",
    }
    skip.add("fallback_models")
    assert {k: v for k, v in ours.items() if k not in skip} == {
        k: v for k, v in theirs.items() if k not in skip
    }
    assert ours["total_cost_usd"] > 0


async def test_nothing_model_facing_names_the_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = {
        "https://a.dev/json": exa.Page(
            "https://a.dev/json", "JSON", "x" * 100, "2026-05-28"
        )
    }
    agent = make_agent(tmp_path, monkeypatch, FakeExa(HITS, pages))
    agent._extractor = FakeLLM(["- a note (https://a.dev/json)"])
    env = SimpleNamespace(
        network_policy=SimpleNamespace(network_mode=NetworkMode.NO_NETWORK)
    )
    # Proximus's own system prompt is unchanged (it speaks of conversation history).
    texts = [
        json.dumps(agent._tool_schemas),
        agent.initial_message("Do it.", "/app", env),  # type: ignore[arg-type]
    ]
    for query in STAGNANT:
        texts.append((await call(agent, "web_search", query=query, goal="g")).content)
    texts.append((await call(agent, "web_fetch", url="https://a.dev/json")).content)
    texts.append((await call(agent, "web_fetch", url="https://gone.dev")).content)
    texts += agent._extractor.prompts  # type: ignore[attr-defined]
    texts += [
        web_agent.UNAVAILABLE,
        web_agent.FAILED,
        web_agent.REJECTED,
        web_agent.NO_TIME,
    ]
    texts += [web_agent.PAGE_FAILED, web_agent.NOTES_HEADER]
    for text in texts:
        assert not LEAKS.search(text), LEAKS.search(text)
    assert "web_search" in texts[1] and "no internet access" in texts[1]

    search_only = make_agent(
        tmp_path / "s", monkeypatch, disallowed_tools=["web_fetch"]
    )
    assert "web_fetch" not in json.dumps(search_only._tool_schemas)
    assert "You also have `web_search` for" in search_only.initial_message(
        "x", "/app", env
    )  # type: ignore[arg-type]
    assert (
        "searches."
        in (await call(search_only, "web_search", query="q", goal="g")).content
    )
    no_web = make_agent(
        tmp_path / "n", monkeypatch, disallowed_tools=["web_search", "web_fetch"]
    )
    assert "You also have" not in no_web.initial_message("x", "/app", env)  # type: ignore[arg-type]

    dated = make_agent(tmp_path / "d", monkeypatch, FakeExa(HITS), web_show_dates=True)
    shown = await call(dated, "web_search", query="parse json python error", goal="g")
    assert "published 2026-05-28" in shown.content


async def test_notes_reorganize_count_in_cost_and_survive_compaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = make_agent(tmp_path, monkeypatch, web_notes_items=2)
    many = "\n".join(f"- finding {i} (https://a.dev/{i})" for i in range(7))
    agent._extractor = FakeLLM(
        [many, "- merged (https://a.dev/0)\n- kept (https://a.dev/1)"]
    )
    await call(agent, "web_search", query="parse json python error", goal="g")
    assert agent._web.notes.items == [
        "merged (https://a.dev/0)",
        "kept (https://a.dev/1)",
    ]
    assert agent._web.reorganizations == 1
    assert agent._run.total_cost_usd == pytest.approx(0.02)
    assert (
        agent._run.last_context_tokens is None
    )  # compaction keeps tracking the main context

    async def summary() -> str:
        return f"my summary, quoting the last handoff: {prompts.HANDOFF_FOOTER}"

    async def progress() -> str:
        return "my progress"

    agent._handoff_summary = summary  # type: ignore[method-assign]
    agent._read_progress = progress  # type: ignore[method-assign]
    agent._run.messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *({"role": "assistant", "content": f"turn {i}"} for i in range(30)),
    ]
    await agent._compact()
    handoff = agent._run.messages[1]["content"]
    assert handoff.startswith(prompts.HANDOFF_HEADER) and handoff.endswith(
        prompts.HANDOFF_FOOTER
    )
    notes = f"{web_agent.NOTES_HEADER}\n- merged (https://a.dev/0)\n- kept (https://a.dev/1)"
    assert (
        "my progress\n\n" in handoff
        and f"{notes}\n\n{prompts.HANDOFF_FOOTER}" in handoff
    )
    assert (
        handoff.count(web_agent.NOTES_HEADER) == 1
    )  # not before the footer in the summary
    # The note about the web tools lived in the replaced first message: it comes along.
    assert f"{agent._tools_note()}\n\n{web_agent.NOTES_HEADER}" in handoff
    assert "You also have `web_search` and `web_fetch`" in handoff
    assert agent._steps[-1].source == "user" and agent._steps[-1].message == handoff
    trajectory = json.loads((tmp_path / "agent" / "trajectory.json").read_text())
    assert trajectory["steps"][-1]["message"] == handoff


# ----- platform wiring -----


async def test_web_is_off_without_a_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-0123456789abcdef")
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    agent = OpenWorldHarness(
        logs_dir=tmp_path / "agent", model_name=MODEL, time_budget_sec=3600
    )
    assert not agent.web_enabled and agent._extractor is None
    assert not {"web_search", "web_fetch"} & set(agent._tools)
    env = SimpleNamespace(
        network_policy=SimpleNamespace(network_mode=NetworkMode.NO_NETWORK)
    )
    assert "You also have" not in agent.initial_message("x", "/app", env)  # type: ignore[arg-type]

    ran: list[str] = []

    async def core_run(self: Any, instruction: str, environment: Any, context: Any):
        ran.append(instruction)

    monkeypatch.setattr(Proximus, "run", core_run)
    context = AgentContext()
    await agent.run("x", env, context)  # type: ignore[arg-type]  # no Exa key needed
    assert ran == ["x"]
    agent._populate_context(context)
    assert context.metadata is not None and "web" not in context.metadata
    assert not (tmp_path / "agent" / web_agent.WEB_LOG).exists()


def test_the_exa_key_is_hidden_from_sandbox_commands_only_with_web(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = make_agent(tmp_path, monkeypatch)
    assert agent._sandbox_hidden_env_keys() == {
        *agent._llm.consumed_env_keys,
        web_agent.EXA_KEY_ENV,
    }
    plain = OpenWorldHarness(
        logs_dir=tmp_path / "plain", model_name=MODEL, time_budget_sec=3600
    )
    assert web_agent.EXA_KEY_ENV not in plain._sandbox_hidden_env_keys()
