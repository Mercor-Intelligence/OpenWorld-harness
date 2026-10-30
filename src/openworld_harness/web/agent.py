"""``web_search`` and ``web_fetch`` over Exa Snapshot, on when ``web_snapshot_as_of``
is set; without it the agent is the plain core.

Ported from the RWE eval platform's WebProximus. The cutoff stays on the host: no
tool description, result, error or note names it, Exa's own error text goes only
to the logs, and published dates are hidden unless ``web_show_dates`` is set. The
tools run in the runner process, so ``EXA_API_KEY`` never enters the sandbox.

The search loop follows the CGDP framing in ``cgdp``: the belief state (research
notes) comes back with web results whenever it changes and is carried across
compaction, and the exhaustion gate halts a stagnant search line. Every web
action, with its gate signals and the notes after it, is appended to
``web_log.jsonl`` in the agent's logs; the run summary goes to the trial's agent
metadata under "web".
"""

import asyncio
import json
import os
import time
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Literal, override
from urllib.parse import urldefrag

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic import BaseModel, ConfigDict, Field

import openworld_harness.prompts as prompts
from openworld_harness import llm, tools
from openworld_harness.config import BUDGET_MARGIN_SEC, OpenWorldConfig
from openworld_harness.proximus import Proximus
from openworld_harness.web import cgdp, exa

# Read in the runner process only; a runner must never pass it to the sandbox.
EXA_KEY_ENV = "EXA_API_KEY"

WEB_LOG = "web_log.jsonl"
type Failure = Literal["rejection", "lasting", "transient"]
# Text requested per page; `web_fetch` windows through it without new requests. Exa cuts
# longer pages at this length.
PAGE_MAX_CHARS = 100_000
HIGHLIGHT_MAX_CHARS = 400
HIGHLIGHTS_PER_RESULT = 3
# Failures in a row (each already retried) that stop the web tools: lasting ones (the account,
# the cutoff) for the rest of the run, transient ones (an outage, a rate limit) for
# COOLDOWN_SEC. Also the refusals in a row after which the cutoff and key are checked again.
MAX_CONSECUTIVE_ERRORS = 3
COOLDOWN_SEC = 300.0
ACCOUNT_STATUSES = frozenset({401, 402, 403})
TRANSIENT_STATUSES = frozenset({408, 425, 429})
# The harness's own search, which tells a refused request from a bad cutoff or key while
# neither has been confirmed by an answer.
PROBE_QUERY = "Wikipedia"
# Longest one extractor call may take. The core's LLM retries provider errors until the run's
# budget is spent, which a best-effort notes update must never do.
EXTRACTOR_TIMEOUT_SEC = 180.0
# llm_call_kwargs are applied last, so these would override web_extractor_reasoning_effort.
EFFORT_KEYS = frozenset({"reasoning_effort", "reasoning", "thinking", "output_config"})

SEARCH_DESCRIPTION = (
    "Search the web. Returns the top results with title, URL and short excerpts. Results "
    "you already saw in earlier searches or fetches are marked (seen) and listed without "
    "excerpts.{read} State in `goal` the specific question this search should answer. "
    "Searches are budgeted; each result shows what is left, and repeating an identical query "
    "costs nothing."
)
FETCH_DESCRIPTION = (
    "Read a web page as plain text{index}. Long pages come in windows: pass `offset` to "
    "continue where the previous window ended. A page that is not in the index returns 'not "
    "available'. Fetches are budgeted; reading further into a page you already fetched costs "
    "nothing."
)
WEB_TOOLS_NOTE = (
    "You also have {tools} for looking things up on the web. They run outside the sandbox, "
    "whatever its network access: they return text to you and change nothing in the sandbox."
)
NOTES_HEADER = (
    "Research notes (kept by the harness from what you found on the web; they persist across "
    "context compaction):"
)
NOTES_UNCHANGED = "Research notes: unchanged since your last web result."
GATE_NOTICE = (
    "Search halted: this query repeats your recent searches, and the last {patience} such "
    "searches surfaced almost no pages you had not already seen ({new} new of {total} this "
    "time). Rephrasing it again is unlikely to find anything new. Continue the task with what "
    "you have, or search a substantially different angle: other names, terms, error messages "
    "or sources."
)
UNAVAILABLE = "The web tools are unavailable in this run."
UNAVAILABLE_NOW = "The web tools are unavailable right now. Try again in a few minutes or continue without them."
FAILED = "{tool} failed: the web service returned an error. Try again later or continue without it."
REJECTED = (
    "web_search failed: the web service rejected this query. Rephrase it or continue."
)
PAGE_FAILED = "web_fetch could not read this page just now: {url}. Try again later or use another source."
NO_TIME = (
    "The run's time budget is nearly used up, so no more web requests can be made. Finish the "
    "task with what you have."
)


class WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="The search query.")
    goal: str = Field(
        description="The specific question this search should answer, in one sentence."
    )
    num_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="How many results to return (1 to 10, default 5).",
    )


class WebFetchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(description="The page's URL (http or https).")
    offset: int = Field(
        default=0,
        ge=0,
        description="Character offset to start reading from (default 0).",
    )
    goal: str = Field(
        default="",
        description="What you want from this page, in one sentence (default: the goal of "
        "your last search).",
    )


@dataclass
class WebRun:
    """The web tools' state for one run."""

    gate: cgdp.Gate
    notes: cgdp.Notes
    seen: set[str] = field(default_factory=set)
    search_cache: dict[tuple[str, int], list[exa.Hit]] = field(default_factory=dict)
    pages: dict[str, exa.Page | None] = field(
        default_factory=dict
    )  # by URL sans fragment
    read: set[tuple[str, int]] = field(
        default_factory=set
    )  # (page, offset) windows shown
    searches: int = 0
    fetches: int = 0
    actions: int = 0
    not_available: int = 0  # pages Exa has no version of
    rejected: int = 0  # requests Exa refused (4xx) as being about the request
    rejected_searches: set[tuple[str, int]] = field(default_factory=set)
    errors: int = 0
    # Whether an answer with content (search results, a page) has shown that Exa takes this
    # run's cutoff and key; until then a refusal is checked with a probe search.
    verified: bool = False
    transient_streak: int = 0
    lasting_streak: int = 0
    rejection_streak: int = 0
    probes: int = 0
    disabled_reason: str | None = None
    cooldown_until: float | None = None  # time.monotonic()
    outages: int = 0
    gate_triggered: list[int] = field(default_factory=list)  # web action numbers (`i`)
    last_goal: str = ""
    notes_shown: int = -1  # the notes version the last web result showed
    reorganizations: int = 0
    extractor: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(
            (
                "calls",
                "failures",
                "skipped",
                "prompt_tokens",
                "completion_tokens",
                "cost_usd",
            ),
            0,
        )
    )
    network_mode: str | None = None


class ExtractorLLM(llm.LLM):
    """The agent's model without the core's prompt-cache breakpoints, which would cache each
    whole extractor prompt for an hour (a 2x input price write on Anthropic) when only its
    stable start repeats. `_extractor_messages` marks that start instead."""

    @override
    def _apply_prompt_caching(self, messages: list[llm.Message]) -> list[llm.Message]:
        return messages


class WebTools(Proximus):
    """The Proximus core plus ``web_search`` and ``web_fetch`` when
    ``web_snapshot_as_of`` is set."""

    options_model = OpenWorldConfig
    config: OpenWorldConfig
    # Keys a runner keeps for a web run but never passes to the sandbox.
    web_host_key_envs = (EXA_KEY_ENV,)

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        cfg = self.config
        self._extractor: ExtractorLLM | None = None
        if self.web_enabled:
            # Same model and key as the agent; no thinking budget unless asked for.
            extractor_config = cfg.model_copy(
                update={
                    "reasoning_effort": cfg.web_extractor_reasoning_effort,
                    "max_thinking_tokens": None,
                    "llm_call_kwargs": _without_effort(cfg.llm_call_kwargs),
                }
            )
            self._extractor = ExtractorLLM(
                str(self.model_name),
                extractor_config,
                self._remaining_budget_sec,
                self.logger,
                extra_env=self.extra_env,
            )
        self._web = self._new_web_run()
        self._exa: exa.SnapshotClient | None = None

    @property
    def web_enabled(self) -> bool:
        return self.config.web_snapshot_as_of is not None

    @override
    def build_tools(self) -> dict[str, tools.Tool]:
        surface = super().build_tools()
        if not self.web_enabled:
            return surface
        disallowed = self.config.disallowed_tools
        read = (
            " Read a result in full with web_fetch."
            if "web_fetch" not in disallowed
            else ""
        )
        index = (
            ", from the same web index as web_search"
            if "web_search" not in disallowed
            else ""
        )
        surface["web_search"] = tools.Tool(
            "web_search",
            SEARCH_DESCRIPTION.format(read=read),
            WebSearchArgs,
            self._web_search,
        )
        surface["web_fetch"] = tools.Tool(
            "web_fetch",
            FETCH_DESCRIPTION.format(index=index),
            WebFetchArgs,
            self._web_fetch,
        )
        return surface

    @override
    def initial_message(
        self, instruction: str, workdir: str, environment: BaseEnvironment
    ) -> str:
        message = super().initial_message(instruction, workdir, environment)
        note = self._tools_note()
        return f"{message}\n\n{note}" if note else message

    def _tools_note(self) -> str:
        names = [f"`{n}`" for n in ("web_search", "web_fetch") if n in self._tools]
        return WEB_TOOLS_NOTE.format(tools=" and ".join(names)) if names else ""

    @override
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if not self.web_enabled:
            await super().run(instruction, environment, context)
            return
        key = os.environ.get(EXA_KEY_ENV)
        if not key:
            # Running on without search would record a no-web trial as a web one.
            raise RuntimeError(
                f"{EXA_KEY_ENV} is not set in the runner; a web run needs it"
            )
        assert self.config.web_snapshot_as_of is not None
        self._web = self._new_web_run()
        self._web.network_mode = environment.network_policy.network_mode.value
        self._exa = exa.SnapshotClient(key, self.config.web_snapshot_as_of)
        try:
            await super().run(instruction, environment, context)
        finally:
            await self._exa.aclose()

    # ----- tools -----

    async def _web_search(self, args: WebSearchArgs) -> tools.ToolResult:
        cfg, w = self.config, self._web
        query = " ".join(args.query.split())
        n = args.num_results
        record: dict[str, Any] = {
            "tool": "web_search",
            "query": query,
            "goal": args.goal,
            "n": n,
        }
        if not query:
            message = "web_search needs a non-empty query."
            return self._finish(record | {"status": "invalid"}, message, is_error=True)
        w.last_goal = args.goal.strip() or w.last_goal
        if (query, n) in w.rejected_searches:  # refused before: not asked again
            return self._finish(
                record | {"status": "rejected"}, REJECTED, is_error=True
            )
        hits = w.search_cache.get((query, n))
        if hits is None:
            refusal = self._refusal("search", w.searches, cfg.web_max_searches)
            if refusal:
                return self._finish(
                    record | {"status": refusal[0]}, refusal[1], is_error=True
                )
            assert self._exa is not None
            try:
                hits = await self._call_exa(self._exa.search(query, n))
            except exa.SnapshotError as e:
                kind = await self._classify(e)
                if kind != "rejection":
                    return self._service_error("web_search", record, e, kind)
                # Exa refused this query: charged, but the tools stay on.
                w.searches += 1
                w.rejected_searches.add((query, n))
                self._rejection("web_search", record, e)
                return self._finish(
                    record | {"status": "rejected"}, REJECTED, is_error=True
                )
            w.searches += 1
            self._answered(verified=True)
            w.search_cache[(query, n)] = hits

        # One result per page: duplicates would be shown twice and skew the novelty.
        unique: dict[str, exa.Hit] = {}
        for hit in hits:
            unique.setdefault(cgdp.normalize_url(hit.url), hit)
        urls, hits = list(unique), list(unique.values())
        new = {u for u in urls if u not in w.seen}
        signal = w.gate.observe(query, len(hits), len(new))
        if signal.fire:
            w.gate_triggered.append(w.actions + 1)  # this action's `i` in the web log
        halted = signal.fire and cfg.web_gate
        added = 0
        if halted:
            # The model never sees these results, so they stay unseen and out of the notes.
            body = GATE_NOTICE.format(
                patience=cfg.web_gate_patience, new=len(new), total=len(hits)
            )
        else:
            body = self._format_hits(query, hits, urls, new)
            w.seen.update(urls)
            if new:  # seen results come without excerpts: nothing for the notes
                added = await self._update_notes("searched the web", body, w.last_goal)
        record |= {
            "status": "halted" if halted else "ok",
            "urls": [hit.url for hit in hits],
            "new": len(new),
            "jaccard": signal.jaccard,
            "novelty": signal.novelty,
            "stagnant": signal.stagnant,
            "gate_triggered": signal.fire,
            "notes_added": added,
        }
        # A halted search comes with the notes: they are what the agent continues with.
        return self._finish(record, body, show_notes=halted)

    async def _web_fetch(self, args: WebFetchArgs) -> tools.ToolResult:
        cfg, w = self.config, self._web
        url = args.url.strip()
        goal = args.goal.strip() or w.last_goal
        record: dict[str, Any] = {
            "tool": "web_fetch",
            "url": url,
            "offset": args.offset,
            "goal": goal,
        }
        if not url.lower().startswith(("http://", "https://")):
            message = "web_fetch needs an http:// or https:// URL."
            return self._finish(record | {"status": "invalid"}, message, is_error=True)
        try:
            # The fragment never reaches a server; any other difference may be a different page.
            key = urldefrag(url).url
        except ValueError:  # e.g. an unbalanced IPv6 bracket
            message = "web_fetch needs a valid http:// or https:// URL."
            return self._finish(record | {"status": "invalid"}, message, is_error=True)
        if key not in w.pages:
            refusal = self._refusal("fetch", w.fetches, cfg.web_max_fetches)
            if refusal:
                return self._finish(
                    record | {"status": refusal[0]}, refusal[1], is_error=True
                )
            assert self._exa is not None
            try:
                fetched = await self._call_exa(self._exa.contents(key, PAGE_MAX_CHARS))
            except exa.SnapshotError as e:
                if e.page_level:
                    # Exa answered but could not read this page now: charged, not remembered.
                    w.fetches += 1
                    w.errors += 1
                    self._answered(verified=False)
                    self.logger.info(f"web_fetch: {e.detail}")
                    record |= {"status": "page_error", "error": e.detail}
                    return self._finish(
                        record, PAGE_FAILED.format(url=url), is_error=True
                    )
                kind = await self._classify(e)
                if kind != "rejection":
                    return self._service_error("web_fetch", record, e, kind)
                # Exa refused this URL: to the model, a page that is not there.
                self._rejection("web_fetch", record, e)
                fetched = None
            else:
                # A page shows the cutoff works; a per-id "no version" does not (a cutoff
                # outside Exa's window looks the same).
                self._answered(verified=fetched is not None)
                if fetched is None:
                    w.not_available += 1
            w.fetches += 1
            w.pages[key] = fetched
        page = w.pages[key]
        if page is None:
            return self._finish(
                record | {"status": "not_available"},
                f"This page is not available: {url}",
                is_error=True,
            )
        # Exa may answer with the page's canonical URL; a search result for either is seen.
        w.seen.update({cgdp.normalize_url(url), cgdp.normalize_url(page.url)})
        total = len(page.text)
        if args.offset and args.offset >= total:
            what = (
                f"the first {total:,} characters of this page, all that is available"
                if page.truncated
                else f"this page ({total:,} characters)"
            )
            return self._finish(
                record | {"status": "bad_offset"},
                f"offset {args.offset} is past the end of {what}.",
                is_error=True,
            )
        end = min(total, args.offset + cfg.web_fetch_chars)
        observation = self._format_page(page, args.offset, end)
        added = 0
        if (key, args.offset) not in w.read:
            w.read.add((key, args.offset))
            added = await self._update_notes(
                "read part of a web page", observation, goal
            )
        record |= {
            "status": "ok",
            "chars": [args.offset, end, total],
            "notes_added": added,
        }
        return self._finish(record, observation)

    def _refusal(self, kind: str, used: int, allowed: int) -> tuple[str, str] | None:
        w = self._web
        if self._exa is None or w.disabled_reason is not None:
            return "unavailable", UNAVAILABLE
        if w.cooldown_until is not None and time.monotonic() < w.cooldown_until:
            return "cooling_down", UNAVAILABLE_NOW
        remaining = self._remaining_budget_sec()
        if remaining is not None and remaining <= BUDGET_MARGIN_SEC:
            return "no_time", NO_TIME
        if used >= allowed:
            return "budget", (
                f"The {kind} budget for this run is used up ({allowed} of {allowed}). "
                "Continue the task with what you have."
            )
        return None

    async def _call_exa[T](self, request: Awaitable[T]) -> T:
        """One Exa request, cut off where the core's end-of-budget margin starts (the client's
        own retries could outlast it), with every failure as a SnapshotError."""
        remaining = self._remaining_budget_sec()
        timeout = None if remaining is None else remaining - BUDGET_MARGIN_SEC
        try:
            return await asyncio.wait_for(request, timeout)
        except exa.SnapshotError:
            raise
        except TimeoutError as e:
            raise exa.SnapshotError(
                "the request outlasted the run's time budget"
            ) from e
        except Exception as e:  # an unexpected response or a client bug: logs only
            raise exa.SnapshotError(f"{type(e).__name__}: {e}") from e

    async def _classify(self, error: exa.SnapshotError) -> Failure:
        """What a failed request says: "rejection" (Exa refused this one request, the input's
        fault), "lasting" (the account, or the cutoff or key) or "transient" (an outage, a
        rate limit, an unexpected response)."""
        kind = _failure(error)
        if kind != "rejection" or self._web.verified:
            return kind
        # Nothing has shown yet that Exa takes this run's cutoff and key, so a refusal may be
        # about them: a plain search under the same cutoff tells.
        return await self._probe()

    async def _probe(self) -> Failure:
        assert self._exa is not None
        self._web.probes += 1
        try:
            await self._call_exa(self._exa.search(PROBE_QUERY, 1))
        except exa.SnapshotError as e:
            self.logger.info(f"web probe: {e.detail}")
            return "transient" if _failure(e) == "transient" else "lasting"
        self._answered(verified=True)
        return "rejection"

    def _answered(self, *, verified: bool) -> None:
        """Exa answered a request; `verified` when the answer had content (search results or
        a page), which shows the cutoff and key work."""
        w = self._web
        w.transient_streak = 0
        if verified:
            w.verified = True
            w.lasting_streak = w.rejection_streak = 0

    def _rejection(
        self, tool: str, record: dict[str, Any], error: exa.SnapshotError
    ) -> None:
        w = self._web
        w.rejected += 1
        self._answered(verified=False)
        w.rejection_streak += 1
        if w.rejection_streak >= MAX_CONSECUTIVE_ERRORS:
            # Refusal after refusal: check the cutoff and key again at the next one (Exa's
            # window moves).
            w.verified, w.rejection_streak = False, 0
        self.logger.info(f"{tool}: {error.detail}")
        record["error"] = error.detail

    def _service_error(
        self, tool: str, record: dict[str, Any], error: exa.SnapshotError, kind: Failure
    ) -> tools.ToolResult:
        w = self._web
        w.errors += 1
        if kind == "lasting":
            w.lasting_streak += 1
            if w.lasting_streak >= MAX_CONSECUTIVE_ERRORS:
                w.disabled_reason = error.detail[:500]
        else:
            w.transient_streak += 1
            if w.transient_streak >= MAX_CONSECUTIVE_ERRORS:
                w.cooldown_until = time.monotonic() + COOLDOWN_SEC
                w.outages += 1
                w.transient_streak = 0
        # The detail can name the snapshot: logs only.
        self.logger.warning(f"{tool}: {error.detail}")
        record |= {"status": "error", "error": error.detail}
        return self._finish(record, FAILED.format(tool=tool), is_error=True)

    def _finish(
        self,
        record: dict[str, Any],
        body: str,
        *,
        is_error: bool = False,
        show_notes: bool = False,
    ) -> tools.ToolResult:
        w = self._web
        w.actions += 1
        entry = {
            "i": w.actions,
            "turn": self._run.n_turns,
            "elapsed_sec": round(self._elapsed(), 1),
            **record,
        }
        if self.config.web_notes:
            entry["notes"] = list(w.notes.items)
        self._log(entry)
        return tools.ToolResult(
            f"{body}\n\n{self._footer(show_notes)}", is_error=is_error
        )

    def _footer(self, show_notes: bool) -> str:
        cfg, w = self.config, self._web
        searches, fetches = cfg.web_max_searches, cfg.web_max_fetches
        budgets = []
        if "web_search" in self._tools:
            budgets.append(f"{searches - w.searches} of {searches} searches")
        if "web_fetch" in self._tools:
            budgets.append(f"{fetches - w.fetches} of {fetches} page fetches")
        parts = [f"Web budget left: {', '.join(budgets)}."]
        if cfg.web_notes:
            # The notes stay in context from the result that last showed them.
            if show_notes or w.notes.version != w.notes_shown:
                parts.append(f"{NOTES_HEADER}\n{w.notes.render()}")
                w.notes_shown = w.notes.version
            else:
                parts.append(NOTES_UNCHANGED)
        return "\n\n".join(parts)

    def _format_hits(
        self, query: str, hits: list[exa.Hit], urls: list[str], new: set[str]
    ) -> str:
        if not hits:
            return f"No results for: {query}"
        lines = [f"{len(hits)} results ({len(new)} new) for: {query}"]
        for i, (hit, url) in enumerate(zip(hits, urls, strict=True), 1):
            title = hit.title or "(untitled)"
            if url not in new:
                lines.append(f"\n[{i}] (seen) {title}\n    {hit.url}")
                continue
            lines.append(f"\n[{i}] {title}\n    {hit.url}")
            if self.config.web_show_dates and hit.published:
                lines.append(f"    published {hit.published[:10]}")
            for highlight in hit.highlights[:HIGHLIGHTS_PER_RESULT]:
                lines.append(
                    f"    > {_clip(' '.join(highlight.split()), HIGHLIGHT_MAX_CHARS)}"
                )
        return "\n".join(lines)

    def _format_page(self, page: exa.Page, start: int, end: int) -> str:
        total = len(page.text)
        cut = page.truncated
        of = f"the first {total:,} available" if cut else f"{total:,}"
        head = [page.title or "(untitled)", page.url]
        if self.config.web_show_dates and page.published:
            head.append(f"published {page.published[:10]}")
        if end < total:
            head.append(
                f"[characters {start:,}-{end:,} of {of}; call web_fetch with offset={end} for more]"
            )
        elif cut:
            head.append(
                f"[characters {start:,}-{end:,}; the page may continue, but only its first "
                f"{total:,} characters are available]"
            )
        elif start:
            head.append(
                f"[characters {start:,}-{end:,} of {total:,}; end of page text]"
            )
        return (
            "\n".join(head)
            + "\n\n"
            + (page.text[start:end] or "(this page has no text)")
        )

    # ----- belief state -----

    async def _update_notes(self, action: str, observation: str, goal: str) -> int:
        """Fold one observation into the notes; returns how many notes it added."""
        if not self.config.web_notes:
            return 0
        notes = self._web.notes
        reply = await self._ask_extractor(
            *cgdp.extraction_prompt(action, self._instruction, goal, notes, observation)
        )
        if reply is None:
            return 0
        added = notes.add(cgdp.parse_notes(reply))
        if notes.needs_reorganizing():
            reply = await self._ask_extractor(
                *cgdp.reorganization_prompt(self._instruction, notes)
            )
            curated = cgdp.parse_notes(reply or "")
            if curated:
                notes.replace(curated)
                self._web.reorganizations += 1
            else:
                notes.keep_newest()
        return added

    async def _ask_extractor(self, stable: str, varying: str) -> str | None:
        stats = self._web.extractor
        timeout = EXTRACTOR_TIMEOUT_SEC
        remaining = self._remaining_budget_sec()
        if remaining is not None:
            timeout = min(timeout, remaining - BUDGET_MARGIN_SEC)
        if timeout <= 0:  # the end of the budget is near: the run needs that time
            stats["skipped"] += 1
            return None
        stats["calls"] += 1
        assert self._extractor is not None
        try:
            parsed = await asyncio.wait_for(
                self._extractor.query(
                    self._extractor_messages(stable, varying), tools=None
                ),
                timeout,
            )
        except Exception as e:  # notes are best-effort; the tool result still returns
            stats["failures"] += 1
            self.logger.warning(
                f"web notes: extractor call failed: {type(e).__name__}: {e}"
            )
            return None
        stats["prompt_tokens"] += parsed.prompt_tokens
        stats["completion_tokens"] += parsed.completion_tokens
        stats["cost_usd"] += parsed.cost_usd
        self._add_extractor_usage(parsed)
        return parsed.content

    def _extractor_messages(self, stable: str, varying: str) -> list[llm.Message]:
        """One user message, its stable start (instructions and task) first. On Anthropic
        that start gets a 5-minute cache breakpoint; other providers cache a repeated prefix
        on their own."""
        assert self._extractor is not None
        if not self._extractor.is_anthropic():
            return [{"role": "user", "content": f"{stable}\n\n{varying}"}]
        content = [
            {"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": varying},
        ]
        return [{"role": "user", "content": content}]

    def _add_extractor_usage(self, parsed: llm.ParsedResponse) -> None:
        """Count an extractor call in the run's totals through the core's own accounting, then
        restore what is per turn: the context size (the small extractor prompt would delay
        compaction) and the fallback counters (it is not a turn)."""
        run = self._run
        kept = (
            run.last_context_tokens,
            run.fallback_attempted_turns,
            run.fallback_turns,
            dict(run.fallback_models),
        )
        self._accumulate_usage(parsed)
        (
            run.last_context_tokens,
            run.fallback_attempted_turns,
            run.fallback_turns,
            run.fallback_models,
        ) = kept

    @override
    async def _compact(self) -> None:
        """Compact as the core does, then add what the replaced first message
        carried (the web tools note) and the notes to the handoff, before its last
        footer."""
        await super()._compact()
        if not self.web_enabled:
            return
        extras = [self._tools_note()]
        if self.config.web_notes and self._web.notes.items:
            extras.append(f"{NOTES_HEADER}\n{self._web.notes.render()}")
        block = "\n\n".join(e for e in extras if e)
        if not block:
            return
        handoff = self._run.messages[1] if len(self._run.messages) > 1 else {}
        content = handoff.get("content")
        if not isinstance(content, str) or not content.startswith(
            prompts.HANDOFF_HEADER
        ):
            self.logger.warning(
                "web notes: no compaction handoff found; notes not re-injected"
            )
            return
        # Before the last footer only: the summary or the instruction may quote it.
        head, footer, tail = content.rpartition(prompts.HANDOFF_FOOTER)
        content = (
            f"{head}{block}\n\n{footer}{tail}" if footer else f"{content}\n\n{block}"
        )
        handoff["content"] = content
        if self._steps and self._steps[-1].source == "user":
            self._steps[-1] = self._steps[-1].model_copy(update={"message": content})
        self._dump_artifacts()

    # ----- bookkeeping -----

    def _new_web_run(self) -> WebRun:
        cfg = self.config
        return WebRun(
            gate=cgdp.Gate(
                cfg.web_gate_jaccard, cfg.web_gate_novelty, cfg.web_gate_patience
            ),
            notes=cgdp.Notes(target=cfg.web_notes_items),
        )

    def _log(self, entry: dict[str, Any]) -> None:
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            with (self.logs_dir / WEB_LOG).open("a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError as e:
            self.logger.debug(f"Could not append to {WEB_LOG}: {e}")

    @override
    def _populate_context(self, context: AgentContext) -> None:
        super()._populate_context(context)
        if not self.web_enabled:
            return
        cfg, w = self.config, self._web
        context.metadata = {
            **(context.metadata or {}),
            "web": {
                "snapshot_as_of": cfg.web_snapshot_as_of,
                "sandbox_network_mode": w.network_mode,
                "searches": w.searches,
                "fetches": w.fetches,
                "actions": w.actions,
                "pages_seen": len(w.seen),
                "not_available": w.not_available,
                "rejected": w.rejected,
                "errors": w.errors,
                "outages": w.outages,
                "probes": w.probes,
                "disabled_reason": w.disabled_reason,
                "gate": {"live": cfg.web_gate, "triggered_at_action": w.gate_triggered},
                "notes": list(w.notes.items) if cfg.web_notes else None,
                "reorganizations": w.reorganizations,
                "extractor": dict(w.extractor),
            },
        }


def _without_effort(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in kwargs.items() if k not in EFFORT_KEYS}
    if isinstance(out.get("extra_body"), dict):
        out["extra_body"] = {
            k: v for k, v in out["extra_body"].items() if k not in EFFORT_KEYS
        }
    return out


def _failure(error: exa.SnapshotError) -> Failure:
    """A failure's kind from its status alone; a "rejection" may still be about the cutoff."""
    status = error.status
    if status in ACCOUNT_STATUSES:
        return "lasting"
    if status is None or not 400 <= status < 500 or status in TRANSIENT_STATUSES:
        return "transient"
    return "rejection"


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
