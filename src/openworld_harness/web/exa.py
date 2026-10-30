"""A minimal async client for Exa Snapshot (`snapshotAsOf` on /search and /contents).

Every request is pinned to one cutoff: Exa returns the newest stored version of each page at
or before it, and leaves out pages with no such version. Ranking still uses Exa's current
signals; only the content is bounded by the cutoff.

Exa's error messages name the snapshot ("Historical content is served from cache only",
"'snapshotAsOf' cannot be in the future"), so `SnapshotError.detail` is for logs only and
never reaches the model.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

BASE_URL = "https://api.exa.ai"
# Snapshot supports auto, fast, instant and lite; deep and deep-reasoning are refused.
SEARCH_TYPE = "auto"
RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
# Longest wait between attempts, whether from backoff or Exa's Retry-After.
MAX_BACKOFF_SEC = 60.0
# Exa's count may differ from Python's (UTF-16 units, a trim to a word boundary), so text this
# close to the requested length (1%, at most 1,000 characters) may have been cut too.
CUT_SLACK_CHARS = 1_000


class SnapshotError(Exception):
    """Exa failed. `detail` may name the snapshot: log it, never show it to the model.

    `page_level` marks a transient failure Exa reported for one page of a request it
    otherwise answered (a per-id status on /contents)."""

    def __init__(
        self,
        detail: str,
        status: int | None = None,
        tag: str | None = None,
        *,
        page_level: bool = False,
    ):
        super().__init__(detail)
        self.detail = detail
        self.status = status
        self.tag = tag
        self.page_level = page_level


@dataclass(frozen=True)
class Hit:
    url: str
    title: str
    highlights: list[str] = field(default_factory=list)
    published: str | None = None


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    text: str
    published: str | None = None
    truncated: bool = False  # Exa may have stopped at the requested length


def normalize_as_of(value: str, now: datetime | None = None) -> str:
    """An ISO 8601 date or date-time as Exa's `YYYY-MM-DDTHH:MM:SSZ` (a date is midnight
    UTC). Raises ValueError for unparseable or future values, which Exa refuses."""
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as e:
        raise ValueError(f"not an ISO 8601 date or date-time: {value!r}") from e
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    if parsed > (now or datetime.now(UTC)):
        raise ValueError(f"{value!r} is in the future")
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


class SnapshotClient:
    def __init__(
        self,
        api_key: str,
        as_of: str,
        *,
        base_url: str = BASE_URL,
        timeout_sec: float = 60.0,
        attempts: int = 5,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.as_of = as_of
        self._attempts = attempts
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            timeout=httpx.Timeout(timeout_sec),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search(self, query: str, num_results: int) -> list[Hit]:
        body = {
            "query": query,
            "type": SEARCH_TYPE,
            "numResults": num_results,
            "contents": {"highlights": True, "snapshotAsOf": self.as_of},
        }
        payload = await self._post("/search", body)
        hits = []
        for r in _dicts(payload.get("results")):
            url = r.get("url") or r.get("id")
            if not url:
                continue
            highlights = r.get("highlights")
            highlights = highlights if isinstance(highlights, list) else []
            hits.append(
                Hit(
                    url=_clean(str(url)),
                    title=_clean(str(r.get("title") or "")).strip(),
                    highlights=[
                        _clean(str(h)).strip() for h in highlights if str(h).strip()
                    ],
                    published=_text(r.get("publishedDate")),
                )
            )
        return hits

    async def contents(self, url: str, max_chars: int) -> Page | None:
        """The page as stored at the cutoff, or None when it has no eligible version."""
        body = {
            "ids": [url],
            "text": {"maxCharacters": max_chars},
            "snapshotAsOf": self.as_of,
        }
        payload = await self._post("/contents", body)
        results = _dicts(payload.get("results"))
        if results:
            r = results[0]
            text = _clean(str(r.get("text") or ""))
            return Page(
                url=_clean(str(r.get("url") or r.get("id") or url)),
                title=_clean(str(r.get("title") or "")).strip(),
                text=text,
                published=_text(r.get("publishedDate")),
                truncated=_utf16_len(text)
                >= max_chars - min(CUT_SLACK_CHARS, max_chars // 100),
            )
        # Exa reports why per id in `statuses`: a missing version is CONTENT_NOT_CACHED (404).
        # Only a 4xx that retrying cannot change means the page is not there.
        errors = [
            s.get("error")
            for s in _dicts(payload.get("statuses"))
            if s.get("status") == "error"
        ]
        error = errors[0] if errors else None
        code = error.get("httpStatusCode") if isinstance(error, dict) else None
        tag = error.get("tag") if isinstance(error, dict) else None
        if isinstance(code, int) and 400 <= code < 500 and code not in RETRY_STATUSES:
            return None
        why = f"HTTP {code} {tag or ''}" if isinstance(error, dict) else repr(error)
        raise SnapshotError(
            f"/contents: {url}: no result ({why.strip()})",
            code if isinstance(code, int) else None,
            tag,
            page_level=True,
        )

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self._attempts):
            last = attempt == self._attempts - 1
            wait: float | None = None
            try:
                response = await self._client.post(path, json=body)
            except httpx.HTTPError as e:
                if last:
                    raise SnapshotError(f"{path}: {type(e).__name__}: {e}") from e
            else:
                if response.status_code == 200:
                    try:
                        payload = response.json()
                    except ValueError as e:
                        raise SnapshotError(f"{path}: response is not JSON", 200) from e
                    if not isinstance(payload, dict):
                        raise SnapshotError(
                            f"{path}: response is not a JSON object", 200
                        )
                    return payload
                if response.status_code not in RETRY_STATUSES or last:
                    raise _http_error(path, response)
                wait = _retry_after(response)
            if wait is None:
                wait = 2 ** (attempt + 1) + random.random()
            await asyncio.sleep(min(MAX_BACKOFF_SEC, wait))
        raise AssertionError("unreachable")


def _dicts(value: Any) -> list[dict[str, Any]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _text(value: Any) -> str | None:
    return _clean(str(value)) if value else None


def _clean(text: str) -> str:
    """Without lone surrogates (JSON can carry them), which no UTF-8 or UTF-16 encode takes."""
    return text.encode("utf-8", "replace").decode("utf-8")


def _utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return max(0.0, float(response.headers.get("retry-after", "")))
    except ValueError:  # absent, or an HTTP date
        return None


def _http_error(path: str, response: httpx.Response) -> SnapshotError:
    try:
        payload = response.json()
    except ValueError:
        payload = {"error": response.text[:500]}
    tag = payload.get("tag") if isinstance(payload, dict) else None
    error = payload.get("error") if isinstance(payload, dict) else payload
    return SnapshotError(
        f"{path}: HTTP {response.status_code} {tag or ''} {error}".strip(),
        response.status_code,
        tag,
    )
