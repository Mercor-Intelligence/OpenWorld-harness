"""The web search loop as a Context Gathering Decision Process (Kausik, Swaminathan and
Kallus, "The Context Gathering Decision Process", arXiv:2605.07042).

A CGDP is a POMDP: a hidden state c (here, the web as Exa stored it at a cutoff the model is
not told), actions (`web_search`, `web_fetch`, and the outer agent's own tools), an
observation function F(a, c) (the snapshot search and contents endpoints), a per-action
cost (a search and fetch budget) and a terminal answer (the agent's submission). The paper's
two orchestrator-side interventions live here, outside the model's prompt:

- **Predicate-based belief state** (paper section 5.1). After each observation a short
  extractor call turns (b_t, o_t) into new freeform notes (freeform beat structured JSON in
  10 of 12 settings). Past `trigger` notes, a reorganization call curates them back to
  `target` (the paper uses 10 and 6).
- **Programmatic exhaustion gate** (section 5.2). A search is stagnant when its query's token
  Jaccard against recent queries is at least `jaccard` and its unique-page rate (results not
  seen before) is at most `novelty`. After `patience` stagnant searches in a row the gate
  fires. The paper's best configuration is J >= 0.6, UPR <= 0.3, p = 2 (f_j0.6_u0.3_p2), and
  it beat both LLM-judged gates.

The paper's gate ends the episode with an answer from b_t. Here search is a sub-loop of a
coding agent, so firing ends only the stagnant search line: the agent gets b_t instead of
the stale results and keeps working.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

TOKEN = re.compile(r"\w+")
# Recent queries the Jaccard signal compares against: 3 catches an A-B-A oscillation, the
# alternating-framings loop of the paper's stagnation trace (appendix A.2, G5/G6).
GATE_WINDOW = 3
NOTE_MAX_CHARS = 600
# A note's source, which the extraction prompt asks for at its end: "... (https://...)". A
# URL may hold one level of parentheses (".../Python_(programming_language)").
SOURCE_SUFFIX = re.compile(r"\s*\((https?://(?:[^()\s]|\([^()\s]*\))+)\)\s*$")
TASK_MAX_CHARS = 4000
NOTHING = "Nothing relevant."
NOTES_MARKER = "NOTES:"


def tokens(text: str) -> frozenset[str]:
    return frozenset(TOKEN.findall(text.lower()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def normalize_url(url: str) -> str:
    """The identity of a page for novelty: no fragment, case-folded host, no trailing /."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:  # e.g. an unbalanced IPv6 bracket: use the URL as given
        return url.strip()
    path = parts.path.rstrip("/")
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, parts.query, "")
    )


@dataclass(frozen=True)
class Signal:
    jaccard: float
    novelty: float  # the unique passage rate: new results / results (0 for no results)
    stagnant: bool
    fire: bool


@dataclass
class Gate:
    jaccard: float = 0.6
    novelty: float = 0.3
    patience: int = 2
    window: int = GATE_WINDOW
    queries: list[frozenset[str]] = field(default_factory=list)
    streak: int = 0

    def observe(self, query: str, n_results: int, n_new: int) -> Signal:
        """Record one search. `fire` says the gate would halt it; the caller decides whether
        the gate is live, so a disabled gate still logs where it would have fired."""
        current = tokens(query)
        overlap = max(
            (jaccard(current, q) for q in self.queries[-self.window :]), default=0.0
        )
        novelty = n_new / n_results if n_results else 0.0
        stagnant = overlap >= self.jaccard and novelty <= self.novelty
        # The streak is not reset on firing: the gate keeps halting until a search that is not
        # stagnant ends the line, so re-sending a halted query cannot reveal its results.
        self.streak = self.streak + 1 if stagnant else 0
        fire = self.streak >= self.patience
        self.queries.append(current)
        del self.queries[: -self.window]  # only the last `window` are ever compared
        return Signal(round(overlap, 3), round(novelty, 3), stagnant, fire)


@dataclass
class Notes:
    """The belief state b_t: freeform notes, one per line, each with its source."""

    target: int = 6
    items: list[str] = field(default_factory=list)
    version: int = (
        0  # bumped on every change, so results can show the notes only when new
    )

    @property
    def trigger(self) -> int:
        return self.target + 4

    def add(self, new: list[str]) -> int:
        before = len(self.items)
        for note in new:
            if note not in self.items:
                self.items.append(note)
        added = len(self.items) - before
        self.version += bool(added)
        return added

    def needs_reorganizing(self) -> bool:
        return len(self.items) > self.trigger

    def replace(self, curated: list[str]) -> None:
        self.items = curated[: self.target]
        self.version += 1

    def keep_newest(self) -> None:
        """The fallback when reorganization fails: stay bounded, keep the latest findings."""
        if len(self.items) > self.trigger:
            self.items = self.items[-self.trigger :]
            self.version += 1

    def render(self) -> str:
        return (
            "\n".join(f"- {item}" for item in self.items)
            if self.items
            else "(none yet)"
        )


def parse_notes(reply: str) -> list[str]:
    """The extractor's notes: the bullet lines after its last NOTES: line, or all bullet lines
    when there is none. Reasoning before the marker, and "Nothing relevant.", are dropped."""
    lines = reply.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        bare = lines[i].strip().strip("*#").strip()
        if bare.startswith(
            NOTES_MARKER
        ):  # as asked, in capitals: not a closing "Notes: ..."
            lines = [bare[len(NOTES_MARKER) :], *lines[i + 1 :]]
            break
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("- ", "* ", "• ")):
            note = stripped[2:].strip()
            if note and note != NOTHING:
                out.append(clip_note(note))
    return out


def clip_note(note: str, limit: int = NOTE_MAX_CHARS) -> str:
    """Shorten an overlong note from the middle, keeping its trailing (source URL)."""
    if len(note) <= limit:
        return note
    source = SOURCE_SUFFIX.search(note)
    tail = (
        source.group(0).strip() if source and len(source.group(0)) <= limit // 2 else ""
    )
    body = note[: source.start()] if tail and source else note
    room = limit - 1 - (len(tail) + 1 if tail else 0)
    return body[:room].rstrip() + "…" + (f" {tail}" if tail else "")


def _abridge(text: str, limit: int = TASK_MAX_CHARS) -> str:
    return (
        text
        if len(text) <= limit
        else text[:limit] + "\n[... task instruction abridged ...]"
    )


# Adapted from the paper's freeform extraction prompt (appendix E.2): it operates on
# (b_t, o_t) only, and asks for new notes alone so the belief state is appended to rather
# than rewritten on every step. The parts that never change in a run (instructions, task)
# come first so providers can cache them; the reasoning goes before a NOTES: line so that
# bullets in it are not taken for notes.
EXTRACTION_PROMPT = (
    "You keep research notes for an agent working on a software task. After each web search "
    "or page read by the agent, you get its current notes and the new observation, and you "
    "record what is new.\n\n"
    "First, what does the observation actually state? Take claims exactly as presented: keep "
    "which project, version, API, function, flag, value or person is involved, and do not "
    "paraphrase in a way that changes attribution or meaning. Then, how do they bear on what "
    "the agent was looking for, the task, and the current notes? What is genuinely new? If a "
    "note already covers it, skip it.\n\n"
    'After that reasoning, write a line containing only "{marker}", followed by ONLY genuinely '
    'new notes, one per line, each starting with "- " and ending with its source URL in '
    "parentheses. Quote exact details verbatim (signatures, commands, error text, numbers). If "
    'the observation contains nothing new beyond the current notes, write "{marker}" and then '
    "exactly: {nothing}\n\n"
    "Task (the agent's instruction):\n{task}"
)
EXTRACTION_OBSERVATION = (
    "The agent just {action}.\n\n"
    "What the agent was looking for:\n{goal}\n\n"
    "Current notes (DO NOT repeat any of these):\n{notes}\n\n"
    "New observation:\n{observation}\n\n"
    'Reason first, then write "{marker}" and the new notes.'
)

# Adapted from the paper's reorganization prompt (appendix E.3).
REORGANIZATION_PROMPT = (
    "You are curating research notes for an agent working on a software task. Produce a "
    "compact, prioritized set.\n\n"
    "Instructions:\n"
    "- Keep at most {target} notes, most important first.\n"
    "- Merge redundant notes into single comprehensive ones.\n"
    "- Drop notes irrelevant to the task.\n"
    "- PRESERVE notes that form multi-step reasoning chains, even if individually they seem "
    "tangential.\n"
    "- Every note keeps its source URL and any verbatim quotes.\n"
    "- Rewrite remaining open questions to reflect what is actually still unknown, and drop "
    "questions the notes now answer.\n\n"
    'Write a line containing only "{marker}", then only the notes, one per line, each starting '
    'with "- ".\n\n'
    "Task (the agent's instruction):\n{task}"
)
REORGANIZATION_NOTES = "Current notes:\n{notes}"


def extraction_prompt(
    action: str, task: str, goal: str, notes: Notes, observation: str
) -> tuple[str, str]:
    """The extractor's prompt as (the part that stays the same all run, the rest)."""
    stable = EXTRACTION_PROMPT.format(
        marker=NOTES_MARKER, nothing=NOTHING, task=_abridge(task)
    )
    varying = EXTRACTION_OBSERVATION.format(
        action=action,
        goal=goal.strip() or "(not stated)",
        notes=notes.render(),
        observation=observation,
        marker=NOTES_MARKER,
    )
    return stable, varying


def reorganization_prompt(task: str, notes: Notes) -> tuple[str, str]:
    stable = REORGANIZATION_PROMPT.format(
        target=notes.target, task=_abridge(task), marker=NOTES_MARKER
    )
    return stable, REORGANIZATION_NOTES.format(notes=notes.render())
