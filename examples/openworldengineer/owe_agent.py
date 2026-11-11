"""OpenWorld-harness with the system prompt the OpenWorldEngineer leaderboard used.

The leaderboard's runner ended the harness's system prompt with a "## Web access"
section that says whether the model has the web tools. This subclass adds the same
section, word for word. Select it as ``owe_agent:OpenWorldEngineerHarness`` with this
directory on ``PYTHONPATH``; ``make_jobs.py`` writes job files that do.
"""

from typing import override

from openworld_harness import OpenWorldHarness

WEB_TOOLS = ("web_search", "web_fetch")
SEARCH_ARM_PROMPT = (
    "You have a web search tool and a page fetch tool. "
    "Use them whenever they would help with the task."
)
NO_SEARCH_ARM_PROMPT = "You do not have web access."
SECTION_HEADER = "## Web access"


class OpenWorldEngineerHarness(OpenWorldHarness):
    """OpenWorld-harness with the web-access section at the end of the system
    prompt."""

    @override
    def system_prompt(self) -> str:
        sections = [
            super().system_prompt().rstrip(),
            f"{SECTION_HEADER}\n\n{self.web_access_prompt()}",
        ]
        return "\n\n".join(sections) + "\n"

    def web_access_prompt(self) -> str:
        """The search line when the model has both web tools, the no-web line when it
        has neither. Raises ValueError for a web run that disallows only one of them,
        which neither line describes (at construction, before any sandbox starts)."""
        disallowed = set(self.config.disallowed_tools)
        available = [t for t in WEB_TOOLS if self.web_enabled and t not in disallowed]
        if not available:
            return NO_SEARCH_ARM_PROMPT
        if len(available) == len(WEB_TOOLS):
            return SEARCH_ARM_PROMPT
        raise ValueError(
            f"disallowed_tools removes only some of the web tools ({', '.join(available)} "
            "left); the web-access section describes both or neither"
        )
