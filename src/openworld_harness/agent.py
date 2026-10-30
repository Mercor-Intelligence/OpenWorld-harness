"""The agent Harbor runs: the Proximus core (proximus.py) plus the fork's additions.

Fixes to the core's behaviour go into proximus.py; options and tools that upstream
Proximus does not have go here, so the core stays easy to merge with upstream.
"""

from typing import override

from openworld_harness.proximus import Proximus


class OpenWorldHarness(Proximus):
    """Long-horizon, native-function-calling agent harness."""

    @staticmethod
    @override
    def name() -> str:
        return "openworld-harness"
