"""Web tools over Exa Snapshot: search and fetch pinned to a hidden date.

`exa` is the snapshot client, `cgdp` the search-loop signals and belief state, and
`agent` the layer that exposes `web_search` and `web_fetch` to the model when
`web_snapshot_as_of` is set. Ported from the RWE eval platform's WebProximus.
"""
