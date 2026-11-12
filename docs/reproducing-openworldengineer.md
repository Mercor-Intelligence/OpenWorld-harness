# Reproducing OpenWorldEngineer runs

The [OpenWorldEngineer](https://openworldengineer.mercor.com) leaderboard ran this
harness before its first public release, as internal versions 0.3.1 and 0.3.2 (the
versions its trajectories record; 0.0.1 behaves as 0.3.2). Its runner set a few more
things per task, and [`examples/openworldengineer/`](../examples/openworldengineer)
does the same with plain Harbor. Run it from a clone, so the locked LiteLLM (1.106.2)
is the one used:

```sh
git clone https://github.com/mercor-code-envs/OpenWorld-harness && cd OpenWorld-harness
uv sync --locked && . .venv/bin/activate

# Once: a static git for the task images that ship none (9 of the 26 tasks).
sh scripts/build-static-git.sh static-git/git

export ANTHROPIC_API_KEY=... EXA_API_KEY=...
python examples/openworldengineer/make_jobs.py --model anthropic/claude-opus-5 \
  --reasoning-effort max --static-git static-git/git --env modal --out owe-runs \
  path/to/OpenWorldEngineer/tasks/*
# It prints a `harbor run -c owe-runs/jobs/<task>.json` command for each task.
```

Each job runs 8 attempts of `owe_agent:OpenWorldEngineerHarness`, which is the harness
plus the leaderboard's system-prompt ending: a `## Web access` section saying whether the
model has the web tools. Its options are, in order (later wins):

- the runner's defaults: `large_input_exclude_mb = 100`, `large_input_total_mb = 256`,
  `sandbox_limits` from the task's `[environment]`, `static_git_path`, and
  `llm_call_kwargs = {"_skip_mcp_handler": true}`;
- the task's `[agent.openworld]` table, if any;
- your options: `reasoning_effort`, `web_snapshot_as_of = "2026-10-01"` (the snapshot
  every leaderboard run searched; `--no-web` drops it), and any `--ak KEY=VALUE`.

`time_budget_sec` is the task's `[agent].timeout_sec`, and Harbor's hard agent timeout
is `hard_timeout_sec` of the options. The job runs a copy of the task under
`owe-runs/tasks/` whose agent phase has no network (`[agent] network_mode =
"no-network"`). Its files are hard links to the task's (full copies across
filesystems), except the rewritten `task.toml` and a marker file. As on the
leaderboard, the web tools are then the agent's only way out of the sandbox. Use the
reasoning effort the leaderboard lists for the model. Model keys and `EXA_API_KEY` are
read from the shell that runs Harbor.

- The leaderboard ran on Modal. Harbor's Docker environment allocates no GPUs, so with
  `--env docker` (the default) the 7 GPU tasks are skipped. On Docker, Harbor enforces
  the offline phase with an nftables egress filter and refuses to start a trial when
  the Docker host's kernel lacks it. Several trials at once on one host also add
  timing noise to the tasks scored on speed.
- Every task ships the leaderboard verifier and its held-out test data in `tests/`, so
  runs are graded the same way as the leaderboard. `dbf-aircraft-design` builds its tools
  service from `environment/tools/` through its `docker-compose.yaml`.
- A job's name holds the task, the model and the web arm. Running the same job again
  resumes it, so give a changed configuration its own `--out`.
- Some models needed more options than these, which the leaderboard passed per model:
  streaming and request settings for a gateway, `model_info`, or a provider's beta
  header. A few models ran through a private LLM gateway. Trials lost to
  infrastructure failures (a sandbox that died, a provider outage) were rerun, or
  regraded from their exported submission, rather than counted.
