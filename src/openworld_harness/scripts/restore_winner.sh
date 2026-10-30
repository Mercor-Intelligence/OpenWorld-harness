# Check out the winning submission so grading and the end-of-run workspace
# capture see the submitted tree, not whatever the last turn left behind. The
# pre-restore HEAD is pinned under refs/proximus/end-state first so the
# kill-time committed state stays reachable in the repository. checkout -f
# rewrites TRACKED files only: untracked files survive the restore.
# Inputs as exec env vars: wd, winner, unwind. stdout contract: `restored`.
# Runs as root (see record_commit.sh).
cd "$wd" || { echo "workspace directory missing" >&2; exit 3; }
git update-ref refs/proximus/end-state HEAD || true
# $unwind is the end-of-run snapshot the harness committed but does not
# grade. Undo it first, keeping the files: what it began tracking is
# untracked again and survives the checkout, as if it had never been taken.
if [ -n "$unwind" ] && [ "$(git rev-parse HEAD 2>/dev/null)" = "$unwind" ] \
    && git rev-parse "$unwind^" >/dev/null 2>&1; then
    git reset -q "$unwind^" || { echo "unwind failed" >&2; exit 4; }
fi
git checkout -q -f "$winner" || { echo "checkout failed" >&2; exit 4; }
echo "restored"
