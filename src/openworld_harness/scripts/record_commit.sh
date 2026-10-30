# Pin $commit as submission #$idx (a submission IS a git commit).
# Inputs arrive as exec ENV VARS — wd, commit, idx, stage_timeout, keep_head —
# never spliced into this script, so there is nothing to shell-quote. Runs as root: the agent user
# often cannot invoke git (task images wrap or hide it), so harness git
# must not inherit default_user.
#
# Empty / HEAD means "snapshot the live workspace": the model cannot commit
# when git is disabled, so HEAD is stale. An explicit ref that resolves is
# pinned as-is; an explicit ref that does not resolve fails (the model
# named a commit that is not in this repo). With keep_head=1 (the exit
# snapshot) the snapshot is staged on a private index and made with
# commit-tree: HEAD and the index stay as the model left them, so restoring
# an older submission keeps the files the model never tracked, and a
# snapshot cut short moves nothing.
# stdout contract (parsed by submissions.py): `subject=<s>` then `hash=<sha>`,
# plus `committed=1` when the harness itself made the commit.
cd "$wd" || { echo "workspace directory missing" >&2; exit 3; }
command -v git >/dev/null 2>&1 || {
    echo "git is not installed in the task image" >&2
    exit 7
}

git config user.email >/dev/null 2>&1 || git config user.email proximus@localhost
git config user.name >/dev/null 2>&1 || git config user.name Proximus

snapshot=0
made_commit=0
case "$commit" in
    ""|HEAD|head) snapshot=1 ;;
esac

if [ "$snapshot" -eq 0 ]; then
    full=$(git rev-parse -q --verify "$commit^{commit}") || {
        echo "'$commit' is not a commit in the workspace repository." \
            "Pass a valid commit hash, or omit commit to snapshot" \
            "the current workspace." >&2
        exit 6
    }
else
    # Stage everything the exclude list allows, then commit if the tree
    # moved; a clean tree with HEAD reuses that commit (confirm-reuse).
    gitdir=$(git rev-parse --git-dir 2>/dev/null) || gitdir=.git
    if [ -e "$gitdir/index.lock" ]; then
        echo "Another git process holds $gitdir/index.lock, so nothing was" \
            "recorded. If no git command of yours is still running, remove" \
            "that file and submit again." >&2
        exit 9
    fi
    index="$gitdir/index"
    if [ "$keep_head" = 1 ]; then
        index="$(cd "$gitdir" && pwd -P)/proximus-snapshot-index"
        rm -f "$index" "$index.lock"
        cp "$gitdir/index" "$index" 2>/dev/null || true  # its stat cache
        GIT_INDEX_FILE=$index
        export GIT_INDEX_FILE
    fi
    # Within stage_timeout seconds, where the image's timeout supports -k:
    # staging a huge workspace must not use up the submission's time, and
    # TERM first lets git remove its own lock. Old busybox stages unbounded.
    if [ -n "$stage_timeout" ] && timeout -k 5 5 true >/dev/null 2>&1; then
        marker=$(mktemp 2>/dev/null) || marker=""
        staged=0
        timeout -k 10 "$stage_timeout" git add -A 2>/dev/null || staged=$?
        # Timed out: GNU timeout exits 124, or 137 after -k; busybox's execs
        # git, which then dies of the TERM itself (143).
        if [ "$staged" -eq 124 ] || [ "$staged" -eq 137 ] \
            || [ "$staged" -eq 143 ]; then
            rm -f "$index.lock"
            if [ "$keep_head" = 1 ]; then
                rm -f "$index"
                unset GIT_INDEX_FILE
            fi
            if [ -n "$marker" ]; then
                timeout 20 find "$gitdir/objects" -type f -name 'tmp_*' \
                    -newer "$marker" -exec rm -f {} + 2>/dev/null
                rm -f "$marker"
            fi
            echo "Staging (git add -A) did not finish within ${stage_timeout}s;" \
                "nothing was recorded and partial staging was cleaned up. If" \
                "large generated files are not part of your result, add them" \
                "to .gitignore and submit again, or commit only what matters" \
                "and submit that commit hash." >&2
            largest=$(timeout 20 sh -c 'git ls-files -o --exclude-standard -z \
                | xargs -0 du -k 2>/dev/null | sort -rn | head -5' 2>/dev/null)
            if [ -n "$largest" ]; then
                printf 'Largest untracked files (KB):\n%s\n' "$largest" >&2
            fi
            exit 8
        fi
        [ -n "$marker" ] && rm -f "$marker"
    else
        git add -A 2>/dev/null || true
    fi
    if [ "$keep_head" = 1 ]; then
        tree=$(git write-tree 2>/dev/null) || tree=""
        rm -f "$index"
        unset GIT_INDEX_FILE
        [ -n "$tree" ] || {
            echo "could not snapshot the current workspace" >&2
            exit 5
        }
        head=$(git rev-parse -q --verify HEAD^{commit}) || head=""
        if [ -n "$head" ] && [ "$tree" = "$(git rev-parse "$head^{tree}")" ]; then
            full=$head
        else
            full=$(git commit-tree "$tree" ${head:+-p "$head"} \
                -m "Proximus submission snapshot" 2>/dev/null) || {
                echo "could not snapshot the current workspace" >&2
                exit 5
            }
            made_commit=1
        fi
    elif git rev-parse -q --verify HEAD^{commit} >/dev/null \
        && git diff --cached --quiet \
        && git diff --quiet; then
        full=$(git rev-parse HEAD)
    else
        git commit -qm "Proximus submission snapshot" --no-verify --allow-empty \
            >/dev/null 2>&1 || {
            echo "could not snapshot the current workspace" >&2
            exit 5
        }
        full=$(git rev-parse HEAD)
        made_commit=1
    fi
fi

git update-ref "refs/proximus/submissions/$idx" "$full" || {
    echo "could not pin refs/proximus/submissions/$idx" >&2
    exit 5
}

# Root-created objects must not lock the agent out of .git on tasks
# where the model can still run git.
if [ -d .git ]; then
    owner=$(stat -c '%u:%g' . 2>/dev/null) || owner=""
    if [ -n "$owner" ]; then
        chown -R "$owner" .git 2>/dev/null || true
    fi
fi

git log -1 --format='subject=%s' "$full"
echo "hash=$full"
if [ "$made_commit" -eq 1 ]; then
    echo "committed=1"
fi
