# Export submission $commit off the sandbox's repository, so a regrade can
# rebuild it in a fresh copy of the task image after the sandbox is gone.
#
# Writes, under $out_dir:
#   sub.tar    the files the commit adds or changes against the run's baseline
#              (refs/proximus/baseline, pinned by ensure_git.sh; the empty
#              tree when the workspace had no commit), read from the commit's
#              objects, not from the live tree;
#   deleted.z  NUL-separated paths the commit deletes against the baseline.
# Prints bytes=<tar size>, files=<n>, deleted=<n> and base=<oid>.
#
# Inputs as exec env vars (submissions.py): wd, commit, out_dir, max_kb.
# Exit 5: the changed files exceed max_kb (nothing written). Exit 6: the
# commit tracks a gitlink (a nested repository), whose contents a tar of this
# repository cannot carry. Runs as root, like record_commit.sh.
cd "$wd" || { echo "workspace directory missing: $wd" >&2; exit 3; }
full=$(git rev-parse -q --verify "$commit^{commit}" 2>/dev/null) || {
    echo "not a commit: $commit" >&2
    exit 4
}
base=$(git rev-parse -q --verify "refs/proximus/baseline^{commit}" 2>/dev/null) ||
    base=$(git hash-object -t tree /dev/null)

rm -rf "$out_dir" && mkdir -p "$out_dir" || exit 3
changes="$out_dir/changes"
# Raw diff lines ":oldmode newmode oldoid newoid status<TAB>path"; paths with
# special characters come C-quoted, which update-index --index-info reads.
git diff-tree -r --no-renames --diff-filter=d "$base" "$full" > "$changes" || exit 3
if awk '{ if ($2 == "160000") found = 1 } END { exit !found }' "$changes"; then
    echo "the submission tracks a gitlink (nested repository)" >&2
    rm -rf "$out_dir"
    exit 6
fi
files=$(wc -l < "$changes" | tr -d ' ')
kb=$(awk '{ print $4 }' "$changes" | git cat-file --batch-check='%(objectsize)' |
    awk '{ s += $1 } END { printf "%d", (s + 1023) / 1024 }')
if [ -n "$max_kb" ] && [ "${kb:-0}" -gt "$max_kb" ]; then
    echo "changed files total ${kb} KB, over the ${max_kb} KB export limit" >&2
    rm -rf "$out_dir"
    exit 5
fi

index="$out_dir/index"
GIT_INDEX_FILE=$index git read-tree --empty || exit 3
awk -F '\t' '{ split($1, f, " "); print f[2] " " f[4] "\t" $2 }' "$changes" |
    GIT_INDEX_FILE=$index git update-index --index-info || exit 3
tree=$(GIT_INDEX_FILE=$index git write-tree) || exit 3
git archive --format=tar -o "$out_dir/sub.tar" "$tree" || exit 3
git diff-tree -r -z --no-renames --diff-filter=D --name-only "$base" "$full" \
    > "$out_dir/deleted.z" || exit 3
deleted=$(tr -cd '\0' < "$out_dir/deleted.z" | wc -c | tr -d ' ')
rm -f "$index" "$changes"
echo "bytes=$(wc -c < "$out_dir/sub.tar" | tr -d ' ')"
echo "files=$files"
echo "deleted=$deleted"
echo "base=$base"
