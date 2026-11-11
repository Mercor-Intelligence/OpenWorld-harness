#!/bin/sh
# The git operations the harness performs on a submission workspace, for checking the static git
# (scripts/build-static-git.sh) inside an image that ships no git. CI runs it as root in a few
# task base images; it ends with a commit by an unprivileged user.
set -eu
command -v git
git --version
mkdir -p /work && cd /work && echo hello > a.txt && mkdir -p .timer && echo t > .timer/x
git -c safe.directory='*' init -q
git config user.email proximus@localhost && git config user.name Proximus
mkdir -p .git/info && printf '/.timer/\n' >> .git/info/exclude
git add -A && git commit -q -m baseline
base=$(git rev-parse HEAD)
echo world >> a.txt && echo new > b.txt
git status --porcelain
git add -A && git commit -q -m submission
git diff --stat "$base" HEAD
git update-ref refs/proximus/winner HEAD
git rm -q --cached b.txt && git ls-files
test "$(git log --oneline | wc -l)" -eq 2
git checkout -q "$base" -- a.txt && test "$(cat a.txt)" = hello
id agent >/dev/null 2>&1 || useradd -m agent 2>/dev/null || adduser -D agent
su agent -s /bin/sh -c 'cd ~ && git init -q r && cd r && git config user.email a@b &&
    git config user.name a && touch f && git add f && git commit -qm x'
echo OK
