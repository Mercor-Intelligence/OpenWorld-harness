#!/bin/sh
# Build a statically linked git (musl libc, zlib) that runs in any x86_64 Linux image.
#
# The harness keeps its submissions in a git repository inside the task workspace and uses only
# git's builtin subcommands, so this single binary is enough. Pass its path as the harness's
# `static_git_path` option, and setup copies it into sandboxes whose image ships no git.
# The binary is git, so it is under git's own license (GPL-2.0).
#
# Needs: curl, xz-utils, make, gcc and musl-tools (musl-gcc). Usage: build-static-git.sh OUT
set -eu

GIT_VERSION=2.56.0
GIT_SHA256=26c56c296b38c0695b26fa95f475f1d01704d2d38e73465ca30b0b2f5dc789d3
ZLIB_VERSION=1.3.2
ZLIB_SHA256=bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16

out=${1:?usage: build-static-git.sh OUT}
# The build runs in a temporary directory, so make OUT absolute first.
case $out in /*) ;; *) out=$PWD/$out ;; esac
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"

curl -fsSL -o git.tar.xz "https://mirrors.edge.kernel.org/pub/software/scm/git/git-$GIT_VERSION.tar.xz"
curl -fsSL -o zlib.tar.gz "https://github.com/madler/zlib/releases/download/v$ZLIB_VERSION/zlib-$ZLIB_VERSION.tar.gz"
echo "$GIT_SHA256  git.tar.xz" | sha256sum -c -
echo "$ZLIB_SHA256  zlib.tar.gz" | sha256sum -c -
tar xf git.tar.xz
tar xf zlib.tar.gz

(cd "zlib-$ZLIB_VERSION" && CC=musl-gcc ./configure --static --prefix="$work/zlib" >/dev/null &&
    make -j"$(nproc)" install >/dev/null)

# The NO_* switches drop everything that needs a library or an interpreter: no remotes over
# HTTP(S), no perl/python commands, no translations, no Rust subsystems. musl's regex lacks
# REG_STARTEND, so git uses its own (NO_REGEX), as Alpine's package does. The fsmonitor
# daemon needs kernel headers that musl-gcc does not see, and nothing here uses it.
cd "git-$GIT_VERSION"
make -j"$(nproc)" CC=musl-gcc prefix=/usr/local \
    CFLAGS="-O2 -I$work/zlib/include" LDFLAGS="-static -L$work/zlib/lib" \
    NO_CURL=1 NO_OPENSSL=1 NO_EXPAT=1 NO_PERL=1 NO_PYTHON=1 NO_TCLTK=1 NO_GETTEXT=1 \
    NO_ICONV=1 NO_RUST=1 NO_REGEX=1 NO_SYS_POLL_H=1 FSMONITOR_DAEMON_BACKEND= FSMONITOR_OS_SETTINGS= \
    git >/dev/null
strip git
install -D -m 0755 git "$out"
"$out" --version
