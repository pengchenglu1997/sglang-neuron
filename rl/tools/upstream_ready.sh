#!/usr/bin/env bash
# Is a branch fit to be sent upstream?  Run it by hand from any checkout of the fork.
#
#   bash rl/tools/upstream_ready.sh <branch> [--base REF] [--skip-tests]
#
# 1. The files the branch changes since its merge base with the base (default origin/upstream-main) hold no TEAM path
#    (rl/**, docs/rl/**, README.md, AGENTS.md, CLAUDE.md, .github/**, .gitattributes). An ancestry test cannot do this,
#    because upstream-main is a single root commit.
# 2. The CPU tests of the branch pass: python -m pytest -q tests/cpu, run in a temporary worktree of the branch
#    (no tests/cpu folder: reported, not a failure; --skip-tests: not run).
# Exit status: 0 ready, 1 not ready, 2 bad input. The checkout and its branches are not changed.
set -u

BRANCH=""
BASE="origin/upstream-main"
SKIP_TESTS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE="${2:-}"; shift 2 || { echo "upstream_ready: --base needs a value" >&2; exit 2; } ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    -*) echo "upstream_ready: unknown option $1" >&2; exit 2 ;;
    *) BRANCH="$1"; shift ;;
  esac
done
[ -n "$BRANCH" ] || { echo "usage: upstream_ready.sh <branch> [--base REF] [--skip-tests]" >&2; exit 2; }

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi
fi

git rev-parse --verify --quiet "$BRANCH^{commit}" >/dev/null || { echo "upstream_ready: no such branch or commit: $BRANCH" >&2; exit 2; }
git rev-parse --verify --quiet "$BASE^{commit}" >/dev/null || { echo "upstream_ready: no such base: $BASE (git fetch origin upstream)" >&2; exit 2; }

status=0
team="$(git -c core.quotepath=false diff --name-only -z --no-renames "$BASE...$BRANCH" | "$PY" "$HERE/check_pr.py" classify --kind team)"
if [ -n "$team" ]; then
  echo "NOT READY: $BRANCH changes TEAM paths against $BASE:"
  printf '%s\n' "$team" | sed 's/^/  /'
  status=1
else
  echo "ok: no TEAM path in the diff of $BRANCH against $BASE"
fi

if [ "$SKIP_TESTS" -eq 1 ]; then
  echo "CPU tests: skipped (--skip-tests)"
elif [ "$status" -eq 0 ]; then
  if git cat-file -e "$BRANCH:tests/cpu" 2>/dev/null; then
    tmp="$(mktemp -d)"
    cleanup() { git worktree remove --force "$tmp/wt" >/dev/null 2>&1; rm -rf "$tmp"; }
    trap cleanup EXIT
    if git worktree add --detach "$tmp/wt" "$BRANCH" >/dev/null 2>&1; then
      if (cd "$tmp/wt" && "$PY" -m pytest -q tests/cpu); then
        echo "ok: CPU tests pass"
      else
        echo "NOT READY: CPU tests failed on $BRANCH"
        status=1
      fi
    else
      echo "upstream_ready: could not create a temporary worktree" >&2
      status=2
    fi
  else
    echo "CPU tests: no tests/cpu folder on $BRANCH, not run"
  fi
fi

[ "$status" -eq 0 ] && echo "READY: $BRANCH"
exit "$status"
