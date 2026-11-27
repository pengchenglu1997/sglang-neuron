#!/usr/bin/env bash
# Is a branch fit to be sent upstream?  Run it by hand from any checkout of the fork.
#
#   bash rl/tools/upstream_ready.sh <branch> [--base REF] [--skip-tests]
#
# 1. The files the branch changes since its merge base with the base (default origin/upstream-main) hold no TEAM path
#    (rl/**, docs/rl/**, README.md, AGENTS.md, CLAUDE.md, .github/**, .gitattributes). An ancestry test cannot do this,
#    because upstream-main is a single root commit. A failure of git (no merge base) or of the interpreter is exit 2.
# 2. The CPU tests of the branch pass, run in a temporary worktree of the branch: python -m pytest -q tests/cpu when
#    that folder exists, and every tests/**/selftest_*.py script as python <script>. When the branch has neither, the
#    tests are NOT CHECKED and the branch is not called READY (exit 3). --skip-tests: not run, and the last line says so.
# Exit status: 0 ready, 1 not ready, 2 bad input or a failed tool, 3 TEAM paths clean but no CPU test found.
# The checkout and its branches are not changed. The script first changes to the top level of the checkout it is run in.
set -u
set -o pipefail

BRANCH=""
BASE="origin/upstream-main"
SKIP_TESTS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE="${2:-}"; shift 2 || { echo "upstream_ready: --base needs a value" >&2; exit 2; } ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
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

# Work from the top of the repository, whatever folder the script is started in (the tests and the paths are relative to it).
TOP="$(git rev-parse --show-toplevel 2>/dev/null)" && [ -n "$TOP" ] || { echo "upstream_ready: not inside a git checkout" >&2; exit 2; }
cd "$TOP" || { echo "upstream_ready: cannot enter $TOP" >&2; exit 2; }

git rev-parse --verify --quiet "$BRANCH^{commit}" >/dev/null || { echo "upstream_ready: no such branch or commit: $BRANCH" >&2; exit 2; }
git rev-parse --verify --quiet "$BASE^{commit}" >/dev/null || { echo "upstream_ready: no such base: $BASE (git fetch origin upstream)" >&2; exit 2; }

tmp="$(mktemp -d)" || { echo "upstream_ready: mktemp failed" >&2; exit 2; }
cleanup() { git worktree remove --force "$tmp/wt" >/dev/null 2>&1; rm -rf "$tmp"; }
trap cleanup EXIT

# Each step is checked on its own: a failed git or a missing interpreter must never read as "no TEAM path".
if ! git -c core.quotepath=false diff --name-only -z --no-renames "$BASE...$BRANCH" > "$tmp/paths.z" 2> "$tmp/err"; then
  echo "upstream_ready: git diff $BASE...$BRANCH failed (unrelated history, no merge base?): $(cat "$tmp/err")" >&2
  exit 2
fi
if ! "$PY" "$HERE/check_pr.py" classify --kind team -z < "$tmp/paths.z" > "$tmp/team.z" 2> "$tmp/err"; then
  echo "upstream_ready: could not classify the changed paths with '$PY': $(cat "$tmp/err")" >&2
  exit 2
fi

status=0
if [ -s "$tmp/team.z" ]; then
  echo "NOT READY: $BRANCH changes TEAM paths against $BASE:"
  tr '\0' '\n' < "$tmp/team.z" | sed 's/^/  /'
  status=1
else
  echo "ok: no TEAM path in the diff of $BRANCH against $BASE"
fi

tests_ran=0
if [ "$SKIP_TESTS" -eq 1 ]; then
  echo "CPU tests: skipped (--skip-tests)"
elif [ "$status" -eq 0 ]; then
  if ! git ls-tree -r --name-only "$BRANCH" -- tests > "$tmp/tree.txt" 2> "$tmp/err"; then
    echo "upstream_ready: git ls-tree failed: $(cat "$tmp/err")" >&2
    exit 2
  fi
  has_cpu=0
  git cat-file -e "$BRANCH:tests/cpu" 2>/dev/null && has_cpu=1
  grep -E '(^|/)selftest_[^/]*\.py$' "$tmp/tree.txt" > "$tmp/selftests.txt"
  rc=$?
  [ "$rc" -le 1 ] || { echo "upstream_ready: grep failed" >&2; exit 2; }
  if [ "$has_cpu" -eq 0 ] && [ ! -s "$tmp/selftests.txt" ]; then
    echo "CPU tests: NOT CHECKED: $BRANCH has no tests/cpu folder and no tests/**/selftest_*.py"
  elif git worktree add --detach "$tmp/wt" "$BRANCH" >/dev/null 2>&1; then
    if [ "$has_cpu" -eq 1 ]; then
      tests_ran=1
      if (cd "$tmp/wt" && "$PY" -m pytest -q tests/cpu); then
        echo "ok: pytest tests/cpu passes"
      else
        echo "NOT READY: pytest tests/cpu failed on $BRANCH"
        status=1
      fi
    fi
    while IFS= read -r f; do
      [ -n "$f" ] || continue
      tests_ran=1
      if (cd "$tmp/wt" && "$PY" "$f"); then
        echo "ok: $f passes"
      else
        echo "NOT READY: $f failed on $BRANCH"
        status=1
      fi
    done < "$tmp/selftests.txt"
  else
    echo "upstream_ready: could not create a temporary worktree" >&2
    exit 2
  fi
fi

if [ "$status" -ne 0 ]; then
  exit "$status"
fi
if [ "$SKIP_TESTS" -eq 1 ]; then
  echo "READY (TEAM-path check only, CPU tests skipped): $BRANCH"
  exit 0
fi
if [ "$tests_ran" -eq 0 ]; then
  echo "NOT READY: CPU tests NOT CHECKED on $BRANCH (no tests/cpu, no tests/**/selftest_*.py); the branch is not called READY"
  exit 3
fi
echo "READY: $BRANCH"
exit 0
