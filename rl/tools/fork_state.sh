#!/usr/bin/env bash
# The state of the fork, read from git and gh. Nothing is written to any file: this command replaces every
# written state table (branches, open pull requests, our upstream pull requests and issues, what main has that
# integration lacks, the upstream check of each upstream head).
#
#   bash rl/tools/fork_state.sh [--remote NAME] [--upstream NAME] [--fetch] [--no-gh]
#
#   --remote NAME    the remote that is the team fork (default origin; its branches are main, integration, upstream-main, ...)
#   --upstream NAME  the remote of sgl-project/sglang-neuron (default upstream), used for the mirror line only
#   --fetch          git fetch the two remotes first (default: read what is already fetched)
#   --no-gh          skip every gh call (and say so)
# Sections: branches (date, merged into main or integration, delete hint), engine paths main has that integration
# lacks, carried upstream pull requests (merge subject "Merge upstream PR #N"), mirror freshness, open pull requests of
# the fork, our upstream pull requests and issues, and per upstream head the file-list check of upstream_ready.sh
# (--skip-tests: it says READY or NOT READY for TEAM paths only; run upstream_ready.sh itself for the CPU tests).
# Exit status: 0 read everything, 2 bad input or git failed, 3 a gh section could not be read (said in its section).
set -u
set -o pipefail

REMOTE=origin
UPSTREAM=upstream
FETCH=0
USE_GH=1
FORK_REPO="${FORK_REPO:-pengchenglu1997/sglang-neuron}"
UP_REPO="${UP_REPO:-sgl-project/sglang-neuron}"
OWNER="${FORK_OWNER:-pengchenglu1997}"
while [ $# -gt 0 ]; do
  case "$1" in
    --remote) REMOTE="${2:-}"; shift 2 || { echo "fork_state: --remote needs a value" >&2; exit 2; } ;;
    --upstream) UPSTREAM="${2:-}"; shift 2 || { echo "fork_state: --upstream needs a value" >&2; exit 2; } ;;
    --fetch) FETCH=1; shift ;;
    --no-gh) USE_GH=0; shift ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "fork_state: unknown option $1" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
TOP="$(git rev-parse --show-toplevel 2>/dev/null)" && [ -n "$TOP" ] || { echo "fork_state: not inside a git checkout" >&2; exit 2; }
cd "$TOP" || exit 2
rc=0

if [ "$FETCH" = 1 ]; then
  git fetch --quiet "$REMOTE" || { echo "fork_state: git fetch $REMOTE failed" >&2; exit 2; }
  git fetch --quiet "$UPSTREAM" 2>/dev/null || echo "fork_state: git fetch $UPSTREAM failed (the mirror line may be stale)"
fi

MAIN="refs/remotes/$REMOTE/main"
INTEG="refs/remotes/$REMOTE/integration"
UPM="refs/remotes/$REMOTE/upstream-main"
git rev-parse --verify --quiet "$MAIN^{commit}" >/dev/null || { echo "fork_state: no $REMOTE/main (git fetch $REMOTE)" >&2; exit 2; }

echo "== branches of $REMOTE (date, merged into main / integration)"
git for-each-ref "refs/remotes/$REMOTE" --format='%(refname:short) %(committerdate:short)' | while read -r ref date; do
  name="${ref#"$REMOTE"/}"
  [ "$name" = HEAD ] && continue
  [ "$ref" = "$REMOTE" ] && continue
  inm=no; ini=no
  git merge-base --is-ancestor "refs/remotes/$ref" "$MAIN" 2>/dev/null && inm=yes
  git rev-parse --verify --quiet "$INTEG^{commit}" >/dev/null && git merge-base --is-ancestor "refs/remotes/$ref" "$INTEG" 2>/dev/null && ini=yes
  hint=""
  case "$name" in
    main|integration|upstream-main) ;;
    *) if [ "$inm" = yes ] || [ "$ini" = yes ]; then hint="  merged: delete after checking it is not the head of an open upstream pull request"; fi ;;
  esac
  printf '%-45s %s  main=%s integration=%s%s\n' "$name" "$date" "$inm" "$ini" "$hint"
done

echo "== engine paths main has that integration lacks"
if git rev-parse --verify --quiet "$INTEG^{commit}" >/dev/null; then
  eng="$(git -c core.quotepath=false diff --name-only "$INTEG...$MAIN" -- . ':!rl' ':!docs/rl' ':!README.md' ':!AGENTS.md' ':!CLAUDE.md' ':!.github' ':!.gitattributes')" || { echo "fork_state: git diff failed" >&2; exit 2; }
  if [ -n "$eng" ]; then printf '%s\n' "$eng"; echo "(merge main into integration, label cpu-only)"; else echo "none"; fi
else
  echo "no $REMOTE/integration"
fi

echo "== carried upstream pull requests (merge subject 'Merge upstream PR #N' on integration, not on upstream-main)"
if git rev-parse --verify --quiet "$INTEG^{commit}" >/dev/null && git rev-parse --verify --quiet "$UPM^{commit}" >/dev/null; then
  carried="$(git log --merges --format=%s "$UPM..$INTEG" | grep -E '^Merge upstream PR #[0-9]+' || true)"
  if [ -n "$carried" ]; then printf '%s\n' "$carried"; else echo "none"; fi
else
  echo "not read: $REMOTE/integration or $REMOTE/upstream-main is missing"
fi

echo "== mirror (upstream-main of $REMOTE against $UPSTREAM/main)"
a="$(git rev-parse --verify --quiet "refs/remotes/$UPSTREAM/main^{commit}")"
b="$(git rev-parse --verify --quiet "$UPM^{commit}")"
if [ -z "$a" ] || [ -z "$b" ]; then
  echo "not read: $UPSTREAM/main or $REMOTE/upstream-main is missing"
elif [ "$a" = "$b" ]; then
  echo "equal ($b)"
else
  echo "DIFFERENT: $UPSTREAM/main $a, $REMOTE/upstream-main $b"
fi

gh_section() {  # title, then the gh arguments; a failure is said, not hidden
  title="$1"; shift
  echo "== $title"
  if [ "$USE_GH" = 0 ]; then echo "not read (--no-gh)"; rc=3; return; fi
  if ! command -v gh >/dev/null 2>&1; then echo "not read: gh is not installed"; rc=3; return; fi
  if ! out="$(gh "$@" 2>&1)"; then echo "not read: gh failed: $out"; rc=3; return; fi
  if [ -n "$out" ]; then printf '%s\n' "$out"; else echo "none"; fi
}

gh_section "open pull requests of $FORK_REPO (number, base, labels, title)" pr list -R "$FORK_REPO" --state open --limit 100 \
  --json number,title,baseRefName,labels --jq '.[] | "#\(.number) -> \(.baseRefName) [\([.labels[].name] | join(","))] \(.title)"'
gh_section "our pull requests in $UP_REPO (number, state, head)" pr list -R "$UP_REPO" --state all --limit 100 \
  --json number,title,state,isDraft,headRefName,headRepositoryOwner \
  --jq ".[] | select(.headRepositoryOwner.login == \"$OWNER\") | \"#\(.number) \(.state)\(if .isDraft then \" draft\" else \"\" end) \(.headRefName) \(.title)\""
gh_section "our issues in $UP_REPO (number, state)" issue list -R "$UP_REPO" --state all --limit 100 --author "$OWNER" \
  --json number,title,state --jq '.[] | "#\(.number) \(.state) \(.title)"'

echo "== upstream cleanliness of each open head of ours (file list against upstream-main; TEAM paths must not appear)"
if [ "$USE_GH" = 0 ] || ! command -v gh >/dev/null 2>&1; then
  echo "not read: the heads come from gh"; rc=3
elif ! heads="$(gh pr list -R "$UP_REPO" --state open --limit 100 --json headRefName,headRepositoryOwner \
      --jq ".[] | select(.headRepositoryOwner.login == \"$OWNER\") | .headRefName" 2>&1)"; then
  echo "not read: gh failed: $heads"; rc=3
elif [ -z "$heads" ]; then
  echo "none"
else
  printf '%s\n' "$heads" | while read -r h; do
    [ -n "$h" ] || continue
    if ! git rev-parse --verify --quiet "refs/remotes/$REMOTE/$h^{commit}" >/dev/null; then
      echo "$h: not read, no $REMOTE/$h (git fetch $REMOTE)"; continue
    fi
    res="$(bash "$HERE/upstream_ready.sh" "refs/remotes/$REMOTE/$h" --base "$UPM" --skip-tests 2>&1)"; st=$?
    echo "$h: exit $st: $(printf '%s\n' "$res" | head -1)"
    if [ "$st" -ne 0 ] && [ "$st" -ne 2 ]; then printf '%s\n' "$res" | sed -n '2,6p' | sed 's/^/    /'; fi
  done
fi
exit "$rc"
