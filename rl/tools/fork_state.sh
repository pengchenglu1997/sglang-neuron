#!/usr/bin/env bash
# The state of the fork, read from git and gh. Nothing is written to any file: this command replaces every
# written state table (branches, open pull requests, our upstream pull requests and issues, what main has that
# integration lacks, the upstream check of each upstream head).
#
#   bash rl/tools/fork_state.sh [--remote NAME] [--upstream NAME] [--no-fetch] [--no-gh]
#
#   --remote NAME    the remote that is the team fork (default origin; its branches are main, integration, upstream-main, ...)
#   --upstream NAME  the remote of sgl-project/sglang-neuron (default upstream), used for the mirror line only
#   --no-fetch       do not fetch; read the remote-tracking refs as they are (the first line says how old they are)
#   --no-gh          skip every gh call (and say so)
# By default the two remotes are fetched first, so the git sections are as fresh as the gh sections.
# Sections: branches (date, merged into main or integration, keep or delete hint), engine paths main has that integration
# lacks, carried upstream pull requests (merge subject "Merge upstream PR N", with or without #), mirror freshness, open pull requests of
# the fork, our upstream pull requests and issues, and per upstream head the file-list check of upstream_ready.sh
# (--skip-tests: it says READY or NOT READY for TEAM paths only; run upstream_ready.sh itself for the CPU tests).
# Exit status: 0 everything read and every upstream head clean; 2 bad input or git failed; 3 something could not be read
# (a gh section, a missing remote or ref, a head without its branch: said as "not read" in its section); 4 everything
# was read and an upstream head holds a TEAM path (3 wins over 4). A caller that trusts the status gets 0 only for a
# complete and clean state.
set -u
set -o pipefail

REMOTE=origin
UPSTREAM=upstream
FETCH=1
USE_GH=1
FORK_REPO="${FORK_REPO:-pengchenglu1997/sglang-neuron}"
UP_REPO="${UP_REPO:-sgl-project/sglang-neuron}"
OWNER="${FORK_OWNER:-pengchenglu1997}"
STALE="${FORK_STATE_STALE_SECONDS:-3600}"
while [ $# -gt 0 ]; do
  case "$1" in
    --remote) REMOTE="${2:-}"; shift 2 || { echo "fork_state: --remote needs a value" >&2; exit 2; } ;;
    --upstream) UPSTREAM="${2:-}"; shift 2 || { echo "fork_state: --upstream needs a value" >&2; exit 2; } ;;
    --no-fetch) FETCH=0; shift ;;
    --no-gh) USE_GH=0; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "fork_state: unknown option $1" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
TOP="$(git rev-parse --show-toplevel 2>/dev/null)" && [ -n "$TOP" ] || { echo "fork_state: not inside a git checkout" >&2; exit 2; }
cd "$TOP" || exit 2
incomplete=0   # something was not read
dirty=0        # an upstream head holds a TEAM path

echo "== fetch"
if [ "$FETCH" = 1 ]; then
  git fetch --quiet "$REMOTE" || { echo "fork_state: git fetch $REMOTE failed" >&2; exit 2; }
  if git fetch --quiet "$UPSTREAM" 2>/dev/null; then
    echo "fetched $REMOTE and $UPSTREAM just now"
  else
    echo "fetched $REMOTE just now; git fetch $UPSTREAM failed (the mirror line may be stale)"; incomplete=1
  fi
else
  fh="$(git rev-parse --git-path FETCH_HEAD 2>/dev/null)"
  mt=""
  [ -n "$fh" ] && [ -e "$fh" ] && mt="$(stat -c %Y "$fh" 2>/dev/null || stat -f %m "$fh" 2>/dev/null || true)"
  if [ -z "$mt" ]; then
    echo "not fetched (--no-fetch); no FETCH_HEAD, so the age of the remote-tracking refs is unknown: they may be stale"
  else
    age=$(( $(date +%s) - mt ))
    if [ "$age" -gt "$STALE" ]; then
      echo "not fetched (--no-fetch); the last fetch was $((age / 60)) minutes ago: the git sections may be stale"
    else
      echo "not fetched (--no-fetch); the last fetch was $((age / 60)) minutes ago"
    fi
  fi
fi

MAIN="refs/remotes/$REMOTE/main"
INTEG="refs/remotes/$REMOTE/integration"
UPM="refs/remotes/$REMOTE/upstream-main"
git rev-parse --verify --quiet "$MAIN^{commit}" >/dev/null || { echo "fork_state: no $REMOTE/main (git fetch $REMOTE)" >&2; exit 2; }
have_integ=0; git rev-parse --verify --quiet "$INTEG^{commit}" >/dev/null && have_integ=1
have_upm=0; git rev-parse --verify --quiet "$UPM^{commit}" >/dev/null && have_upm=1

# Open upstream pull requests of ours, read first: "<number> <head branch>" per line.
heads=""; heads_ok=0; heads_why=""
if [ "$USE_GH" = 0 ]; then heads_why="--no-gh"
elif ! command -v gh >/dev/null 2>&1; then heads_why="gh is not installed"
elif ! heads="$(gh pr list -R "$UP_REPO" --state open --limit 100 --json number,headRefName,headRepositoryOwner \
      --jq ".[] | select(.headRepositoryOwner.login == \"$OWNER\") | \"\(.number) \(.headRefName)\"" 2>&1)"; then
  heads_why="gh failed: $heads"; heads=""
else heads_ok=1; fi

echo "== branches of $REMOTE (date, merged into main / integration, keep or delete)"
git for-each-ref "refs/remotes/$REMOTE" --format='%(refname:short) %(committerdate:short)' | while read -r ref date; do
  name="${ref#"$REMOTE"/}"
  [ "$name" = HEAD ] && continue
  [ "$ref" = "$REMOTE" ] && continue
  inm=no; ini=no
  git merge-base --is-ancestor "refs/remotes/$ref" "$MAIN" 2>/dev/null && inm=yes
  [ "$have_integ" = 1 ] && git merge-base --is-ancestor "refs/remotes/$ref" "$INTEG" 2>/dev/null && ini=yes
  hint=""
  case "$name" in
    main|integration|upstream-main) ;;
    *) if [ "$inm" = yes ] || [ "$ini" = yes ]; then
         num="$(printf '%s\n' "$heads" | awk -v h="$name" '$2 == h { print $1; exit }')"
         if [ -n "$num" ]; then
           hint="  head of upstream #$num: keep"
         elif [ "$heads_ok" = 0 ]; then
           hint="  merged: delete only after checking it is not the head of an open upstream pull request (not read: $heads_why)"
         elif [ "$have_upm" = 1 ] && git merge-base --is-ancestor "$UPM" "refs/remotes/$ref" 2>/dev/null \
              && bash "$HERE/upstream_ready.sh" "refs/remotes/$ref" --base "$UPM" --skip-tests >/dev/null 2>&1; then
           hint="  upstream-bound (rooted on upstream-main, engine paths only): keep"
         else
           hint="  merged, not the head of an open upstream pull request: delete"
         fi
       fi ;;
  esac
  printf '%-45s %s  main=%s integration=%s%s\n' "$name" "$date" "$inm" "$ini" "$hint"
done

echo "== engine paths main has that integration lacks"
if [ "$have_integ" = 1 ]; then
  eng="$(git -c core.quotepath=false diff --name-only "$INTEG...$MAIN" -- . ':!rl' ':!docs/rl' ':!README.md' ':!AGENTS.md' ':!CLAUDE.md' ':!.github' ':!.gitattributes')" || { echo "fork_state: git diff failed" >&2; exit 2; }
  if [ -n "$eng" ]; then printf '%s\n' "$eng"; echo "(merge main into integration, label cpu-only)"; else echo "none"; fi
else
  echo "not read: no $REMOTE/integration"; incomplete=1
fi

echo "== carried upstream pull requests (merge subject 'Merge upstream PR N' on integration, not on upstream-main)"
if [ "$have_integ" = 1 ] && [ "$have_upm" = 1 ]; then
  carried="$(git log --merges --format=%s "$UPM..$INTEG" | grep -E '^Merge upstream PR #?[0-9]+' || true)"
  if [ -n "$carried" ]; then printf '%s\n' "$carried"; else echo "none"; fi
else
  echo "not read: $REMOTE/integration or $REMOTE/upstream-main is missing"; incomplete=1
fi

echo "== mirror (upstream-main of $REMOTE against $UPSTREAM/main)"
a="$(git rev-parse --verify --quiet "refs/remotes/$UPSTREAM/main^{commit}")"
b="$(git rev-parse --verify --quiet "$UPM^{commit}")"
if [ -z "$a" ] || [ -z "$b" ]; then
  echo "not read: $UPSTREAM/main or $REMOTE/upstream-main is missing"; incomplete=1
elif [ "$a" = "$b" ]; then
  echo "equal ($b)"
else
  echo "DIFFERENT: $UPSTREAM/main $a, $REMOTE/upstream-main $b"
fi

gh_section() {  # title, then the gh arguments; a failure is said, not hidden
  title="$1"; shift
  echo "== $title"
  if [ "$USE_GH" = 0 ]; then echo "not read (--no-gh)"; incomplete=1; return; fi
  if ! command -v gh >/dev/null 2>&1; then echo "not read: gh is not installed"; incomplete=1; return; fi
  if ! out="$(gh "$@" 2>&1)"; then echo "not read: gh failed: $out"; incomplete=1; return; fi
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
if [ "$heads_ok" = 0 ]; then
  echo "not read: the heads come from gh ($heads_why)"; incomplete=1
elif [ "$have_upm" = 0 ]; then
  echo "not read: no $REMOTE/upstream-main"; incomplete=1
elif [ -z "$heads" ]; then
  echo "none"
else
  # No pipe: the loop must run in this shell, or incomplete and dirty would be lost with the subshell.
  while read -r num h; do
    [ -n "$h" ] || continue
    if ! git rev-parse --verify --quiet "refs/remotes/$REMOTE/$h^{commit}" >/dev/null; then
      echo "$h (upstream #$num): not read, no $REMOTE/$h (git fetch $REMOTE)"; incomplete=1; continue
    fi
    res="$(bash "$HERE/upstream_ready.sh" "refs/remotes/$REMOTE/$h" --base "$UPM" --skip-tests 2>&1)"; st=$?
    echo "$h (upstream #$num): exit $st: $(printf '%s\n' "$res" | head -1)"
    case "$st" in
      0) ;;
      1) dirty=1; printf '%s\n' "$res" | sed -n '2,6p' | sed 's/^/    /' ;;
      *) echo "    not read: upstream_ready.sh could not decide"; incomplete=1 ;;
    esac
  done <<EOF
$heads
EOF
fi
if [ "$incomplete" = 1 ]; then exit 3; fi
if [ "$dirty" = 1 ]; then exit 4; fi
exit 0
