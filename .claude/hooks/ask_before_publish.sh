#!/usr/bin/env bash
# PreToolUse hook of Claude Code for this repository (Bash and PowerShell tools), named in .claude/settings.json.
# It reads the tool call as JSON on stdin and answers "ask" (the human confirms in the permission prompt) for:
#   - every gh write (pr/issue create, new, comment, edit, ready, review, merge, close ...; release, label, repo,
#     workflow, run, secret, variable writes) unless -R/--repo names pengchenglu1997/ explicitly: without -R gh
#     writes to its default repository, which is sgl-project in a clone of the fork; and any gh write naming sgl-project;
#   - every gh api call with a write method or with fields (-f, -F, --field, --raw-field, --input) unless the endpoint is
#     repos/pengchenglu1997/...;
#   - any git push (also git -C dir push, git -c k=v push, git --git-dir d push) and any gh pr create;
#   - cloud launches: aws ... run-instances, start-instances, request-spot-instances, request-spot-fleet, create-fleet.
# Every other call gets no decision (exit 0, no output), so the normal permission rules apply.
# It matches text, so it covers the usual forms only (not git aliases, variables holding a command, or code that calls
# an API): a prompt is not a boundary and not the lead's yes (AGENTS.md section 0).
# Bash only, no jq: it matches the raw JSON text, so it runs the same on Linux and in Git Bash on Windows.
set -u
input="$(cat)"
# JSON escapes of newline, return and tab become spaces, an escaped quote becomes a quote, real newlines become spaces.
flat="$(printf '%s' "$input" | tr '\n\r\t' '   ' | sed -e 's/\\[nrt]/ /g' -e 's/\\"/"/g')"
ask() {
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"%s"}}\n' "$1"
  exit 0
}
S='(^|[^a-z0-9_.-])'
V="(\"[^\"]*\"|'[^']*'|[^ ]+)"
GH="${S}gh(\\.exe)?( +-[^ ]+( +[^- ][^ ]*)?)* +"
GIT="${S}git(\\.exe)?( +-[cC] *${V}| +--(git-dir|work-tree|namespace|exec-path|config-env|super-prefix)( +|=)${V}| +--?[a-z][a-z-]*(=${V})?)* +"
GHW="${GH}((pr|issue) +(create|new|comment|edit|ready|review|merge|close|reopen|lock|unlock|delete|transfer|pin|unpin|develop|update-branch|revert)"
GHW="${GHW}|release +(create|new|edit|delete|delete-asset|upload)|label +(create|edit|delete|clone)"
GHW="${GHW}|repo +(create|new|edit|delete|rename|archive|unarchive|sync|fork|deploy-key)|workflow +(run|enable|disable)"
GHW="${GHW}|run +(rerun|cancel|delete)|(secret|variable) +(set|delete)|cache +delete)([^a-z0-9_-]|$)"
OWN="(-R|--repo)( +|=)?[\"']?pengchenglu1997/"
API="${GH}api([^a-z0-9_-]|$)"
WRITE_METHOD="(-X|--method)( +|=)?[\"']?(POST|PUT|PATCH|DELETE)"
FIELDS=" (-[fF]|--field|--raw-field|--input)"
GET_METHOD="(-X|--method)( +|=)?[\"']?GET"
OWN_API="repos/pengchenglu1997/"

upstream_check() {  # $1: a text (the whole call or one segment of it)
  local t="$1"
  m() { printf '%s' "$t" | grep -Eiq -- "$1"; }
  if m "$GHW"; then
    m 'sgl-project' && ask "This writes to sgl-project (upstream). It needs the lead's yes typed in this session (AGENTS.md section 4)."
    m "$OWN" || ask "A gh write without -R pengchenglu1997/...: gh writes to its default repository, which is upstream in a clone of the fork. It needs the lead's yes."
  fi
  if m "$API"; then
    if m "$WRITE_METHOD" || { m "$FIELDS" && ! m "$GET_METHOD"; }; then
      m 'sgl-project' && ask "gh api with a write method or fields on sgl-project (upstream). It needs the lead's yes typed in this session."
      m "$OWN_API" || ask "gh api with a write method or fields outside repos/pengchenglu1997/ (a {owner} placeholder is upstream). It needs the lead's yes."
    fi
  fi
  return 0
}

has() { printf '%s' "$flat" | grep -Eiq -- "$1"; }
upstream_check "$flat"
# The same check on each segment between ; & | so that one -R pengchenglu1997/ does not cover a second gh call.
while IFS= read -r seg; do
  [ -n "$seg" ] && upstream_check "$seg"
done <<EOF
$(printf '%s' "$flat" | tr ';&|' '\n\n\n')
EOF
has "${GIT}push([^a-z0-9_-]|$)" \
  && ask "A push publishes. When you act as the lead (AGENTS.md section 0) it needs the lead's yes typed in this session."
has "${GH}pr +(create|new)([^a-z0-9_-]|$)" \
  && ask "A pull request. Without -R pengchenglu1997/sglang-neuron gh opens it upstream; as the lead it needs the lead's yes."
has "${S}aws(\\.exe)? .*(run-instances|start-instances|request-spot-instances|request-spot-fleet|create-fleet)" \
  && ask "A cloud launch. Every launch needs a human yes typed in this session (AGENTS.md section 1)."
exit 0
