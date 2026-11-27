#!/usr/bin/env bash
# PreToolUse hook of Claude Code for this repository (Bash and PowerShell tools), named in .claude/settings.json.
# It reads the tool call as JSON on stdin and answers "ask" (the human confirms in the permission prompt) for:
#   - any write to an sgl-project repository through gh: pr/issue create, comment, edit, ready, review, merge, close ...;
#     release, label or repo writes; gh api with a write method or with fields; and any git push naming sgl-project;
#   - any git push and any gh pr create (also with options before the subcommand: git -C dir push, /usr/bin/gh ...);
#   - cloud launches: aws ... run-instances, start-instances, request-spot-instances, request-spot-fleet, create-fleet.
# Every other call gets no decision (exit 0, no output), so the normal permission rules apply. The permission rules
# "ask" of .claude/settings.json cover the plain forms of the same commands; this hook adds the forms a rule cannot match.
# A prompt is not the lead's yes (AGENTS.md section 0): it only makes sure a human sees the call before it runs.
# Bash only, no jq: it matches the raw JSON text, so it runs the same on Linux and in Git Bash on Windows.
set -u
input="$(cat)"
flat="$(printf '%s' "$input" | tr '\n\r' '  ')"
has() { printf '%s' "$flat" | grep -Eiq -- "$1"; }
ask() {
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"%s"}}\n' "$1"
  exit 0
}
GH='(^|[^a-z0-9_.-])gh(\.exe)?( +-[^ ]+( +[^- ][^ ]*)?)* +'
GIT='(^|[^a-z0-9_.-])git(\.exe)?( +-[cC] +[^ ]+| +--?[a-z][a-z-]*(=[^ ]+)?)* +'
if has 'sgl-project'; then
  has "${GH}(pr|issue) +(create|comment|edit|ready|review|merge|close|reopen|lock|unlock|delete|transfer|pin|unpin|develop)" \
    && ask "This writes to sgl-project (upstream). It needs the lead's yes typed in this session (AGENTS.md section 4)."
  has "${GH}(release|label|repo) +(create|edit|delete|upload|rename|archive|unarchive|sync|fork|clone-url)" \
    && ask "This writes to sgl-project (upstream). It needs the lead's yes typed in this session (AGENTS.md section 4)."
  has "${GH}api .*(-X|--method)[ =]*(POST|PUT|PATCH|DELETE)" \
    && ask "gh api with a write method on sgl-project (upstream). It needs the lead's yes typed in this session."
  has "${GH}api .*( -f | -F | --field | --raw-field | --input )" \
    && ask "gh api with fields defaults to POST: a write on sgl-project (upstream). It needs the lead's yes typed in this session."
  has "${GIT}push" \
    && ask "A push to sgl-project (upstream). It needs the lead's yes typed in this session (AGENTS.md section 4)."
fi
has "${GIT}push( |\"|$)" \
  && ask "A push publishes. When you act as the lead (AGENTS.md section 0) it needs the lead's yes typed in this session."
has "${GH}pr +create" \
  && ask "A pull request. Without -R pengchenglu1997/sglang-neuron gh opens it upstream; as the lead it needs the lead's yes."
has '(^|[^a-z0-9_.-])aws(\.exe)? .*(run-instances|start-instances|request-spot-instances|request-spot-fleet|create-fleet)' \
  && ask "A cloud launch. Every launch needs a human yes typed in this session (AGENTS.md section 1)."
exit 0
