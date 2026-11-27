# AGENTS.md (this fork, public)

For team members and their coding agents; this fork follows sgl-project/sglang-neuron. Team process (tasks, claims, ledger,
the machine): the team hub (private, team members only) https://github.com/pengchenglu1997/trainium-rl-hub, its AGENTS.md first.
Without hub access, act as an outside reader: read, and propose by pull request from your own fork.

## 0 Setup

```
gh repo fork pengchenglu1997/sglang-neuron --clone=false    # once per account; skip it if you have write access here
gh repo clone pengchenglu1997/sglang-neuron -- --filter=blob:none && cd sglang-neuron   # origin = the team fork
git remote add upstream https://github.com/sgl-project/sglang-neuron   # skip it if gh added it
git remote add mine https://github.com/<you>/sglang-neuron.git; git config user.email <id>+<you>@users.noreply.github.com   # mine: push here only
git fetch --multiple origin upstream && bash rl/tools/fork_state.sh     # branches, pull requests, our upstream pull requests, mirror
```

State is read, never written here. A command and a document that disagree: on a fact the command wins; on a permission the stricter applies.

## 1 Route by intent

| Intent | Read | Run | Stop and ask |
|---|---|---|---|
| reproduce a number | `rl/REPRODUCE.md` | `python rl/make_patch_dir.py --ref <commit> --out <dir>`, then the commands there | each server launch needs a human yes in this session |
| best numbers, switches, how a result was measured | `rl/METHOD.md`; `rl/data/<session>/README.md` | `python rl/tools/best.py`; `python rl/tools/switches.py` | - |
| change engine code | section 3 (`<base>` is `integration` when it did not run on a device, else `main`) | `gh pr create -R pengchenglu1997/sglang-neuron --base <base> --head <you>:<branch>` | the label: the lead sets it |
| add session data | section 5 | `python rl/tools/check_data.py --base origin/main --head HEAD` | - |
| prepare an upstream pull request | section 4 | `bash rl/tools/upstream_ready.sh <branch>` | any upstream push, pull request, comment or body edit: the lead's yes |
| change `rl/METHOD.md`, `rl/best.tsv`, `rl/tools/`, `.github/`, this file | - | - | the lead decides |

## 2 Contributing without write access

Push to your own fork (`mine`) and open a pull request from it. You cannot label, assign, merge or re-run workflows: the
lead sets the label, approves and merges with a merge commit. Name the label you ask for in the pull request body; never
pass `--label` to `gh pr create`. Never push to the head of another pull request. After merge: `git push mine --delete <branch>`.

## 3 Branches and the label rule

- `main`: upstream, plus engine changes that ran on a device, plus the team paths. `integration`: `main` plus engine changes
  that did not run on a device, plus carried upstream pull requests. `upstream-main`: a copy of upstream, pushed by the lead.
- Team paths: `rl/**`, `docs/rl/**`, `.github/**`, `README.md`, `AGENTS.md`, `CLAUDE.md`, `.gitattributes`. Every other path is
  engine (`sglang_neuron/`, `tests/`, `docs/` outside `docs/rl/`, `scripts/`, `tools/`, `patches/`, `docker/`, `examples/`, `pyproject.toml`).
- Label rule, as check F1 implements it (`rl/tools/check_pr.py`): a pull request that changes team paths only needs no label
  and targets `main`. A pull request that changes any engine path carries exactly one label: `hardware-verified` (it ran
  on a device; base `main` or `integration`) or `cpu-only` (it did not; base `integration`, never `main`, never sent upstream).
  Never both. Into `integration`, a team path may change only as a sync from `main` (every team path equals `main`); a
  `hardware-verified` pull request there may also carry its own `rl/data/<session>/` folder and `docs/rl/` lines.
- Edit `rl/`, `docs/rl/`, `README.md` and this file on `main` only. `fork_state.sh` shows the engine paths `integration` lacks.
- Someone other than the author reviews before merge (not the author's own agent). A change that makes a README wrong fixes it
  in the same pull request, and opens the paired hub pull request if the hub goes stale.

## 4 Upstream

- A branch for upstream starts from `origin/upstream-main`, keeps its `feat/` or `fix/` name, and changes engine paths only.
  `bash rl/tools/upstream_ready.sh <branch>` must print READY (exit 3, "CPU tests: NOT CHECKED", is not READY). The same
  head is merged into `main` and sent upstream; do not cherry-pick it to another name.
- Evidence links in an upstream text: `https://github.com/pengchenglu1997/sglang-neuron/tree/<sha>/rl/data/...`. `gh pr create`
  defaults to the upstream repository: for a team pull request always pass `-R pengchenglu1997/sglang-neuron`.
- Every upstream action (a head pushed, a pull request opened or made ready, a comment, a body edit) needs the lead's yes
  typed in the session; a comment written with the lead's token records that yes, it is not the yes.

## 5 Data folders

One folder per session, `rl/data/<date>_<tag>/`: a README (one line per sub-folder: question, files, outcome; the commits
served; the tag glossary), `PATCH_MANIFEST.txt` of every patch folder served, `SHA256SUMS` of the published files. Sub-folders
are named by topic. Merged files are never edited. At most 2 MB per file, 10 MB per folder. A `hardware-verified` pull
request cites its folder; its `PATCH_MANIFEST.txt` must match every changed `sglang_neuron/` file (F2).

## 6 Public content

This repository is public. No person names, bylines or @handles (the fork owner's handle inside a URL is allowed); no chat
content, prices, instance, spot or account ids, availability zones, local paths, other people's unpublished numbers, hub
issue or row ids. CPU or simulator work: say the test exists and give its command; quote no quantity. Every public number
carries its tier and a path under `rl/data/`. Before every push (also to `mine`) run `python rl/tools/scrub_check.py
<changed paths> --words ../trainium-rl-hub/tools/public_words.txt` (CI runs it without the names list).

## 7 Stop

- A rejected push to `origin` (expected; never work around it); any write to `sgl-project/...` without the lead's yes; a
  server launch or the machine without a human yes typed in this session; a check you cannot satisfy (tell the lead, do not
  edit it in the same pull request). F1 red only for a missing label is the lead's step.
