# AGENTS.md (this fork)

For team members and their coding agents. This fork follows sgl-project/sglang-neuron.

1. Before any work here, follow the team protocol in the team hub (private, team members only):
   https://github.com/pengchenglu1997/trainium-rl-hub (read its AGENTS.md; search its issues, `ledger/` and `lessons/` first; claim an issue).
2. This repository is public. Push only our own measured results and code. No names, no chat content, no prices, no instance or account ids, no local paths, no numbers measured by others.
3. Pull requests: `gh pr create` in a fork targets the upstream repository by default. For a team pull request always pass `--repo pengchenglu1997/sglang-neuron`. A pull request to upstream needs the team lead's yes first.
4. A branch meant for upstream starts from upstream `main` (copied here as branch `upstream-main`), not from this fork's `main`, so that this file, the README and `docs/rl/` do not travel with it. Branch layout: README.md.
5. Results are recorded in the team hub's ledger; `docs/rl/RESULTS.md` here is the long public write-up.
6. Every pull request to this fork carries exactly one of the labels `cpu-only` or `hardware-verified`. A `cpu-only` pull request targets the `integration` branch, never `main`, and is never sent upstream. No number from cpu-only work is quoted here, upstream or anywhere outside the team hub. A change to this file, the README or `docs/rl/` only may target `main`.
7. Before merge, someone other than the author (or that person's agent) reviews. A review by the author's own agent does not count.
8. README files are kept current. A pull request that makes any README here wrong (the front page's branch table, its list of what `main` adds, its results table, or a sub-folder README such as `tests/rlkit_sampler/README.md`) fixes it in the same pull request. If the change also makes the team hub's README or AGENTS.md wrong, open the paired pull request there.
