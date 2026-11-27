# sglang-neuron (RL post-training fork)

A working fork of [sgl-project/sglang-neuron](https://github.com/sgl-project/sglang-neuron), the SGLang plugin for AWS
Trainium. We use it as the rollout server for reinforcement-learning post-training on Trainium2, and we send what
holds up on hardware back upstream.

What the plugin itself is, how to install it and which models it serves: the
[upstream README](https://github.com/pengchenglu1997/sglang-neuron/blob/upstream-main/README.md) and [docs/](docs/index.md).
This page covers only what is different in the fork.

## Start here

| You are | Read |
|---|---|
| A team member or a team member's coding agent | [AGENTS.md](AGENTS.md), then the team hub (private, team members only): https://github.com/pengchenglu1997/trainium-rl-hub. Tasks, the results ledger, lessons and the working rules are there, not here. |
| Anyone else | [docs/rl/RESULTS.md](docs/rl/RESULTS.md) for what we measured, [docs/rl/KERNELS.md](docs/rl/KERNELS.md) for the kernels, [docs/rl/UPSTREAM_PRS.md](docs/rl/UPSTREAM_PRS.md) for the upstream pull requests we carry and why. |

## Branches

| Branch | What it holds | Who writes to it |
|---|---|---|
| `main` | Upstream v0.1.0, plus our changes that have run on Trainium2 hardware (list below), plus `docs/rl/`. The default branch; start new work from here. | Pull request with the label `hardware-verified` |
| `integration` | `main`, plus changes that have not run on hardware yet, plus upstream pull requests 20, 4, 9, 16, 17, 18 that upstream has not merged. | Pull request with the label `cpu-only` |
| `upstream-main` | The upstream `main`, unchanged (commit `4e24326`, v0.1.0). Moved by hand when upstream moves. | Nobody; it is a copy |
| `feat/*`, `fix/*` | One open piece of work each. Deleted after merge, unless an upstream pull request still uses the branch. | The person who claimed the hub issue |

A branch meant for an upstream pull request starts from `upstream-main`, so that this page, `AGENTS.md` and `docs/rl/`
do not travel with it.

Open work branches today:

| Branch | What | State |
|---|---|---|
| `fix/compile-cache-stamp-race` | the start-up fix below, on upstream `main` | merged into `main`; kept because upstream pull request [#22](https://github.com/sgl-project/sglang-neuron/pull/22) uses it |
| `feat/sampler-kernel` | the sampler kernel below, on upstream `main` | merged into `main`; kept because upstream draft pull request [#25](https://github.com/sgl-project/sglang-neuron/pull/25) uses it |
| `feat/inplace-adapter-sync` | in-place weight update and LoRA adapter sync | ran on trn2 (Qwen3-0.6B and Qwen3-8B). Not in `main` or `integration`: it and upstream pull request 20 change the same function |
| `feat/lora-merge-kernel` | LoRA shard-merge NKI kernel, on top of `feat/inplace-adapter-sync`, off by default | first hardware run at Qwen3-0.6B on 2026-11-24; acceptance rule open |
| `feat/lm-head-pretransposed` | opt-in pre-transposed LM head weight, on upstream v0.1.0 | CPU only; its draft pull request conflicts with `integration` and is superseded by the branch below |
| `feat/lm-head-pretransposed-integration` | the same change on `integration`, where the in-place weight reload also writes the LM head's own copy of a tied model | CPU only; pull request to `integration` |
| `feat/sampler-kernel-topk-topp` | the sampler kernel in the `topk_topp` mode, switch `RLKIT_SAMPLER_KERNEL_TOPK=1`, off by default ([tests/rlkit_sampler/README.md](tests/rlkit_sampler/README.md)) | CPU and NKI simulator only; pull request to `integration` |

To read or run an upstream pull request that is not in `integration`, fetch its head from upstream:

```
git fetch https://github.com/sgl-project/sglang-neuron pull/<N>/head:upstream-pr/<N>
```

## What `main` adds to upstream

| Change | Switch | Run on hardware |
|---|---|---|
| Compile cache: the stamp's temporary file gets a name of its own, so two servers that start together on one empty cache do not stop at start-up (`sglang_neuron/backend/compile_cache.py`) | none, always on | trn2, 2026-11-24, 10 trials per arm: released code 1 of 10 failed, with the fix 0 of 10 |
| Sampler NKI kernel for the `temperature` sampler (`sglang_neuron/rlkit_sampler_kernel.py`, one guarded branch in `sglang_neuron/sampling.py`, tests in `tests/rlkit_sampler/`) | `RLKIT_SAMPLER_KERNEL=1`; unset, the released code path runs | trn2, 2026-11-24, Qwen3-0.6B and Qwen3-8B (numbers below). Not run inside a full RL loop |

Each of the two was run on its own branch. `main` with both together has not been started on hardware.

## Measured

Trn2.48xlarge (Trainium2), Neuron SDK 2.33, v0.1.0 image, one chip as TP4 unless the row says otherwise. One run each
unless the row says otherwise. Every row, with its conditions and source file, is in
[docs/rl/RESULTS.md](docs/rl/RESULTS.md).

| What | Result |
|---|---|
| Baseline pair, Qwen3-0.6B, prompt 1024, 256 output tokens, concurrency 32 | vllm-neuron 0.29.0.1.2.0: 2,375.3 tok/s; SGLang-Neuron v0.1.0 released defaults: 2,460.6 tok/s |
| Sampler kernel, Qwen3-0.6B, same prompt and concurrency, two passes | 2,730 and 2,734 tok/s, against 2,455 and 2,454 for the released `temperature` sampler |
| Sampler kernel, Qwen3-8B, two passes | 1,489 and 1,496 tok/s, against 1,421 and 1,426 (about 5 percent) |
| Sampler kernel with `--dp-size 4` of TP1 in one server, Qwen3-0.6B | 3,026 tok/s; the released sampler with `--dp-size 4`: 2,701 (one measurement) |
| New weights on Qwen3-0.6B: restart, against an in-place update (`feat/inplace-adapter-sync`) | 455-460 s, against 0.34-0.41 s |
| Upstream pull request 20 on its own branch: in-place update against a fresh start on the same checkpoint | same text, largest token logprob difference 0.0, 20 prompts, Qwen3-0.6B and Qwen3-8B |

## Sent upstream

- Issue [#15](https://github.com/sgl-project/sglang-neuron/issues/15): in-place weight updates for RL rollout servers,
  with measurements. Upstream pull request 20 now implements the full-checkpoint part.
- Pull request [#22](https://github.com/sgl-project/sglang-neuron/pull/22), ready for review since 2026-11-25: the compile-cache start-up fix, with
  the trn2 measurement of 2026-11-24 in its text.
- A comment on upstream pull request [#20](https://github.com/sgl-project/sglang-neuron/pull/20#issuecomment-7182387872):
  our trn2 run of its head on Qwen3-0.6B and Qwen3-8B.
- Issue [#24](https://github.com/sgl-project/sglang-neuron/issues/24): the released `temperature` and `greedy` samplers
  give different tokens at temperature 0 on 3 of 20 prompts (Qwen3-0.6B, TP4).
- Draft pull request [#25](https://github.com/sgl-project/sglang-neuron/pull/25): the sampler kernel, with its limits
  and a question to the maintainers on whether and in what shape they want it.

## Pull requests to this fork

- `gh pr create` in a fork targets upstream by default. Always pass `--repo pengchenglu1997/sglang-neuron`.
- Exactly one label: `hardware-verified` (target `main`) or `cpu-only` (target `integration`). A check fails without it.
  A change to this page, `AGENTS.md` or `docs/rl/` only may target `main`.
- Someone other than the author reviews before merge.
- The full rules are in [AGENTS.md](AGENTS.md).

## Where things live

| What | Where |
|---|---|
| Tasks, results ledger, lessons, plans, status | team hub (private): https://github.com/pengchenglu1997/trainium-rl-hub |
| Engine code, kernels, the public write-up of results | this repository |
| Run scripts and raw result files | https://github.com/pengchenglu1997/trainium-sglang-rl |
| The plugin's own documentation | [docs/](docs/index.md), unchanged from upstream |
