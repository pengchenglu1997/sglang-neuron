# Upstream pull requests: what we carry and why

Upstream is sgl-project/sglang-neuron. Its `main` is one squashed commit (4e24326, tag v0.1.0). When this was written, 20 pull requests were open (numbers 1 to 14 and 16 to 21) and none was merged. Number 15 is our own issue, not a pull request. The count of 21 elsewhere in this fork includes that issue. Row 15 is in the table so that every number from 1 to 21 has a line.

## Rules for reading this file

- Every number in the "Author reports" column is the author's own, copied from the PR text. **We have not run any of them.** The column "Run by us" is "no" for every row until a hardware task in [ROADMAP.md](https://github.com/pengchenglu1997/trainium-sglang-rl/blob/main/docs/plan/ROADMAP.md) writes a result next to it.
- The authors ran on different hardware. Many ran on trn3. We use trn2, so a trn3 result does not tell us the result on trn2. The column says which one the author used.
- Decisions:
  - **integrated**: merged into our branch `integration`, in the order given below.
  - **tracked branch**: kept as its own branch `upstream-pr/<N>`, the unchanged upstream PR head, for reference. Not in `integration`. Branches that stack on another PR say so.
  - **not carried**: no branch (PRs 3, 14 and 19). Ideas worth keeping are listed at the end of this file.
- Reading depth: the decisions for 20, 4, 9, 16, 17 and 18 come from reading the full diff. The others come from the PR text and the file list. Where we only read the text, a later diff read may change the decision. Every number in the "Author reports" column was checked against the PR text on 2026-11-23.

## Table of all 21 numbers

| PR | Author | What it does | Our decision | Reason in one line | Author reports (validation) | Run by us |
|---|---|---|---|---|---|---|
| 20 | @Zhichenzzz | Reload Qwen3 and Qwen3-MoE weights from disk in place (`update_weights_from_disk`) | integrated (first) | It is the upstream route for our issue 15 and the base for full-checkpoint sync. | trn3.48xlarge. Qwen3-4B TP4: update returns in about 1.9 s, greedy output identical before and after reloading the same weights. Miles GRPO with disk-delta sync on Qwen3-4B and Qwen3-30B-A3B: trainer-vs-rollout logprob abs diff 0.007 to 0.015, 30+ syncs on the 30B run. | no |
| 4 | @whn09 | Paged decode: stop passing `max_context_len` to nkilib attention | integrated | It fixes wrong output when the decode prior is above nkilib's 8K flash-attention tile, and the dense Qwen3 decode path uses this file. | trn2.48xlarge, Qwen3-30B-A3B TP8 EP8. A needle test at 8,132 / 8,204 / 9,644 / 24,044 / 76,844 / 240,044 tokens with and without the argument. | no |
| 9 | @ttpks1010 | Add Llama 3.1 8B on the shared dense transformer | integrated | New package only, changes no shared runtime file, and gives a second dense target for weight sync (G1) and a common benchmark model (G2). | trn2.3xlarge, TP4, SDK 2.33. Greedy tokens identical to Hugging Face in 151 of 151 at a 4096 context. GSM8K first 100 questions, five-shot: 0.78. | no |
| 16 | @zcnrex | Cache a failing torch_mlir import and traceback formatting at trace time | integrated | It changes only trace-time work, is easy to revert, and shortens every launch. | trn3.48xlarge, Qwen3-30B-A3B. Warm launch ready 496 s down to 311 s (author states -37%), measured with PR 12's trace-ahead warm-up included. Author states zero neuronx-cc compiles on an existing cache and identical tokens. | no |
| 17 | @jimburtoft | Qwen3-MoE prefill experts in the hybrid `moe_cte` form on trn2 | integrated | Gated by an environment switch, touches only MoE prefill, and it is our 30B-rung hardware class (G1, G2). | trn2.3xlarge, Qwen3-30B-A3B TP4 EP4. Output tok/s at 2048 in / 128 out, concurrency 64: 220.2 to 371.0 (author states +68%). Decode unchanged, c=1 TPOT 45.8 ms on both. | no |
| 18 | @jimburtoft | Qwen3-MoE decode experts run selectively for batches of up to 4 tokens | integrated | Default-safe (a switch restores main), MoE only, cheap to carry. It helps single-request latency, not a throughput-bound rollout. | trn2.3xlarge, Qwen3-30B-A3B TP4 EP4. 128 in / 512 out, c=1: TPOT 45.8 ms to 17.1 ms (author states 2.6x). No change above 4 rows. | no |
| 21 | @Zhichenzzz | `POST /pull_weights`: pull published weight deltas into a host-local checkpoint | tracked branch (on top of PR 20) | Matters only for full-parameter RL at 27 to 32B. It adds a 352-line copied module and new dependencies we have not confirmed in the image, and it needs a trainer format we do not produce. | trn3.48xlarge. Miles GRPO with disk-delta transfer on Qwen3-4B and Qwen3-30B-A3B. Every sync pulls and reloads, checksums verify on each pull, 30+ syncs on the 30B run. | no |
| 11 | @whn09 | Warm-up: queue every graph compile first, trace each shape once | tracked branch | It is measured on our configuration but is on by default, rewrites warm-up for every model, and has a known SPMD ordering risk. Excludes 12. | "Measured on main"; PR 11 does not name the instance (PR 10, same author, gives the same 540 s cache-off row on a trn2.48xlarge). Qwen3-8B TP4, empty NEFF cache. Ready 1467 s to 540 s. GSM8K 0.940 and TPOT 12.71 ms the same both ways. | no |
| 10 | @whn09 | Keep dynamo traces across launches (`SGLANG_NEURON_PRECOMPILE_CACHE=1`) | tracked branch (stacked on 11) | Largest restart win on our hardware, but 385 lines on torch-neuronx caching internals, contains PR 11's commits and conflicts with 12. | trn2.48xlarge, Qwen3-8B TP4, SDK 2.33. Ready: cache off 540 s, first launch 530 s, restored 92 s. GSM8K 0.940 and 26 NEFFs in all three. | no |
| 12 | @zcnrex | Compile warm-up shapes concurrently (trace every shape before warming any) | tracked branch | Cleaner design than 11, but evidence is trn3 and MoE only and there is no opt-out switch. Fallback if 11 shows SPMD mismatches. | trn3.48xlarge, Qwen3-30B-A3B TP4 EP4, cold cache. 22 shapes, launch to ready about 35 min down to 18 min. 4-layer override, 12 shapes: 307 s to 197 s. | no |
| 6 | @matheus-1618 | Qwen3.8 27B text model on a trn2.3xlarge | tracked branch | It is the dense 27B-class rung, but 27 files, shared-code edits, and no `reload_weights`, so PR 20 refuses it and our patch would fail. | trn2.3xlarge, TP8. GSM8K 0.990. TPOT 27.7 to 28.2 ms by configuration. HF teacher-forced 120 of 120. | no |
| 2 | @matheus-1618 | Gemma 4 31B text model on a trn2.3xlarge | tracked branch | Fallback 31B rung. Has `load_weights`, no `reload_weights`, slow, long cold compile (author-stated 74 min), thin test range. | trn2.3xlarge, TP4. GSM8K first 100: 0.820. Single-client TPOT 35.2 ms with the device sampler. | no |
| 13 | @zcnrex | Qwen3-MoE FP8 (e4m3) KV cache | tracked branch | Costs accuracy and the evidence is trn3 only. RL parity work must use a bf16 KV cache first. | trn3.48xlarge, Qwen3-30B-A3B, 4K in / 4K out. Output tok/s 936 (bf16) to 1,271 (fp8), +36%. GSM8K (all 1,319, 5-shot, greedy): 0.907 with bf16 KV, 0.810 and 0.863 with fp8 KV (two runs), a cost of about 0.04 to 0.10. The author reports a similar drop on SGLang CUDA (H200). | no |
| 5 | @whn09 | Refuse the `moe_cte` expert shard of 1536 on trn2 (Qwen3-235B layouts) | tracked branch | A correct guard, but it does nothing for Qwen3-30B-A3B (shard 768). Conflicts with 17. Merge after 17 only if someone runs a 235B-class model. | trn2.48xlarge. Shard 1536 (TP32 EP32): garbage from 800 tokens, GSM8K 0.000, server stays broken. Shard 768 layouts right, GSM8K 0.955 and 0.965. | no |
| 7 | @whn09 | Hy3-FP8 model: FP8 experts and KV, DCP, sparse decode, MTP, PD (draft) | tracked branch (reference only) | 6,907 added lines with shared-file edits and no CPU tests. Kept for its FP8 MoE kernels and recipe notes. Not for merging. | trn2.48xlarge, SDK 2.33. Decode TPOT 21.6 ms at c=1. GSM8K 0.933 with MTP. | no |
| 8 | @whn09 | MiMo-V2.6-Flash-RL model with fused decode kernels (draft) | tracked branch (reference only) | 18k lines. Kept for its kernel design doc and profiling tools. Conflicts with 19. Check where code came from before copying anything. | trn2.48xlarge, SDK 2.33. One node (16 chips): 17.71 output tok/s per chip; 1P1D on two nodes: 9.39 per chip. | no |
| 1 | @whn09 | Mooncake (EFA) as a PD KV transfer backend | tracked branch | We do not run prefill-decode split rollout today, and the author measured parity with NIXL, so there is no gain now. | trn2.48xlarge pair, Qwen3-30B-A3B. Same results as NIXL: GSM8K-200 0.925 both; 1024/256 c=64 output 511.0 vs 511.1 tok/s. | no |
| 3 | @whn09 | HiCache L3 on Mooncake Store (persistent prefix KV) | not carried | No gain on one server (author-reported), and a persistent prefix cache is unsafe across RL weight updates unless keyed by weight version. | trn2.48xlarge, Qwen3-8B TP4. Cross-engine and restart prefix hits give identical output. Throughput with L2: 50.66 off vs 50.82 tok/s. | no |
| 14 | @jimburtoft | MiniMax-H3 video model on one trn2 device | not carried | Not related to Qwen3 RL. | One trn2 device, image v0.1.0, 22 CPU tests. | no |
| 19 | @whn09 | Hy4-Flash-FP8 model (draft) | not carried | One day old when read, and the author states performance is far from target. | trn2.48xlarge. Greedy output matches the reference for the first 32 tokens. c=1 TPOT 42.2 ms against 6.82 ms for the cited 4 x H200 reference. | no |
| 15 | (ours) | Issue, not a PR: proposes in-place weight update | not applicable | PR 20 and 21 now cover the full-checkpoint part. Our adapter route and kernel are not covered. | Not applicable. | not applicable |

Notes on the table:

- PR 13 accuracy cost: read from the PR's accuracy table on 2026-11-23 (0.907 against 0.810 / 0.863).
- PR 6 TPOT is a range over the GDN state dtype and tile settings in the author's table, not one number.
- The PR 14 row lists no performance figure; the PR text gives test counts (22 CPU tests). Open the PR before citing any speed.
- PR 6 has a host `load_weights` and a module-level `hf_weight_mappings(graph)` helper, but no `reload_weights`. That is why PR 20 refuses it (checked in the diff).
- PR 9's `LlamaGraph` subclasses `TransformerForCausalLM` (checked in the diff), so it inherits PR 20's `reload_weights`. Not run.

## Merge order for the `integration` branch

`main` is upstream v0.1.0 plus these documents, with no code change. Branch `integration` is `main` plus PRs 20, 4, 9, 16, 17, 18 in that order. None of them has been run by us. Each PR also lives as its own branch `upstream-pr/<N>`, the unchanged upstream PR head.

Conflicts to expect, and what to do:

- 20 on main: none.
- 4 after 20: none. Our own earlier copy of the same change is not carried; the fork uses PR 4.
- 9 after 4: none in this set. Doc-table conflicts appear only if PR 2 or PR 6 are merged later (`README.md`, `examples/README.md`, `docs/model-recipes/index.md`, `feature-model-compatibility.md`, and `catalog.py` with PR 6).
- 16 then 17: textual conflict in `sglang_neuron/backend/stack.py`. 16 adds `cache_builtin_dialect_lookup` and `cache_traceback_frame_formatting`. 17 adds `prime_target_family` and target-family helpers. Keep both sets. Not yet checked: whether 17's helpers sit next to 16's insertion point. After resolving, import `backend.stack` with `SGLANG_NEURON_CPU_MODE=1`.
- 18 after 17: conflicts in `sglang_neuron/envs.py` (both append after `deepseek_v2_max_context`; keep both), `sglang_neuron/env_registry.py` (keep both entries) and `docs/guides/reference-configuration.md` (one row each; keep both).
- Cache cost: PRs 4, 16 and 17 each change a file in the kernel-source digest (`attention_decode_paged.py`, `backend/stack.py`, `moe_cte_bf16.py`). After `integration` is built, the existing NEFF cache is refused or must be restamped, and every graph recompiles once. This is not a git conflict. Plan it as a one-time launch cost.

Conflicts for tracked branches against `integration`:

- 21 conflicts with 20 on adjacent lines in `_register_launch_hooks` (`general_plugin.py`). Keep both registrations.
- 5 conflicts with 17 in `moe_cte_bf16.py` (two hunks) and `qwen3-moe.md`.
- 13 conflicts with 4 in `attention_decode_paged.py` and with 17 in `qwen3-moe.md`.
- 10 and 11 conflict with 12 in `graph_runner.py`. 10 and 16 touch neighbouring hunks in `backend/init.py`.
- 6 conflicts with 9 in `catalog.py` and docs.
- 8 conflicts with 19 in `general_plugin.py`, `env_registry.py` and `catalog.py`.
- 7 and 8 touch the same `_register_launch_hooks` block as 20.

## Check that must pass before we rely on each integrated PR

All checks are on trn2 with SDK 2.33. None has been run. "Fresh start" means a new server started on that checkpoint.

| PR | Check before relying |
|---|---|
| 20 | Qwen3-0.6B TP4 and Qwen3-8B TP4. Update to a different checkpoint and compare token logprobs with a fresh start on that checkpoint. Update back and compare with the first start. Check that the radix cache is flushed after the update. |
| 4 | Qwen3-8B TP4. Greedy tokens identical to main on a short GSM8K prompt (below 8K). Needle test at 8,204 / 9,644 / 24,044 tokens with and without the PR. Record the recompile time after the cache restamp. The author's text implies no change below 8K, but nobody has measured that, and long-context decode now reads every tile (the author did not time this). |
| 9 | TP4. Greedy token parity with Hugging Face on a 4096-token prompt (the PR reports 151 of 151) and GSM8K first 100 (the PR reports 0.78). |
| 16 | Qwen3-8B TP4, warm cache. Ready time with and without the PR, greedy tokens equal, zero neuronx-cc compiles, and a log line showing whether the stub installed in our image. |
| 17 | Qwen3-30B-A3B TP4 EP4 on a trn2.3xlarge. Main vs PR vs `SGLANG_NEURON_MOE_PREFILL=block` on the same build at 2048/128, c=32 and c=64. Then a logprob difference of block vs hybrid prefill on 20 prompts. |
| 18 | Same model and host. c=1 TPOT for main vs PR vs `SGLANG_NEURON_MOE_DECODE_SELECTIVE=0`. Then an on-device logprob difference of selective vs all-expert decode at batch 1, 2 and 4. |

## How our patch relates to PR 20 and PR 21

Our patch is branch `feat/inplace-adapter-sync`, on top of v0.1.0. It changes `sglang_neuron/general_plugin.py`, `sglang_neuron/models/host.py` and `sglang_neuron/models/qwen3/sglang_model.py`. The kernel files `sglang_neuron/models/rlkit_kernel_path.py` and `sglang_neuron/models/rlkit_lora_merge.py`, the kernel branch in `host.py` and the CPU tests in `tests/rlkit/` are on branch `feat/lora-merge-kernel`, on top of it. The patch was hardware-tested on trn2 for Qwen3-0.6B and Qwen3-8B (full update and adapter sync). The kernel write mode was not run on hardware. Other model families, pipeline parallel and MoE were not tested with the adapter path.

Overlap with PR 20:

- Both hook the same function, `WeightUpdater.update_weights_from_disk`, and both pin the same md5 of the replaced upstream function. Two AROUND hooks on one target would be wrong.
- PR 20 refuses by capability (`reloads_weights_in_place`) and pins the md5. Ours does not.
- PR 20 keeps the live graph's loaders after `load_state_dict(assign=True)`. Ours stages a second graph on the CPU. At 8B our full update takes 4.7 to 6.2 s, of which the loaders take 3.6 to 4.7 s per rank.

Decision:

- PR 20 is the base for full-checkpoint sync. **Our staged-CPU-graph full-reload path is retired, not merged.**
- What upstream does not cover, and we keep:
  - The LoRA adapter route: `adapter_config.json` in the folder, host base copies, a pre-write coverage check, refusal of rslora and dora, and the lazy low-rank slice.
  - Our extra checks: dtype, every live parameter refreshed, buffer coverage log, and `drain_launches` before the write.
  - The LoRA shard-merge NKI kernel (simulator only).
  - The `RLKIT_DECODE_LADDER` switch in `models/qwen3/sglang_model.py`, a separate small item.
- PR 21 (delta pull) is a tracked branch on top of PR 20. It needs a trainer that publishes the `weight_v{N:06d}` format, and ours does not.
- PR 20 does not check dtype, does not check that every live parameter was refreshed, and does not call `drain_launches`. It does reject unknown tensors and shape mismatches before the first `copy_`.

Plan for the fork, in order:

1. Branch `main` = upstream v0.1.0 plus these documents, no code change. Branch `integration` = `main` + 20, 4, 9, 16, 17, 18, conflicts resolved as listed above.
2. Our patch is branch `feat/inplace-adapter-sync` on top of v0.1.0. It is not on `integration`, because it and PR 20 hook the same function (see above).
3. The LoRA merge kernel is on its own branch `feat/lora-merge-kernel`, on top of `feat/inplace-adapter-sync`, marked "NKI 0.7.0 CPU simulator only, not run on hardware". Its CPU tests are in `tests/rlkit/` (`bash tests/rlkit/run_all.sh`).
4. The unchanged heads of upstream PRs 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 20 and 21 are branches `upstream-pr/<N>`. The experiment scripts, launchers and session records stay in https://github.com/pengchenglu1997/trainium-sglang-rl .
5. Later, as named tasks in our roadmap (not public) and not done blind:
   - `feat/adapter-sync-on-pr20`: the hook becomes a thin wrapper. If `adapter_config.json` is present, call our adapter function. Otherwise fall through to PR 20's reload. The adapter function uses the live first graph's `hf_weight_mappings()` and the loaders PR 20 keeps, instead of `build_graph`. Base copies are cleared on every full reload. Check that the kept loaders accept `_LowRankSlice`. Done only with the Qwen3-0.6B and Qwen3-8B adapter re-tests (adapter 1, 2, 1, zero adapter, each equal to a fresh start on the host-merged checkpoint).
   - A small upstream follow-up to PR 20 with the dtype and every-parameter-refreshed checks and a CPU-mode test. `drain_launches` is proposed only if an async-execution test shows it is needed.
   - Propose the adapter route upstream as a separate PR once the re-base passes, linked from issue 15.
   - Port the kernel into the adapter write path only after a hardware run of the kernel.
   - A `reload_weights` path for PR 6 (it has a host `load_weights` and a module-level `hf_weight_mappings(graph)` but no `reload_weights`) and the `k_eq_v` check for PR 2, only if the 27 to 32B rung needs them.

## Ideas worth borrowing from PRs we do not merge

- **PR 20.** Re-attach weight loaders after `load_state_dict(assign=True)` so the live graph reloads without building a second graph. Refuse by capability, not by model name. Validate unknown tensors and shapes before the first `copy_`.
- **PR 21.** Per-tensor checksums that fail loudly, an applied-version marker, and a per-host file lock so each rank's pull is idempotent.
- **PR 4.** The needle-test grid (8,132 / 8,204 / 9,644 / 24,044 / 76,844 / 240,044 tokens), pinned on `--context-length`, as a cheap correctness gate for any decode kernel change.
- **PR 10 and 11.** The restart acceptance table (cache off, first launch, restored) with GSM8K, TPOT and NEFF count held constant. Queue all compiles before executing any.
- **PR 12.** An `unsubmitted()` scope to trace without executing, tracing in the same order on all ranks, the SPMD NEFF check inside the trace pass, and `NEURON_PARALLEL_COMPILE_WORKERS=1` for small hosts.
- **PR 16.** Confirm zero neuronx-cc compiles as proof that a change leaves the lowered program untouched. Use py-spy on one rank's trace to find fixed per-node costs.
- **PR 17 and 5.** Resolve platform facts once at model construction or import time, not inside a traced forward. Use tables of validated shapes with an explicit refusal. We can reuse this for LoRA merge kernel dispatch.
- **PR 18.** The filler-expert trick (an unchosen local expert at zero affinity) whenever a kernel needs a fixed index count. Its stated gap, that 8 to 32 rows still read all experts, is a candidate for a mid-batch MoE decode kernel.
- **PR 5.** A canary prompt after each weight update. A bad kernel can poison every later request, and `flush_cache` does not recover it.
- **PR 9.** Launch refusals for unsupported RoPE scaling and biases. Its acceptance list (Hugging Face token parity, needle at three depths, prefix-cache parity, cancel soak) as a checklist template.
- **PR 2.** Teacher-force against a Hugging Face CPU reference, and check separately that the reference shows the same loop, to tell a kernel bug from model behaviour.
- **PR 6.** A working bank for recurrent state, and the prefill ladder (512 and 1024 tiles), which cut short-prompt TTFT in the author's table. Relevant to RL rollouts with short prompts.
- **PR 8.** A per-kernel check pattern (CPU emulate reference, `nki.simulate`, one core against torch, with controls that must fail). A `profile_step`-style per-engine busy table. Negative results are recorded with their numbers (the flash-attention early-exit patch was slower on a mixed workload).
- **PR 3.** Page-major contiguous staging beat per-row transfers by a wide margin in the author's microbenchmark. Treat that as a rule for any weight or KV shipping we build. Any prefix cache must be flushed or keyed by weight version on a weight update.
- **PR 14.** One-rank gloo group helpers for CPU unit tests of weight-update code without hardware.
- **PR 19.** Compile rank 0 first to avoid host-memory exhaustion at high TP. NeuronCore int64 multiply, XOR and remainder are not exact, so use int32.
- **PR 13.** FP8 KV is a possible later lever for rollout capacity, only after bf16 parity is shown.
- **PR 7.** The Qwen3-235B recipe notes and FP8 MoE kernels, as reference material for kernel work (G3).
