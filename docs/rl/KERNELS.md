# Kernels

Status of the NKI kernel work for RL post-training on Trainium2 with SGLang-Neuron, as of 2026-11-25. Changed 2026-11-25:
both kernels ran on hardware on 2026-11-24 (trn2.48xlarge, Trainium2); earlier versions of this file said that no kernel
had run on a NeuronCore.

Read this first:

- The LoRA merge kernel (branch `feat/lora-merge-kernel`) was tested in the NKI 0.7.0 CPU simulator on 2026-11-23 and ran
  on hardware for the first time on 2026-11-24 (Qwen3-0.6B in a server, 8B shards outside a server). It is not called
  validated: its acceptance rule is open (section 1).
- The sampler kernel (branch `feat/sampler-kernel`) was built and run on hardware on 2026-11-24, at Qwen3-0.6B and
  Qwen3-8B, for the `temperature` sampler only (section 2). Changed 2026-11-26: the same kernel is used in the
  `topk_topp` mode on branch `feat/sampler-kernel-topk-topp`, in the CPU tests and the NKI simulator only (section 2,
  "`topk_topp` mode").
- Every number has a condition and a source file. Numbers marked "estimate" are not measurements. A "target" is a
  target, with its basis.
- All profile numbers in section 5 are Qwen3-0.6B, TP4, one run each, from 2026-11-21. The hardware results of 2026-11-24
  are in [RESULTS.md](RESULTS.md) section 8.
- Measured results that do not involve a kernel are in [RESULTS.md](RESULTS.md).

Where this fits. Goal G3 is one or two NKI kernel optimisations with before/after numbers. The merge kernel serves the
adapter update of the RL loop (section 1). The sampler kernel was the second candidate. It was not scheduled when this
file was written on 2026-11-23, and it was built and run on 2026-11-24 (section 2). Sections 3 and 4 record what was
looked at and not taken, so nobody repeats the search.

## 1. LoRA shard-merge kernel

### What it computes

`lora_merge(base[IN,OUT] bf16, a_stack[K,IN], bt[K,OUT]) -> out[IN,OUT]`, with
`out = bf16(float32(base) + a_stack.T @ bt)`. One cast to bf16, at the end.

It runs on the device, per tensor-parallel rank and per adapted weight shard, from a resident copy of the base shard.
Only the LoRA factors go from host to device. The serving graphs are not changed and not recompiled, so a decode or
prefill step costs what it cost before.

### Why

At Qwen3-8B, TP4, rank 16 on q, k, v, o, gate, up and down, an adapter update on a running server takes 13.9-14.0 s in
steady state (first update 35.8 s). Of that, 13.5-13.6 s is forming the delta matrices on the host and 0.3 s is the
copy to the device. A full-checkpoint in-place update at 8B takes 4.7-6.2 s, so at 8B the adapter path is slower than
the full path. (Source: `docs/NIGHT_RECORD_2026-11-21.md` section 4; one run each.)

### Design in short

- A bf16 copy of each adapted base shard stays on the device. The kernel reads it and writes the merged shard. The base
  copy is never overwritten, so swapping adapter 1 for 2 for 1 returns the same weights by construction.
- The factors are built per rank through the engine's own weight loaders (`weights.py` of SGLang-Neuron v0.1.0, commit
  4e24326). That makes the fused qkv and gate_up layouts, the replicated KV heads, and the scale `alpha / r` follow the
  engine and not a copy of its logic. The scale is folded into `bt`.
- K (the stacked LoRA rank, 16 to 48 here) is the partition axis of both factor operands, so no transpose is needed on
  the device. Fused segments (q, k, v) become a block-diagonal stack.
- Tiles: 128 rows of IN, 512 columns of OUT (one fp32 PSUM bank). K above 128 is cut into chunks of 128 and
  accumulated. The base tile cannot be preloaded into PSUM (undefined on this hardware generation per the NKI docs), so
  the add of the base is done on the vector engine.
- The cost: one bf16 copy of each adapted shard stays on the device, about 3.5 GB per rank at 8B (plan estimate; the
  patch logs the real figure per update as "base copies kept on the device").
- The design was chosen against two rejected designs; the rejected ones are not described here.

### Where it sits in the patch

The in-place update and adapter sync patch is on branch `feat/inplace-adapter-sync` (files
`sglang_neuron/general_plugin.py`, `sglang_neuron/models/host.py`, `sglang_neuron/models/qwen3/sglang_model.py`). The
kernel is on branch `feat/lora-merge-kernel`, on top of it: `sglang_neuron/models/rlkit_lora_merge.py`,
`sglang_neuron/models/rlkit_kernel_path.py`, the kernel branch in `host.py`, and the CPU tests in `tests/rlkit/`. The host
path stays the default.

| Switch | Values | Meaning |
|---|---|---|
| `RLKIT_DEVICE_WRITE=kernel` | unset (host path) or `kernel` | use the kernel path for adapter updates |
| `RLKIT_KERNEL_IMPL` | `auto`, `nki`, `torch`, `sim` | how the kernel runs: NKI on device, the same arithmetic in torch, or the simulator |
| `RLKIT_KERNEL_FACTORS` | `fp32`, `split` | factor dtype; `split` carries fp32 factors as three bf16 parts (hi, hi, lo), less accurate than native fp32 in the simulator |
| `RLKIT_KERNEL_CHECK` | `first`, `all`, `none` | read the live parameter back after the write: once per call form, shard shape, K and factor dtype, or every call, or never |
| `RLKIT_ADAPTER_ON_DEVICE=1` | on/off | the older eager torch merge on the device (no NKI) |

Guards. A parameter the kernel guard or the layout probe refuses (unaligned shard, K above 128, non-float32 factors)
is merged on the host in the same update. Any failure of the kernel call or of the read-back check redoes the whole
update on the host and turns the kernel mode off for that server. Two changes apply to both modes: adapter coverage is
checked for all parameters before the first write, and adapters that set `use_rslora`, `use_dora`, `rank_pattern` or
`alpha_pattern` are refused (the old code applied one scale `alpha / r` silently).

### Status: CPU tests (2026-11-23) and first hardware run (2026-11-24)

CPU tests first. Environment: WSL Ubuntu, NKI 0.7.0 simulator, Python 3.12, torch 2.15.0 CPU, 2026-11-23. Target platform override
trn2. No `neuronx-cc` compile and no device run. The test scripts are in `tests/rlkit/` on branch
`feat/lora-merge-kernel`; run them with `bash tests/rlkit/run_all.sh` (see `tests/rlkit/README.md`).

| Test | What it checks | Result |
|---|---|---|
| `test_merge_kernel_sim.py` | zero adapter; integer factors; random factors; 20 alternating swaps; K 16, 32, 48; partial tiles; all 0.6B and 8B per-rank shapes (4 kinds each); LNC 1 vs 2 | all pass. Kernel equals the float32 reference on every entry. Against the float64 sum rounded once: 255 of 6,291,456 entries differ (8B qkv) to 735 of 25,165,824 (8B gate_up) |
| `test_k_chunks.py` | K 144 and 200 (two accumulating matmuls), partial tiles, LNC 1 and 2, guards | 43 pass, 0 failed |
| `test_factor_layout.py` | per-rank factors through the engine's loaders; integer factors bit-exact | 78 pass, 0 failed. qkv, o, gate_up, down at TP 1, 2, 4, 8, 16, every rank, 0.6B and 8B |
| `negative_control_factor_layout.py` | wrong rank, forgotten scale, summed segments must fail | each mistake fails on 4 of 4 ranks |
| `test_end_to_end_review.py` | loader, then builders, then kernel, against loader(base + full delta in float64) rounded once | 0 failed; 7 to 52 entries differ per shard, at most 1 to 18 units |
| `test_host_kernel_path.py` | the patched `host.py` end to end on adapter folders on disk; real engine loaders, stubbed engine imports | 47 pass, 0 failed. At 0.6B TP4 in the simulator, live parameters equal an independent float32 reference on every entry (0 of 3,933,184 differ) and equal the patch's host path on every entry; they differ from the float64 sum rounded once in 255 (rank 0) and 244 (rank 3) entries |
| mutation test of the wiring | 21 single faults planted | the test of that time caught 15. Of the 6 missed, 3 gave wrong weights; tests were added and now catch all 3. The other 3 changed no result in any scenario built (write before the check, the bf16 dtype guard, the probe-cache key) and have no test |

What these tests do not show:

- The output is "one bf16 rounding of an fp32 sum", not "the exact sum rounded once". About 1e-5 to 1e-4 of entries
  differ from the exact rounding, mostly by 1 unit.
- `test_host_kernel_path.py` is not a server run. SGLang is not installed in that environment, so the plugin's CPU
  mode was not run.
- Not done: a compile-only run with `neuronx-cc`; the call through torch (`wrap_nki`); reading a real PEFT adapter
  (key names, dtypes); a performance number (the simulator is not a timing model).
- Open for hardware only: whether a device tensor copied into a live parameter serves correctly. The eager torch device
  merge (`RLKIT_ADAPTER_ON_DEVICE=1`) wrote values that differ from the host path by single units in under 0.005 % of
  entries, yet the served output did not equal a restart on the host-merged checkpoint: same greedy tokens on two of
  three test prompts with token logprobs 0.005-0.011 apart on average, and a fork at a near tie on the third
  (`patches/PR_DRAFT_inplace_weight_update.md` in the public repo). The cause was not found. The kernel does not by
  itself explain or remove that difference. Changed 2026-11-25: the kernel's own hardware run (below) shows a difference of
  the same kind against a restart on the host-merged checkpoint.

### Hardware, 2026-11-24 (first run on a NeuronCore)

Hardware: trn2.48xlarge (Trainium2), NKI 0.7.0, SGLang-Neuron v0.1.0 image. Full tables and conditions are in
[RESULTS.md](RESULTS.md) section 8.6. In short:

- H1, the kernel alone on one core (eager call and compiled call): it runs on the device outside a serving graph. Integer
  factors are bit-exact in 4 shapes. Float factors equal eager torch on the device element for element. Per Qwen3-8B
  shard, warm: kernel 0.19 to 0.67 ms, torch on the device 0.50 to 1.80 ms over the five warm calls per shape, host merge 24 to 112 ms.
- H2, Qwen3-0.6B TP4 in a server, kernel mode: the kernel ran in all 4 updates on all 4 ranks (112 tensors each, no
  fallback to the host). Adapter 1, 2, 1 identical (logprob difference 0.0); the zero adapter equals the base (0.0). Update
  time 12.16 s for the first update, then 0.54 to 0.60 s.
- Against a fresh start on the host-merged checkpoint the result is not equal. On 2 of 3 prompts the text is the same,
  with a mean logprob difference of 0.007 to 0.008 before any fork (largest 0.05 to 0.08). The third prompt forks at
  token 0.
- Control: a copy of the Qwen3-0.6B base checkpoint with 700 elements of each of 198 matrices moved by one unit in the last
  place changes the logprobs of all 20 prompts tested and the length of 5 answers. Largest difference on the first prompt
  0.074; per-token difference of the sums: median 0.002, mean 0.010. This is the same order as the kernel difference. It
  does not show that one-ulp differences are the only cause.

The kernel runs on the device and is self-consistent. It has not been timed at Qwen3-8B in a server (H3 not run), and the
host-merged restart differs from it by an amount that the control puts in the range of one-ulp weight differences.

Open acceptance question: what difference between the kernel server and a restart on the host-merged checkpoint is
accepted? The row "vs a restart on a checkpoint merged by the same kernel: token logprob difference 0.0" below was written
for step H4, which has not run. Without H4 the open question is whether the 0.007 to 0.008 mean logprob difference comes
only from rounding of the merged weights or also from the in-place write. No acceptance rule for this has been set.

### Target and acceptance (proposed, fixed before any hardware run)

| Quantity (Qwen3-8B, TP4, rank 16 on seven projections) | Today | Target (proposed) |
|---|---|---|
| adapter swap, steady state | 13.9 s (host merge, measured 2026-11-21) | at most 1.5 s |
| same adapter swapped back in | n/a | bit-identical weights and outputs |
| vs a restart on a checkpoint merged by the same kernel | n/a | token logprob difference 0.0 |
| decode step before and after a swap | same | same (compile-cache keys unchanged) |

The 1.5 s is a proposed target and an estimate. Its basis is one measured number: 16.4 GB of 8B weights overwritten on
the device in 1.19 s (plain TorchNeuron process, no server; `patches/PR_DRAFT_inplace_weight_update.md`). It is not a
measurement of this kernel.

Baselines that go into every table: full-checkpoint in-place update 4.7-6.2 s at 8B (0.34-0.41 s at 0.6B); host-merge
adapter 13.9 s at 8B (0.67 s at 0.6B); the eager torch device merge already in the patch (about 0.13 s at 0.6B TP4, PR draft;
not measured at 8B); the lazy host delta (`_LowRankSlice`, CPU-tested only).

### Hardware steps and stop rules

The H1 script is `tests/rlkit/hw_merge_kernel_bench.py` on branch `feat/lora-merge-kernel`. `50_adapter_server_test.py`
(H2, H3) is an experiment script of the public repo https://github.com/pengchenglu1997/trainium-sglang-rl, not of this
fork. Changed 2026-11-25: the "Status" column is new; H1 and H2 ran on 2026-11-24.

| Step | Script | Question | Stop rule | Status |
|---|---|---|---|---|
| H1 | `tests/rlkit/hw_merge_kernel_bench.py`, one core | Can the kernel be called on device tensors outside a serving graph (eager, or a small compiled helper)? Time per 8B shard; read back against the reference; integer factors bit-exact; time of the eager torch device merge and of the host merge at the same shapes | exit code 3 means it cannot be called: stop the kernel part and report the torch-device and host numbers only | ran 2026-11-24; the kernel can be called (eager and compiled); integer factors bit-exact; per-shard times above |
| H2 | `50_adapter_server_test.py`, 0.6B TP4, kernel mode, starting with `RLKIT_KERNEL_CHECK=all` | Swap time split (factors, upload, kernels, copy, sync); adapter 1, 2, 1 identical; zero adapter equals base | any check fails: kernel mode stays off | ran 2026-11-24; adapter 1, 2, 1 identical; zero adapter equals base; swap 0.54 to 0.60 s after a first update of 12.16 s; differs from a host-merged restart (above) |
| H3 | same test at 8B TP4 | The headline number; device memory after the base copy; KV pool size | swap above 1.5 s: try one launch per kind over stacked layers, then report what was measured | not run |
| H4 | restart on the kernel-merged checkpoint | Logprob difference against the swapped server | not 0.0: the difference stays unexplained and is reported so | not run |
| H5 | compile-cache keys and decode step time before and after a swap | Zero forward overhead | none | not run |

H4 tests one explanation of the eager-merge difference above: a restart on the checkpoint the kernel itself wrote must give
0.0. If it does, the earlier difference is the few-entry rounding difference between two merge arithmetics, not a fault
of in-place writing.

### What a write-up may and may not say

May: swap time before and after at 8B with all baselines in one table; no change to the serving graphs; identical
weights when the same adapter returns; the shard-space layout (K on the partition axis of both operands, no transpose on
the device, block-diagonal fused segments); the memory cost of the base copies.

May not: credit the kernel with the whole drop from 13.9 s (most of it comes from not forming full matrices on the
host); call it the first LoRA on Trainium (NxD Inference serves multi-LoRA with torch ops); claim hardware correctness
from simulator runs; call the kernel validated while the acceptance rule in the hardware section above is open.

## 2. Sampler kernel (built and run on hardware, 2026-11-24)

Changed 2026-11-25. Until 2026-11-23 this section read "designed, not built, not scheduled". The kernel was built on
2026-11-24 and run on hardware the same day. The sections below that were written before the build are kept and marked.

### What it replaces

In `SGLANG_NEURON_SAMPLER=temperature` mode with no top-logprobs, the shard-local stage of the engine's sampler
(`sampling.py`): the per-column hash noise, the Gumbel score, the per-shard maximum, the winner id, the raw logit at the
winner, and the log-sum-exp. One kernel per rank does this over the rank's 37,984 vocabulary columns (151,936 divided by
4; Qwen3-0.6B and 8B have the same shard width at TP4). The kernel views the shard as `[rows x S, 37984 / S]` so that the
rows fill the 128 partitions, walks the columns in chunks of 2,048, and writes 4 partial values per partition and chunk.
The merge of the partials, the raw logit at the winner, the all-gather and the global merge stay the engine's code.
`topk_topp`, greedy-only mode and top-logprobs above zero keep the torch path (top-logprobs above zero with the switch on
raises an error). Changed 2026-11-26: `topk_topp` has a switch of its own on a CPU-only branch (subsection "`topk_topp`
mode" below).

Where it is. Branch `feat/sampler-kernel` of this fork, on top of v0.1.0: `sglang_neuron/sampling.py` (the released file
plus one import-time constant and one guarded branch) and a new file `sglang_neuron/rlkit_sampler_kernel.py`. Switch: `RLKIT_SAMPLER_KERNEL=1`. Unset or 0, the released code path runs. Kernel
sources are part of the compile-cache digest, so a server with the kernel needs its own compile-cache folder.

### Relevance to a published RL setting

The published RL rollout setting on another accelerator that states its sampling settings, the vLLM TPU RL write-up of
2026-11-12 (https://vllm.ai/blog/2026-11-12-rl-small-dense-tpu-v7), samples with top-k and top-p. In `sampling.py` of SGLang-Neuron
(`origin/main`), `topk_topp` mode is `temperature` plus SGLang's top-k / top-p / min-p filters, applied to a pool of the
global top 256 tokens per row (`TOP_K_MAX`); `temperature` mode does not apply these filters and rejects such requests at
admission. On a device-sampling server, a request with top-k and top-p therefore runs in `topk_topp` mode. The kernel designed here replaces the
`temperature` mode only, so it would not run under that setting. Changed 2026-11-25: the kernel as built replaces the
`temperature` mode only, so it still does not cover that setting. Covering it needs a `topk_topp` mode kernel, which does
not exist. Changed 2026-11-26: a `topk_topp` path with the same kernel exists in the CPU simulator only (subsection
"`topk_topp` mode" below); on hardware the setting is still not covered.

The conclusion that stood here on 2026-11-23, "the sampler kernel is not scheduled; the second kernel is chosen from a
device profile", no longer holds: the kernel was built and run on 2026-11-24. The device profiles taken in that session
are not read yet (see [RESULTS.md](RESULTS.md) section 8.8).

### Hardware results, 2026-11-24

Hardware: trn2.48xlarge (Trainium2), one chip = TP4 on 4 logical cores, SGLang-Neuron v0.1.0 image, NKI 0.7.0. Full
tables, conditions and file names are in [RESULTS.md](RESULTS.md) section 8.5. Prompt 1024, 256 output tokens, tok/s at
concurrency 1 / 8 / 32, two passes each:

| | Qwen3-0.6B | Qwen3-8B |
|---|---|---|
| released `temperature` sampler | 211 / 1,160 / 2,455 and 211 / 1,171 / 2,454 | 109 / 604 / 1,421 and 110 / 607 / 1,426 |
| sampler kernel | 319 / 1,624 / 2,730 and 319 / 1,636 / 2,734 | 114 / 642 / 1,489 and 114 / 647 / 1,496 |
| released `greedy` sampler (one pass) | 311 / 1,608 / 2,781 | 113 / 639 / 1,533 |

- At 8B the gain is 4.5 to 6.6 percent over the six cells (unrounded values in `82_sampler_kernel_r12.md`; about 5 percent; at concurrency 32, 1,421 to 1,489
  and 1,426 to 1,496, which is 4.8 and 4.9 percent).
- Sampled rollout as the RL loop asks (Qwen3-0.6B, temperature 0.7, 64 questions x 8 answers, up to 256 tokens, three
  passes): released 2,283 / 2,335 / 2,295 tok/s, kernel 2,472 / 2,484 / 2,484.
- Combinations at 0.6B: kernel with `--dp-size 4` of TP1 in one server 223 / 1,420 / 3,026 (repeats at concurrency 32:
  3,039, 3,048, 3,035); kernel with 4 containers of TP1 223 / 1,438 / 3,098 (one measurement). The released `temperature`
  sampler with `--dp-size 4` gave 192 / 1,172 / 2,701.
- Correctness. 0.6B at temperature 0, 20 prompts: the kernel equals the released `greedy` sampler on 20 of 20 prompts
  (largest logprob difference 5.7e-6). The released engine's own `temperature` and `greedy` samplers differ on 3 of 20
  prompts at temperature 0, and the kernel agrees with the `greedy` one, not with the `temperature` one, on those 3. At 8B
  the kernel equals the released `temperature` sampler on 20 of 20 (largest 3.8e-6). At temperature 0.8 with request seeds
  (0.6B, 60 requests of 48 tokens) the kernel and the released sampler give equal whole sequences on 44 of 60. The first
  token of one prompt over 3,000 draws with the same seeds has the same counts for every token.
- Simulator tests before the run: 11 of 11 pass.

Limits:

- the `temperature` sampler only; the `topk_topp` mode is not covered on hardware (a CPU-simulator path exists, below);
- Qwen3-0.6B and Qwen3-8B only; nothing at 27 to 32B;
- the gain at 8B is about 5 percent;
- not run inside a full RL loop (a sampled rollout bench ran);
- at concurrency 32 the kernel is below the released `greedy` sampler at both sizes (2,730 against 2,781 at 0.6B; 1,489
  against 1,533 at 8B);
- the device profiles of the session are not read, so there is no instruction-time figure for the sampler scope with and
  without the kernel.

What remains: a `topk_topp` mode kernel; a full RL loop with the kernel on; larger models; a proposal upstream (nothing
has been sent). Changed 2026-11-26: the `topk_topp` path below is written and CPU-tested; its hardware run remains.

### `topk_topp` mode (CPU simulator only, 2026-11-26)

Status: written and tested on the CPU and in the NKI 0.7.0 simulator only. Not run on a device. No speed claim. The test
results are recorded in the team hub, not here (cpu-only work). Branch `feat/sampler-kernel-topk-topp`, a `cpu-only`
pull request to `integration`; switch `RLKIT_SAMPLER_KERNEL_TOPK=1` (unset or 0: the released function body runs).

What it replaces. Of the shard-local stage of `sample_filtered_sharded_logits` (`sampling.py` v0.1.0, lines 372-386)
only the unfiltered draw: the per-column hash noise, the Gumbel score, the shard maximum and its smallest id. These come
from `gumbel_partials`, the `temperature` mode's kernel, called unchanged; the merge is the engine's tie rule, as in that
mode. New file `sglang_neuron/rlkit_sampler_topk.py`; it holds no NKI code. The temperature kernel's file is not changed,
so its upstream draft pull request is not touched.

What stays the engine's torch code, and why:
- both log-sum-exps. The tempered one gives the pool probabilities that the top-p and min-p tests compare; a kernel
  exp-sum, which differs from torch in the last bits, could move a row across a top-p boundary;
- the shard pool top-k (`_pool_topk`: on a device nkilib's rotational kernel where one fits the shape, else the
  reduction-based `_exact_topk`), the raw-logit gather, and every line after
  the payload (all-gather, shard winner, pool sort, cumulative probability, keep mask, pool draw with torch noise,
  logprobs, top logprobs), copied verbatim from v0.1.0 lines 387-416; a test checks the text.

Consequence. A restricted sampled row (top-k below SGLang's "all" value, top-p below 1 or min-p above 0, temperature
above 0) gets its token and logprob from the released code on the same inputs: the kernel's result is not read for it
(a test puts garbage in place of the kernel's result and these rows stay bit-equal). The kernel's result is read for
temperature-0 rows (max and smallest id: exact) and for unrestricted sampled rows (noise from the kernel's own `log`,
which in the simulator differs from `torch.log` in the last bits for some inputs, so a near-tie can be drawn
differently; on hardware the `temperature` mode gave 44 of 60 equal sequences at the same seeds, section above).

Tests (`tests/rlkit_sampler/selftest_sampler_topk.py`; method and list in `tests/rlkit_sampler/README.md`): against the
released `topk_topp` path over a simulated TP4 group, bit-equal outputs with the kernel's CPU reference for top-k 1 / 50 /
256 / off, top-p 0.5 / 0.95 / 1, min-p 0 / 0.1, temperature-0 rows, mixed per-row settings, ties, rows 1 / 8 / 32; the
switch unset leaves the released path; planted faults are caught; in the simulator, restricted rows bit-equal and
temperature-0 tokens equal. Whether these hold on a device is not known.

Compile flags. `models/host.py` (`neuron_compiler_args`) keeps the default arguments for a decode that traces the device
top-k, which `topk_topp` always does, because `-O2` miscompiles that top-k (section "The compile-flag question" below).
This branch does not change that rule. So the kernel would compile at the default arguments in this mode, a
configuration none of its hardware runs used (those were `temperature` decodes, by the code at `-O2`). Not known until
run.

Compile cache. `sampling.py` and `rlkit_sampler_topk.py` are kernel sources of the cache digest (`kernel_sources` follows
the imports of a kernel file), so a server built from this branch gets another digest than one built from
`integration`, with the switch on or off; a cache folder stamped by an `integration` server is refused.

Not known until run on a device:
- whether the decode graph with the kernel next to the rotational top-k loads and runs: the engine's comment records
  that a GpSimd top-k hung the GpSimd engine when co-resident with vector work in one NEFF, and this kernel uses GpSimd
  integer multiplies;
- whether restricted rows give the released server's tokens (expected by construction; another graph may order float
  work differently);
- any speed. In the RL setting every sampled row is restricted, so the kernel's noise pass and its log-sum-exp
  partials are computed and thrown away; the path removes the torch noise pass and the torch full-width maximum, and
  whether that saves any time is not known.
  The released `topk_topp` sampler's speed is in [RESULTS.md](RESULTS.md) section 8.4.

Designs considered and not taken (2026-11-26):
- a fused filter kernel (shard top-k by `max8` rounds, cumulative mass by a vector scan, a cut-off value instead of the
  pool sort): more device work removed, but its cumulative sum is not the released matmul bit for bit, so it could not
  give the released tokens at top-p boundaries, and it replaces the rotational top-k, whose tie order the released path
  leaves unspecified;
- a certified unfiltered draw (top-8 candidates per chunk from the kernel, rescored with the released torch ops, plus a
  per-row certificate that the released winner is among them): would make unrestricted sampled rows exact too, needs new
  NKI ops (`max8`, `nc_match_replace8`) and a device bound on the `log` error; a follow-up candidate;
- a runtime skip of the noise pass when every sampled row is restricted (a register-loaded loop bound): simulator-only
  feature, not used here.

### Measured basis (written 2026-11-23, before the kernel existed)

| Quantity | Value | Condition | Source |
|---|---|---|---|
| speed, `temperature` mode | 2,183-2,218 tok/s | Qwen3-0.6B TP4, 256 requests of up to 256 tokens, 32 in flight, three passes, one server | `NIGHT_RECORD` section 3 |
| speed, `topk_topp` mode, full vocabulary | 2,713-2,730 tok/s | same | same |
| sampler subtree of the decode step | 2.62 ms of 26.0 ms instruction time per step (10.1 %) | device profile of rank 0, 4 decode steps, `temperature` mode with no top logprobs (so, by the code, a decode compiled at `-O2`). Instruction time is summed over engines; it is not wall time (the step is 9.2-9.3 ms) | our analysis notes (not published); the join script was not saved |

Which mode the RL loop used: the launcher `srv_sglang_neuron.sh` defaults to `temperature`, and
`scripts/run_trn2_seeds.sh` sets neither `SGL_SAMPLER` nor `SGL_TOP_LOGPROBS`, so by the scripts the 0.6B loop ran
`temperature` with top logprobs 0. This was not read back from the run's own server log. If the loop ran greedy, this
kernel does not touch the loop.

### The compile-flag question (corrected on review)

In `models/host.py` (`neuron_compiler_args`, same code on upstream `main` and in our patch), the decode graph is
compiled at `-O2`, without the modular-flow threshold, except when it traces the device top-k: when
`SGLANG_NEURON_MAX_TOP_LOGPROBS` is above 0, or the mode is `topk_topp`. Then it keeps the default arguments (`-O1` plus
the threshold, by our reading of the code). The code comment says `-O2` miscompiles that top-k.

An earlier version of this file said the probe compared `temperature` at `-O2` with `topk_topp` at the default. That is
not what the scripts did. `58_sampler_probe.py` starts every server with `SGL_TOP_LOGPROBS=20` (its default), so by the
code both probe servers kept the default arguments, and the 20 % gap was measured between two programs compiled with the
same flags, both with the top-logprob path on. Not read back from the server logs.

What follows:

- The probe gap (2,183-2,218 against 2,713-2,730 tok/s) is not explained by the `-O2` flag, as far as the code shows.
- The configuration the RL loop runs and the kernel would replace (`temperature`, top logprobs 0, decode at `-O2`) has
  no measured speed. The 2.62 ms profile figure comes from that configuration; the throughput gap does not.
- The two numbers may still agree by coincidence: one is a wall-time gap, the other is instruction time summed over
  engines.

### Step 0, before any kernel work (written 2026-11-23; not run as written)

Changed 2026-11-25: step 0 was not run as described here. The session of 2026-11-24 measured the released `temperature`,
`greedy` and `topk_topp` servers at Qwen3-0.6B with another harness ([RESULTS.md](RESULTS.md) section 8.4), and the kernel
was built. The text below is what was written before.

Rerun `58_sampler_probe.py` for three passes on Qwen3-0.6B TP4 with three servers: `topk_topp` (top logprobs 20, as
before), `temperature` with `--top-logprobs 0` (the loop's configuration, `-O2`), and `temperature` with top logprobs 0
and the default flags (a one-line change in our patched `host.py`; there is no environment override). Read the compiler
arguments back from each server log.

- If `temperature` at top logprobs 0 already reaches 2,600 tok/s or more at either flag setting, the probe gap came from
  the top-logprob path or the flags, not from the sampler operations. Report that as the finding. The kernel stays
  unscheduled; it comes back as a candidate only if a device profile of that configuration still shows the sampler
  scope at 1.0 ms or more per step.
- If it stays near 2,200 tok/s, the kernel stays unscheduled for the reason in "Relevance to a published RL setting";
  the device profile decides which kernel comes next.
- These thresholds are proposed, not measured; they come from our analysis notes (not published).

### Bound of the gain (written 2026-11-23)

Changed 2026-11-25: the kernel's measured gain at 0.6B in a sampled rollout bench is 6.4 to 8.3 percent (section above);
the gain in an RL iteration is not measured, because no RL loop ran with the kernel. The bound below was written before
that. At most 1.09x of the 0.6B RL iteration if the kernel closes the gap, 1.14x as a hard ceiling (our calculation in
our analysis notes, not published; basis not re-derived here). Qwen3-0.6B only. The 8B figure
is unmeasured because no 8B loop has run. The kernel is therefore a small gain on its own; its value is also the
engineering evidence (G3), if step 0 does not remove the reason.

### Design in short, and acceptance (targets written 2026-11-23, before the build)

Changed 2026-11-25: how the targets stand after the hardware run.

- The throughput target (at least 2,600 tok/s) was set for the `58_sampler_probe.py` harness. The session used llmbench and
  a sampled-rollout bench, so the target is not scored. For the record, the kernel server gave 2,730 and 2,734 tok/s at
  concurrency 32 in llmbench at 0.6B.
- Sampler-scope instruction time at most 0.5 ms: not measured; the device profiles are not read.
- Sampled tokens equal to the engine's path on 99.9 percent of replayed steps: not run in that form. What ran is in the
  hardware results above (equal to the released `greedy` sampler at temperature 0 on 20 of 20 prompts; 44 of 60 equal
  sequences at temperature 0.8).
- Trainer-versus-server logprob difference: not run.
- The open technical risks listed last (integer multiply wrap, `ln` against `torch.log`, passing per-rank constants into
  the traced graph) were not examined one by one. The kernel compiled and ran in the decode graph at both sizes; the log shows it for the 1-row graph only (it logs once
  per process), so the 8-row and 32-row graphs are not confirmed by the log.

The design as written before the build:

- Per rank, per decode step, input logits `[B, 37984]` bf16, B in 1 to 32; a per-rank hash table of the columns built
  once at load time (152 KB); the hash and noise exactly as the engine's oracle code computes them.
- Rows are spread over the 128 partitions in segments; columns are walked in tiles of 2,048; a running maximum with the
  engine's tie rule (smallest global id) and a running log-sum-exp are merged across tiles and segments.
- Proof on CPU: the kernel is checked bit for bit against a copy of the engine's own oracle in the simulator, with
  ties, greedy rows, non-finite rows, every batch size, and five planted faults that the tests must catch.
- Hardware acceptance, fixed in advance and estimates where stated: at least 2,600 tok/s on the same sampler harness
  (this closes at least 80 % of the 2,200 to 2,720 gap; estimate); sampler-scope instruction time at most 0.5 ms
  against 2.62 ms; sampled tokens equal to the engine's `temperature` path on at least 99.9 % of 255 steps x 32 rows
  replayed from identical logits; trainer-vs-server logprob difference no worse than the measured values (mean absolute
  difference 0.015 before and 0.017 after an adapter update, Qwen3-0.6B; PR draft in the public repo). All of these are
  proposed targets.
- Open technical risks, unverified: whether NKI int32 multiplication wraps exactly modulo 2^32 (the main risk; a
  12-bit limb fallback is designed); whether the simulator's `ln` equals `torch.log` in fp32; how `wrap_nki` passes
  per-rank constants into a traced decode graph. Hardware is needed for the first and third.

## 3. Directions examined and not taken

### Live-length decode attention

Idea: make the decode attention read only the live part of the paged KV cache, so a step does not pay for the compiled
context.

Not taken now:

- The measured slope is real: with one live row the step is 9.3 / 13.7 / 25.2 ms at compiled context 1024 / 4096 / 8192
  (`NIGHT_RECORD` section 6). But the design's central saving (fewer HBM bytes for dead blocks) is probably already
  in the library kernel. The source of the library's decode attention (nkilib `attention_tkg.py` on its main branch,
  read second-hand through a web summary; the version in SDK 2.33 was not opened) skips out-of-range block entries in
  its indirect loads. What remains is compute and instruction issue over dead tiles, which nobody measured. The cause of
  the context slope is therefore unexplained, not excluded.
- At the RL loop's context of 1024 the bound is about 1.04x of the iteration, by the design's own arithmetic.
- The context ladder (short rows run a smaller-context graph) is expected to remove much of the long-context cost with
  no kernel. That is an estimate; the measurement with the ladder on (`docs/experiment-plan-2026-11-22.md`, W1.2) has
  not been run.
- Swapping only the attention core out of the fused attention block loses its fused QKV, RoPE and output projection.
- An upstream author is working on early exit in attention (upstream PRs 4 and 8).

Revisit only if a no-kernel device profile at compiled context 1024 and 8192 (ladder off) shows a large cost per dead
tile, and long-context rollouts become the workload. That profile can be taken in the same session as sampler step 0.

### Device-side full-checkpoint shard rebuild

Idea: build the per-rank shards on the device for a full-checkpoint update, instead of in host loaders.

Not taken:

- Its share of any RL loop we run is zero: all of them use the adapter path.
- The loader time (3.6-4.7 s per rank at 8B) was never broken down, so it is not known that data movement is the cost.
- Upstream PR 20 reports a full in-place reload of about 1.9 s on Qwen3-4B TP4 on a trn3.48xlarge, a host path (different
  model and machine). If that holds at 8B, a host fix would beat a kernel.
- The kernel would be pure data movement with no arithmetic, so it has little kernel content.

First step if anyone picks this up: a host profile of one loader run.

## 4. Candidates from other platforms (candidate, not measured)

None of these has been tried against our code. Each line says where the idea comes from and what the first check is.

| Candidate | Source of the idea | What would be checked | Status |
|---|---|---|---|
| Separate fetch tile and compute tile in the decode attention (tune the DMA tile and the SBUF compute tile independently) | The vLLM TPU write-up for small-dense-model RL (vllm.ai/blog/2026-11-12-rl-small-dense-tpu-v7) reports that decoupling HBM fetch geometry from VMEM compute tiling in its attention kernel raised decode throughput by 48 % at TP4 (64,929 to 96,273 tok/s; page opened 2026-11-23). This is a TPU number reported by the source, not reproduced, and does not transfer as a number. | Whether the library decode kernel's fetch and compute tile sizes can be set separately, and what the K-cache load path costs. Every server log of ours warns that `chunk_num_blocks=4 is not a multiple of 16` and falls back to per-block copy plus transpose for the K cache. The link between that warning and the profile (transposes are 49 % of hardware flops) is not traced | candidate, not measured |
| Batch small tiles per DMA | AWS Neuron DMA guidance as read in the NKI DMA bandwidth guide and the AWS agent skill pack: about 2 KiB or more contiguous per partition is the recommended minimum and 4 KiB or more saturates; many small DMAs to adjacent addresses are slower than one large one | Our merge kernel moves 128 x 512 bf16 tiles, which is 1 KiB per partition. That is below the 4 KiB guidance, although the whole tile is 128 KiB. Batching four OUT tiles per `dma_copy` would reach 4 KiB. Check on hardware with a device profile of the merge kernel. The same check applies to the KV reads (head dim 128 in bf16 is 256 B per row of one head) | candidate, not measured |
| Multi-step decode in one compiled loop | The same vLLM TPU write-up moves many decode iterations into one compiled device loop. On Neuron the upstream repository has a whole-model multi-step decode kernel for Qwen3.5 (`qwen35_hybrid_multistep`), but it is greedy argmax only and refuses sampling and logprobs, which RL needs. Dense Qwen3 does not use it | Whether `neuronx-cc` can compile a multi-token decode loop with sampling and logprobs. Our profile shows the device runs back to back (gaps of 6-9 µs between executions), so the gain at 0.6B would not come from removing host gaps | candidate, not measured |

## 5. Where the time goes today

All rows: Qwen3-0.6B, TP4, four logical cores, context 1024, one decode bucket of 32 unless stated, one run each.
Profile of a rollout pass of 32 requests of up to 256 tokens. Nothing here is Qwen3-8B.

| Quantity | Value | Condition | Source |
|---|---|---|---|
| Decode step, device execution | median 9.13 ms (system profile); 9.2-9.3 ms (device profile) | system profile: 287 executions per core in a 2.66 s window, two of four ranks read; device profile: 4 executions, rank 0 | `NIGHT_RECORD` section 5 |
| Executions as share of the window | 2,653 ms of a 2,658 ms span per core | same system profile | same |
| Gap between executions | median 6-9 µs, about 5 ms in total | same | same |
| Collective barrier waits | 9.8 ms in total over the window; collective active time 0.66 ms per step | system profile; device profile (2.65 ms over 4 steps) | `NIGHT_RECORD` section 5; our analysis notes (not published) |
| Wall time per step at 32 live rows | 10.4 ms | one bucket, context 1024 | `NIGHT_RECORD` section 6 (A) |
| Fused attention call, share of instruction time | 18.10 ms of 26.02 ms per step (69.7 %); 28 layers, about 0.65 ms per layer | device profile, instruction durations summed over engines, joined to HLO scopes. Not wall time | our analysis notes (not published); the join script was not saved |
| Other scopes, instruction time per step | sampler 2.62 ms (10.1 %); MLP 1.70 ms (6.5 %); lm_head 0.52 ms (2.0 %); RMSNorm 0.52 ms (2.0 %); not attributed 2.45 ms (9.4 %) | same | same |
| Engine active time per step (4 executions divided by 4) | DMA 5.4 ms; GpSimd 3.5; scalar 3.3; sync 2.9; vector 2.6; tensor (matmul) 2.2 | device profile; engines overlap, so these do not add up to the step | `NIGHT_RECORD` section 5 (totals 21.7 / 14.0 / 13.2 / 11.7 / 10.4 / 8.9 ms over 4 executions, 39.5 ms) |
| DMA share | 21.7 ms of the 39.5 ms window, by plain division 55 % | as above. Our analysis notes (not published) write "21.7 % of window"; the two do not agree and the denominator is not settled. The profiler also warns that dynamic DMA values may be inaccurate | `NIGHT_RECORD` section 5; our analysis notes (not published) |
| Matmul work | transposes 45.2e9 of 92.3e9 hardware flops (49 % by division); matmul engine is the least busy of the six queues | device profile, 4 steps | `NIGHT_RECORD` section 5; our analysis notes (not published) |
| Step against compiled context, 32 live rows | 10.4 / 14.8 / 26.5 ms at 1024 / 4096 / 8192 | one bucket, context ladder off | `NIGHT_RECORD` section 6 (A, C, D) |
| Step against compiled context, 1 live row | 9.3 / 13.7 / 25.2-25.4 ms at 1024 / 4096 / 8192 | same | same |
| Step with default buckets, 1 live row | 4.6 ms (215 tok/s) against 9.3 ms (108 tok/s) with one bucket | context 1024 | `NIGHT_RECORD` section 6 (B, A) |
| Row fill of one group of 32 answers | 0.58-0.63 | same short GSM8K requests | `NIGHT_RECORD` section 6 |

Reading, with limits:

- The device runs back to back, and half of the matmul flops are transposes. The fused attention call holds about 70 %
  of instruction time. These facts point at the decode attention path, but instruction time is not wall time, and the
  wall share of that call is unknown.
- The context slope (+4.4 ms from 1024 to 4096, +11.7 ms from 4096 to 8192 at 32 rows, by subtraction of the table)
  has no explanation yet.
- Per-operation attribution needs the raw profile files. They are not in this fork or the public repo (size); the record gives their
  checksums.
- One run each, one model size, one tensor-parallel layout. None of this is a statement about 8B or about 27-32B.
