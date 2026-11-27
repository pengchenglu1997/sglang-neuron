# Kernels

Design of the NKI kernels for RL post-training on Trainium2 with SGLang-Neuron: why each is built as it is, what was
rejected, and the hardware results behind each claim. The results tables with conditions are in [RESULTS.md](RESULTS.md).

Read this first:

- The LoRA merge kernel (branch `feat/lora-merge-kernel`) ran on hardware once: Qwen3-0.6B in a server, Qwen3-8B shards
  outside a server (RESULTS.md section 8.6). It is not called validated: its acceptance rule is undecided (section 1).
- The sampler kernel (branch `feat/sampler-kernel`) ran on hardware at Qwen3-0.6B and Qwen3-8B for the `temperature`
  sampler (section 2). The same kernel in the `topk_topp` mode is on `integration` and ran at Qwen3-0.6B (section 2,
  RESULTS.md section 10).
- Every number has a condition and a source file. Numbers marked "estimate" are not measurements. A "target" is a
  target, with its basis. Tier of every measured number: `screen` or `single run` ([`rl/METHOD.md`](../../rl/METHOD.md) section 1);
  a table says which.
- Measured results that do not involve a kernel are in [RESULTS.md](RESULTS.md). Sections 3 and 4 record what was
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
the full path. (Source: `rl/docs/NIGHT_RECORD_2026-11-21.md` section 4; tier: single run.)

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
`sglang_neuron/models/rlkit_kernel_path.py`, the kernel branch in `host.py`, and the tests in `tests/rlkit/`. The host
path stays the default.

| Switch | Values | Meaning |
|---|---|---|
| `RLKIT_DEVICE_WRITE=kernel` | unset (host path) or `kernel` | use the kernel path for adapter updates |
| `RLKIT_KERNEL_IMPL` | `auto`, `nki`, `torch`, `sim` | how the kernel runs: NKI on device, the same arithmetic in torch, or the simulator |
| `RLKIT_KERNEL_FACTORS` | `fp32`, `split` | factor dtype; `split` carries fp32 factors as three bf16 parts (hi, hi, lo), less accurate than native fp32 |
| `RLKIT_KERNEL_CHECK` | `first`, `all`, `none` | read the live parameter back after the write: once per call form, shard shape, K and factor dtype, or every call, or never |
| `RLKIT_ADAPTER_ON_DEVICE=1` | on/off | the older eager torch merge on the device (no NKI) |

Guards. A parameter the kernel guard or the layout probe refuses (unaligned shard, K above 128, non-float32 factors)
is merged on the host in the same update. Any failure of the kernel call or of the read-back check redoes the whole
update on the host and turns the kernel mode off for that server. Two changes apply to both modes: adapter coverage is
checked for all parameters before the first write, and adapters that set `use_rslora`, `use_dora`, `rank_pattern` or
`alpha_pattern` are refused (the old code applied one scale `alpha / r` silently).

### Tests off the device

Run all of them with `bash tests/rlkit/run_all.sh` (`tests/rlkit/README.md`) on branch `feat/lora-merge-kernel`. They
check, without a device and without timing:

| Test | What it checks |
|---|---|
| `test_merge_kernel_sim.py` | zero adapter; integer factors; random factors; alternating swaps; K 16, 32, 48; partial tiles; all 0.6B and 8B per-rank shapes; both logical-core settings. Against the float32 reference and against the float64 sum rounded once |
| `test_k_chunks.py` | K above 128 (two accumulating matmuls), partial tiles, guards |
| `test_factor_layout.py` | per-rank factors through the engine's loaders at several TP sizes, every rank, 0.6B and 8B; integer factors bit-exact |
| `negative_control_factor_layout.py` | wrong rank, forgotten scale, summed segments must fail |
| `test_end_to_end_review.py` | loader, then builders, then kernel, against loader(base + full delta in float64) rounded once |
| `test_host_kernel_path.py` | the patched `host.py` end to end on adapter folders on disk; real engine loaders, stubbed engine imports |
| mutation test of the wiring | single planted faults; each fault that changes a result must be caught by a test |

What these tests do not show:

- The output is "one bf16 rounding of an fp32 sum", not "the exact sum rounded once". About 1e-5 to 1e-4 of entries
  differ from the exact rounding, mostly by 1 unit.
- `test_host_kernel_path.py` is not a server run. SGLang is not installed in that environment, so the plugin's own CPU
  mode was not run.
- Not done: a compile-only run with `neuronx-cc`; the call through torch (`wrap_nki`); reading a real PEFT adapter
  (key names, dtypes); a performance number (the simulator is not a timing model).
- Whether a device tensor copied into a live parameter serves correctly is a question for hardware only. The eager torch
  device merge (`RLKIT_ADAPTER_ON_DEVICE=1`) wrote values that differ from the host path by single units in under 0.005 %
  of entries, yet the served output did not equal a restart on the host-merged checkpoint: same greedy tokens on two of
  three test prompts with token logprobs 0.005-0.011 apart on average, and a fork at a near tie on the third
  (`rl/patches/PR_DRAFT_inplace_weight_update.md`). The cause was not found. The kernel's own hardware run (below) shows a
  difference of the same kind.

### Hardware (tier: single run)

Hardware: trn2.48xlarge (Trainium2), NKI 0.7.0, SGLang-Neuron v0.1.0 image. Full tables and conditions: RESULTS.md section 8.6.

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

The kernel runs on the device and is self-consistent. It was not timed at Qwen3-8B in a server (H3), and the
host-merged restart differs from it by an amount that the control puts in the range of one-ulp weight differences.

Open acceptance question: what difference between the kernel server and a restart on the host-merged checkpoint is
accepted? The row "vs a restart on a checkpoint merged by the same kernel: token logprob difference 0.0" below was written
for step H4, which did not run. Without H4 it is undecided whether the 0.007 to 0.008 mean logprob difference comes
only from rounding of the merged weights or also from the in-place write.

### Target and acceptance (proposed, fixed before any hardware run)

| Quantity (Qwen3-8B, TP4, rank 16 on seven projections) | Measured | Target (proposed) |
|---|---|---|
| adapter swap, steady state | 13.9 s (host merge, `rl/data/2026-11-21_night/50_adapter_q8.jsonl`; tier: single run) | at most 1.5 s |
| same adapter swapped back in | n/a | bit-identical weights and outputs |
| vs a restart on a checkpoint merged by the same kernel | n/a | token logprob difference 0.0 |
| decode step before and after a swap | same | same (compile-cache keys unchanged) |

The 1.5 s is a proposed target and an estimate. Its basis is one measured number: 16.4 GB of 8B weights overwritten on
the device in 1.19 s (plain TorchNeuron process, no server; `rl/patches/PR_DRAFT_inplace_weight_update.md`). It is not a
measurement of this kernel.

Baselines that go into every table: full-checkpoint in-place update 4.7-6.2 s at 8B (0.34-0.41 s at 0.6B); host-merge
adapter 13.9 s at 8B (0.67 s at 0.6B); the eager torch device merge already in the patch (about 0.13 s at 0.6B TP4, PR draft;
not measured at 8B); the lazy host delta (`_LowRankSlice`; tested off the device only).

### Hardware steps and stop rules

The H1 script is `tests/rlkit/hw_merge_kernel_bench.py` on branch `feat/lora-merge-kernel`. `rl/50_adapter_server_test.py`
(H2, H3) is an experiment script of `rl/`, not of the engine.

| Step | Script | Question | Stop rule |
|---|---|---|---|
| H1 | `tests/rlkit/hw_merge_kernel_bench.py`, one core | Can the kernel be called on device tensors outside a serving graph (eager, or a small compiled helper)? Time per 8B shard; read back against the reference; integer factors bit-exact; time of the eager torch device merge and of the host merge at the same shapes | exit code 3 means it cannot be called: stop the kernel part and report the torch-device and host numbers only |
| H2 | `rl/50_adapter_server_test.py`, 0.6B TP4, kernel mode, starting with `RLKIT_KERNEL_CHECK=all` | Swap time split (factors, upload, kernels, copy, sync); adapter 1, 2, 1 identical; zero adapter equals base | any check fails: kernel mode stays off |
| H3 | same test at 8B TP4 | The headline number; device memory after the base copy; KV pool size | swap above 1.5 s: try one launch per kind over stacked layers, then report what was measured |
| H4 | restart on the kernel-merged checkpoint | Logprob difference against the swapped server | not 0.0: the difference stays unexplained and is reported so |
| H5 | compile-cache keys and decode step time before and after a swap | Zero forward overhead | none |

H4 tests one explanation of the eager-merge difference above: a restart on the checkpoint the kernel itself wrote must give
0.0. If it does, the earlier difference is the few-entry rounding difference between two merge arithmetics, not a fault
of in-place writing.

### What a write-up may and may not say

May: swap time before and after at 8B with all baselines in one table; no change to the serving graphs; identical
weights when the same adapter returns; the shard-space layout (K on the partition axis of both operands, no transpose on
the device, block-diagonal fused segments); the memory cost of the base copies.

May not: credit the kernel with the whole drop from 13.9 s (most of it comes from not forming full matrices on the
host); call it the first LoRA on Trainium (NxD Inference serves multi-LoRA with torch ops); claim hardware correctness
from simulator runs; call the kernel validated while the acceptance rule in the hardware section above is undecided.

## 2. Sampler kernel

### What it replaces

In `SGLANG_NEURON_SAMPLER=temperature` mode with no top-logprobs, the shard-local stage of the engine's sampler
(`sampling.py`): the per-column hash noise, the Gumbel score, the per-shard maximum, the winner id, the raw logit at the
winner, and the log-sum-exp. One kernel per rank does this over the rank's 37,984 vocabulary columns (151,936 divided by
4; Qwen3-0.6B and 8B have the same shard width at TP4). The kernel views the shard as `[rows x S, 37984 / S]` so that the
rows fill the 128 partitions, walks the columns in chunks of 2,048, and writes 4 partial values per partition and chunk.
The merge of the partials, the raw logit at the winner, the all-gather and the global merge stay the engine's code.
Greedy-only mode and top-logprobs above zero keep the torch path (top-logprobs above zero with the switch on raises an
error). The `topk_topp` mode has a switch of its own (subsection below).

Where it is. Branch `feat/sampler-kernel` of this fork, on top of v0.1.0: `sglang_neuron/sampling.py` (the released file
plus one import-time constant and one guarded branch) and a new file `sglang_neuron/rlkit_sampler_kernel.py`. Switch:
`RLKIT_SAMPLER_KERNEL=1`. Unset or 0, the released code path runs. Kernel sources are part of the compile-cache digest,
so a server with the kernel needs its own compile-cache folder.

### Relevance to a published RL setting

The published RL rollout setting on another accelerator that states its sampling settings, the vLLM TPU RL write-up of
2026-11-12 (https://vllm.ai/blog/2026-11-12-rl-small-dense-tpu-v7), samples with top-k and top-p. In `sampling.py` of SGLang-Neuron
(`origin/main`), `topk_topp` mode is `temperature` plus SGLang's top-k / top-p / min-p filters, applied to a pool of the
global top 256 tokens per row (`TOP_K_MAX`); `temperature` mode does not apply these filters and rejects such requests at
admission. On a device-sampling server, a request with top-k and top-p therefore runs in `topk_topp` mode. The
`temperature` kernel does not run under that setting. The `topk_topp` path with the same kernel (subsection below) is the
one that covers it; the comparison of restricted sampled rows token by token against the released path was not run.

### Hardware results (tier: screen unless stated)

Hardware: trn2.48xlarge (Trainium2), one chip = TP4 on 4 logical cores, SGLang-Neuron v0.1.0 image, NKI 0.7.0. Full
tables, conditions and file names: RESULTS.md section 8.5. Prompt 1024, 256 output tokens, tok/s at
concurrency 1 / 8 / 32, two passes each:

| | Qwen3-0.6B | Qwen3-8B |
|---|---|---|
| released `temperature` sampler | 211 / 1,160 / 2,455 and 211 / 1,171 / 2,454 | 109 / 604 / 1,421 and 110 / 607 / 1,426 |
| sampler kernel | 319 / 1,624 / 2,730 and 319 / 1,636 / 2,734 | 114 / 642 / 1,489 and 114 / 647 / 1,496 |
| released `greedy` sampler (one pass: single run) | 311 / 1,608 / 2,781 | 113 / 639 / 1,533 |

- At 8B the kernel is 4.5 to 6.6 percent above the released `temperature` sampler over the six cells (unrounded values in
  `82_sampler_kernel_r12.md`): not distinguished at screen tier. At 0.6B it is 10 percent or more above at all three concurrencies.
- At concurrency 32 the kernel and the released `greedy` sampler are not distinguished at screen tier (2,730 against 2,781
  at 0.6B; 1,489 against 1,533 at 8B).
- Sampled rollout as the RL loop asks (Qwen3-0.6B, temperature 0.7, 64 questions x 8 answers, up to 256 tokens, three
  passes): released 2,283 / 2,335 / 2,295 tok/s, kernel 2,472 / 2,484 / 2,484; not distinguished at screen tier.
- Combinations at 0.6B (single run): kernel with `--dp-size 4` of TP1 in one server 223 / 1,420 / 3,026 (repeats at
  concurrency 32: 3,039, 3,048, 3,035); kernel with 4 containers of TP1 223 / 1,438 / 3,098 (one measurement). The released
  `temperature` sampler with `--dp-size 4` gave 192 / 1,172 / 2,701.
- Correctness (single run each). 0.6B at temperature 0, 20 prompts: the kernel equals the released `greedy` sampler on 20 of
  20 prompts (largest logprob difference 5.7e-6). The released engine's own `temperature` and `greedy` samplers differ on 3
  of 20 prompts at temperature 0, and the kernel agrees with the `greedy` one, not with the `temperature` one, on those 3.
  At 8B the kernel equals the released `temperature` sampler on 20 of 20 (largest 3.8e-6). At temperature 0.8 with request
  seeds (0.6B, 60 requests of 48 tokens) the kernel and the released sampler give equal whole sequences on 44 of 60. The
  first token of one prompt over 3,000 draws with the same seeds has the same counts for every token.
- Simulator test: `python tests/rlkit_sampler/selftest_sampler_kernel.py`.

Limits:

- the `temperature` sampler only on `main`; Qwen3-0.6B and Qwen3-8B only; nothing at 27 to 32B;
- not run inside a full RL loop (a sampled rollout bench ran);
- the device profiles of the session are not read, so there is no instruction-time figure for the sampler scope with and
  without the kernel.

What remains: a full RL loop with the kernel on; larger models; a proposal upstream (draft pull request 25 upstream
carries the temperature-mode kernel).

### `topk_topp` mode

Branch `feat/sampler-kernel-topk-topp` (on `integration`); switch `RLKIT_SAMPLER_KERNEL_TOPK=1` (unset or 0: the released
function body runs). Tested off the device with `python tests/rlkit_sampler/selftest_sampler_topk.py`; run on a device once,
at Qwen3-0.6B, TP4 (RESULTS.md section 10; tier: single run unless stated there).

Hardware (Qwen3-0.6B, TP4, temperature 0, 20 prompts, batch 1). The first device run, with the vocabulary mask applied
only when a padded column exists, gave other logprobs than the switch off: 0 of 20 first-token logprobs bit-equal (the
version "before the fix"). The kernel path now applies the mask always, as the released path does. The device ran that
change as the same two statements under `if True:`; this exact file text was not itself run. With the mask applied,
kernel on and kernel off are bit-equal on the 20 prompts: first-token logprobs 20 of 20, top-20 vectors 20 of 20, every
generated position 878 of 878, token sequences 20 of 20. Speed with the mask applied at concurrency 32, temperature 0.8,
top-k 50, top-p 0.95 (screen): 2,822.6 and 2,822.4 tok/s with the kernel on, 2,617.6 and 2,616.9 with it off, which is below
10 percent apart and so not distinguished at screen tier (one cold start per server, other jobs on the machine). The cause of
the difference when the mask was skipped was not determined: removing the mask on the released path alone changed nothing.
Not run: temperature above 0 compared token by token, batch above 1 for the comparison, Qwen3-8B, a padded vocabulary on a
device.

What it replaces. Of the shard-local stage of `sample_filtered_sharded_logits` (`sampling.py` v0.1.0, lines 364-378)
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
  logprobs, top logprobs), copied verbatim from v0.1.0 lines 379-408; a test checks the text.

Limits. At most 128 rows per call (the kernel's partitions; it has no row tiling), so a server with the switch on needs
`--max-running-requests` of 128 or less. More rows raise a `RuntimeError` that names `RLKIT_SAMPLER_KERNEL_TOPK`; the
released `topk_topp` path has no such limit. Row keys must lie in [0, 2**31), as `noise_row_keys` makes them.

Consequence. A restricted sampled row (top-k below SGLang's "all" value, top-p below 1 or min-p above 0, temperature
above 0) gets its token and logprob from the released code on the same inputs: the kernel's result is not read for it
(a test puts garbage in place of the kernel's result and these rows stay bit-equal). The kernel's result is read for
temperature-0 rows (max and smallest id: exact) and for unrestricted sampled rows (noise from the kernel's own `log`;
in the simulator that `log` differs from `torch.log` in the last bits for some inputs, and a simulator run can then
draw another token for such a row). On hardware the `temperature` mode gave 44 of 60 equal sequences at the same seeds
(above); that is an observed difference whose cause was not examined, and nothing here explains it.

Tests (`tests/rlkit_sampler/selftest_sampler_topk.py`; method and list in `tests/rlkit_sampler/README.md`): against the
released `topk_topp` path over a simulated TP4 group, bit-equal outputs with the kernel's reference for several top-k,
top-p and min-p settings, temperature-0 rows, mixed per-row settings, ties, several row counts, a padded vocabulary and
non-finite logits; the switch unset leaves the released path and the switch on gives the released output; more rows than
the limit raise the error that names the switch; planted faults are caught. The device comparison is the hardware paragraph above.

Compile flags. `models/host.py` (`neuron_compiler_args`) keeps the default arguments for a decode that traces the device
top-k, which `topk_topp` always does, because `-O2` miscompiles that top-k (section "The compile-flag question" below).
This branch does not change that rule. So the kernel compiles at the default arguments in this mode, a
configuration none of the `temperature` mode's hardware runs used (those were `temperature` decodes, by the code at
`-O2`). The `topk_topp` server of the device run compiled and served at this configuration; how it compares with `-O2`
was not run.

Compile cache. `sampling.py` and `rlkit_sampler_topk.py` are kernel sources of the cache digest (`kernel_sources` follows
the imports of a kernel file), so a server built from this branch gets another digest than one built from
`integration`, with the switch on or off; a cache folder stamped by an `integration` server is refused.

Open after the device run (it showed that the graph loads and serves at Qwen3-0.6B, TP4, and gave the speed cells of
RESULTS.md section 10.3):
- whether the decode graph with the kernel next to the rotational top-k stays correct at other shapes: the engine's
  comment records that a GpSimd top-k hung the GpSimd engine when co-resident with vector work in one NEFF, and this
  kernel uses GpSimd integer multiplies;
- whether restricted rows give the released server's tokens at temperature above 0 (expected by construction; another
  graph may order float work differently);
- any speed that is distinguished at screen tier. In the RL setting every sampled row is restricted, so the kernel's noise
  pass and its log-sum-exp partials are computed and thrown away; the path removes the torch noise pass and the torch
  full-width maximum. The released `topk_topp` sampler's speed is in RESULTS.md section 8.4.

Designs considered and not taken:
- a fused filter kernel (shard top-k by `max8` rounds, cumulative mass by a vector scan, a cut-off value instead of the
  pool sort): more device work removed, but its cumulative sum is not the released matmul bit for bit, so it could not
  give the released tokens at top-p boundaries, and it replaces the rotational top-k, whose tie order the released path
  leaves unspecified;
- a certified unfiltered draw (top-8 candidates per chunk from the kernel, rescored with the released torch ops, plus a
  per-row certificate that the released winner is among them): would make unrestricted sampled rows exact too, needs new
  NKI ops (`max8`, `nc_match_replace8`) and a device bound on the `log` error; a follow-up candidate;
- a runtime skip of the noise pass when every sampled row is restricted (a register-loaded loop bound): simulator-only
  feature, not used here.

### Measured basis (before the kernel existed)

| Quantity | Value | Condition | Source |
|---|---|---|---|
| speed, `temperature` mode | 2,183-2,218 tok/s | Qwen3-0.6B TP4, 256 requests of up to 256 tokens, 32 in flight, three passes, one server (tier: screen) | `NIGHT_RECORD` section 3 |
| speed, `topk_topp` mode, full vocabulary | 2,713-2,730 tok/s | same | same |
| sampler subtree of the decode step | 2.62 ms of 26.0 ms instruction time per step (10.1 %) | device profile of rank 0, 4 decode steps, `temperature` mode with no top logprobs (so, by the code, a decode compiled at `-O2`). Instruction time is summed over engines; it is not wall time (the step is 9.2-9.3 ms) | our analysis notes (not published); the join script was not saved |

Which mode the RL loop used: the launcher `rl/srv_sglang_neuron.sh` defaults to `temperature`, and
`rl/scripts/run_trn2_seeds.sh` sets neither `SGL_SAMPLER` nor `SGL_TOP_LOGPROBS`, so by the scripts the 0.6B loop ran
`temperature` with top logprobs 0. This was not read back from the run's own server log. If the loop ran greedy, this
kernel does not touch the loop.

### The compile-flag question

In `models/host.py` (`neuron_compiler_args`, same code on upstream `main` and in our patch), the decode graph is
compiled at `-O2`, without the modular-flow threshold, except when it traces the device top-k: when
`SGLANG_NEURON_MAX_TOP_LOGPROBS` is above 0, or the mode is `topk_topp`. Then it keeps the default arguments (`-O1` plus
the threshold, by our reading of the code). The code comment says `-O2` miscompiles that top-k.

`rl/58_sampler_probe.py` starts every server with `SGL_TOP_LOGPROBS=20` (its default), so by the code both probe servers
kept the default arguments, and the 20 % gap was measured between two programs compiled with the same flags, both with
the top-logprob path on. Not read back from the server logs.

What follows:

- The probe gap (2,183-2,218 against 2,713-2,730 tok/s) is not explained by the `-O2` flag, as far as the code shows.
- The configuration the RL loop runs and the kernel replaces (`temperature`, top logprobs 0, decode at `-O2`) has
  no measured speed on the probe harness. The 2.62 ms profile figure comes from that configuration; the throughput gap does not.
- The two numbers may still agree by coincidence: one is a wall-time gap, the other is instruction time summed over
  engines.

## 3. Directions examined and not taken

### Live-length decode attention

Idea: make the decode attention read only the live part of the paged KV cache, so a step does not pay for the compiled
context.

Not taken:

- The measured slope is real: with one live row the step is 9.3 / 13.7 / 25.2 ms at compiled context 1024 / 4096 / 8192
  (`NIGHT_RECORD` section 6; tier: single run). But the design's central saving (fewer HBM bytes for dead blocks) is probably already
  in the library kernel. The source of the library's decode attention (nkilib `attention_tkg.py` on its main branch,
  read second-hand through a web summary; the version in SDK 2.33 was not opened) skips out-of-range block entries in
  its indirect loads. What remains is compute and instruction issue over dead tiles, which nobody measured. The cause of
  the context slope is therefore unexplained, not excluded.
- At the RL loop's context of 1024 the bound is about 1.04x of the iteration, by the design's own arithmetic.
- The context ladder (short rows run a smaller-context graph) is expected to remove much of the long-context cost with
  no kernel. That is an estimate; the measurement with the ladder on (step W1.2 of `rl/docs/experiment-plan-2026-11-22.md`) was not run.
- Swapping only the attention core out of the fused attention block loses its fused QKV, RoPE and output projection.
- An upstream author is working on early exit in attention (upstream PRs 4 and 8).

Revisit only if a no-kernel device profile at compiled context 1024 and 8192 (ladder off) shows a large cost per dead
tile, and long-context rollouts become the workload.

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

| Candidate | Source of the idea | What would be checked |
|---|---|---|
| Separate fetch tile and compute tile in the decode attention (tune the DMA tile and the SBUF compute tile independently) | The vLLM TPU write-up for small-dense-model RL (vllm.ai/blog/2026-11-12-rl-small-dense-tpu-v7) reports that decoupling HBM fetch geometry from VMEM compute tiling in its attention kernel raised decode throughput by 48 % at TP4 (64,929 to 96,273 tok/s; page opened 2026-11-23). This is a TPU number reported by the source, not reproduced, and does not transfer as a number. | Whether the library decode kernel's fetch and compute tile sizes can be set separately, and what the K-cache load path costs. Every server log of ours warns that `chunk_num_blocks=4 is not a multiple of 16` and falls back to per-block copy plus transpose for the K cache. The link between that warning and the profile (transposes are 49 % of hardware flops) is not traced |
| Batch small tiles per DMA | AWS Neuron DMA guidance as read in the NKI DMA bandwidth guide and the AWS agent skill pack: about 2 KiB or more contiguous per partition is the recommended minimum and 4 KiB or more saturates; many small DMAs to adjacent addresses are slower than one large one | Our merge kernel moves 128 x 512 bf16 tiles, which is 1 KiB per partition. That is below the 4 KiB guidance, although the whole tile is 128 KiB. Batching four OUT tiles per `dma_copy` would reach 4 KiB. Check on hardware with a device profile of the merge kernel. The same check applies to the KV reads (head dim 128 in bf16 is 256 B per row of one head) |
| Multi-step decode in one compiled loop | The same vLLM TPU write-up moves many decode iterations into one compiled device loop. On Neuron the upstream repository has a whole-model multi-step decode kernel for Qwen3.5 (`qwen35_hybrid_multistep`), but it is greedy argmax only and refuses sampling and logprobs, which RL needs. Dense Qwen3 does not use it | Whether `neuronx-cc` can compile a multi-token decode loop with sampling and logprobs. Our profile shows the device runs back to back (gaps of 6-9 µs between executions), so the gain at 0.6B would not come from removing host gaps |

## 5. Where the time goes

All rows: Qwen3-0.6B, TP4, four logical cores, context 1024, one decode bucket of 32 unless stated. Tier: single run.
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

`NIGHT_RECORD` is [`rl/docs/NIGHT_RECORD_2026-11-21.md`](../../rl/docs/NIGHT_RECORD_2026-11-21.md).

Reading, with limits:

- The device runs back to back, and half of the matmul flops are transposes. The fused attention call holds about 70 %
  of instruction time. These facts point at the decode attention path, but instruction time is not wall time, and the
  wall share of that call is unknown.
- The context slope (+4.4 ms from 1024 to 4096, +11.7 ms from 4096 to 8192 at 32 rows, by subtraction of the table)
  has no explanation.
- Per-operation attribution needs the raw profile files. They are not in this fork (size); the record gives their
  checksums.
- One run each, one model size, one tensor-parallel layout. None of this is a statement about 8B or about 27-32B.
- The LM head's weight layout is a switch on `main`, `SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1` (no kernel of its own;
  RESULTS.md section 9 has its trn2 speed and correctness rows). Device profiles of separate starts with the same switch
  (`lmh_on`, `lmh_off`, in `devprof/` of [`rl/data/2026-11-26_trn2/`](../../rl/data/2026-11-26_trn2/)) were analysed; the profiles
  of the section 9 starts themselves were not. This table has no row for it and does not state whether the per-step
  transpose of the vocabulary shard is gone.
