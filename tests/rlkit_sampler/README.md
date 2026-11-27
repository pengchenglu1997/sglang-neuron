# Sampler kernel for SGLang-Neuron v0.1.0 (temperature sampler; topk_topp mode in the CPU simulator only)

One NKI kernel that replaces the shard-local stage of the engine's temperature sampler. It is off by default. Changed
2026-11-25: it was sent upstream as draft pull request 25 (sgl-project/sglang-neuron) from branch `feat/sampler-kernel`.

Changed 2026-11-26: branch `feat/sampler-kernel-topk-topp` (a `cpu-only` pull request to `integration`) uses the same
kernel, unchanged, in the `topk_topp` mode behind a second switch, `RLKIT_SAMPLER_KERNEL_TOPK=1`. That part has run in
the CPU tests and the NKI simulator only; nothing of it has run on a device. See [topk_topp mode](#topk_topp-mode-cpu-and-simulator-only)
below. Everything else on this page is about the `temperature` mode.

Where it is: written on branch `feat/sampler-kernel`, on top of `4e24326`, the v0.1.0 snapshot (the commit
`feat/inplace-adapter-sync` and `feat/lora-merge-kernel` also start from; the kernel does not depend on either of
them). Since 2026-11-25 it is also in `main` and `integration`. Every measurement below was taken on the
`feat/sampler-kernel` build.

| File | What |
|---|---|
| `sglang_neuron/sampling.py` | The released file plus one import-time constant and one guarded branch (8 added lines, 0 removed). On `feat/sampler-kernel-topk-topp`: a second constant and a second guarded branch, in `sample_filtered_sharded_logits` (14 added lines in all, 0 removed). |
| `sglang_neuron/rlkit_sampler_kernel.py` | New. The NKI kernel, a CPU reference of it, and the function the guarded branch calls. |
| `sglang_neuron/rlkit_sampler_topk.py` | New on `feat/sampler-kernel-topk-topp`. The `topk_topp` function its guarded branch calls (no NKI code of its own). |
| `tests/rlkit_sampler/selftest_sampler_kernel.py` | CPU tests and NKI simulator tests. |
| `tests/rlkit_sampler/selftest_sampler_topk.py` | New on `feat/sampler-kernel-topk-topp`. CPU and NKI simulator tests of the `topk_topp` path. |
| `tests/rlkit_sampler/released_sampling_v0.1.0.py` | The released `sampling.py`, for the selftest where git cannot read this checkout. |
| `tests/rlkit_sampler/hardware/` | The three scripts of the trn2 runs described below, and `85_sampler_topk_hw.py` for the `topk_topp` path (not run). |

## What it replaces

`sampling.py` of v0.1.0 has two sampler modes that matter here: `temperature` and `greedy`. In `temperature` mode each
rank, per decode step, does this on its vocabulary shard `[R, 37984]` (R = rows being sampled; 37,984 = 151,936 / 4 for
Qwen3 at TP4):

- builds the global column ids and masks padding,
- computes a per-column and a per-row hash (integer operations) and turns it into uniform noise,
- takes `-log(-log(u))` of it (Gumbel noise),
- forms the score `logits / T + noise` for sampled rows and `logits` for temperature-0 rows,
- finds the best score and the smallest column id that holds it,
- computes the log-normalizer (`logsumexp`) of the logits.

`greedy` mode does the mask, the max, and the log-normalizer, and none of the noise work. That is the extra work the
kernel targets.

The kernel does all of the above in one pass per chunk, on chip. It views the shard as `[R*S, 37984/S]`, where S is
the largest divisor of 37,984 with `R*S <= 128` (S = 32, 16, 4 at R = 1, 8, 32), so that the rows fill the partition
axis. It works in chunks of 2,048 columns and writes 4 partial values per (partition, chunk): best score, a code for
the smallest id holding it, the chunk maximum of the logits, and the sum of `exp(logits * scale - max)`.

What stays the engine's: the merge of the partials per row (`_max_smallest_id`), the gather of the raw logit at the
winner, the payload, the all-gather, `_shard_winner` and the final logprob (the lines of `sample_sharded_logits` after
the shard-local stage are copied unchanged). `greedy` mode and top logprobs > 0 are not touched by this switch, and
neither is `topk_topp` mode (it has its own switch, below). With the switch on, a request for top logprobs > 0 raises.

The kernel computes the noise itself, with the engine's hash. Passing the engine's noise in from torch would keep most
of the cost the kernel is meant to remove.

## How to switch it on

```
RLKIT_SAMPLER_KERNEL=1
```

in the server's environment, with `SGLANG_NEURON_SAMPLER=temperature`. The value is read once at import. Unset or `0`
runs the released code path. With the switch on, every rank logs a line `[rlkit] sampler by the kernel: rows=...
shard_width=...` when the graph is traced.

The kernel sources are part of the digest of the
compile cache (`backend/compile_cache.py` in v0.1.0). A server with the kernel refuses to start on a folder stamped by
a released server, so give it a compile-cache folder of its own. The runs below used one folder for the kernel server and
separate folders for the released servers.

## What was measured, 2026-11-24

All rows: hardware, trn2.48xlarge (Trainium2), one chip = TP4 on 4 logical cores, unless the row says otherwise.
SGLang-Neuron v0.1.0 image, Neuron SDK 2.33. Server flags of the speed tables, the correctness checks and the rollout
bench: `--context-length 4096 --chunked-prefill-size 2048 --max-running-requests 32 --max-total-tokens 131072
--cuda-graph-bs-decode 1 8 32`. The layout rows use `--tp-size 1 --max-total-tokens 65536 --max-running-requests 8` per
replica (plus `--dp-size 4` in one server), and the row with the image's default flags uses the example script's own
flags (`--tp-size 4 --context-length 4096 --chunked-prefill-size 2048 --max-running-requests 64 --max-total-tokens 266240
--mem-fraction-static 0.4`).

Speed: llm-bench, prompt 1024 tokens, 256 output tokens, 64 requests per cell after 4 warm-up requests. Cells are output
tokens per second at concurrency 1 / 8 / 32. The three servers of a run were up at the same time on three chips; the
benchmark ran against one server at a time. "Pass 1" and "pass 2" are two runs of the same server, one start. The
released greedy sampler was run once per model.

Qwen3-0.6B (`82_sampler_kernel_r10.md`):

| server | pass 1 | pass 2 |
|---|---|---|
| released temperature sampler | 211 / 1,160 / 2,455 | 211 / 1,171 / 2,454 |
| kernel | 319 / 1,624 / 2,730 | 319 / 1,636 / 2,734 |
| released greedy sampler | 311 / 1,608 / 2,781 | not run |

Qwen3-8B (`82_sampler_kernel_r12.md`):

| server | pass 1 | pass 2 |
|---|---|---|
| released temperature sampler | 109 / 604 / 1,421 | 110 / 607 / 1,426 |
| kernel | 114 / 642 / 1,489 | 114 / 647 / 1,496 |
| released greedy sampler | 113 / 639 / 1,533 | not run |

Change of the kernel against the released temperature sampler, computed from the rounded values above, per pass:

| model | concurrency 1 | concurrency 8 | concurrency 32 |
|---|---|---|---|
| Qwen3-0.6B | +51.2 % (211 to 319), +51.2 % | +40.0 % (1,160 to 1,624), +39.7 % (1,171 to 1,636) | +11.2 % (2,455 to 2,730), +11.4 % (2,454 to 2,734) |
| Qwen3-8B | +4.6 % (109 to 114), +3.6 % (110 to 114) | +6.3 % (604 to 642), +6.6 % (607 to 647) | +4.8 % (1,421 to 1,489), +4.9 % (1,426 to 1,496) |

At 0.6B the kernel's numbers are near the released greedy sampler's: +2.6 %, +1.0 %, -1.8 % at concurrency 1 / 8 / 32
(311 / 1,608 / 2,781 against 319 / 1,624 / 2,730; one pass of the greedy server). At 8B the same comparison gives
+0.9 %, +0.5 %, -2.9 % (113 / 639 / 1,533 against 114 / 642 / 1,489).

Sampled rollout as an RL loop asks for it (our rollout client; temperature 0.7; 64 questions x 8 answers each; up to 256
new tokens; Qwen3-0.6B; one server start per arm; three passes), output tokens per second:

| | pass 1 | pass 2 | pass 3 |
|---|---|---|---|
| released temperature sampler | 2,283 | 2,335 | 2,295 |
| kernel | 2,472 | 2,484 | 2,484 |
| pass by pass | +8.3 % | +6.4 % | +8.2 % |

This is generation only. No training step and no weight update ran in it.

Other layouts, Qwen3-0.6B, kernel on, same benchmark, concurrency 1 / 8 / 32 (these are different serving layouts of
one chip, not TP4):

| layout | tokens per second | TTFT p50 at 32 | repeats |
|---|---|---|---|
| kernel, `--dp-size 4` of TP1 in one server | 223 / 1,420 / 3,026 | 0.160 to 0.170 s | concurrency 32 again: 3,039 and 3,048 (with stream interval 4; with in-flight launches 4; neither changed it); 223 / 1,425 / 3,035 alone on the machine |
| kernel, 4 x TP1 containers (sum of four clients) | 223 / 1,438 / 3,098 | 0.138 s | one measurement |
| released temperature sampler, `--dp-size 4` of TP1 in one server | 192 / 1,172 / 2,701 | 0.166 s | one measurement |
| released temperature sampler, 4 x TP1 containers (sum of four clients) | 192 / 1,189 / 2,756 | 0.140 s (median of four clients) | one measurement |

Kernel with the image's default flags, TP4: 321 / 1,629 / 2,744, TTFT p50 0.025 / 0.075 / 0.254 s (one measurement).
Logical-core configuration 1 with 8 replicas was refused by the engine at start (`set_logical_nc_config`; this model's
layout runs 2).

The kernel ran at shard width 37,984 (TP4) and at shard width 151,936 (TP1, in the `--dp-size 4` server; 128
partitions, 128 segments). The flags ask for decode graphs of 1, 8 and 32 rows, and the speed cells use concurrency 1, 8
and 32. The kernel logs once per process, at the first trace, and that line says `rows=1`; the log does not show the
8-row and 32-row graphs being traced.

## Correctness evidence

Files: `82_sampler_kernel_r10.jsonl`, `83_sampler_check2_r11.jsonl`, `82_sampler_kernel_r12.jsonl` (companion repository).
Hardware, trn2.48xlarge, TP4.

- The line `sampler by the kernel` is in the log of every rank of the kernel server and absent from the released one.
- Qwen3-0.6B, temperature 0, 20 prompts, 48 tokens:
  - two released temperature servers (same image, same flags, other chip): 20 of 20 same tokens, logprobs bit for bit
    equal;
  - released temperature against released greedy: 17 of 20 same tokens; the three others fork at token 8, 8 and 16;
    largest logprob difference before a fork 0.103;
  - released temperature against kernel: the same 17 of 20, the same fork positions, largest difference 0.103;
  - kernel against released greedy: 20 of 20 same tokens, largest logprob difference 5.7e-6.
  So at temperature 0 the released engine's own temperature and greedy samplers differ on 3 of 20 prompts, and the
  kernel's output matches the greedy sampler's.
- Qwen3-8B, temperature 0, 20 prompts: kernel against released temperature, 20 of 20 same tokens, largest logprob
  difference 3.8e-6.
- Qwen3-0.6B, temperature 0.8, request seeds, 60 requests of 48 tokens:
  - the released server against itself, same seeds: 60 of 60 same; with other seeds: 1 of 60;
  - two released servers, same seeds: 60 of 60;
  - the kernel server against itself, same seeds: 60 of 60;
  - released against kernel, same seeds: 44 of 60 whole sequences the same; 16 fork later (fork positions 2 to 46);
    mean logprob difference before a fork 0.012.
  So with the kernel on, the sampled tokens are not those of the released sampler for every request.
- Qwen3-0.6B, first token of one prompt, temperature 1.0, 3,000 draws with the same seeds: the kernel and the released
  server give the same count for every token (total variation 0.0). A second draw of the released server with other
  seeds differs from the first by total variation 0.019. Observed counts against the probabilities the server reports:
  chi-square 5.65 on about 7 degrees of freedom (kernel and released server alike).
- Concurrent requests, temperature 0, information only (the two servers may batch the requests differently):
  Qwen3-0.6B 8 of 8 and 28 of 32 same tokens; Qwen3-8B 8 of 8 and 20 of 32.
- The first hardware run (r10) sent its sampled requests without `top_p 1, top_k -1, min_p 0`; the temperature sampler
  refused them (HTTP 400), so the sampled check of that run says nothing. `83_sampler_check2.py` sends them correctly
  and gave the sampled results above.

Tests before the hardware run, `tests/rlkit_sampler/selftest_sampler_kernel.py`, WSL with the NKI 0.7.0 simulator and
CPU torch: 11 of 11 passed (also run again on this branch, same result).
- T1: against the released `sample_sharded_logits` over a simulated TP4 group, same token ids and logprobs within 1e-5,
  rows 1 / 8 / 32, temperatures 0 / 0.6 / 1.0 and mixed.
- T2: per-chunk best score bit-equal to the engine's, noise included.
- T3: ties, constant rows, very negative and -inf logits.
- T4: three planted faults (wrong noise column, wrong row key, no tie rule) are caught.
- T5: `sampling.py` equals the released file plus the guarded branch only.
- SIM: the NKI kernel in the simulator against the CPU reference at 4, 32 and 128 partitions; the noise term of single
  columns is compared with torch and may differ in the last bits (the simulator's `log`).

## Limits

- Temperature sampler only on hardware. `greedy` is untouched, and top logprobs > 0 raises with the switch on. The
  `topk_topp` path has run in the CPU tests and the simulator only (section below).
- Qwen3-0.6B and Qwen3-8B only, TP4 (and TP1 at 0.6B), on one trn2.48xlarge. Other models, other vocabulary sizes and
  padded vocabularies were not run (a padded shard keeps the mask in torch; not run).
- The gain at 8B is small: 4.5 to 6.6 percent over the six cells above (from the unrounded values; the rounded table
  values give 3.6 percent for pass 2 at concurrency 1), 4.8 and 4.9 percent at concurrency 32.
- It was not run inside a full RL loop. The rollout benchmark above is generation only.
- At the same seeds the kernel does not give the released sampler's tokens for every request (44 of 60 sequences equal at
  0.6B).
- Concurrency above 32 was not run.
- Each speed cell is one run of 64 requests. The repeat of the released defaults on this machine agreed within 0.2
  percent in tokens per second at the cells repeated; pass 1 and pass 2 above differ by 0.9 percent in the 0.6B
  released temperature cell at concurrency 8 (1,160 and 1,171).

## Not verified

- Whether the device's `log` is the same function as the one the engine's `torch.log` is lowered to. The sampled
  tokens differ from the released sampler's in 16 of 60 requests, which fits a difference in the last bits of the noise
  at near-ties; no run isolated the cause.
- The integer multiply of the hash and the integer immediates on the device were not tested in isolation. The first-token
  counts of the 3,000 draws agree with the released server's, which is consistent with a correct hash.
- Where the time of the released temperature sampler goes (shard-local passes against fixed per-step costs, such as the
  extra graph inputs of temperature mode). Device profiles of four decode steps (released, kernel, kernel with
  `--dp-size 4`) were captured and have not been read.
- Whether GpSimd integer multiplies cost more at high row counts, and whether the custom call changes how the compiler
  overlaps the sampler with the end of the LM head. Not examined.
- The optimization level of the decode graph with the kernel was not read from the logs of these runs.
- Whether the kernel ran in the 8-row and 32-row graphs. The speed cells at concurrency 8 and 32 differ from the
  released sampler's, and the token comparison under concurrency (above) differs on some requests, but the log shows the
  kernel for one row count only.

## topk_topp mode (CPU and simulator only)

Branch `feat/sampler-kernel-topk-topp`, a `cpu-only` pull request to `integration`. Nothing here has run on a device,
and there is no speed claim. The CPU and simulator results are recorded in the team hub, not here (AGENTS.md rule 6).

Switch: `RLKIT_SAMPLER_KERNEL_TOPK=1` with `SGLANG_NEURON_SAMPLER=topk_topp`, read once at import. Unset or `0` runs the
released function body; `RLKIT_SAMPLER_KERNEL=1` does not reach the `topk_topp` mode.

What it does. The released `sample_filtered_sharded_logits` runs, per rank over `[R, 37984]`: the padded-vocabulary
mask, the Gumbel noise and score, the shard maximum and its smallest id (the unfiltered draw), the raw logit at the
winner, two log-sum-exps and the shard's top 256 (`_pool_topk`); then, after the all-gather, the pool sort, the filter
probabilities, the keep mask and the draw over the kept pool. With the switch on, only the unfiltered draw is replaced:
by `gumbel_partials`, the temperature mode's kernel, called unchanged, and merged with the engine's tie rule as in the
temperature mode. Everything else is the engine's torch code: the log-sum-exps (torch.logsumexp, as released), the pool
top-k, the raw-logit gather, and the lines after the payload, copied verbatim (a test checks the text).

Why only that. The tempered log-sum-exp feeds the probabilities that the top-p and min-p tests compare, so a kernel
exp-sum, which differs from torch in the last bits, could move a row across a top-p boundary. Keeping it in torch means
that a restricted sampled row (top-k below the "all" value, top-p below 1 or min-p above 0) gets its token and logprob
from the released code on the same inputs, whatever the kernel computes. The kernel's result is read only for
temperature-0 rows (a max and a smallest id, exact) and for unrestricted sampled rows (noise from the kernel's own `log`;
the simulator's `log` is not `torch.log` bit for bit, so in the simulator such a row can draw another token). On
hardware the temperature mode gave a difference in whole sequences whose cause was not examined (RESULTS.md); this page
does not explain it. The kernel's log-sum-exp partials are still computed and not used. In the RL setting
(every sampled row restricted) the kernel's noise pass is work whose result is thrown away; removing it needs a
kernel without the noise, or a runtime skip, which this branch does not have.

Limits of this mode: at most 128 rows per call (the kernel has no row tiling), so a server with the switch on needs
`--max-running-requests` of 128 or less; more rows raise a `RuntimeError` naming `RLKIT_SAMPLER_KERNEL_TOPK`, while the
released path has no such limit. Row keys must lie in [0, 2**31), as `noise_row_keys` makes them.

Tests, `tests/rlkit_sampler/selftest_sampler_topk.py` (CPU; `--sim` adds the NKI simulator), against the released
`sample_filtered_sharded_logits` over a simulated TP4 group:
- K0: the switch unset or `0` runs the released body (bit-equal outputs, the new module never imported); `1` routes to
  the kernel path, whose output (CPU reference stage) is bit-equal to the released output.
- K1, K2: with the CPU reference of the kernel, the whole output (token ids, logprobs, top logprobs) is bit-equal for
  top-k 1 / 50 / 256 / off, top-p 0.5 / 0.95 / 1, min-p 0 / 0.1, temperatures 0 / 0.6 / 0.8 / 1.0, rows 1 / 8 / 32,
  mixed per-row settings in one batch, top logprobs 0 and 20, and `SGLANG_RETURN_ORIGINAL_LOGPROB` off and on.
- K3: ties (at the top of the pool, across the top-k cut, equal maxima of greedy rows across ranks), constant rows,
  -inf columns.
- K4: with garbage in place of the kernel's result, restricted sampled rows stay bit-equal and the other rows change.
- K5: planted faults that must be caught (wrong noise column, wrong row key, no tie rule, filters ignored). The noise
  faults are visible only on unrestricted sampled rows, so those batches use them.
- K6: `sampling.py` is the released file plus the two guarded branches; its other functions have the released bytecode;
  the copied tail is the released text.
- K7: `backend/compile_cache.py` counts `rlkit_sampler_topk.py` among the kernel sources.
- K8: a padded vocabulary (width times world above the vocabulary size, large values in the padded columns) and
  non-finite logits (-inf columns, a whole -inf shard, +inf, NaN): the whole output is bit-equal.
- K9: 128 rows run; 129 rows raise an error that names `RLKIT_SAMPLER_KERNEL_TOPK`.
- SIM: the kernel stage is the NKI kernel in `nki.simulate`. Restricted rows must be bit-equal and temperature-0 rows
  must have equal tokens and logprobs, also on the K8 inputs; unrestricted sampled rows are counted, not required.
`selftest_sampler_kernel.py`'s T5 now allows the second guarded branch.

Compile cache. `sampling.py` and the new file are kernel sources in the cache digest (`kernel_sources` follows the
imports of a kernel file), so a server built from this branch has another digest than one built from `integration`,
with the switch on or off. Give it a compile-cache folder of its own.

Compile flags. `models/host.py` keeps the decode graph at the default arguments (not `-O2`) whenever the mode is
`topk_topp`, because `-O2` miscompiles the device top-k; this branch does not change that. The temperature kernel's
hardware runs were in a decode that, by the code, compiled at `-O2`; how the kernel compiles and runs at the default
arguments is not known until run.

Not known until run on a device: whether the decode graph with the kernel and the rotational top-k together loads and
runs (the engine records that a GpSimd top-k hung when co-resident with vector work in one NEFF; this kernel uses GpSimd
integer multiplies); whether restricted rows then give the released server's tokens; any speed. The script
`hardware/85_sampler_topk_hw.py` checks these (not run): released `topk_topp` servers A and A2 (control) against the
kernel server B, temperature 0, the RL setting and a min-p setting with seeds (must match), unfiltered sampled requests
(counted), the kernel's log line, concurrent requests (information only), and an optional llmbench.

## Run it

CPU and simulator test (WSL with a venv that has NKI 0.7.0 and CPU torch; no device):

```
python tests/rlkit_sampler/selftest_sampler_kernel.py --sim
python tests/rlkit_sampler/selftest_sampler_topk.py --sim
```

Without `--sim` the NKI simulator is skipped. The released `sampling.py` is read with `git show v0.1.0:...`; if git
cannot read the checkout, the copy in the tests folder is used.

Hardware scripts, `tests/rlkit_sampler/hardware/`: `82_sampler_kernel_hw.py` (three servers: released temperature,
kernel, released greedy; checks and benchmarks; Qwen3-0.6B), `83_sampler_check2.py` (the second correctness check, with
controls), `84_sampler_kernel_8b.py` (the same as 82 on Qwen3-8B), `85_sampler_topk_hw.py` (the `topk_topp` path, not run yet; it
imports 82 and mounts `patches/sglang_neuron_v0.1.0_sampler_kernel_topk/` holding `sampling.py`,
`rlkit_sampler_kernel.py` and `rlkit_sampler_topk.py`). They are not self-contained. They import the launcher
kit of the companion repository, [pengchenglu1997/trainium-sglang-rl](https://github.com/pengchenglu1997/trainium-sglang-rl)
(`81_opt_sweep.py`, `server_ctl.py`, `rl_common.py`, `rollout_client.py`, `backends.json`, `srv_sglang_neuron.sh`,
`22b_llmbench_grid.sh`, and the scripts these call), which is not copied here. Put them in that kit's folder, with the two
changed package files under `patches/sglang_neuron_v0.1.0_sampler_kernel/` (`sampling.py`, `rlkit_sampler_kernel.py`,
the paths relative to `sglang_neuron/`); `82_sampler_kernel_hw.py` mounts that folder over the installed package.
