# Results

Public claims of the RL post-training work on Trainium2 with SGLang-Neuron, and the file each number comes from.
The headline numbers and what they are compared with: the block in section 9, printed by `python rl/tools/best.py`.

How to read this file:

- Every number has a tier and, except where the text says its source is not published, a source file under [`rl/data/`](../../rl/data/) (rules: [`rl/METHOD.md`](../../rl/METHOD.md) section 1).
  `screen` is at least 2 passes on one server start per arm, in one session. `once` is one pass or one start: a
  record, not a claim. A difference of less than 10 percent is written "not distinguished at screen tier".
- A row names the model, the layout, what was measured, the number, the condition and the source file. Nothing here is a
  cost-normalised comparison.
- Dates are Seattle local dates of the run. Conditions of each session: the README of its folder under `rl/data/`.
  A bare file name in sections 8 and 9 is a file of the folder named at the top of that section.
- TP4 is tensor parallel over 4 NeuronCores (one rollout server). "Layout" names the machine layout and the sync path.
- H1 to H5 are the hardware steps of the LoRA merge kernel ([KERNELS.md](KERNELS.md) section 1). `rNN` in a file name
  (`82_sampler_kernel_r12.md`) is the run that wrote the file; the scripts are indexed in [`rl/docs/EXPERIMENTS.md`](../../rl/docs/EXPERIMENTS.md).
- Server logs and raw profiles of the 2026-11-21 session are not published (size);
  [`rl/docs/NIGHT_RECORD_2026-11-21.md`](../../rl/docs/NIGHT_RECORD_2026-11-21.md) gives their checksums.

## 1. RL loop at Qwen3-0.6B

Loop: `rl/30_rl_loop.py`, GRPO-style policy gradient with a LoRA trainer, GSM8K. Layout for the trn2 rows: one trn2.48xlarge,
8 rollout servers at TP4, a 16-process data-parallel trainer, adapter sync through `update_weights_from_disk` into the
patched server (in place), 32 questions x 8 answers per step, up to 256 new tokens, temperature 0.7, learning rate 1e-4,
150 steps, not overlapped. Result files `rl/data/2026-11-20/trn2_seeds/30_rl_loop_plain_s{2,3,4}.jsonl`, run 2026-11-20.
Tier: once per seed (three seeds). Figures are medians over the 150 step rows of each seed.

| What | Seed 2 | Seed 3 | Seed 4 | Note |
|---|---|---|---|---|
| step wall time | 5.33 s | 5.31 s | 5.32 s | means over all 150 rows are 6.89 / 5.72 / 5.74 s because step 1 compiles the trainer (step 1 train time 217.5 / 45.3 / 46.2 s); means over steps 2-150, as the files' summary lines give them, are 5.40 / 5.43 / 5.44 s |
| rollout | 2.70 s | 2.70 s | 2.70 s | about 51 % of the step |
| train (16 processes) | 1.13 s | 1.15 s | 1.13 s | about 21 % of the step |
| sync (in-place adapter update, fan-out to 8 servers) | 1.22 s | 1.21 s | 1.21 s | about 23 % of the step |
| export | 0.08 s | 0.08 s | 0.08 s | |
| reward (string match) | 0.001 s | 0.002 s | 0.002 s | |
| greedy accuracy, 500 GSM8K questions, before | 0.572 | 0.574 | 0.572 | |
| greedy accuracy, after step 150 | 0.712 | 0.728 | 0.702 | evaluated every 25 steps; the seven values per seed are in the files |

Other rows from the same loop files and the plan (tier: once each):

| Model | Layout | What | Number | Condition | Date | Source |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | one patched server | adapter update in place | 0.67-0.70 s after a first update of 7-9 s | rank-16 LoRA on 7 projections, TP4 | 2026-11-20 | `rl/patches/PR_DRAFT_inplace_weight_update.md`; `rl/docs/experiment-plan-2026-11-22.md` section 3 |
| Qwen3-0.6B | one patched server | full-checkpoint update in place | 0.34-0.41 s | TP4 | 2026-11-20 | same |
| Qwen3-0.6B | one RTX PRO 6000, standard SGLang, one replica, merged-weight sync | RL step, 150 steps, seed 1 | 6.69 s wall (6.40 s summed parts), mean of steps 2-150; accuracy 0.574 to 0.714 | sync path, trainer and layout differ from the trn2 rows: not a like-for-like comparison | 2026-11-20 | `rl/data/2026-11-20/cuda/30_rl_loop_cuda_long_b4.jsonl` (summary line) |
| Qwen3-0.6B | trn2, 8 replicas, earlier trainer (before shared memory), seed 1 | RL step, 150 steps | 6.83 s summed parts (no wall figure in the file); accuracy 0.574 to 0.748 | the fourth 150-step run; train time 2.83 s against 1.2 s in the seed runs | 2026-11-20 | `rl/data/2026-11-20/trn2_sgl_final/30_rl_loop_long1.jsonl` |

Limits: the RL loop was run on Qwen3-0.6B only. The four 150-step trn2 runs ended at 0.702 / 0.712 / 0.728 / 0.748.
Sampler: by the scripts (`rl/srv_sglang_neuron.sh` defaults to `temperature`; `rl/scripts/run_trn2_seeds.sh` sets neither
`SGL_SAMPLER` nor `SGL_TOP_LOGPROBS`) the loop ran `temperature` with no top logprobs; not read back from the run's server logs.

## 2. In-place update across model families

Our in-place update patch (branch `feat/inplace-adapter-sync` of this fork, on SGLang-Neuron v0.1.0), trn2, TP4, context 1024,
one decode bucket of 32. Script `rl/48_inplace_server_test.py`, result files `rl/data/2026-11-21_night/48_inplace_*.jsonl`, run
2026-11-21. The test compares the server after an in-place update with a fresh start on the same weights, by text and
token logprobs. Tier: once. Source of the table: `rl/docs/NIGHT_RECORD_2026-11-21.md` section 2.

| Model | Graph | Live tensors covered | In-place full update | Back to base | In place vs fresh start |
|---|---|---|---|---|---|
| Qwen3-8B | shared dense transformer | 291 parameters, 1 buffer (`rotary_emb.inv_freq`) | 4.7-6.2 s (12 updates, two servers) | same text, logprob difference 0.0 | same text, logprob difference 0.0 |
| GLM-4-9B-0414 | shared dense transformer | 363 parameters, 1 buffer | 4.9 s (2 updates) | same text, 0.0 | same text, 0.0 |
| DeepSeek-V2-Lite | its own graph (MLA, MoE) | 351 parameters, 1 buffer | 10.3-10.6 s (2 updates) | same text, 0.0 | same text, 0.0 |

- The patch was not changed for GLM-4 or DeepSeek-V2. In all three the only buffer is `inv_freq`, equal in the staged
  and the live graph, so walking `named_parameters()` covers every weight of these families.
- GLM-4: the perturbed checkpoint changed the logprobs (largest difference 0.087) but not the 48-token texts.
- DeepSeek-V2-Lite: the first attempt died at start on a compile-cache race, not on a refusal. A start with the
  example's own flags (context 4096) was still compiling after one hour and was stopped by hand; the reduced shapes
  were ready in 953 s.
- Not covered: Qwen3-MoE, GPT-OSS, GLM-5.2, Qwen3.5, Kimi-K3 (kernel models with packed or quantized weights).
- Qwen3-8B where the time goes (same record, section 4): per rank, host graph built in 0.00-0.02 s, loaders 3.6-4.7 s,
  copy to the device 0.3-1.1 s. Keeping the host graph between updates changes nothing.

## 3. Adapter update at Qwen3-8B

Script `rl/50_adapter_server_test.py`, result file `rl/data/2026-11-21_night/50_adapter_q8.jsonl`, run 2026-11-21. Qwen3-8B,
TP4, rank 16 on seven projections, host-merge path (no kernel). Tier: once. Source: `rl/docs/NIGHT_RECORD_2026-11-21.md` section 4.

| What | Number | Note |
|---|---|---|
| first adapter update | 35.8 s | includes first-use cost |
| later updates | 13.9-14.0 s | 13.5-13.6 s forming the deltas on the host, 0.3 s copy |
| correctness | equal to a fresh start on the merged checkpoint: same text, logprob difference 0.0 | adapters replace each other; the zero adapter gives the base back |
| comparison | full-checkpoint update at 8B takes 5-6 s | so at 8B the adapter path is slower than the full path |

The shard-only adapter delta (`_LowRankSlice`) and the LoRA merge kernel aim at this number. The kernel ran at Qwen3-0.6B
in a server (section 8.6); at Qwen3-8B it was not timed in a server, and the shard-only delta did not run on hardware
([KERNELS.md](KERNELS.md)).

## 4. Sampler modes

Qwen3-0.6B, TP4, `rl/58_sampler_probe.py`, result files `rl/data/2026-11-21_night/58_sampler_{s1,b1}.jsonl`, run 2026-11-21.
Speed: 256 requests of up to 256 tokens, 32 in flight, three passes, one server per mode. Tier: screen.

| Mode | Setting | Speed | HTTP result |
|---|---|---|---|
| `topk_topp` | full vocabulary | 2,713-2,730 tok/s | 200 |
| `topk_topp` | top_p 0.95, top_k 20 | 2,700-2,764 tok/s | 200 for all five tested (top_p, top_k) settings |
| `temperature` | full vocabulary | 2,183-2,218 tok/s | 200 |
| `temperature` | any of four filtered settings | not applicable | 400 (message: use a `topk_topp` server for filtered sampling) |

Top-p cut: 8 tests on 4 prompts, 400 one-token samples each at temperature 1, with top_p set inside the mass of the token
of rank j. In all 8 the number of distinct tokens returned was j: the token that crosses p is kept, as in vLLM and
standard SGLang. Source: `rl/docs/NIGHT_RECORD_2026-11-21.md` section 3.

Caveat: the probe started both servers with `SGL_TOP_LOGPROBS=20` (the default of `58_sampler_probe.py`). With top
logprobs above 0, `neuron_compiler_args` in `models/host.py` keeps the default compiler arguments for both modes, so
by the code the two probe servers were compiled alike. The RL loop's `temperature` server has top logprobs 0, and in
that case the decode compiles at `-O2`. So the probe's `temperature` row is not the configuration the loop runs, and
the loop's configuration has no measured speed here ([KERNELS.md](KERNELS.md) section 2). Not read back from server logs.

## 5. Decode profile

Qwen3-0.6B, TP4, 32 requests of up to 256 tokens, run 2026-11-21. Sampler `temperature` with no top logprobs (the
launcher defaults; `59_decode_profile.py` sets neither), so by the code the decode was compiled at `-O2`; not read back
from the server log. Tier: once. Source: `rl/docs/NIGHT_RECORD_2026-11-21.md` section 5; files
`59_decode_profile_p1.jsonl`, `59b_profile_sum.txt`, `59d_devprof_sum.txt` in `rl/data/2026-11-21_night/`.

| What | Number | Condition |
|---|---|---|
| pass wall time | 2.67 s with and without the profile window | |
| decode executions | 287 per physical core in a 2.66 s window; median 9.13 ms; sum 2,653 ms of a 2,658 ms span | system profile, two of four ranks read |
| gap between executions | median 6-9 µs, about 5 ms in total | same |
| collective barrier waits | 9.8 ms in total | same |
| device profile, one execution | 9.2-9.3 ms | rank 0, 4 decode steps |
| hardware flops / model flops / transpose flops | 92.3e9 / 48.4e9 / 45.2e9 | 4 steps |
| MFU / MBU as printed by the profiler | 0.0031 / 0.091 | unit (percent or fraction) not confirmed |
| active time over 4 executions (39.5 ms) | DMA 21.7 ms, GpSimd 14.0, scalar 13.2, sync 11.7, vector 10.4, tensor (matmul) 8.9 | engines overlap |

The profiler warns that dynamic DMA values may be inaccurate without DGE notifications, and the compiled program has no
compiler metrics. Per-operation attribution is not in the record; our computation from the raw files is in
[KERNELS.md](KERNELS.md) section 5, with its limits.

## 6. Shape cost

Qwen3-0.6B, TP4, context ladder off, `rl/60_shape_cost.py`, files `rl/data/2026-11-21_night/60_shape_cost_{A,B,C,D}.jsonl`, run
2026-11-21. Step = wall time divided by the longest answer. Same GSM8K questions at every level; answers up to 256 tokens.
Tier: once. Source: `rl/docs/NIGHT_RECORD_2026-11-21.md` section 6.

| Concurrent requests | A: one bucket (32), context 1024 | B: default buckets, context 1024 | C: one bucket, context 4096 | D: one bucket, context 8192 |
|---|---|---|---|---|
| 32 | 10.4 ms | 9.3 | 14.8 | 26.5 |
| 16 | 9.8 | 7.1 | 14.2 | 25.9 |
| 8 | 9.5 | 5.8 | 13.9 | 25.5 |
| 4 | 9.3 | 5.1 | 13.7 | 25.4 |
| 2 | 9.3 | 5.0 | 13.6 | 25.3 |
| 1 | 9.3 | 4.6 | 13.7 | 25.2-25.4 |

- With one bucket a step costs the same whether 1 or 32 rows are live. With the default buckets one request runs at
  4.6 ms a step (215 tok/s) against 9.3 ms (108 tok/s).
- The step grows with the compiled context although the requests are the same short ones: 10.4 / 14.8 / 26.5 ms at
  1024 / 4096 / 8192.
- Row fill of one group of 32 answers: 0.58-0.63 (tokens divided by 32 times the longest answer). A single group runs at
  1,790-2,030 tok/s; 256 queued requests at 2,713-2,730 tok/s.
- Server ready time: A 109 s, B 350 s, C 215 s, D 230 s (the cache already held some shapes).

## 7. Tests off the device

No hardware. Public text names a test by its command and quotes no quantity from it.

| Suite | Command | Branch |
|---|---|---|
| LoRA merge kernel in the NKI simulator, factor layout through the engine's loaders, patched `host.py` end to end, mutation test of the wiring | `bash tests/rlkit/run_all.sh` (`tests/rlkit/README.md`) | `feat/lora-merge-kernel` |
| sampler kernel in the NKI simulator | `python tests/rlkit_sampler/selftest_sampler_kernel.py` | `feat/sampler-kernel` |
| top-k sampler kernel path | `python tests/rlkit_sampler/selftest_sampler_topk.py` | `integration` |

Not covered by these tests: a compile with `neuronx-cc`, any device run, any timing, a real PEFT adapter, the plugin's
own CPU mode. The simulator is not a timing model. What each LoRA merge test checks: the tests table of [KERNELS.md](KERNELS.md).

## 8. trn2 session of 2026-11-24

Files: [`rl/data/2026-11-24_trn2/`](../../rl/data/2026-11-24_trn2/); its README maps each file to a subsection here.
Hardware for every row of this section: trn2.48xlarge spot (Trainium2), one chip = TP4 on 4 logical cores, unless a row
names another layout. Images: `pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04` and
`pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.1-ubuntu26.04`; NKI 0.7.0 in the SGLang image. In the
SGLang image `general_plugin.py`, `models/host.py` and `models/transformer/model.py` have the md5 of tag v0.1.0;
`backend/compile_cache.py` differs from the tag in one comment line only. Not published: per-request benchmark files,
server logs, `.out` files and device profiles. The probe scripts 73 to 81 are at the top level of `rl/`; the patch folders
they mount are branches of this fork.

Run conditions that bear on the numbers:

- Many runs went on at the same time on other chips of the same machine. Repeats on a quieter machine agreed within 1
  percent (released defaults, greedy sampler, sampler kernel with `--dp-size 4`).
- Every throughput cell is one run of 64 requests on one server start, 256 output tokens, unless the row says more (tier:
  once). Throughput is total output tokens over the wall time of the batch (llmbench at commit 08f4163, column
  `output_token_throughput_tps`). TTFT is the median time to the first streamed chunk.
- The first hardware test of the sampler kernel sent sampled requests without `top_p 1, top_k -1, min_p 0`. The
  `temperature` sampler refused them with HTTP 400. The check script was changed to send them; the results below are from
  the corrected script.

### 8.1 Upstream PR 20 on its own branch

Hardware, trn2.48xlarge, TP4. Three files of branch `upstream-pr/20` (`general_plugin.py`, `models/host.py`,
`models/transformer/model.py`) over the v0.1.0 image; the md5 of each file is recorded in each result file. Launch flags:
`--context-length 1024 --chunked-prefill-size 1024 --max-running-requests 32 --max-total-tokens 32768
--cuda-graph-bs-decode 32`. The check compares answers and token logprobs of 20 prompts, 48 new tokens each (960 tokens).
Run 2026-11-24. Files `75_pr20_check_pr20_q06.jsonl`, `75_pr20_check_pr20_q8.jsonl`. Tier: once.

| What | Qwen3-0.6B | Qwen3-8B |
|---|---|---|
| in-place update to the other checkpoint, against a fresh start on that checkpoint | identical; largest logprob difference 0.0 | identical; 0.0 |
| in-place update back, against the first start | identical; 0.0 | identical; 0.0 |
| update time | 0.23 to 0.26 s (12 updates) | 5.0 s and 5.45 s (two updates) |
| first start / restart on the other checkpoint | 265 s / 139 s | 360 s / 164 s |

Smoke test at 0.6B, not a load test: 10 updates in a row all succeeded. One update with 8 short requests in flight gave 0
HTTP errors, took 0.59 s, and the answers after it equal the reference of the new weights. Not run: 100 updates in a row,
an update under a sustained load. Radix cache: only an indirect check ran (the first prompt, cached before the update,
gave the fresh-start answer after it, difference 0.0); this shows no stale entry was served for that prompt, not that the
cache was flushed. These results were posted to upstream pull request 20 on 2026-11-25 (Seattle time) as a
[comment](https://github.com/sgl-project/sglang-neuron/pull/20#issuecomment-7182387872).

### 8.2 Compile-cache stamp race

Hardware, trn2.48xlarge. Two containers started together on one empty shared compile cache, 10 trials per arm. Run
2026-11-24. File `74_stamp_race_r1.jsonl`. The fix is the branch `fix/compile-cache-stamp-race` (upstream pull request 22).

| Arm | Trials that failed |
|---|---|
| released code (v0.1.0) | 1 of 10 |
| with the fix | 0 of 10 |

In the failed trial the second container exited with code 1 and `FileNotFoundError: '/cache/neff/.KERNEL_SOURCES.195' ->
'/cache/neff/KERNEL_SOURCES'` at line 196 of `compile_cache.py`. Process id 195 existed in both containers (ids 192 to 197 and
193 to 198). The process-id ranges of the two containers overlapped in 10 of 10 trials of both arms. With a failure rate
near 1 in 10, 0 of 10 does not prove the fix by itself. The fix removes the equal temporary name by construction.

### 8.3 Baseline pair: vllm-neuron and SGLang-Neuron

Hardware, trn2.48xlarge, one chip, TP4. Qwen3-0.6B, prompt 1024 tokens, 256 output tokens, 64 requests per cell, one run
per cell. Cell: tok/s / TTFT p50. Run 2026-11-24. Tier: once, except the repeat pass named below (screen).

- vllm-neuron 0.29.0.1.2.0: TP4, max-model-len 4096, max-num-seqs 32.
- SGLang-Neuron v0.1.0 released defaults: the image's example script `qwen3_0p6b/run_server.sh` with its own flags
  (`--tp-size 4 --context-length 4096 --chunked-prefill-size 2048 --max-running-requests 64 --max-total-tokens 266240
  --mem-fraction-static 0.4`), sampler `temperature`.

| Concurrency | vllm-neuron 0.29.0.1.2.0 | SGLang-Neuron v0.1.0 |
|---|---|---|
| 1 | 209.9 / 0.021 s | 211.1 / 0.025 s |
| 8 | 1,230.3 / 0.076 s | 1,168.0 / 0.075 s |
| 32 | 2,375.3 / 0.276 s | 2,460.6 / 0.260 s |

The full grid has 18 cells per engine (prompts 128, 1024 and 3072; concurrency 1, 2, 4, 8, 16, 32). It is in
`77_pair_q06.md`; every cell there was run once except the repeat rows of the 1024 prompt. Repeat pass of the 1024-token
row: tok/s within 0.2 percent for both engines (at concurrency 32: 2,375.3 and 2,372.2 for vllm-neuron, 2,460.6 and
2,456.7 for SGLang-Neuron). SGLang-Neuron released defaults measured again later in the session: 211 / 1,161 / 2,459 tok/s
(`77_pair_q06d.md`). The order of the two engines changes with concurrency in this table, and the two engines are not
distinguished at screen tier at concurrency 32.

Qwen3-8B, prompt 1024, 256 output tokens, SGLang-Neuron v0.1.0 released defaults (file `77_pair_q8.md`; tier: once
with a repeat pass):

| Concurrency | tok/s | TTFT p50 | Repeat pass tok/s |
|---|---|---|---|
| 1 | 109.3 | 0.067 s | 109.7 |
| 8 | 604.8 | 0.211 s | 606.8 |
| 32 | 1,419.7 | 0.744 s | 1,425.9 |

vllm-neuron on Qwen3-8B did not start in two attempts on 2026-11-24 (TP4, max-model-len 4096, container). The compiler
`neuronx-cc` exited with code 70 and the error `NCC_EVRF059`: the kernel file of `vendored_kernels.rotational_topk` "does
not exist on the host". The start tests of 2026-11-26 showed that a stale NKI compile-result index in a reused cache folder
produces this error (`i12_vllm_start/`, `i34_nki_index/` in [`rl/data/2026-11-26_trn2/`](../../rl/data/2026-11-26_trn2/)); whether
it caused the 2026-11-24 failures was not checked. On an empty cache folder vllm-neuron ran the same benchmark:
`rl/data/2026-11-26_trn2/serving/llmbench_b_q8_vllm_p{1,2}.csv`, listed with its tier in section 9.

### 8.4 Serving changes, Qwen3-0.6B

Hardware, trn2.48xlarge, Qwen3-0.6B, prompt 1024, 256 output tokens, run 2026-11-24. Cell: tok/s at concurrency 1 / 8 /
32, one run per cell unless the row says more (tier: once). "Text" is the answer to five fixed prompts at temperature 0 (48 tokens)
against the released-defaults server. Files `opt_sweep_r9.md`, `81_opt_sweep_r9.jsonl`.

Changes that moved the numbers:

| Change | tok/s at 1 / 8 / 32 | TTFT p50 at 32 | Text |
|---|---|---|---|
| released defaults (reference) | 211 / 1,168 / 2,461 | 0.260 s | reference |
| sampler `greedy` | 312 / 1,609 / 2,782 (second run: 312 / 1,610 / 2,783) | 0.259 s | same on 5 of 5 |
| sampler `topk_topp` (decode compiled with the default arguments) | 218 / 1,266 / 2,607 | 0.271 s | same on 5 of 5 |
| `--dp-size 4` of TP1 in one server, on the same four cores | 192 / 1,172 / 2,701 | 0.166 s | same on 5 of 5 |
| 4 containers of TP1 on the same four cores (sum of four clients) | 192 / 1,189 / 2,756 | 0.140 s (median of four) | same on 5 of 5 |
| 2 containers of TP2 (sum of two clients) | 129 / 461 / 1,513 | 0.203 s | differs on 1 of 5 |

At concurrency 32 the rows `greedy` (2,782), 4 containers of TP1 (2,756) and 2 containers of TP2 (1,513) differ from the
reference by 10 percent or more; `topk_topp` (2,607) and `--dp-size 4` (2,701) are not distinguished from it at single-run
tier. The concurrency-1 cell of a replica layout is one replica alone, which is why it is lower than the TP4 cell.

Not distinguished from the released defaults (once per cell, below screen tier):

| Change | tok/s at 1 / 8 / 32 | TTFT p50 at 32 | Text |
|---|---|---|---|
| `--chunked-prefill-size 1024` | 211 / 1,162 / 2,459; prompt 3072 at concurrency 1 and 32: 202.3 and 1,972.9 against 203.6 and 1,974.8 | 0.267 s | same on 5 of 5 |
| `NEURON_RT_XU_WORKER_ENABLE_LOW_LATENCY_COMPLETION=1` (platform default 0) | 211 / 1,160 / 2,456 | 0.261 s | same on 5 of 5 |

Two passes on one server start of the released defaults differ by 0.2 percent or less (section 8.3). A second start
later in the session gave 211 / 1,161 / 2,459 (`77_pair_q06d.md`).

Not ready within 30 minutes, cause not read: greedy sampler with `--dp-size 4`, greedy sampler with 4 containers. Not run:
chunk size 512, in-flight launches alone, and the controls that run the same graph as the released defaults. Replica cells
sum one client process per replica; the processes start together and are not time-aligned beyond that.

### 8.5 Sampler kernel

Hardware, trn2.48xlarge, one chip, TP4, run 2026-11-24. One NKI kernel replaces the shard-local stage of the
`temperature` sampler (the per-column hash noise, the Gumbel score, the per-shard maximum and winner, and the
log-sum-exp). The all-gather and the global merge stay the engine's code. Switch `RLKIT_SAMPLER_KERNEL=1`; unset, the
released code path runs. Branch `feat/sampler-kernel` (design: section 2 of [KERNELS.md](KERNELS.md)); its test off the
device is in section 7.

Speed, prompt 1024, 256 output tokens, tok/s at concurrency 1 / 8 / 32, two passes each unless stated (tier: screen). Files
`82_sampler_kernel_r10.md`, `82_sampler_kernel_r12.md`.

| Server | Qwen3-0.6B | Qwen3-8B |
|---|---|---|
| released `temperature` sampler | 211 / 1,160 / 2,455 and 211 / 1,171 / 2,454 | 109 / 604 / 1,421 and 110 / 607 / 1,426 |
| sampler kernel | 319 / 1,624 / 2,730 and 319 / 1,636 / 2,734 | 114 / 642 / 1,489 and 114 / 647 / 1,496 |
| released `greedy` sampler (one pass: once) | 311 / 1,608 / 2,781 | 113 / 639 / 1,533 |

At 0.6B the kernel is 10 percent or more above the released `temperature` sampler at all three concurrencies (at 32: 2,730
and 2,734 against 2,455 and 2,454). At 8B the kernel is 4.5 to 6.6 percent above over the six cells of the table (from the
unrounded values in `82_sampler_kernel_r12.md`), which is not distinguished at screen tier. At concurrency 32 the kernel and
the released `greedy` sampler are not distinguished at screen tier (2,730 against 2,781 at 0.6B; 1,489 against 1,533 at 8B).

Sampled rollout as the RL loop asks for it: Qwen3-0.6B, temperature 0.7, 64 questions with 8 answers each, up to 256 new
tokens, three passes (file `55_rollout_bench_roll_*.jsonl`; tier: screen).

| Server | tok/s in the three passes |
|---|---|
| released `temperature` sampler | 2,283 / 2,335 / 2,295 |
| sampler kernel | 2,472 / 2,484 / 2,484 |

Pass by pass the kernel is 8.3, 6.4 and 8.2 percent above: not distinguished at screen tier. The rollout is not a full RL loop.

Combinations at Qwen3-0.6B (files `opt_sweep_r13.md`, `opt_sweep_r15.md`, `opt_sweep_r16.md`; tier: once):

| Server | tok/s at 1 / 8 / 32 | TTFT p50 at 32 |
|---|---|---|
| kernel, released default flags | 321 / 1,629 / 2,744 (TTFT 0.025 / 0.075 / 0.254 s) | 0.254 s |
| kernel and `--dp-size 4` of TP1 in one server | 223 / 1,420 / 3,026 | 0.160 to 0.170 s |
| the same, repeated | 3,039 and 3,048 at concurrency 32 (with `--stream-interval 4`; with 4 in-flight launches; neither changed the result), and 223 / 1,425 / 3,035 with the machine otherwise idle | |
| kernel and 4 containers of TP1 (sum of four clients, one measurement) | 223 / 1,438 / 3,098 | 0.138 s |

For comparison, the released `temperature` sampler with `--dp-size 4` gave 192 / 1,172 / 2,701 (section 8.4), and with 4
containers of TP1 192 / 1,189 / 2,756. Logical-core configuration 1 with 8 replicas on one chip was refused by the engine
at start (`set_logical_nc_config`; this model's layout runs with 2).
Correctness evidence (files `82_sampler_kernel_r10.jsonl`, `82_sampler_kernel_r12.jsonl`, `83_sampler_check2_r11.jsonl`; once each):

- The log line "sampler by the kernel" is in the log of every rank of the kernel server and absent from the released one.
- Qwen3-0.6B, temperature 0, 20 prompts. Two released `temperature` servers agree bit for bit. The released `temperature`
  sampler against the released `greedy` sampler: 17 of 20 prompts give the same tokens, largest logprob difference before
  a fork 0.103. The released `temperature` sampler against the kernel: the same 17 of 20, the same fork positions, 0.103.
  The kernel against the released `greedy` sampler: 20 of 20, largest logprob difference 5.7e-6. So the released engine's
  `temperature` and `greedy` samplers differ at temperature 0 on 3 of 20 prompts, and the kernel output matches the
  `greedy` one.
- Qwen3-0.6B, temperature 0.8, request seeds, 60 requests of 48 tokens. The same server twice: 60 of 60 equal. Other
  seeds: 1 of 60 equal. Two released servers: 60 of 60. Released against kernel: 44 of 60 whole sequences equal; 16 fork
  later; mean logprob difference before a fork 0.012. The cause of the forks was not examined.
- Qwen3-0.6B, first token of one prompt, temperature 1.0, 3,000 draws with the same seeds. Kernel and released give the
  same count for every token (total variation 0.0). A second draw of the released server with other seeds differs from
  the first by 0.019. Observed counts against the probabilities the server reports: chi-square 5.65 on about 7 degrees of
  freedom.
- Qwen3-8B, temperature 0, 20 prompts: the kernel equals the released `temperature` sampler on 20 of 20, largest logprob
  difference 3.8e-6.
- Concurrent requests, for information only (the two servers may batch the requests differently): at 0.6B 8 of 8 and 28
  of 32 same tokens, at 8B 8 of 8 and 20 of 32.

Limits: the `temperature` sampler only (not `topk_topp`; top logprobs above zero is refused with the switch on); Qwen3-0.6B
and Qwen3-8B only; not run inside a full RL loop; the released and kernel sequences are not equal at temperature 0.8 for
16 of 60 requests.

### 8.6 LoRA merge kernel, first hardware run

Hardware, trn2.48xlarge, run 2026-11-24. The kernel is on branch `feat/lora-merge-kernel`, run through the patched server
with `RLKIT_DEVICE_WRITE=kernel`, `RLKIT_KERNEL_IMPL=nki`, `RLKIT_KERNEL_CHECK=all`, fp32 factors. NKI 0.7.0. The
step-1 bench ran on one core of the same NUMA node as other runs of the session. This is the first run of this kernel on
hardware. It is not called validated here. Tier: once.

Step H1 (`tests/rlkit/hw_merge_kernel_bench.py`, one core, eager call; files `78_kernel_h1.jsonl`,
`73_merge_kernel_bench.jsonl`): the kernel can be called on a NeuronCore outside a serving graph, eager and compiled. The
bench ran to its end: 21 rows, none failed. Integer factors are bit-exact in 4 shapes. Float factors give the same result
as eager torch on the device, element for element. Per Qwen3-8B shard, warm: kernel 0.19 to 0.67 ms, torch on the device
0.50 to 1.80 ms over the five warm calls per shape, host merge 24 to 112 ms. These are single calls on one shard, not an
adapter update of a server.

Step H2 (`rl/50_adapter_server_test.py`, Qwen3-0.6B, TP4, kernel mode; files `78_kernel_h2.jsonl`, `50_adapter_h2_06b.jsonl`):

- The kernel ran in all 4 updates on all 4 ranks, 112 tensors each, with no fallback to the host.
- Adapter 1, adapter 2, adapter 1 again: identical (logprob difference 0.0). The zero adapter equals the base (0.0).
- Update time: 12.16 s for the first update, then 0.54 to 0.60 s.
- Against a restart on the host-merged checkpoint the result is not equal: on 2 of 3 prompts the text is the same and the
  mean logprob difference is 0.007 to 0.008 before any fork (largest 0.05 to 0.08). On the third prompt the answers fork
  at token 0 (logprobs -1.056 against -0.985).

Control (`75_pr20_check_ulp1.jsonl`; Qwen3-0.6B, 20 prompts, same check script as section 8.1): a copy of the base
checkpoint in which 700 elements of each of 198 matrices were moved by one unit in the last place. Its logprobs differ from
the base on all 20 prompts (first prompt: largest difference 0.074; per-token difference of the sums over the prompts:
median 0.002, mean 0.010), and the length of 5 answers changes. This is the same order as the kernel-against-restart
difference above. The reading: a difference of this size is what one-ulp weight differences alone produce in bf16 serving.
That the one-ulp difference is the only cause is not proven.

What this shows: the kernel runs on the device and is self-consistent (adapter 1, 2, 1 identical; zero adapter equals the
base). Open: the rule for accepting its result against a restart on the host-merged checkpoint (step H4, a restart on a
checkpoint merged by the kernel itself, did not run). It was not timed at Qwen3-8B in a server (steps H3 to H5 did not run).

### 8.7 Qwen3-30B-A3B start probe

Hardware, trn2.48xlarge, TP4 EP4, our `integration` branch (which carries upstream PRs 17 and 18), run 2026-11-24. One
probe, not a baseline; tier: once. The server was ready after 2,770 s (cold compile cache). The prompt "17 * 23" was
answered 391. One cell: prompt 1024, concurrency 8, 256 output tokens, 32 requests: 153 tok/s, TTFT p50 0.508 s (file
`80_moe30b_r8.jsonl`, `llmbench_moe30b_r8.csv`). No other cell, no comparison with the PRs switched off.

### 8.8 Device profiles

Captured, not analysed. Four decode steps of TP rank 0, three servers: released `temperature` sampler, sampler kernel,
sampler kernel with `--dp-size 4` (four replicas). They were converted to parquet on the machine. Nothing in this file is
derived from them.

## 9. trn2 session of 2026-11-26: pre-transposed LM head on `main`

Files, conditions, tag glossary and limits: [`rl/data/2026-11-26_trn2/README.md`](../../rl/data/2026-11-26_trn2/README.md); every
cell (concurrency 1, 8, 32, with TTFT p50) is in `tables.txt` section 1 and the seconds to ready are in its section 2, both
read from the files by `make_tables.py`. Sub-folders used here: `lm_head/` (benchmarks, `cc/` correctness), `serving/`, `i12_vllm_start/`.

What ran: the tree of branch `feat/lm-head-pretransposed-main`, which is `main` (commit `1c8b314`) plus the opt-in
pre-transposed LM head (`SGLANG_NEURON_LM_HEAD_PRETRANSPOSED`, default off; code of commit `9857f66`, later commits of
the branch change documents only). The seven files under `sglang_neuron/` in which the branch differs from
`upstream-main` (`backend/compile_cache.py`, `env_registry.py`, `models/transformer/config.py`, `model.py`,
`weights.py`, `rlkit_sampler_kernel.py`, `sampling.py`) were mounted over the installed package of the v0.1.0 image, so
the server ran this tree. `main` has no in-place weight reload; none was run.

Conditions in short (the README has the full list): one trn2.48xlarge chip as TP4, sampler `temperature`, llmbench-1024
([`rl/METHOD.md`](../../rl/METHOD.md) section 7), two passes on one server start per row (tier: screen), other jobs on the same
machine during every row, every request of every cell succeeded. "Switch 0" is `SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=0`,
"switch 1" is `=1`; "kernel" is `RLKIT_SAMPLER_KERNEL=1` (section 8.5).

Headline rows, concurrency 32, tier screen (generated from [`rl/best.tsv`](../../rl/best.tsv); the generator reads the CSV files):

<!-- best:begin generated by rl/tools/best.py from rl/best.tsv; do not edit -->
| Key | Model | Configuration | Value | Baseline | Tier | Files |
|---|---|---|---|---|---|---|
| q06_lmh_on | Qwen3-0.6B | temperature sampler kernel and pre-transposed LM head on | 3,337.3 and 3,326.7 tok/s at concurrency 32 | q06_vllm: 2,378.2 and 2,376.0 tok/s at concurrency 32 | screen | `rl/data/2026-11-26_trn2/lm_head/results/`: `llmbench_b_lmh_q06_s1k_t0_p1.csv`, `llmbench_b_lmh_q06_s1k_t0_p2.csv` |
| q06_kernel | Qwen3-0.6B | temperature sampler kernel on, LM head switch off | 2,738.6 and 2,738.5 tok/s at concurrency 32 | q06_off: 2,452.9 and 2,446.6 tok/s at concurrency 32 | screen | `rl/data/2026-11-26_trn2/lm_head/results/`: `llmbench_b_lmh_q06_s0k_t0_p1.csv`, `llmbench_b_lmh_q06_s0k_t0_p2.csv` |
| q06_off | Qwen3-0.6B | both switches off | 2,452.9 and 2,446.6 tok/s at concurrency 32 | - | screen | `rl/data/2026-11-26_trn2/lm_head/results/`: `llmbench_b_lmh_q06_s0n_t0_p1.csv`, `llmbench_b_lmh_q06_s0n_t0_p2.csv` |
| q06_vllm | Qwen3-0.6B | vllm-neuron 0.29.0.1.2.0 | 2,378.2 and 2,376.0 tok/s at concurrency 32 | - | screen | `rl/data/2026-11-26_trn2/i12_vllm_start/`: `llmbench_b_q06_vllm_p1.csv`, `llmbench_b_q06_vllm_p2.csv` |
| q8_kernel | Qwen3-8B | temperature sampler kernel on, LM head switch off | 1,501.1 and 1,500.2 tok/s at concurrency 32 | q8_vllm: 1,126.9 and 1,112.4 tok/s at concurrency 32 | screen | `rl/data/2026-11-26_trn2/lm_head/results/`: `llmbench_b_lmh_q8_s0k_t0_p1.csv`, `llmbench_b_lmh_q8_s0k_t0_p2.csv` |
| q8_lmh_on | Qwen3-8B | temperature sampler kernel and pre-transposed LM head on | 1,497.5 and 1,504.1 tok/s at concurrency 32; not distinguished at screen tier | q8_kernel: 1,501.1 and 1,500.2 tok/s at concurrency 32 | screen | `rl/data/2026-11-26_trn2/lm_head/results/`: `llmbench_b_lmh_q8_s1k_t0_p1.csv`, `llmbench_b_lmh_q8_s1k_t0_p2.csv` |
| q8_vllm | Qwen3-8B | vllm-neuron 0.29.0.1.2.0 | 1,126.9 and 1,112.4 tok/s at concurrency 32 | - | screen | `rl/data/2026-11-26_trn2/serving/`: `llmbench_b_q8_vllm_p1.csv`, `llmbench_b_q8_vllm_p2.csv` |
| q8_rel | Qwen3-8B | released image (SGLang-Neuron v0.1.0) without fork patches | 1,421.0 and 1,421.3 tok/s at concurrency 32 | - | screen | `rl/data/2026-11-26_trn2/serving/`: `llmbench_b_q8_rel_t0_p1.csv`, `llmbench_b_q8_rel_t0_p2.csv` |
<!-- best:end -->

Reading (screen tier; differences below 10 percent are not distinguished):

- Qwen3-0.6B with the kernel, switch 1 against switch 0, concurrency 32: 21.9 and 21.5 percent above, pass against pass.
  At concurrency 1 and 8 the two are not distinguished at screen tier, and with one start per server a difference between
  starts is not separated from the effect of the switch.
- Qwen3-0.6B without the kernel (switch 1 on two starts): the switch is 54 to 60 percent above switch 0 at concurrency 1,
  38 to 42 percent at concurrency 8 and 21 percent at concurrency 32. Switch 0 without the kernel had one start, run later
  in the session than the kernel rows. With the kernel on, the switch adds nothing distinguished at concurrency 1 and 8.
  Why the effects overlap at low concurrency was not examined.
- Qwen3-8B with the kernel: switch 1 and switch 0 are not distinguished at screen tier. Qwen3-8B does not tie its embeddings.
- `--dp-size 4` of TP1 with both switches on (3,099.3 and 3,089.3) is not distinguished from TP4 with the same switches
  (3,347.4 and 3,344.2) at screen tier; 2 x TP2 gave 2,334.1 and 2,332.9 (`dp_layouts/`; `tables.txt` section 1).
- vllm-neuron on Qwen3-8B (start on an empty cache folder): `serving/llmbench_b_q8_vllm_p{1,2}.csv`, in the table above.

Correctness, switch 0 against switch 1, both with the kernel (method of section 8.5: 20 fixed prompts, at most 48 new tokens
(`max_tokens` 48; on Qwen3-0.6B 6 prompts and on Qwen3-8B 1 prompt end earlier at the end of the sequence, so 860 and
932 tokens are compared), temperature 0, one request at a time; a fork is the first position where the tokens differ,
logprob differences are taken before a fork; files `lm_head/cc/results/`, comparison lines in `tables.txt` section 4; tier: once):

| Check | Qwen3-0.6B | Qwen3-8B |
|---|---|---|
| prompts with the same tokens | 20 of 20 (same text 20 of 20) | 20 of 20 (same text 20 of 20) |
| token logprobs compared / bit-equal | 860 / 826 | 932 / 876 |
| largest logprob difference | 0.0252 | 0.0320 |
| GSM8K test, first 200 questions, temperature 0, up to 512 tokens | 0.565 with either switch; the same 113 questions right, McNemar p 1.0 | not run |

The log line "sampler by the kernel" is in the log of every rank of each server run with the kernel (4 lines per log,
one per rank) and absent from the logs of the no-kernel servers, their compile starts included.

Start-up, seconds to ready (cold: empty compile cache, four servers compiling at once on four chips; warm: the
benchmark start from the server's own cache; the warm column is in `tables.txt` section 2; the cold column comes from server logs that are not published; tier: once):

| Server | Cold | Warm |
|---|---|---|
| Qwen3-0.6B switch 0, kernel | 561.6 | 284.9 |
| Qwen3-0.6B switch 1, kernel | 556.6 | 279.8 |
| Qwen3-0.6B switch 1, no kernel | 581.7 (compiled later, with two other servers of this section running) | 279.8 |
| Qwen3-0.6B switch 0, no kernel | 606.4 (compiled with two other servers of this section running) | 254.6 |
| Qwen3-0.6B switch 1, no kernel, second start | (cache copy) | 264.8 |
| Qwen3-8B switch 0, kernel | 833.1 | 335.0 |
| Qwen3-8B switch 1, kernel | 828.1 | 340.1 |

Limits: one session on a shared machine; two passes on one start per server, except switch 1 without the kernel (two
starts); temperature 0 only in the speed rows; `main` has no in-place weight reload, so a weight update with the switch on
was not run; whether the decode graph drops the per-step transpose is not stated here; profiles of separate starts with
the switch off and on (`lmh_off_nd`, `lmh_on_nd`) were analysed in `devprof/`, the result is not stated here; the profiles
of the starts in this section were not analysed; not run inside an RL loop. The switch adds one vocabulary shard per rank
on a tied model (Qwen3-0.6B); Qwen3-8B is untied, its LM head only changes layout.

## 10. trn2 session of 2026-11-26: top-k sampler kernel (`RLKIT_SAMPLER_KERNEL_TOPK=1`) and its vocabulary mask

What ran: the `topk_topp` kernel path of `integration` (`sglang_neuron/rlkit_sampler_topk.py` as of commit `897d927`;
the file is unchanged up to the commit before this change). The files were mounted over the installed package of the
v0.1.0 image. The mask runs below used that file with one line changed (the condition `if width * group.world_size >
vocabulary_size:` replaced by `if True:`, so the two statements under it always run). This change writes the two
statements without the condition; that exact text was not itself run on the device.

Conditions for every row of this section:

- trn2.48xlarge (Trainium2), v0.1.0 image, Qwen3-0.6B, TP4 on 4 logical cores (shard width 37,984, vocabulary 151,936:
  no padded column), `SGLANG_NEURON_SAMPLER=topk_topp`, `SGLANG_NEURON_MAX_TOP_LOGPROBS=20`, launch flags as in
  [`rl/data/2026-11-26_trn2/README.md`](../../rl/data/2026-11-26_trn2/README.md). "Kernel on" is `RLKIT_SAMPLER_KERNEL_TOPK=1`,
  "kernel off" is the switch unset.
- One session, one start per server (a cache stamped by other kernel sources is refused). The three servers of 10.1
  (kernel on, kernel off, and the second kernel-off server) each started from a copy of the same earlier kernel-off
  compile cache (94 files). The servers of 10.2 started on empty caches (0 files); the 10.3 servers each had a cache
  directory of their own, and the file count at start was not recorded. Other jobs ran on the same machine, on other
  chips. No row was repeated on a quiet machine.
- Result files are in [`rl/data/2026-11-26_trn2/`](../../rl/data/2026-11-26_trn2/): `i37_topk_kernel/`, `correctness/` and `serving/`.

### 10.1 Before the fix: the mask skipped on the kernel path (information)

The kernel path applied the vocabulary mask only when a padded column exists, the released path always. At temperature 0
(20 prompts, one at a time, 48 tokens, 20 top logprobs, thinking off), kernel on against kernel off, one server each
(`cmp_r1_koff_kon.txt`; tier: once):

| What | Kernel on against kernel off |
|---|---|
| first-token logprob bit-equal | 0 of 20 |
| top-20 vectors bit-equal | 0 of 20 |
| same top-20 order | 4 of 20 |
| same token sequences | 14 of 20 |
| mean / max absolute difference of the first-token logprob | 0.01402 / 0.04043 |

Two kernel-off servers of the same code agreed with each other on the first-token logprob (20 of 20,
`cmp_r1_koff_r2_koff_b.txt`). Position-0 gaps between top-20 logprobs lay on a 1/16 grid in 380 of 380 cases with the
kernel on and mask skipped, and in 2 of 380 (about the chance rate for the 2e-4 tolerance) on every other server of this
section, including the released path with the mask removed (10.2) (`grid_*.txt`); that is consistent with bf16-valued
logits and does not prove it. Speed numbers of the same code, kernel off 2,610.9 and 2,620.8 tok/s, kernel on 2,769.7 and
2,863.2 tok/s at concurrency 32 (`llmbench_b_q06_topk_koff_k50_p1/p2.csv`, `llmbench_b_q06_topk_kon_k50_p1/p2.csv` in `serving/`;
tier: screen), are from before the fix and are not comparable with the table in 10.3 beyond what 10.3 says.

### 10.2 With the mask applied on the kernel path

Same 20 prompts and settings, kernel on against kernel off, both servers running the file with the mask always applied
(`cmp_r3_koff_mask_kon_mask.txt`; tier: once):

| What | Kernel on against kernel off |
|---|---|
| first-token logprob bit-equal | 20 of 20 |
| top-20 vectors bit-equal | 20 of 20 |
| same top-20 order | 20 of 20 |
| sampled logprob of every generated position (up to 48 per prompt) bit-equal | 878 of 878 |
| same token sequences | 20 of 20 |

The kernel-on server with the mask also equals the earlier kernel-off server without the patch on 20 of 20 prompts at
every position (`cmp_r1_koff_r3_kon_mask.txt`), and the kernel-off server with the mask, which started on an empty cache,
equals the warm-copy kernel-off server of 10.1 on 20 of 20 prompts at every position (`cmp_r1_koff_r3_koff_mask.txt`).
The log line "topk_topp sampler by the kernel" appeared 4 times
(one per rank, `rows=1`) in the kernel-on server; it is logged when the graph is traced, so it shows that the kernel
branch was traced into the one-row graph, not that each request ran it.

What this shows: on the kernel path, applying the skipped mask removes the whole position-0 difference on these 20
prompts. What it does not show:

- Why. A control on the released path (kernel off) with its mask removed gave the same outputs as with the mask, 20 of 20
  prompts at every position (`cmp_r1_koff_r5_koff_nomask.txt`, one run), so removing the mask alone does not produce
  the difference. A run with the kernel's winner unused but the kernel still in the graph, and the mask not applied to the
  logits that feed the logprobs (an all-false `masked_fill` sits on the winner branch only), gave the kernel-on outputs
  of 10.1 at position 0 (`cmp_r1_kon_r4_kon_dead.txt`). Why the compiled graph differs when the kernel
  is present and the mask absent was not determined (no HLO or NEFF was read).
- Not run: temperature above 0 compared token by token, batch above 1 for this comparison, Qwen3-8B,
  `RLKIT_SAMPLER_KERNEL=1` together with this fix, a padded vocabulary on a device.
- Later positions are not always reproducible even with the kernel off: in one kernel-off server the first answer to one
  prompt differed from another kernel-off server's answer from position 1 on, while its repeat of that prompt on the same
  server matched it. Position 0 was bit-equal in every repeat.

### 10.3 Speed with the mask applied

llmbench, prompt 1024 tokens, 256 output tokens (`ignore_eos`), 64 requests per cell after 4 warm-up requests, streaming
`/v1/completions`; output tokens per second. Pass "top-k 50" is temperature 0.8, top-k 50, top-p 0.95, run twice on
one server start (tier: screen); "no filter" is temperature 0.8, top-k off, top-p 1, min-p 0, run once (tier: once).
Kernel on and kernel off ran on separate sets of 4 cores, started together, and the two benchmarks ran interleaved: by the
recorded request start times, 5 of the 6 concurrency-32 cells overlapped a concurrency-1 cell of the other server
(`llmbench_b_…_kon_*.csv`, `llmbench_b_…_koff_*.csv`, and their raw per-request files in `i37_topk_kernel/extra/i37x/bench/`).

| Server | Pass | Concurrency 1 | Concurrency 8 | Concurrency 32 |
|---|---|---|---|---|
| kernel off, mask applied | top-k 50, 1 | 214.9 | 1,260.5 | 2,617.6 |
| kernel off, mask applied | top-k 50, 2 | 214.7 | 1,258.7 | 2,616.9 |
| kernel off, mask applied | no filter | 214.4 | 1,264.4 | 2,601.0 |
| kernel on, mask applied | top-k 50, 1 | 241.6 | 1,362.0 | 2,822.6 |
| kernel on, mask applied | top-k 50, 2 | 241.5 | 1,363.2 | 2,822.4 |
| kernel on, mask applied | no filter | 241.5 | 1,363.4 | 2,825.2 |

At concurrency 1 the kernel-on rows are 10 percent or more above the kernel-off rows; at concurrency 8 and 32 the two are
not distinguished at screen tier. Every request succeeded (0 errors). Limits: one cold start per server and one session; no A/A pair of servers, so the
run-to-run spread between starts is not measured here; other jobs on the machine; Qwen3-0.6B only; the cause of the
difference between the two servers was not isolated; the 8B model was not run.

## Not measured

- The RL loop at Qwen3-8B. No 8B loop has run.
- 8B training memory and speed. Weights alone take 16.38 GB of 25.8 GB on one core (from the plan); the 8B trainer
  (`rl/trainer8b.py`) has a CPU test only.
- Anything at 27-32B beyond one start probe: Qwen3-30B-A3B started once (section 8.7); Qwen3.8-27B or similar: nothing has
  run. Our patch covers the Qwen3 dense family only.
- A weight update while requests are running with our own script (`rl/68_update_under_load.py`). One update with 8
  requests in flight ran on upstream PR 20's branch (section 8.1).
- Any cost-normalised comparison with GPU, TPU or AMD accelerators. The one GPU figure above is a different layout and a
  different sync path.
- Kernels on hardware: the LoRA merge kernel and the sampler kernel ran on 2026-11-24 (sections 8.5 and 8.6). Not run: the
  LoRA merge kernel in a server at Qwen3-8B, the sampler kernel's `topk_topp` mode at Qwen3-8B or with temperature above 0
  compared token by token (section 10), either kernel inside a full RL loop, any attention change.
- The adapter update at 8B with the shard-only delta, or with the kernel.
- Long-answer cost with the context ladder on; accuracy on a model other than Qwen3-0.6B.
