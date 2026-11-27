# Results so far

Measured results of the RL post-training work on Trainium2 with SGLang-Neuron, as of 2026-11-26. Section 8 (the trn2
session of 2026-11-24) was added on 2026-11-25. Sentences that were out of date after that session were changed and say so.
Section 9 (the pre-transposed LM head on `main`, trn2 session of 2026-11-26) was added on 2026-11-26. Section 10 (the
top-k sampler kernel and its vocabulary mask, same session) was added on 2026-11-26.

Rules for reading this file:

- Every row has the model, the layout, what was measured, the number, the condition, the date and the source file.
- One run each unless a row says otherwise. Nothing here is a cost-normalised comparison.
- Dates are Seattle local dates of the run. Raw result files are under `data/` of the public repo
  https://github.com/pengchenglu1997/trainium-sglang-rl, as named. Server logs and raw profiles of
  the 2026-11-21 session are not in the public repo (size); `docs/NIGHT_RECORD_2026-11-21.md` gives their checksums.
- The last section lists what has not been measured.

Abbreviations: TP4 is tensor parallel over 4 NeuronCores (one rollout server). "Layout" names the machine layout and the
sync path.

Names in file and step labels:

- H1 to H5: the hardware steps of the LoRA merge kernel, table in [KERNELS.md](KERNELS.md) section 1.
- W1.2 and similar: numbered steps of `docs/experiment-plan-2026-11-22.md` in the run repository.
- `rNN` in a result file name (`82_sampler_kernel_r12.md`): the run that wrote the file; the scripts are in the run
  repository, `docs/EXPERIMENTS.md`.
- `s4_*.sh` and folders such as `i12_vllm_start/`: the scripts and result folders of the 2026-11-26 session in the run
  repository; the README of [data/2026-11-26_trn2](https://github.com/pengchenglu1997/trainium-sglang-rl/tree/main/data/2026-11-26_trn2) says what each folder holds.

## 1. RL loop at Qwen3-0.6B

Loop: `30_rl_loop.py`, GRPO-style policy gradient with a LoRA trainer, GSM8K. Layout for the trn2 rows: one trn2.48xlarge,
8 rollout servers at TP4, a 16-process data-parallel trainer, adapter sync through `update_weights_from_disk` into the
patched server (in place), 32 questions x 8 answers per step, up to 256 new tokens, temperature 0.7, learning rate 1e-4,
150 steps, not overlapped. Result files `data/2026-11-20/trn2_seeds/30_rl_loop_plain_s{2,3,4}.jsonl`, run 2026-11-20.
Figures below are medians over the 150 step rows of each seed, computed from those files. Three seeds, one run each.

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

Other rows from the same loop files and the plan:

| Model | Layout | What | Number | Condition | Date | Source |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | one patched server | adapter update in place | 0.67-0.70 s after a first update of 7-9 s | rank-16 LoRA on 7 projections, TP4 | 2026-11-20 | `patches/PR_DRAFT_inplace_weight_update.md`; `docs/experiment-plan-2026-11-22.md` section 3 |
| Qwen3-0.6B | one patched server | full-checkpoint update in place | 0.34-0.41 s | TP4 | 2026-11-20 | same |
| Qwen3-0.6B | one RTX PRO 6000, standard SGLang, one replica, merged-weight sync | RL step, 150 steps, seed 1 | 6.69 s wall (6.40 s summed parts), mean of steps 2-150; accuracy 0.574 to 0.714 | sync path, trainer and layout differ from the trn2 rows, so this is not a like-for-like comparison | 2026-11-20 | `data/2026-11-20/cuda/30_rl_loop_cuda_long_b4.jsonl` (summary line) |
| Qwen3-0.6B | trn2, 8 replicas, earlier trainer (before shared memory), seed 1 | RL step, 150 steps | 6.83 s summed parts (no wall figure in the file); accuracy 0.574 to 0.748 | the fourth 150-step run; train time 2.83 s against 1.2 s in the seed runs | 2026-11-20 | `data/2026-11-20/trn2_sgl_final/30_rl_loop_long1.jsonl` |

Limits: the RL loop was run on Qwen3-0.6B only. The four 150-step trn2 runs ended at 0.702 / 0.712 / 0.728 / 0.748.
Sampler: the launcher `srv_sglang_neuron.sh` defaults to `temperature`, and `scripts/run_trn2_seeds.sh` does not set
`SGL_SAMPLER` or `SGL_TOP_LOGPROBS`, so by the scripts the loop ran `temperature` with no top logprobs. This was not
read back from the run's own server logs (they are not in the public repo).

## 2. In-place update across model families

Our in-place update patch (branch `feat/inplace-adapter-sync` of this fork, on SGLang-Neuron v0.1.0), trn2, TP4, context 1024, one decode bucket of 32.
Script `48_inplace_server_test.py`, result files `data/2026-11-21_night/48_inplace_*.jsonl`, run 2026-11-21. The test
compares the server after an in-place update with a fresh start on the same weights, by text and token logprobs.

| Model | Graph | Live tensors covered | In-place full update | Back to base | In place vs fresh start |
|---|---|---|---|---|---|
| Qwen3-8B | shared dense transformer | 291 parameters, 1 buffer (`rotary_emb.inv_freq`) | 4.7-6.2 s (12 updates, two servers) | same text, logprob difference 0.0 | same text, logprob difference 0.0 |
| GLM-4-9B-0414 | shared dense transformer | 363 parameters, 1 buffer | 4.9 s (2 updates) | same text, 0.0 | same text, 0.0 |
| DeepSeek-V2-Lite | its own graph (MLA, MoE) | 351 parameters, 1 buffer | 10.3-10.6 s (2 updates) | same text, 0.0 | same text, 0.0 |

Source for the table: `docs/NIGHT_RECORD_2026-11-21.md` section 2.

- The patch was not changed for GLM-4 or DeepSeek-V2. In all three the only buffer is `inv_freq`, equal in the staged
  and the live graph, so walking `named_parameters()` covers every weight of these families.
- GLM-4: the perturbed checkpoint changed the logprobs (largest difference 0.087) but not the 48-token texts.
- DeepSeek-V2-Lite: the first attempt died at start on a compile-cache race, not on a refusal. A start with the
  example's own flags (context 4096) was still compiling after one hour and was stopped by hand; the reduced shapes
  were ready in 953 s.
- Not covered: Qwen3-MoE, GPT-OSS, GLM-5.2, Qwen3.5, Kimi-K3 (kernel models with packed or quantized weights).
- Qwen3-8B where the time goes (same section 4 of the record): per rank, host graph built in 0.00-0.02 s, loaders
  3.6-4.7 s, copy to the device 0.3-1.1 s. Keeping the host graph between updates changes nothing.

## 3. Adapter update at Qwen3-8B

Script `50_adapter_server_test.py`, result file `data/2026-11-21_night/50_adapter_q8.jsonl`, run 2026-11-21.
Qwen3-8B, TP4, rank 16 on seven projections, host-merge path (no kernel).

| What | Number | Note |
|---|---|---|
| first adapter update | 35.8 s | includes first-use cost |
| later updates | 13.9-14.0 s | 13.5-13.6 s forming the deltas on the host, 0.3 s copy |
| correctness | equal to a fresh start on the merged checkpoint: same text, logprob difference 0.0 | adapters replace each other; the zero adapter gives the base back |
| comparison | full-checkpoint update at 8B takes 5-6 s | so at 8B the adapter path is slower than the full path |

Source: `docs/NIGHT_RECORD_2026-11-21.md` section 4. The shard-only adapter delta (`_LowRankSlice`) and the LoRA merge
kernel aim at this number. Changed 2026-11-25: the kernel ran on hardware on 2026-11-24 at Qwen3-0.6B (section 8.6); it
has not been timed at Qwen3-8B in a server, and the shard-only delta has not run on hardware (see
[KERNELS.md](KERNELS.md)).

## 4. Sampler modes

Qwen3-0.6B, TP4, `58_sampler_probe.py`, result files `data/2026-11-21_night/58_sampler_{s1,b1}.jsonl`, run 2026-11-21.
Speed: 256 requests of up to 256 tokens, 32 in flight, three passes, one server per mode.

| Mode | Setting | Speed | HTTP result |
|---|---|---|---|
| `topk_topp` | full vocabulary | 2,713-2,730 tok/s | 200 |
| `topk_topp` | top_p 0.95, top_k 20 | 2,700-2,764 tok/s | 200 for all five tested (top_p, top_k) settings |
| `temperature` | full vocabulary | 2,183-2,218 tok/s | 200 |
| `temperature` | any of four filtered settings | not applicable | 400 (message: use a `topk_topp` server for filtered sampling) |

Top-p cut: 8 tests on 4 prompts, 400 one-token samples each at temperature 1, with top_p set inside the mass of the token
of rank j. In all 8 the number of distinct tokens returned was j: the token that crosses p is kept, as in vLLM and
standard SGLang. Source: `docs/NIGHT_RECORD_2026-11-21.md` section 3.

Caveat: the probe started both servers with `SGL_TOP_LOGPROBS=20` (the default of `58_sampler_probe.py`). With top
logprobs above 0, `neuron_compiler_args` in `models/host.py` keeps the default compiler arguments for both modes, so
by the code the two probe servers were compiled alike. The RL loop's `temperature` server has top logprobs 0, and in
that case the decode compiles at `-O2`. So the probe's `temperature` row is not the configuration the loop runs, and
the loop's configuration has no measured speed. See [KERNELS.md](KERNELS.md) section 2. Not read back from server logs.

Changed 2026-11-25: the session of 2026-11-24 measured the `temperature`, `greedy` and `topk_topp` servers at Qwen3-0.6B
with another harness (llmbench, prompt 1024, 64 requests per cell; section 8.4). Whether those servers ran with top
logprobs 0 was not read back from their logs, so the sentence above about the loop's configuration still holds.

## 5. Decode profile

Qwen3-0.6B, TP4, 32 requests of up to 256 tokens, run 2026-11-21. Sampler `temperature` with no top logprobs (the
launcher defaults; `59_decode_profile.py` sets neither), so by the code the decode was compiled at `-O2`; not read back
from the server log. Source: `docs/NIGHT_RECORD_2026-11-21.md` section 5;
files `59_decode_profile_p1.jsonl`, `59b_profile_sum.txt`, `59d_devprof_sum.txt`.

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

Qwen3-0.6B, TP4, context ladder off, `60_shape_cost.py`, files `data/2026-11-21_night/60_shape_cost_{A,B,C,D}.jsonl`, run
2026-11-21. Step = wall time divided by the longest answer. Same GSM8K questions at every level; answers up to 256 tokens.

| Concurrent requests | A: one bucket (32), context 1024 | B: default buckets, context 1024 | C: one bucket, context 4096 | D: one bucket, context 8192 |
|---|---|---|---|---|
| 32 | 10.4 ms | 9.3 | 14.8 | 26.5 |
| 16 | 9.8 | 7.1 | 14.2 | 25.9 |
| 8 | 9.5 | 5.8 | 13.9 | 25.5 |
| 4 | 9.3 | 5.1 | 13.7 | 25.4 |
| 2 | 9.3 | 5.0 | 13.6 | 25.3 |
| 1 | 9.3 | 4.6 | 13.7 | 25.2-25.4 |

Source: `docs/NIGHT_RECORD_2026-11-21.md` section 6.

- With one bucket a step costs the same whether 1 or 32 rows are live. With the default buckets one request runs at
  4.6 ms a step (215 tok/s) against 9.3 ms (108 tok/s).
- The step grows with the compiled context although the requests are the same short ones: 10.4 / 14.8 / 26.5 ms at
  1024 / 4096 / 8192.
- Row fill of one group of 32 answers: 0.58-0.63 (tokens divided by 32 times the longest answer). A single group runs at
  1,790-2,030 tok/s; 256 queued requests at 2,713-2,730 tok/s.
- Server ready time: A 109 s, B 350 s, C 215 s, D 230 s (the cache already held some shapes).

## 7. CPU-only kernel tests

Not hardware. NKI 0.7.0 CPU simulator and CPU torch, WSL Ubuntu, 2026-11-23. Source: the test scripts in `tests/rlkit/` on branch
`feat/lora-merge-kernel` (`bash tests/rlkit/run_all.sh`; `tests/rlkit/README.md`); all seven scripts pass. Full table
with conditions: [KERNELS.md](KERNELS.md) section 1.

| What | Result |
|---|---|
| LoRA merge kernel in the simulator, all 0.6B and 8B per-rank shapes (4 kinds each), K 16 / 32 / 48 | all pass; equal to the float32 reference on every entry |
| distance from the float64 sum rounded once | 255 of 6,291,456 entries (8B qkv) to 735 of 25,165,824 (8B gate_up) differ, largest 32 and 192 units of the last place respectively |
| K 144 and 200 (two accumulating matmuls), partial tiles, LNC 1 and 2 | 43 pass, 0 failed |
| factors through the engine's own loaders, TP 1 / 2 / 4 / 8 / 16, 0.6B and 8B | 78 pass, 0 failed; integer factors bit-exact on every rank |
| negative control (wrong rank, forgotten scale, summed segments) | each fails on 4 of 4 ranks |
| patched `host.py` end to end at 0.6B TP4, simulator | 47 pass, 0 failed; 0 of 3,933,184 live entries differ from the float32 reference |
| mutation test of the wiring | 21 faults planted; 15 caught at first, 18 after the added tests; 3 have no test |

What is not covered: compile with `neuronx-cc`, any device run, any timing, a real PEFT adapter, a run of the plugin's own
CPU mode. The simulator is not a timing model. Changed 2026-11-25: the first device runs of the LoRA merge kernel were
made on 2026-11-24 (section 8.6); these CPU tests are unchanged.

## 8. trn2 session of 2026-11-24

Hardware for every row of this section: trn2.48xlarge spot (Trainium2), one chip = TP4 on 4 logical cores, unless a row
names another layout. Images: `pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04` and
`pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.1-ubuntu26.04`; NKI 0.7.0 in the SGLang image. In the
SGLang image `general_plugin.py`, `models/host.py` and `models/transformer/model.py` have the md5 of tag v0.1.0;
`backend/compile_cache.py` differs from the tag in one comment line only. Dates are Seattle local dates. The result files
named below are from that session. Changed 2026-11-25: they are in `data/2026-11-24_trn2/` of the public repo
(https://github.com/pengchenglu1997/trainium-sglang-rl), whose README maps each file to a subsection here. Not there:
per-request benchmark files, server logs, `.out` files and device profiles. The probe scripts 73 to 81 are at the top
level of that repo (index: its `docs/EXPERIMENTS.md` section 8); the patch folders they mount are branches of this fork.

Run conditions that bear on the numbers:

- Many runs went on at the same time on other chips of the same machine. Repeats on a quieter machine agree within 1
  percent (released defaults, greedy sampler, sampler kernel with `--dp-size 4`).
- Every throughput cell is one run of 64 requests on one server start, 256 output tokens, unless the row says more.
  Throughput is total output tokens over the wall time of the batch (llmbench at commit 08f4163, column
  `output_token_throughput_tps`). TTFT is the median time to the first streamed chunk.
- The first hardware test of the sampler kernel sent sampled requests without `top_p 1, top_k -1, min_p 0`. The
  `temperature` sampler refused them with HTTP 400. The check script was changed to send them; the results below are from
  the corrected script.

### 8.1 Upstream PR 20 on its own branch

Hardware, trn2.48xlarge, TP4. Three files of branch `upstream-pr/20` (`general_plugin.py`, `models/host.py`,
`models/transformer/model.py`) over the v0.1.0 image; the md5 of each file is recorded in each result file. Launch flags:
`--context-length 1024 --chunked-prefill-size 1024 --max-running-requests 32 --max-total-tokens 32768
--cuda-graph-bs-decode 32`. The check compares answers and token logprobs of 20 prompts, 48 new tokens each (960 tokens).
Run 2026-11-24. Files `75_pr20_check_pr20_q06.jsonl`, `75_pr20_check_pr20_q8.jsonl`.

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
2026-11-24. File `74_stamp_race_r1.jsonl`. The fix is the branch `fix/compile-cache-stamp-race` (upstream pull request 22; a draft until 2026-11-25, ready for review since).

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
per cell. Cell: tok/s / TTFT p50. Run 2026-11-24.

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
(`77_pair_q06d.md`). The order of the two engines changes with concurrency in this table.

Qwen3-8B, prompt 1024, 256 output tokens, SGLang-Neuron v0.1.0 released defaults (file `77_pair_q8.md`):

| Concurrency | tok/s | TTFT p50 | Repeat pass tok/s |
|---|---|---|---|
| 1 | 109.3 | 0.067 s | 109.7 |
| 8 | 604.8 | 0.211 s | 606.8 |
| 32 | 1,419.7 | 0.744 s | 1,425.9 |

vllm-neuron on Qwen3-8B: it did not start in two attempts on 2026-11-24 (TP4, max-model-len 4096, container). The
compiler `neuronx-cc` exited with code 70 and the error `NCC_EVRF059`: the kernel file of
`vendored_kernels.rotational_topk` "does not exist on the host". The start tests of 2026-11-26 found that this error comes from a stale NKI
compile-result index in a reused cache folder (`i12_vllm_start/`, `i34_nki_index/` in the public data folder), and
vllm-neuron 0.29.0.1.2.0 on Qwen3-8B (TP4, max-model-len 4096, start on an empty cache folder) ran the same benchmark:
1,126.9 and 1,112.4 tok/s at concurrency 32, TTFT p50 0.909 s and 0.955 s (two passes; files
`serving/llmbench_b_q8_vllm_p{1,2}.csv` in [data/2026-11-26_trn2](https://github.com/pengchenglu1997/trainium-sglang-rl/tree/main/data/2026-11-26_trn2)). The SGLang-Neuron rows of the same session are in
section 9.

### 8.4 Serving changes, Qwen3-0.6B

Hardware, trn2.48xlarge, Qwen3-0.6B, prompt 1024, 256 output tokens, run 2026-11-24. Cell: tok/s at concurrency 1 / 8 /
32, one run per cell unless the row says more. "Text" is the answer to five fixed prompts at temperature 0 (48 tokens)
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

Changes that did not move the numbers:

| Change | tok/s at 1 / 8 / 32 | TTFT p50 at 32 | Text |
|---|---|---|---|
| `--chunked-prefill-size 1024` | 211 / 1,162 / 2,459; prompt 3072 at concurrency 1 and 32: 202.3 and 1,972.9 against 203.6 and 1,974.8 | 0.267 s | same on 5 of 5 |
| `NEURON_RT_XU_WORKER_ENABLE_LOW_LATENCY_COMPLETION=1` (platform default 0) | 211 / 1,160 / 2,456 | 0.261 s | same on 5 of 5 |

Both differ from the released defaults by less than 1 percent in every cell (largest: 1,160 against 1,168 at concurrency
8). Two passes on one server start of the released defaults differ by 0.2 percent or less (section 8.3). A second start
later in the session gave 211 / 1,161 / 2,459 (`77_pair_q06d.md`), up to 0.6 percent from the first (1,161.1 against
1,168.0 at concurrency 8).

Not ready within 30 minutes, cause not read: greedy sampler with `--dp-size 4`, greedy sampler with 4 containers. Not run:
chunk size 512, in-flight launches alone, and the controls that run the same graph as the released defaults. The
concurrency-1 cell of a replica layout is one replica alone, which is why it is lower than the TP4 cell. Replica cells
sum one client process per replica; the processes start together and are not time-aligned beyond that.

### 8.5 Sampler kernel

Hardware, trn2.48xlarge, one chip, TP4, run 2026-11-24. One NKI kernel replaces the shard-local stage of the
`temperature` sampler (the per-column hash noise, the Gumbel score, the per-shard maximum and winner, and the
log-sum-exp). The all-gather and the global merge stay the engine's code. Switch `RLKIT_SAMPLER_KERNEL=1`; unset, the
released code path runs. Branch `feat/sampler-kernel` ([KERNELS.md](KERNELS.md) section 2). Simulator tests before the
hardware run: 11 of 11 pass (`selftest_sampler_kernel.py`, NKI 0.7.0, CPU).

Speed, prompt 1024, 256 output tokens, tok/s at concurrency 1 / 8 / 32, two passes each unless stated. Files
`82_sampler_kernel_r10.md`, `82_sampler_kernel_r12.md`.

| Server | Qwen3-0.6B | Qwen3-8B |
|---|---|---|
| released `temperature` sampler | 211 / 1,160 / 2,455 and 211 / 1,171 / 2,454 | 109 / 604 / 1,421 and 110 / 607 / 1,426 |
| sampler kernel | 319 / 1,624 / 2,730 and 319 / 1,636 / 2,734 | 114 / 642 / 1,489 and 114 / 647 / 1,496 |
| released `greedy` sampler (one pass) | 311 / 1,608 / 2,781 | 113 / 639 / 1,533 |

At 8B the kernel is 4.5 to 6.6 percent above the released `temperature` sampler over the six cells of the table, from
the unrounded values in `82_sampler_kernel_r12.md`. The lowest cell is pass 2 at concurrency 1 (109.5 to 114.5; the rounded
table values, 110 to 114, would give 3.6) and the highest is pass 2 at concurrency 8 (606.9 to 647.3). At concurrency 32:
1,421 to 1,489 and 1,426 to 1,496, which is 4.8 and 4.9 percent. At 0.6B and concurrency 32 the kernel
(2,730) is below the released `greedy` sampler (2,781), and at 8B too (1,489 against 1,533).

Sampled rollout as the RL loop asks for it: Qwen3-0.6B, temperature 0.7, 64 questions with 8 answers each, up to 256 new
tokens, three passes (file `55_rollout_bench_roll_*.jsonl`).

| Server | tok/s in the three passes |
|---|---|
| released `temperature` sampler | 2,283 / 2,335 / 2,295 |
| sampler kernel | 2,472 / 2,484 / 2,484 |

Pass by pass this is 8.3, 6.4 and 8.2 percent. The rollout is not a full RL loop.

Combinations at Qwen3-0.6B (file `opt_sweep_r13.md`, `opt_sweep_r15.md`, `opt_sweep_r16.md`):

| Server | tok/s at 1 / 8 / 32 | TTFT p50 at 32 |
|---|---|---|
| kernel, released default flags | 321 / 1,629 / 2,744 (TTFT 0.025 / 0.075 / 0.254 s) | 0.254 s |
| kernel and `--dp-size 4` of TP1 in one server | 223 / 1,420 / 3,026 | 0.160 to 0.170 s |
| the same, repeated | 3,039 and 3,048 at concurrency 32 (with `--stream-interval 4`; with 4 in-flight launches; neither changed the result), and 223 / 1,425 / 3,035 with the machine otherwise idle | |
| kernel and 4 containers of TP1 (sum of four clients, one measurement) | 223 / 1,438 / 3,098 | 0.138 s |

For comparison, the released `temperature` sampler with `--dp-size 4` gave 192 / 1,172 / 2,701 (section 8.4), and the
released `temperature` sampler with 4 containers of TP1 gave 192 / 1,189 / 2,756. Logical-core configuration 1 with 8
replicas on one chip was refused by the engine at start (`set_logical_nc_config`; this model's layout runs with 2).

Correctness evidence (files `82_sampler_kernel_r10.jsonl`, `82_sampler_kernel_r12.jsonl`, `83_sampler_check2_r11.jsonl`):

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
and Qwen3-8B only; about 5 percent at 8B; not run inside a full RL loop; the released and kernel sequences are not equal
at temperature 0.8 for 16 of 60 requests.

### 8.6 LoRA merge kernel, first hardware run

Hardware, trn2.48xlarge, run 2026-11-24. The kernel is on branch `feat/lora-merge-kernel`, run through the patched server
with `RLKIT_DEVICE_WRITE=kernel`, `RLKIT_KERNEL_IMPL=nki`, `RLKIT_KERNEL_CHECK=all`, fp32 factors. NKI 0.7.0. The
step-1 bench ran on one core of the same NUMA node as other runs of the session. This is the first run of this kernel on
hardware. It is not called validated here.

Step H1 (`tests/rlkit/hw_merge_kernel_bench.py`, one core, eager call; files `78_kernel_h1.jsonl`,
`78_kernel_h1_bench.out`, `73_merge_kernel_bench.jsonl`): the kernel can be called on a NeuronCore outside a serving graph,
eager and compiled. The bench ran to its end: 21 rows, none failed. Integer factors are bit-exact in 4 shapes. Float
factors give the same result as eager torch on the device, element for element. Per Qwen3-8B shard, warm: kernel 0.19 to 0.67
ms, torch on the device 0.50 to 1.80 ms over the five warm calls per shape, host merge 24 to 112 ms. These are single calls on one shard, not an adapter
update of a server.

Step H2 (`50_adapter_server_test.py`, Qwen3-0.6B, TP4, kernel mode; files `78_kernel_h2.jsonl`,
`50_adapter_h2_06b.jsonl`):

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
base). What is open: the rule for accepting its result against a restart on the host-merged checkpoint (restart on a
checkpoint merged by the kernel itself, step H4, was not run). It has not been timed at Qwen3-8B in a server (steps H3 to
H5 not run).

### 8.7 Qwen3-30B-A3B start probe

Hardware, trn2.48xlarge, TP4 EP4, our `integration` branch (which carries upstream PRs 17 and 18), run 2026-11-24. One
probe, not a baseline. The server was ready after 2,770 s (cold compile cache). The prompt "17 * 23" was answered 391. One
cell: prompt 1024, concurrency 8, 256 output tokens, 32 requests: 153 tok/s, TTFT p50 0.508 s (file
`80_moe30b_r8.jsonl`, `llmbench_moe30b_r8.csv`). No other cell, no comparison with the PRs switched off.

### 8.8 Device profiles

Captured, not analysed. Four decode steps of TP rank 0, three servers: released `temperature` sampler, sampler kernel,
sampler kernel with `--dp-size 4` (four replicas). They were converted to parquet on the machine. Nothing in this file is
derived from them. The existing script `59d_devprof_sum.py` does not read this layout (there is no `summary.json`).

## 9. trn2 session of 2026-11-26: pre-transposed LM head on `main`

What ran: the tree of branch `feat/lm-head-pretransposed-main`, which is `main` (commit `1c8b314`) plus the opt-in
pre-transposed LM head (`SGLANG_NEURON_LM_HEAD_PRETRANSPOSED`, default off; code of commit `9857f66`, later commits of
the branch change documents only). The seven files under `sglang_neuron/` in which the branch differs from
`upstream-main` (`backend/compile_cache.py`, `env_registry.py`, `models/transformer/config.py`, `model.py`,
`weights.py`, `rlkit_sampler_kernel.py`, `sampling.py`) were mounted over the installed package of the v0.1.0 image, so
the server ran this tree. `main` has no in-place weight reload; none was run.

Conditions for every row of this section:

- trn2.48xlarge (Trainium2), image `pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04`, one
  chip as TP4 (4 logical cores), sampler `temperature`.
- Launch: the image's example script of the model (`qwen3_0p6b/run_server.sh`, `qwen3_8b/run_server.sh`) with
  `--context-length 4096 --chunked-prefill-size 2048 --max-running-requests 32 --max-total-tokens 131072
  --cuda-graph-bs-decode 1 8 32` appended.
- Switches: "switch 0" is `SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=0`, "switch 1" is `=1`. "Kernel" is
  `RLKIT_SAMPLER_KERNEL=1` (section 8.5). Switch 0 with the kernel is `main` as it is used today.
- llmbench at commit 08f4163: prompt 1024 tokens, 256 output tokens (`ignore_eos`), temperature 0, streaming
  `/v1/completions`, 64 requests per cell after 4 warm-up requests, concurrency 1, 8 and 32. Cell: tok/s
  (`output_token_throughput_tps`) / TTFT p50. Two passes on one server start, one after the other. The servers of a
  comparison ran one after the other on the same chip, from a warm compile cache of their own.
- One session. Other jobs (compiles and benchmarks of other work) ran on the same machine during every row; which chips
  they used was not recorded. The container list before and after each benchmark is in the result logs. No row was
  repeated on a quiet machine.
- Provenance: the result files do not record the image tag, the instance type or the llmbench commit. They are stated in
  the "Conditions" of the README of the [public data folder](https://github.com/pengchenglu1997/trainium-sglang-rl/tree/main/data/2026-11-26_trn2).
- Every request of every cell succeeded (0 errors).
- The result files of this section are in the [public data folder](https://github.com/pengchenglu1997/trainium-sglang-rl/tree/main/data/2026-11-26_trn2): llmbench summaries in `lm_head/results/`, the
  correctness check in `lm_head/cc/results/`, start-up records in `serving/`. Server logs are not published.

Qwen3-0.6B:

| Server | Pass | Concurrency 1 | Concurrency 8 | Concurrency 32 |
|---|---|---|---|---|
| switch 0, kernel | 1 | 319.5 / 0.027 s | 1,613.9 / 0.078 s | 2,738.6 / 0.257 s |
| switch 0, kernel | 2 | 317.7 / 0.028 s | 1,636.6 / 0.077 s | 2,738.5 / 0.261 s |
| switch 1, kernel | 1 | 325.8 / 0.027 s | 1,639.9 / 0.078 s | 3,337.3 / 0.260 s |
| switch 1, kernel | 2 | 326.3 / 0.028 s | 1,654.6 / 0.074 s | 3,326.7 / 0.261 s |
| switch 1, no kernel (first start) | 1 | 329.7 / 0.025 s | 1,629.0 / 0.074 s | 2,972.7 / 0.254 s |
| switch 1, no kernel (first start) | 2 | 328.6 / 0.026 s | 1,640.2 / 0.074 s | 2,971.1 / 0.257 s |
| switch 0, no kernel | 1 | 205.7 / 0.031 s | 1,155.1 / 0.081 s | 2,452.9 / 0.269 s |
| switch 0, no kernel | 2 | 210.5 / 0.029 s | 1,166.7 / 0.078 s | 2,446.6 / 0.265 s |
| switch 1, no kernel (second start) | 1 | 325.1 / 0.027 s | 1,612.3 / 0.080 s | 2,969.0 / 0.261 s |
| switch 1, no kernel (second start) | 2 | 324.3 / 0.029 s | 1,639.4 / 0.075 s | 2,974.0 / 0.255 s |

The rows ran in this order: the two kernel servers, the first start of switch 1 without the kernel, the two Qwen3-8B
servers, then switch 0 without the kernel and a second start of switch 1 without the kernel (from a copy of its cache),
back to back. Switch 0 without the kernel is the released code path apart from the compile-cache fix.

Qwen3-8B:

| Server | Pass | Concurrency 1 | Concurrency 8 | Concurrency 32 |
|---|---|---|---|---|
| switch 0, kernel | 1 | 114.2 / 0.068 s | 644.9 / 0.213 s | 1,501.1 / 0.720 s |
| switch 0, kernel | 2 | 114.7 / 0.057 s | 646.5 / 0.212 s | 1,500.2 / 0.723 s |
| switch 1, kernel | 1 | 115.1 / 0.067 s | 638.5 / 0.207 s | 1,497.5 / 0.738 s |
| switch 1, kernel | 2 | 115.6 / 0.055 s | 640.6 / 0.207 s | 1,504.1 / 0.719 s |

Reading, by division of the table values:

- Qwen3-0.6B with the kernel, switch 1 against switch 0, pass against pass: 2.0 and 2.7 percent above at concurrency 1
  (325.8 and 326.3 against 319.5 and 317.7) and 1.6 and 1.1 percent at concurrency 8 (1,639.9 and 1,654.6 against
  1,613.9 and 1,636.6). The two passes of one server differ by 0.6 percent (switch 0) and 0.1 percent (switch 1) at
  concurrency 1, and by 1.4 and 0.9 percent at concurrency 8. So the gain at concurrency 1 is larger than the
  difference between the passes, and at concurrency 8 about the size of it. With one start per server, a difference
  between starts is not separated from the effect of the switch. At concurrency 32 it is 21.9 and 21.5 percent above
  (3,337.3 and 3,326.7 against 2,738.6 and 2,738.5). TTFT p50 does not move by more than 0.004 s.
- Qwen3-0.6B without the kernel: switch 1 is 54 to 60 percent above switch 0 at concurrency 1, 38 to 42 percent at
  concurrency 8 and 21 percent at concurrency 32 (both starts of switch 1). So at concurrency 1 and 8 the switch and the
  kernel give about the same speed, alone or together; at concurrency 32 the two add. Switch 0 without the kernel had
  one start, run later in the session than the kernel rows. Why they overlap at low concurrency was not examined.
- Qwen3-8B with the kernel: no change (concurrency 32: 1,497.5 and 1,504.1 against 1,501.1 and 1,500.2; pass against
  pass, every cell within 1 percent). Qwen3-8B does not tie its embeddings.

Correctness, switch 0 against switch 1, both with the kernel (method of section 8.5: 20 fixed prompts, at most 48 new tokens
(`max_tokens` 48; on Qwen3-0.6B 6 prompts and on Qwen3-8B 1 prompt end earlier at the end of the sequence, so 860 and
932 tokens are compared), temperature 0, one request at a time; a fork is the first position where the tokens differ, logprob differences are
taken before a fork):

| Check | Qwen3-0.6B | Qwen3-8B |
|---|---|---|
| prompts with the same tokens | 20 of 20 (same text 20 of 20) | 20 of 20 (same text 20 of 20) |
| token logprobs compared / bit-equal | 860 / 826 | 932 / 876 |
| largest logprob difference | 0.0252 | 0.0320 |
| GSM8K test, first 200 questions, temperature 0, up to 512 tokens | 0.565 with either switch; the same 113 questions right, McNemar p 1.0 | not run |

The log line "sampler by the kernel" is in the log of every rank of each server run with the kernel (4 lines per log,
one per rank) and absent from the logs of the no-kernel servers (switch 1, both starts; switch 0), their compile starts
included.

Start-up, seconds to ready (cold: empty compile cache, four servers compiling at once on four chips; warm: the
benchmark start from the server's own cache):

| Server | Cold | Warm |
|---|---|---|
| Qwen3-0.6B switch 0, kernel | 561.6 | 284.9 |
| Qwen3-0.6B switch 1, kernel | 556.6 | 279.8 |
| Qwen3-0.6B switch 1, no kernel | 581.7 (compiled later, with two other servers of this section running) | 279.8 |
| Qwen3-0.6B switch 0, no kernel | 606.4 (compiled with two other servers of this section running) | 254.6 |
| Qwen3-0.6B switch 1, no kernel, second start | (cache copy) | 264.8 |
| Qwen3-8B switch 0, kernel | 833.1 | 335.0 |
| Qwen3-8B switch 1, kernel | 828.1 | 340.1 |

Limits: one session on a shared machine; two passes on one start per server, except switch 1 without the kernel (two starts); temperature 0 only in the
speed rows; `main` has no in-place weight reload, so a weight update with the switch on was not run; whether the
decode graph drops the per-step transpose is not stated here; the device profiles of this session were analysed later (`devprof/` and `prof_all/` in the
public data folder, for the transpose `devprof/out/g04_transpose_where.txt`);
not run inside an RL loop. The switch adds one vocabulary shard per rank on a tied model (Qwen3-0.6B); Qwen3-8B is
untied, its LM head only changes layout.

## 10. trn2 session of 2026-11-26: top-k sampler kernel (`RLKIT_SAMPLER_KERNEL_TOPK=1`) and its vocabulary mask

What ran: the `topk_topp` kernel path of `integration` (`sglang_neuron/rlkit_sampler_topk.py` as of commit `897d927`;
the file is unchanged up to the commit before this change). The files were mounted over the installed package of the
v0.1.0 image. The mask runs below used that file with one line changed (the condition `if width * group.world_size >
vocabulary_size:` replaced by `if True:`, so the two statements under it always run). This change writes the two
statements without the condition; that exact text was not itself run on the device.

Conditions for every row of this section:

- trn2.48xlarge (Trainium2), v0.1.0 image, Qwen3-0.6B, TP4 on 4 logical cores (shard width 37,984, vocabulary 151,936:
  no padded column), `SGLANG_NEURON_SAMPLER=topk_topp`, `SGLANG_NEURON_MAX_TOP_LOGPROBS=20`, launch flags as in section 9.
  "Kernel on" is `RLKIT_SAMPLER_KERNEL_TOPK=1`, "kernel off" is the switch unset.
- One session, one start per server (a cache stamped by other kernel sources is refused). The three servers of 10.1
  (kernel on, kernel off, and the second kernel-off server) each started from a copy of the same earlier kernel-off
  compile cache (94 files). The servers of 10.2 started on empty caches (0 files); the 10.3 servers each had a cache
  directory of their own, and the file count at start was not recorded. Other jobs ran on the same machine, on other
  chips. No row was repeated on a quiet machine.
- Result files are in the [public data folder](https://github.com/pengchenglu1997/trainium-sglang-rl/tree/main/data/2026-11-26_trn2): `i37_topk_kernel/`, `correctness/` and `serving/`.

### 10.1 Before the fix: the mask skipped on the kernel path (information)

The kernel path applied the vocabulary mask only when a padded column exists, the released path always. At temperature 0
(20 prompts, one at a time, 48 tokens, 20 top logprobs, thinking off), kernel on against kernel off, one server each
(`cmp_r1_koff_kon.txt`):

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
2,863.2 tok/s at concurrency 32 (`llmbench_b_q06_topk_koff_k50_p1/p2.csv`, `llmbench_b_q06_topk_kon_k50_p1/p2.csv`),
are from before the fix and are not comparable with the table in 10.3 beyond what 10.3 says.

### 10.2 With the mask applied on the kernel path

Same 20 prompts and settings, kernel on against kernel off, both servers running the file with the mask always applied
(`cmp_r3_koff_mask_kon_mask.txt`, one run each):

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
one server start; "no filter" is temperature 0.8, top-k off, top-p 1, min-p 0, run once. Kernel on and kernel off ran on
separate sets of 4 cores, started together, and the two benchmarks ran interleaved: by the recorded request start times,
5 of the 6 concurrency-32 cells overlapped a concurrency-1 cell of the other server (`llmbench_b_…_kon_*.csv`,
`llmbench_b_…_koff_*.csv`, and their raw per-request files).

| Server | Pass | Concurrency 1 | Concurrency 8 | Concurrency 32 |
|---|---|---|---|---|
| kernel off, mask applied | top-k 50, 1 | 214.9 | 1,260.5 | 2,617.6 |
| kernel off, mask applied | top-k 50, 2 | 214.7 | 1,258.7 | 2,616.9 |
| kernel off, mask applied | no filter | 214.4 | 1,264.4 | 2,601.0 |
| kernel on, mask applied | top-k 50, 1 | 241.6 | 1,362.0 | 2,822.6 |
| kernel on, mask applied | top-k 50, 2 | 241.5 | 1,363.2 | 2,822.4 |
| kernel on, mask applied | no filter | 241.5 | 1,363.4 | 2,825.2 |

Every request succeeded (0 errors). Limits: one cold start per server and one session; no A/A pair of servers, so the
run-to-run spread between starts is not measured here; other jobs on the machine; Qwen3-0.6B only; the cause of the
difference between the two servers was not isolated; the 8B model was not run.

## Not measured yet

Changed 2026-11-25: the list below was edited after the session of 2026-11-24; items that session ran are marked.
Changed 2026-11-26: the sampler kernel's `topk_topp` mode ran on 2026-11-26 (section 10); its remaining gaps are listed there.

- The RL loop at Qwen3-8B. No 8B loop has run.
- 8B training memory and speed. Weights alone take 16.38 GB of 25.8 GB on one core (from the plan); the 8B trainer
  (`trainer8b.py`) has been tested on CPU only.
- Anything at 27-32B beyond one start probe: Qwen3-30B-A3B started once (section 8.7); Qwen3.8-27B or similar: nothing has
  run. Our patch covers the Qwen3 dense family only.
- A weight update while requests are running with our own script (`68_update_under_load.py` has not run). One update with 8
  requests in flight ran on upstream PR 20's branch (section 8.1).
- Any cost-normalised comparison with GPU, TPU or AMD accelerators. The one GPU figure above is a different layout and a
  different sync path.
- Kernels on hardware: the LoRA merge kernel and the sampler kernel ran on 2026-11-24 (sections 8.5 and 8.6). Not run: the
  LoRA merge kernel in a server at Qwen3-8B, the sampler kernel's `topk_topp` mode at Qwen3-8B or with temperature above 0
  compared token by token (section 10), either kernel inside a full RL loop, any attention change.
- The adapter update at 8B with the shard-only delta, or with the kernel.
- Long-answer cost with the context ladder on; accuracy on a model other than Qwen3-0.6B.
