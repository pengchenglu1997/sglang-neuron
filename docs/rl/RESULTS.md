# Results so far

Measured results of the RL post-training work on Trainium2 with SGLang-Neuron, as of 2026-11-23.

Rules for reading this file:

- Every row has the model, the layout, what was measured, the number, the condition, the date and the source file.
- One run each unless a row says otherwise. Nothing here is a cost-normalised comparison.
- Dates are Seattle local dates of the run. Raw result files are under `data/` of the public repo
  https://github.com/pengchenglu1997/trainium-sglang-rl, as named. Server logs and raw profiles of
  the 2026-11-21 session are not in the public repo (size); `docs/NIGHT_RECORD_2026-11-21.md` gives their checksums.
- The last section lists what has not been measured.

Abbreviations: TP4 is tensor parallel over 4 NeuronCores (one rollout server). "Layout" names the machine layout and the
sync path.

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
kernel aim at this number; neither has run on hardware (see [KERNELS.md](KERNELS.md)).

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
CPU mode. The simulator is not a timing model.

## Not measured yet

- The RL loop at Qwen3-8B. No 8B loop has run.
- 8B training memory and speed. Weights alone take 16.38 GB of 25.8 GB on one core (from the plan); the 8B trainer
  (`trainer8b.py`) has been tested on CPU only.
- Anything at 27-32B (Qwen3-30B-A3B, Qwen3.8-27B or similar): nothing has run, and the patch covers the Qwen3 dense family
  only.
- A weight update while requests are running (`68_update_under_load.py` exists and has not run on hardware).
- Any cost-normalised comparison with GPU, TPU or AMD accelerators. The one GPU figure above is a different layout and a
  different sync path.
- Any kernel on hardware: the LoRA merge kernel, the sampler kernel, any attention change.
- The adapter update at 8B with the shard-only delta, or with the kernel.
- Long-answer cost with the context ladder on; the sampler at 8B; accuracy on a model other than Qwen3-0.6B.
