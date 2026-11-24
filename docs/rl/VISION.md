# Vision, goals and KPIs

Status: draft of 2026-11-23. This fork follows sgl-project/sglang-neuron (Apache-2.0; upstream `main` is one squashed commit, 4e24326, tag v0.1.0). Everything here is a plan with a measured starting point. Where a number is not measured, the text says so.

How evidence is marked in this document:

| Mark | Meaning |
|---|---|
| **HW** | measured on Trainium2 hardware (trn2.48xlarge, Neuron SDK 2.33, v0.1.0 image), one run each unless a row says otherwise |
| **SIM** | run only in the NKI 0.7.0 CPU simulator. Says nothing about speed, and does not show that the kernel compiles or runs on a NeuronCore |
| **CPU** | run only as CPU tests (PyTorch on CPU, no Trainium) |
| **UP** | taken from the text of an upstream pull request. We did not reproduce it |
| **none** | not done |

## 1. Vision

RL post-training on Trainium should be a first-class path.

- A rollout server on SGLang-Neuron accepts new weights and new LoRA adapters in place, with no restart and no recompile.
- It serves models from 0.6B to the 30B class.
- It runs at a speed and cost comparable to GPU, TPU and AMD accelerators.
- It carries kernel-level optimisations whose gains are proven by measurement.
- The work goes upstream where upstream wants it.

Why in place: v0.1.0 refuses every `update_weights_*` call, so new weights mean a restart. For Qwen3-0.6B on TP4 a restart takes 455-460 s with the example configuration, and 139 s with one batch bucket (**HW**, `PR_DRAFT_inplace_weight_update.md`). One RL step takes about 5.4 s wall time (section 3, G1). A restart per step is not workable.

## 2. Where we stand

| Piece | State | Evidence |
|---|---|---|
| RL loop (GRPO-style, GSM8K, LoRA trainer), Qwen3-0.6B, 150 steps | done, 4 runs (3 seeds of one configuration, 1 run of an earlier trainer configuration) | **HW** |
| In-place full-checkpoint update, Qwen3-0.6B, Qwen3-8B, GLM-4-9B-0414, DeepSeek-V2-Lite | measured; equals a fresh start | **HW** |
| In-place LoRA adapter update, Qwen3-0.6B and Qwen3-8B | measured; equals a fresh start; slow at 8B (13.9 s) | **HW** |
| Adapter delta formed per shard on the host (`_LowRankSlice`) | written; 24 of 24 loader cases bit-identical to the full-matrix path | **CPU** only |
| LoRA shard-merge NKI kernel | built, reviewed, wired into the patch behind `RLKIT_DEVICE_WRITE=kernel` | **SIM** and **CPU** only; no `neuronx-cc` compile, no device run, no timing |
| 8B RL loop; 8B trainer | no loop run; `trainer8b.py` passes 36 of 36 checks against a random 3-layer model | **CPU** only |
| 27-32B class model | nothing of ours | **none** |
| Comparison with GPU, TPU, AMD | one GPU run, not cost-normalised; no TPU or AMD number collected | see G2 |
| Upstream | 1 issue (15), 1 comment, no maintainer reply as of 2026-11-23 | read with `gh` |

## 3. Goals

### G1. RL post-training end to end on a larger model on Trainium2

Ladder:

1. Qwen3-0.6B: done.
2. Qwen3-8B (dense): next.
3. A 27-32B class model: Qwen3-30B-A3B (MoE) or Qwen3.8-27B (hybrid with Gated DeltaNet layers).

Where it stands today:

- **0.6B loop, HW.** 32 questions x 8 answers per step, temperature 0.7, up to 256 new tokens, one LoRA update per step, adapter sync to 8 TP4 replicas, 16-process trainer, all on one trn2.48xlarge. Means over steps 2-150 (step 1 compiles the trainer), seeds 2, 3, 4: rollout 2.71 s, train 1.21-1.27 s, export 0.08-0.09 s, sync 1.21-1.22 s, sum of parts 5.22-5.27 s, step wall time 5.40-5.44 s. Greedy accuracy on 500 GSM8K test questions went from 0.572-0.574 to 0.712, 0.728 and 0.702. A fourth 150-step run with an earlier trainer configuration (train 2.83 s a step) went from 0.574 to 0.748. One loop of 150 steps took 1,088 s (seed 2). The same loop on one RTX PRO 6000 (standard SGLang, one replica, merged-weights sync, one seed) took 6.40 s per step summed (6.69 s wall) and went from 0.574 to 0.714. See the GPU caveat under G2. Source: summary lines of `data/2026-11-20/trn2_seeds/30_rl_loop_plain_s{2,3,4}.jsonl`, `trn2_sgl_final/30_rl_loop_long1.jsonl`, `cuda/30_rl_loop_cuda_long_b4.jsonl`.
- **8B, HW, serving side only.** Full in-place update 4.7-6.2 s (12 updates, two servers). Per rank: host graph 0.00-0.02 s, loaders 3.6-4.7 s, copy to device 0.3-1.1 s. Adapter update (rank 16 on seven projections): first update 35.8 s, then 13.9-14.0 s, of which 13.5-13.6 s forms full delta matrices on the host and 0.3 s is the copy. In place equals a fresh start on the same checkpoint: same text, token logprob difference 0.0. No 8B rollout throughput, no 8B trainer run, no 8B RL step.
- **8B training memory.** Weights on one core take 16.38 GB of 25.77 GB (plan figure). A design that holds one 8B copy per core with per-layer gradient checkpointing is untested on hardware (**CPU** only).
- **27-32B.** Upstream PR 20 states that a Miles GRPO run (torchtitan trainer, disk-delta sync) reloads in place on Qwen3-4B TP4 and on Qwen3-30B-A3B TP4/EP4, with 30+ syncs on the 30B run, on trn3.48xlarge (**UP**; Trainium3, not Trainium2). Upstream PRs 17 and 18 report Qwen3-30B-A3B serving on a trn2.3xlarge (TP4, EP4): on `main`, 21.7 output tok/s at concurrency 1 and 793.0 tok/s at concurrency 64 for 128 input / 512 output tokens (**UP**). Our patch covers Qwen3 dense only. It does not cover Qwen3-MoE or the Gated DeltaNet family. For Qwen3.8-27B, we found forward kernels but no backward kernel for the Gated DeltaNet layers, so training it is blocked until one exists.

What finishes each rung:

- 8B: one 8B RL step table (K1) with every part measured, 3 seeds x 150 steps with accuracy above the step-0 noise (K10), trainer-versus-server logprob agreement (K4).
- 27-32B class: a written choice between Qwen3-30B-A3B and Qwen3.8-27B, with the trainer route named. Likely first pick is Qwen3-30B-A3B, because PR 20 covers Qwen3-MoE and PRs 17/18 serve it on trn2. This is an assessment, not a measurement.
- Open for every rung: which trainer runs on Trainium. The probe of TorchTitan Neuron (listed in SDK 2.33.1 as beta) has not been run.

### G2. Performance at least comparable to Google TPU and AMD accelerators, like for like

What "like for like" means here is set in [METHOD.md](METHOD.md), section 5. In short: one protocol on every platform, the full table and not a peak, accuracy gate on every row, cost-normalised.

Where it stands today:

- **GPU row, not comparable.** The only comparison we hold is the 0.6B loop above: a whole trn2.48xlarge (8 serving replicas and a 16-process trainer) against one GPU. The Trainium side syncs a LoRA adapter, the GPU side exports and syncs merged weights, and the trainers differ. The two rows are **not cost-normalised and must not be quoted as parity.**
- **TPU and AMD: no number collected.** We have read how TPU serving handles the same static-shape problem. vLLM TPU precompiles shapes at warm-up, keeps a disk cache, and pads to power-of-two or linear buckets with a tunable gap ([docs](https://docs.vllm.ai/en/v0.11.1/configuration/tpu/)). The vLLM TPU RL write-up of 2026-11-12 ([post](https://vllm.ai/blog/2026-11-12-rl-small-dense-tpu-v7)) lists: many decode steps in one compiled device loop, keeping the KV allocation across a weight update, no padding past the batch token limit, vocab-sharded sampling, and logprob post-processing overlapped with decode. Several match our own findings (section 4, G3). We took no number from these pages.
- **Our own measured waste that a TPU-style design would remove (HW, 0.6B).** One decode bucket costs the same whether 1 or 32 rows are live: 9.3 ms at 1 row against 10.4 ms at 32 rows (compiled context 1024). With the default buckets one row costs 4.6 ms. A step grows with compiled context for the same short requests: 10.4 / 14.8 / 26.5 ms at context 1024 / 4096 / 8192 (32 rows). Row fill of one group of 32 answers is 0.58-0.63. About 54 % of the 576-token training length is padding in the first of four saved runs (prompt lengths exist only for that run).

What finishes G2: the protocol written and fixed (K5, K6); the same protocol run on trn2, one GPU and one AMD or TPU system; a table with all concurrency levels, TTFT and TPOT percentiles, the accuracy gate result, and instance price per hour with date and source.

### G3. One or two NKI kernel optimisations, with before and after numbers

Candidates, in the order we would build them:

1. **LoRA shard-merge kernel.** Computes `out = bf16(float32(base) + a_stack.T @ bt)` on the device, per rank and per adapted shard, from a resident bf16 copy of the base shard. Only the LoRA factors cross from host to device. Serving graphs are unchanged and not recompiled.
   - **SIM:** kernel equals the float32 reference on every entry in all tests (0 differing entries, K 16, 32, 48, 144, 200; partial tiles; LNC 1 and 2; all 0.6B and 8B per-rank shapes). Against the exact float64 sum rounded once, 255 of 6,291,456 entries differ at 8B qkv up to 735 of 25,165,824 at 8B gate_up. Factor layout through the engine's own loaders: 78 tests pass, integer factors bit-exact for TP 1, 2, 4, 8, 16, every rank. Each of three layout mistakes fails on 4 of 4 ranks.
   - **CPU:** the host path with kernel mode, 47 tests pass; at 0.6B TP4 the live parameters equal an independent float32 reference on every entry (0 of 3,933,184).
   - **none:** no `neuronx-cc` compile, no device run, no timing. Open on hardware: whether a device tensor copied into a live parameter serves correctly. The earlier eager device merge wrote values within single units of the host path in under 0.005 % of entries, yet the served output differed from a restart on the host-merged checkpoint. The cause was not found. The kernel does not by itself remove that difference.
   - **What the target rests on:** one measured number, 16.4 GB of 8B weights overwritten on the device in 1.19 s (**HW**, plain TorchNeuron process, no server; PR draft). The proposed 1.5 s target for the 8B adapter swap is an estimate from that number, not a measurement of the kernel.
   - **What may not be claimed:** that the kernel explains the whole drop from 13.9 s (most of it comes from not forming full matrices on the host); that it is the first LoRA on Trainium (NxD Inference serves multi-LoRA with torch ops); any hardware correctness from simulator runs.
2. **Keyed-Gumbel sampler tail kernel** (temperature sampling, Qwen3 dense, TP4). Designed on paper only (**none**: no code, no test). Basis for choosing it: on Qwen3-0.6B, temperature mode runs 2,183-2,218 tok/s against 2,713-2,730 for `topk_topp` (**HW**, 256 requests, 32 in flight, one server). A device profile read by our analysis puts the sampler subtree at 2.62 ms of 26.0 ms summed instruction time per step. The basis is weaker than it looks. The probe started both servers with 20 top logprobs, so by the code both kept the default compiler flags and both ran the top-logprob path. The loop's configuration (temperature, no top logprobs, decode at -O2 by the code) was never timed. The 2.62 ms profile figure comes from that configuration; the throughput gap does not. Hardware step 0 is therefore a kill test with no kernel: time temperature mode with top logprobs 0 at -O2 and at the default flags, next to `topk_topp`. If it reaches 2,600 tok/s (proposed threshold), the gap was not the sampler operations, and the kernel waits. Upper bound on the 0.6B iteration: 1.09x if the gap closes, 1.14x ceiling (estimate from our analysis notes, not published). By the launch scripts the loop ran `temperature` with no top logprobs; not read back from its server logs.
3. **Not chosen now.** Live-length paged decode attention: the fused attention call is 18.1 of 26.0 ms of summed instruction time per step (70 %) on 0.6B, and the step grows with compiled context. But nkilib main appears to skip dead blocks already (SDK 2.33 version unverified), the bound at context 1024 is about 1.04x, and upstream PR 4 (author whn09, open) already works on the early-exit path of this attention call. Revisit after a no-kernel profile at context 1024 against 8192. Device-side shard rebuild for full-checkpoint sync: no loop of ours uses that path, and upstream PR 20 reports about 1.9 s per update on a host path (Qwen3-4B, trn3), so a host fix comes first.

A lesson from public work on Qwen3.8 on Trainium2 ([vibesys-playground/qwen3.8-trainium2-vibesys](https://github.com/vibesys-playground/qwen3.8-trainium2-vibesys), no licence file, so we describe and do not copy): putting a kernel boundary into a graph changes how the compiler lowers the neighbouring operations, so a correct kernel can still change outputs. Every kernel of ours gets an end-to-end check, not only a kernel check.

### G4. Work with upstream, not beside it

Where things stand (read 2026-11-23):

- Upstream PR 20 (author Zhichenzzz, open): `TransformerForCausalLM.reload_weights` copies a new checkpoint into the live parameters in place for Qwen3 and Qwen3-MoE. It hooks `WeightUpdater.update_weights_from_disk` with an AROUND hook, md5-pins the replaced function, and keeps other update paths refused. Claimed: Qwen3-4B TP4, success in about 1.9 s; Miles GRPO trainer-versus-rollout logprob absolute difference 0.007-0.015 across syncs (**UP**, trn3.48xlarge).
- Upstream PR 21 (same author, open): `POST /pull_weights`, which brings a host-local checkpoint to a published version from full checkpoints or zstd deltas with per-tensor checksums. It pairs with PR 20.
- Our issue 15 proposed the same in-place update on 2026-11-22 with the measurements above. It has one comment (ours) and no maintainer reply.
- **Not covered by upstream and not by PR 20 or 21:** the LoRA adapter update path (including the shard-only delta), the NKI shard-merge kernel, and the measured equality with a fresh start on GLM-4-9B-0414 and DeepSeek-V2-Lite.

What we do:

- Test PR 20 and PR 21 on trn2 against our checks (equality with a fresh start, 100 updates in a row, updates under load) and post the results there. This needs hardware and a yes from the owner before anything is posted.
- Once PR 20 lands, rebase the adapter path on it, and send the adapter path as a second PR. Kernel PR only after hardware tests pass.
- This fork carries upstream PRs usable for our goals, one branch per change (`upstream-pr/<N>`, unchanged upstream heads; branch `integration` merges PRs 20, 4, 9, 16, 17, 18), so the fork can run them before they merge. None of them has been run by us. When upstream merges one, the fork drops its copy. Branch layout: [ROADMAP.md](ROADMAP.md), task N1.
- Nothing is posted anywhere without an explicit yes.

## 4. KPIs

Targets are marked **proposed**. A proposed target is a starting point for the team to change before the first run. Acceptance numbers are fixed before a run (principles, section 6).

| # | KPI | Definition (exact measurement) | Baseline today | Target (proposed) | How it is checked |
|---|---|---|---|---|---|
| K1 | RL iteration wall time, split into rollout / train / export / sync, per model size | One RL step: 32 questions x 8 answers, GSM8K, temperature 0.7, up to 256 new tokens, one LoRA update, export, sync to every replica. Mean over steps 2 to N (step 1 compiles shapes). Report each part and the step | 0.6B (**HW**, `30_rl_loop.py`, 3 seeds, 150 steps, means of steps 2-150): rollout 2.71 s, train 1.21-1.27 s, export 0.08-0.09 s, sync 1.21-1.22 s, sum of parts 5.22-5.27 s, wall 5.40-5.44 s. Spread of the summed step inside the seed-2 run: 4.63-5.76 s. 8B: not measured. 27-32B: not measured | 8B and 27-32B: table filled with every part measured. 8B step time within 20 % of the sum of parts measured earlier (basis: plan W3.2). No speed target until the table exists | Summary event at the end of each `30_rl_loop_*.jsonl` file; 3 seeds; script and commit named in the record |
| K2 | Adapter swap time at 8B | Wall time of `update_weights_from_disk` on an adapter folder, steady state (after the first update), 8B TP4, rank 16 on q, k, v, o, gate, up, down; median of at least 5 swaps; split into factors, upload, kernels, copy, sync | 13.9-14.0 s host merge (**HW**); first update 35.8 s. 0.6B: 0.67-0.70 s on one server, 1.21-1.22 s per sync in the 8-replica loop | At most 1.5 s. This is an estimate from one measured device-overwrite number (16.4 GB in 1.19 s), not from the kernel. Intermediate: shard-only host delta 2.5-3.5 s (estimate from a 5.3x ratio of host arithmetic on a laptop; **CPU** only) | `50_adapter_server_test.py`, kernel mode on; first run with `RLKIT_KERNEL_CHECK=all`; stop rule (see [KERNELS.md](KERNELS.md)): above 1.5 s, try one launch per kind over stacked layers, then report what was measured |
| K3 | Swap versus restart: token logprob difference | Same prompts and same sampled tokens scored by (a) the server after the in-place swap, (b) a fresh server started on a checkpoint merged by the same method. Max absolute token logprob difference, and equality of greedy text | 0.0 and same text for host-merge full and adapter updates at 0.6B and 8B, and for full updates on GLM-4-9B-0414 and DeepSeek-V2-Lite (**HW**). Device-side eager merge: not equal (greedy tokens equal on 2 of 3 prompts with mean logprob difference 0.005-0.011; fork at a near tie on the third; PR draft). Kernel path: not measured | Exactly 0.0 on the kernel path; the same adapter swapped back in gives bit-identical weights and outputs; decode step time and compiled-graph keys before and after a swap unchanged | `48_inplace_server_test.py`, `50_adapter_server_test.py`; hardware steps H2 to H5 in [ROADMAP.md](ROADMAP.md); kernel mode stays off if any check fails |
| K4 | Trainer versus server logprob agreement | Mean absolute difference of token logprobs between the trainer and the rollout server on the same sampled answers, before and after an adapter update; also p95 | 0.6B (**HW**): 0.015 before, 0.017 after an update; 0.088 with mismatched weights. 8B: not measured. For scale, PR 20 reports 0.007-0.015 with its own trainer on trn3 (**UP**) | At 8B and for the 27-32B model: no worse than the 0.6B floor of 0.017 (mean). Report p95 | `51_logprob_parity.py` |
| K5 | Rollout tokens per second per chip, full concurrency table | Output tokens/s divided by the number of chips used, at every concurrency level the engine supports (1, 2, 4, 8, ... up to the compiled limit), with fixed input and output lengths. Report TTFT and TPOT p50 and p95 and the whole table, not a peak. State chips used and whether one TP4 server is one chip | 0.6B, partial (**HW**): 2,713-2,730 tok/s per TP4 server at 32 in flight (256 requests of up to 256 tokens, `topk_topp`); step time by live rows from 1 to 32 (`NIGHT_RECORD`, section 6). 8B and 27-32B: not measured. For scale, Qwen3-30B-A3B on a trn2.3xlarge: 21.7 tok/s at concurrency 1, 793.0 at 64 (**UP**, PR 17/18, `main`) | Table filled at 8B and for the 27-32B model with the same protocol on a GPU row and an AMD or TPU row (K6). No ratio set before those rows exist | `55_rollout_bench.py`, `57_ttft_probe.py`, `60_shape_cost.py`; one scripted job, one commit |
| K6 | Cost-normalised throughput | Tokens per second per instance-dollar-hour, from K5 at the same concurrency and the same accuracy gate. Price per hour from a named public page with date; spot and on-demand shown separately | **Not measured.** No price was collected | Set before the first run, from the measured GPU row. The comparison counts as "comparable" only on this KPI, never on the whole-instance step times of K1 | A table with prices, dates and sources next to the K5 rows |
| K7 | Kernel gain, paired A/B | For each kernel: interleaved baseline and candidate runs on the same machine, at least 7 pairs after 1 warm-up (proposed), median of paired differences, at several model sizes and several shapes. The gain must exceed the measured noise floor. No cell may regress. The kernel name must appear in a device profile, and the compiled program must differ | Noise floor: **not measured**. Context: 0.6B decode step 9.13 ms median execution (**HW**, 287 executions per core in a 2.66 s window) | LoRA merge: K2 and K3 hold, decode step unchanged. Sampler: at least 2,600 tok/s on 0.6B (temperature mode with top logprobs 0, one bucket, context 1024; closes at least 80 % of the 2,200 to 2,720 gap; estimate; the 2,200 baseline was measured with top logprobs 20, so ROADMAP B4 first measures the baseline of this configuration), sampler-scope instruction time at most 0.5 ms against 2.62 ms, sampled-token agreement with the XLA path at least 99.9 % over 255 steps x 32 rows, K4 no worse | Hardware steps in [ROADMAP.md](ROADMAP.md); a paired-verdict script tested first on synthetic series; the noise floor is measured before the first claim |
| K8 | Update under load | Full and adapter updates while requests are running, including long ones. Count HTTP errors and answers produced on neither weight set | 0.6B, short requests (**HW**): 40 updates with 8 sender threads, 688 answers, 320 on old and 368 on new weights, none on neither, no HTTP error. 100 updates in a row, 0.26-0.38 s each, no drift. Long requests (2,000+ tokens): not tested | Zero errors; every answer on one weight set; wait or pause time reported, with 8 requests of 2,000+ tokens in flight | `49_sync_stress.py`, `68_update_under_load.py` |
| K9 | Upstream | Counts on GitHub: our issue 15 answered; our review comments with data on PR 20 and PR 21; our PRs opened and merged; fork-only changes with no upstream issue or PR | Issue 15: open, 1 comment (ours), 0 maintainer replies. Upstream PRs 1-21 (15 is an issue): all open, none merged. Our PRs: 0 | One reply on issue 15. One data comment on each of PR 20 and PR 21 after we test them. Adapter-path PR opened after PR 20 lands. Zero fork-only changes without an upstream issue or PR | `gh` read of the issue and PRs, dated |
| K10 | Learning signal | Greedy accuracy on 500 GSM8K test questions, before and after 150 steps, 3 seeds, same evaluation set | 0.6B (**HW**): 0.572-0.574 before; 0.702-0.748 after in four runs. CUDA, one seed: 0.714. Step-0 spread across runs: 0.572-0.598 | 8B: accuracy rises above the step-0 spread in 3 of 3 seeds. No size of rise promised | `21_gsm8k_acc.py` and the loop's eval events |
| K11 | 8B trainer fits and runs | One 8B LoRA step (rank 16, seven projections, micro-batch 1) completes on trn2 without out-of-memory; peak device memory and step time recorded | Not run on hardware. Weights alone: 16.38 GB of 25.77 GB on one core | Completes with the design chosen from the probes. Expected peak 17.3-18.5 GB and under 2 s per compiled micro-batch are estimates only | `70_mem_probe.py` and the 8B trainer probes |
| K12 | Measurement hygiene | Share of reported numbers that come from one scripted job on one commit, with script, commit id and result file named; one run reported as one run | 0.6B loop and sync numbers: result files in the public repo. Some 2026-11-21 server logs and raw profiles are not in the repo (size); the record lists checksums. Kernel numbers: CPU test scripts in `tests/rlkit/` (branch `feat/lora-merge-kernel`) | 100 % of numbers in any KPI table carry script, commit id and result file. A number without them is marked unverified | A check of each table before it is shared |

Notes on the table:

- K1 and the GPU row. The 5.40-5.44 s per step (wall) of the trn2 loop against 6.69 s on one GPU is not a cost-normalised comparison. It must not be quoted as parity.
- K3 and K4 use the engine's own path and a restart as the reference, not only our own reference code.
- Estimates in this table: the 1.5 s adapter target, the 2.5-3.5 s intermediate, the 2,600 tok/s sampler target, the 17.3-18.5 GB peak, and the 2 s micro-batch.

## 5. Non-goals

- Competing on raw inference throughput with engines built for that. Our speed goal (G2) is comparability on an RL workload, measured on a stated protocol.
- Building a from-scratch serving engine. Public from-scratch engines for Qwen3.8 on Trainium2 exist; we use SGLang-Neuron and improve it.
- Anything that needs a serving-graph recompile per RL step. An update that recompiles is a restart under another name.
- New RL algorithms. The loop (GRPO-style policy gradient, GSM8K, LoRA trainer) is a fixed baseline that exercises the infrastructure.

## 6. Working principles

1. An experiment runs only with one stated question and an expected result, written down before the run, with what we do if it holds and if it does not.
2. Acceptance numbers are fixed before the run. A number chosen after seeing the result is reported as such.
3. Correctness is checked against the engine's own path and against a restart on the same weights, not only against our own reference code. Correctness comes before speed.
4. Hardware claims never rest on simulator runs. The simulator checks function, not speed, and not that the compiler accepts the kernel.
5. Survey other people's work before building: upstream pull requests, other accelerators such as TPU, and public kernel libraries. Several of our first ideas changed after reading them.
6. Every reported number comes from one scripted job on one commit, with its condition and its source. A single run is reported as a single run.
7. State limits in the same sentence as the claim: tested on hardware, tested only in the simulator, or not tested.
8. Work with upstream. Prefer an upstream issue or PR over a private patch. Nothing is posted without an explicit yes.

The measurement rules in detail are in [METHOD.md](METHOD.md).

## 7. Open points

- **PR count.** Upstream numbers 1 to 21 are 20 pull requests and 1 issue (15), read with `gh` on 2026-11-23. All 20 are open, none merged.
- **One chip per TP4 server?** The loop uses 8 TP4 replicas on 32 logical cores. We did not confirm that one TP4 server is one chip. K5 needs it.
- **Which sampler the loop ran.** By the launch scripts, `temperature` with no top logprobs (decode at -O2 by the code). Not read back from the loop's server logs, which are not in the public repo. The sampler probe ran a different configuration (top logprobs 20).
- **The 2.62 ms and 26.0 ms profile figures** come from a join of device-profile instructions to compiler operations done by our analysis. The raw profiles are not in the public repo. The instruction time is summed across engines and may not lie on the critical path.
- **GPU row details.** "One RTX PRO 6000" is from the experiment plan. The GPU price and the AMD and TPU rows are not collected.
- **Trainium3 numbers** in PR 20 are from a different chip. They show the route works there. They say nothing about Trainium2 speed.
- **27-32B choice and the trainer route** are not settled (G1).
- **Hardware access.** Every hardware step needs a machine. Access is not in place yet, and each launch needs the owner's yes.
- **Paths.** Experiment scripts and records are named as in the public repo, not in this fork. Code paths are those of the fork branches.

## 8. Sources

| Short name | Where |
|---|---|
| Public repo | https://github.com/pengchenglu1997/trainium-sglang-rl (README, `docs/NIGHT_RECORD_2026-11-21.md`, `docs/experiment-plan-2026-11-22.md`, `data/2026-11-20/` loop files, `patches/PR_DRAFT_inplace_weight_update.md`) |
| Kernel plan and results | [KERNELS.md](KERNELS.md), [RESULTS.md](RESULTS.md); code `sglang_neuron/models/rlkit_lora_merge.py` and tests `tests/rlkit/` (README `tests/rlkit/README.md`) on branch `feat/lora-merge-kernel` |
| Kernel direction analysis | our analysis notes (not published) |
| Public-work check on Qwen3.8 and TPU practice | our analysis notes (not published); summary in [LANDSCAPE.md](LANDSCAPE.md) |
| Upstream | https://github.com/sgl-project/sglang-neuron: issue 15; pull requests 4, 6, 17, 18, 20, 21, read with `gh` on 2026-11-23 |
| TPU pages opened | https://docs.vllm.ai/en/v0.11.1/configuration/tpu/ and https://vllm.ai/blog/2026-11-12-rl-small-dense-tpu-v7 |
