# Landscape: what exists that we build on

Status: draft, written 2026-11-23. Purpose: so that nobody on the team rebuilds something that already exists, and so
that everyone knows where our own work stops and other people's work starts.

This file does not decide what we merge. The merge decisions are in [UPSTREAM_PRS.md](UPSTREAM_PRS.md).

How to read the evidence marks:

- **Opened** means we opened the page, PR text or file ourselves in this pass.
- **Earlier reading** means the item comes from a reading saved earlier (our analysis notes, not published).
  The sources were opened then, but we did not re-open them for this file. Treat these as second-hand.
- **Not opened by us** means we only saw a title, a search snippet or a summary relayed by someone else.
- Numbers are written with their condition. "Reported by the source" means we did not reproduce it.

Our goals, for reference: G1 RL post-training end to end on larger models on Trainium2 (Qwen3-0.6B done, Qwen3-8B
next, then a 27-32B class model); G2 performance at least comparable to Google TPU and AMD on a like-for-like
protocol; G3 one or two NKI kernel optimisations with before/after numbers; G4 work with upstream, not beside it.

---

## 1. Upstream sglang-neuron

Repository: https://github.com/sgl-project/sglang-neuron (Apache-2.0 plugin of SGLang for AWS Trainium).

State, opened on 2026-11-23: `main` is one squashed release commit (`4e24326`, "SGLang-Neuron: squashed release
snapshot"), tag `v0.1.0`. No pull request has been merged. GitHub lists **20 open pull requests** (numbers 1 to 21
except 15). Number 15 is an open **issue**, ours (in-place weight updates for RL rollout servers). GitHub's
open-item counter shows 21 because it counts that issue. Any earlier note that says "21 open PRs" counts the issue.
Because none is merged, everything below is a proposal by its author, not shipped code.

Titles and authors come from `gh pr list -R sgl-project/sglang-neuron --state all`. The right-hand column is our
reading of what the PR means for RL work. In the review pass of 2026-11-23 every PR body was read with `gh pr view`;
the numbers each PR reports, with conditions, are in [UPSTREAM_PRS.md](UPSTREAM_PRS.md). "Opened" and "Earlier reading" keep the marks
of the first draft.

| # | Title | Author | Meaning for RL work |
|---|---|---|---|
| 1 | Add Mooncake (EFA) as a PD KV transfer backend | whn09 | Prefill/decode split with KV transfer. Not needed for a single-server rollout. PR text read in review; see UPSTREAM_PRS.md. |
| 2 | Add Gemma 4 31B (text, bf16) on a trn2.3xlarge | matheus-1618 | A 31B model on one trn2.3xlarge, so a possible G1 rung. Serving only: it has `load_weights` but no `reload_weights` (read in the diff), so PR 20 would refuse an update. PR text read in review; see UPSTREAM_PRS.md. |
| 3 | HiCache L3 on Mooncake Store | whn09 | A persistent KV tier. KV in it would go stale after a weight update unless it is flushed (our issue 15 says this). PR text read in review; see UPSTREAM_PRS.md. |
| 4 | Paged decode: do not pass max_context_len (nkilib FA early exit is wrong past 8K) | whn09 | Correctness fix for long contexts on dense Qwen3, Qwen3-MoE and GLM-4. Cost: decode reads every tile of the compiled context. Opened. |
| 5 | Qwen3-235B-A22B on trn2: refuse the moe_cte expert shard of 1536, document the layouts | whn09 | Very large Qwen3-MoE on trn2. Only relevant to a far G1 rung. Earlier reading, body not re-opened. |
| 6 | Add Qwen3.8 27B (text, bf16) on a trn2.3xlarge | matheus-1618 | A 27B model served by SGLang-Neuron on one chip. This is the nearest existing G1 rung in the plugin. No weight update: it has a host `load_weights` and a module-level `hf_weight_mappings(graph)` but no `reload_weights`. Opened. |
| 7 | Hy3-FP8 (HYV3ForCausalLM): FP8 experts and KV cache, DCP, sparse decode, MTP, PD | whn09 | Large FP8 model with its own kernels. Not RL. PR text read in review; see UPSTREAM_PRS.md. |
| 8 | MiMo-V2.6-Flash-RL on a trn2.48xlarge: TP64/EP64, lossless FP8 experts, fused decode kernels, up to 1M context | whn09 | Source of useful kernel patterns: a fused decode sampler, KV write kernel, and a patch for the nkilib early-exit bug (off by default). Earlier reading. |
| 9 | feat: add Llama 3.1 (LlamaForCausalLM) on the shared dense transformer | ttpks1010 | One more model on the shared dense transformer. Its `LlamaGraph` subclasses `TransformerForCausalLM` (read in the diff), so it inherits PR 20's `reload_weights`; not run. |
| 10 | Keep dynamo traces across launches (SGLANG_NEURON_PRECOMPILE_CACHE=1) | whn09 | Cuts restart time. Matters for any RL setup that still restarts. Opened. |
| 11 | Warm-up: queue every graph compile first, and trace each shape once | whn09 | Cuts cold start. Opened. |
| 12 | Compile warm-up shapes concurrently (trace every shape before warming any) | zcnrex | Cuts cold start. Overlaps with 11 in purpose. Opened. |
| 13 | Qwen3-MoE: serve an FP8 (e4m3) KV cache | zcnrex | More KV capacity for Qwen3-MoE only, with an accuracy cost. Not dense Qwen3. Earlier reading. |
| 14 | MiniMax-H3: run on one trn2 device (TP4) with opt-in single-device options | jimburtoft | Another model on one chip. PR text read in review; see UPSTREAM_PRS.md. |
| 16 | Cache failed torch_mlir builtin-dialect import and traceback frame formatting (~37% faster warm launch) | zcnrex | Cuts warm launch time. Opened. |
| 17 | [Perf] [Trainium] qwen3_moe: Run the prefill experts on moe_cte's hybrid form on Trainium2 | jimburtoft | Prefill speed for Qwen3-MoE (matters if the 27-32B rung is Qwen3-30B-A3B). Earlier reading. |
| 18 | [Perf] qwen3_moe: Run the decode experts selectively for small batches | jimburtoft | Decode speed for Qwen3-MoE at small batch. Earlier reading. |
| 19 | Hy4-Flash-FP8 (HYV4FlashForConditionalGeneration) on half a trn2.48xlarge: TP32/EP32, IndexPool sparse MLA, Engram, up to 1M context | whn09 | Large model with its own kernels. Not RL. PR text read in review; see UPSTREAM_PRS.md. |
| 20 | Reload transformer-family weights from disk in place | Zhichenzzz | In-place full-checkpoint reload for Qwen3 and Qwen3-MoE. The core of our weight-sync goal. Opened. |
| 21 | Serve /pull_weights: pull published weight deltas into a host-local checkpoint | Zhichenzzz | Delta pull route used by Miles disk-delta sync. Pairs with 20. Opened. |

Details of the PRs we opened (numbers are what each PR states, not our measurements):

- **PR 4.** Test model Qwen3-30B-A3B-Instruct-2507, TP8 EP8, trn2.48xlarge, needle test with greedy decoding. With
  `max_context_len` passed, retrieval fails at some lengths above 8K tokens. Without it, it passes at the lengths
  tested. The author says the defect inside nkilib's early-exit loop is not yet located. Our own measurement of the
  same cost is in section 6 (step time grows with compiled context).
- **PR 6.** Qwen3.8 27B, TP8 on the eight NeuronCores of a trn2.3xlarge, radix cache, 32K context. Recipe table: TTFT
  (64 to 512 tokens) 0.20 s, TPOT 28.2 ms, GSM8K 0.990 with fp32 GDN state and the 512 tile; bf16 state variant TPOT
  27.7 ms, GSM8K 0.990. One server at a time, reported by the source.
- **PR 10.** Qwen3-8B, TP4, trn2.48xlarge, empty NEFF cache, ready time: cache off 540 s, cache on and restored 92 s.
  Decode warm-up 412 s down to 13.2 s. GSM8K 0.940 and TPOT 12.71 ms unchanged. Reported by the source. Off by
  default. Not covered: PD, PP, DP attention.
- **PR 11.** Qwen3-8B, TP4, empty NEFF cache, on main (box not named in the PR): ready 1467 s to 540 s, decode warm-up (21 shapes) 1314 s to 412 s.
- **PR 12.** Qwen3-30B-A3B TP4 EP4 on trn3.48xlarge, cold cache: full model, 22 shapes, launch to ready about 35 min
  to 18 min. Reported by the source.
- **PR 16.** Qwen3-30B-A3B on trn3.48xlarge: warm launch 496 s to 311 s (-37%) with the PR 12 warm-up included.
- **PR 20 and 21.** See section 2.

Rungs of G1 that already show up in upstream (none carries an RL claim except through PR 20 and 21):

- Qwen3-30B-A3B: used as the 30B test in PR 20 and 21 with Miles GRPO on trn3.48xlarge (TP4/EP4); also the test model of
  PRs 4, 12, 16 and 18.
- Qwen3.8 27B: PR 6.
- Gemma 4 31B: PR 2.

Note on hardware: PR 20 and 21 validate on **trn3.48xlarge**. Our measurements are on **trn2.48xlarge**. A
result on one does not stand in for the other.

---

## 2. Weight sync for RL on Trainium

### 2.1 Upstream PR 20 and PR 21 (opened)

**What they do.** PR 20 (opened 2026-11-23) lets `update_weights_from_disk` work for transformer-family models
(Qwen3, Qwen3-MoE). The host reads this rank's shards with the graph's own weight mappings and `copy_()`s them into the
device parameters the compiled graphs already share. No graph is rebuilt. It refuses models whose graphs do not copy a
checkpoint (packed or giga-kernel weights) and pipelined servers. The replaced upstream function is md5-pinned. Other
update paths (tensor, distributed, IPC) stay refused.
PR 21 (opened 2026-11-23) adds `POST /pull_weights {local_checkpoint_dir, source_dir, target_version}`. A trainer
publishes each version under `source_dir/weight_v{N:06d}/` as a full HF checkpoint or a zstd delta against version N-1
with per-tensor checksums. The pull seeds from the server's model path, applies deltas in order, verifies checksums,
and takes a file lock. The delta code is copied from the `sglang-miles` branch of sgl-project/sglang. The engine then
reloads with PR 20's route.

**How validated, as the PRs state it.** trn3.48xlarge, `public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0`
plus the change. Qwen3-4B TP4: update returns success in about 1.9 s, greedy output identical before and after
reloading the same weights. Miles GRPO (torchtitan trainer, disk-delta sync) on Qwen3-4B TP4 and Qwen3-30B-A3B TP4/EP4;
trainer-vs-rollout `train_rollout_logprob_abs_diff` stays at 0.007-0.015 across syncs; a 30B run passed more than 30
syncs. Reported by the source, not reproduced by us.

**What they lack for our goals.**

- Models other than Qwen3 and Qwen3-MoE keep refusing. Data-parallel engines not covered (PR 20 text).
- No LoRA adapter path. Only full checkpoints and full-checkpoint deltas.
- The delta is applied on the host. No device-side apply and no kernel.
- No test of an update while requests are in flight is reported in either PR.
- Hardware is trn3, not trn2. No 8B number is given.
- The two PRs touch adjacent lines of `general_plugin.py`, so whichever lands second needs a rebase (PR 21 text).

### 2.2 Our issue 15 and in-place patch (our public repo, opened)

Issue: https://github.com/sgl-project/sglang-neuron/issues/15 (opened 2026-11-22, our proposal; at the time we read
it, it holds the issue text and one comment, both ours, and no maintainer reply).
Public repo: https://github.com/pengchenglu1997/trainium-sglang-rl (Apache-2.0; `README.md`,
`docs/NIGHT_RECORD_2026-11-21.md`, `docs/experiment-plan-2026-11-22.md`, `patches/`).

**What it does.** A patch of three modified plugin files plus two new ones against v0.1.0. `update_weights_from_disk` is
no longer refused. It drains the device queue, runs the model's own loaders on the new checkpoint on the host, and
`copy_`s each tensor into the live parameter of the same name. No parameter object is replaced; nothing is compiled
or warmed. A second path takes a LoRA adapter (a PEFT directory) and writes `scale * (B @ A)` into the same
parameters, starting from a host copy of the base shard.

**How validated** (trn2.48xlarge, SDK 2.33, v0.1.0 image, one run each unless stated; source: our PR draft and issue
text):

| check | result |
|---|---|
| restart-based sync, Qwen3-0.6B TP4 | 455-460 s at 21 decode shapes; 139 s at one batch bucket (of which weight loading 0.45 s) |
| full update, Qwen3-0.6B TP4, over HTTP | 0.34-0.41 s |
| full update, Qwen3-8B TP4 | 4.7-6.2 s (loaders 3.6-4.7 s per rank, copy to device 0.3-1.1 s) |
| in place vs fresh start, same checkpoint | same text, token logprobs identical (0.6B, 8B, GLM-4-9B-0414, DeepSeek-V2-Lite) |
| 100 updates in a row (0.6B) | all ok, every answer equal to its reference, 0.26-0.38 s each |
| 40 updates with 8 threads sending requests (0.6B) | 688 answers, each on old or new weights, no HTTP error; short requests only |
| adapter update, rank 16, 7 projections (0.6B) | 0.67-0.70 s after a first of 7-9 s |
| adapter update (8B), host path | 13.9 s, 13.5 s of it forming full matrices |
| adapter 1, then 2, then 1; zero adapter | third equals first exactly; zero adapter gives the base exactly |
| RL loop, 0.6B, 8 replicas, adapter sync, 16-process LoRA trainer | three seeds of the final configuration: 5.40-5.44 s a step (wall, mean of steps 2-150), accuracy on 500 GSM8K test questions 0.572-0.574 to 0.702-0.728. An earlier run with a slower trainer: 6.83 s a step (summed parts), 0.574 to 0.748 (one run). The PR draft's "5.6 s a step" for that run does not match its result file; the file is the source here |

Not run on hardware (stated in the repo): the shard-only adapter delta (`_LowRankSlice` in `models/host.py`); on CPU
it is bit-identical to the full-matrix path for 24 of 24 loader cases. Also `66_pull_weights_shim.py`, our local shim
of the `/pull_weights` route that Miles disk-delta calls, and the 8B trainer.

**What it lacks.**

- Written against v0.1.0 without the open PRs. No unit tests. Not in the shape of a PR (draft lists what it still needs).
- Qwen3 dense, GLM-4 and DeepSeek-V2-Lite only. Kernel models with packed or quantized weights untried.
- `weight_version` and the prefix-cache flush after an update are left to upstream's scheduler. We checked only that
  answers after an update match a fresh start. A persistent KV tier (PR 3) would go stale.
- Long requests in flight during an update were not tested.
- The patch ties the compile cache to a hash of plugin sources, so every patch edit forces a cold compile (4-5 min each
  for us).
- The attention change of PR 4 is not carried; apply that PR's diff to try it.

### 2.3 Overlap between upstream and us

| Capability | Upstream PR 20 / 21 | Our patch |
|---|---|---|
| full checkpoint, in place, from disk | yes (Qwen3, Qwen3-MoE) | yes (Qwen3 dense, GLM-4, DeepSeek-V2-Lite tested) |
| per-model capability flag, md5-pinned replaced function | yes | proposed in the draft, not built |
| `/pull_weights` delta route | PR 21 | local shim only, not run on hardware |
| LoRA adapter update in place | no | yes, trn2 hardware for 0.6B; 8B shard-only path CPU only |
| update with requests in flight | not reported | 40 updates with 8 threads, short requests |
| validated hardware | trn3.48xlarge | trn2.48xlarge |
| RL trainer in the loop | Miles GRPO, torchtitan trainer | our own LoRA GRPO loop on GSM8K |

### 2.4 vllm-neuron weight transfer (opened; public pages only)

PR: https://github.com/vllm-project/vllm-neuron/pull/98, "Add Qwen3.5 hybrid model, neuron weight-transfer, and
sleep/wake_up offload for RL rollout", author susmit-holbox, open, created 2026-11-12.
Public fork: https://github.com/susmit-holbox/vllm-neuron. Branches that show the feature split we follow:
`feat/qwen3-5-gdn-a1`, `feat/neuron-weight-transfer-backend`, `feat/neuron-sleep-wake`, `feat/rollout-engine`
(commit titles seen: "Add neuron weight-transfer backend (RL weight sync, A2)", "Add neuron sleep/wake-up backend
(RL weight offload, A3)", "Add Qwen3.5 decode path: full-attention KV cache + GDN recurrent state").

**What it does (PR text).** A1: a native Qwen3.5 hybrid model (gated attention plus GatedDeltaNet) in eager PyTorch.
A2: a weight-transfer backend so a trainer can push updated weights into a serving rollout in place, without a restart
and without invalidating the compiled graph. A3: sleep/wake_up that copies weights to CPU and back in place,
keeping the NEFF's tensor addresses.

**How validated (PR text).** A1 verified token for token against `transformers` (greedy) on a trn2.48xlarge. A2 and A3
device-tested: weight round trip, and offload to CPU and restore exact. The PR gives no timing numbers.

**What it lacks for us.** Phase 1 only: eager single core (TP=1), text only, `max_num_seqs=1`. KV and GDN state are
per-sequence buffers, not paged. GDN runs as eager PyTorch (the NKI kernel is a follow-up). `sleep` does not
reclaim HBM. No LoRA path mentioned. It targets vLLM, not SGLang, so none of it drops into our server, but it is the
closest public design to in-place weight transfer on Neuron and the branch layout is the one we copy.

### 2.5 Miles (opened: README and docs through the GitHub API)

Repository: https://github.com/radixark/miles (Apache-2.0). An RL post-training framework that pairs SGLang for
rollout with Megatron-LM, FSDP2 or torchtitan for training.

**What it does for weight sync.** Transfer modes in the argument parser: `broadcast`, `broadcast_packed`, `p2p`
(RDMA), `disk-delta`, `ici`, `http-lora`. Disk-delta (docs/advanced/disaggregated-rollout.md): the trainer publishes
versioned deltas of the canonical checkpoint to shared storage; each rollout host pulls the version and patches its
host-local checkpoint, verifies per-tensor checksums of the resulting state, reloads, advances the engine weight
version and resumes. Default delta encoding is XOR, which must be applied exactly once to the declared base version.
For LoRA, adapter tensors are pushed and each push registers an immutable version under its own engine-side name, so
in-flight sampling on an older version is not disturbed (docs/advanced/lora.md).

**How validated.** Registered end-to-end test: Qwen3-4B Megatron trainer (TP2 on two GPUs) with two TP1 SGLang engines
on the other two GPUs of one 4-GPU node. The README lists NVIDIA and AMD GPUs. Trainium does not appear in the README.
Neuron rollout with Miles is shown only in the PR 20 and 21 text.

**What it lacks for us.**

- The docs state that current `main` rejects disk-delta with LoRA, with `--colocate` and with PD. So the disk-delta
  route and the adapter route do not combine. This is the exact place where our adapter path is useful.
- LoRA push modes are described for SGLang on GPU. Whether they work against a Neuron engine is not shown anywhere we
  opened. Not opened by us: the `http-lora` mode details.
- Needs a rollout engine that answers its HTTP routes. On Neuron these are PR 20 and 21 (and our shim).

### 2.6 NxD Inference multi-LoRA (earlier reading, not re-opened)

Repository: https://github.com/aws-neuron/neuronx-distributed-inference. From the saved upstream reading:

- `modules/lora_serving/lora_layer.py` gathers per-request adapter weights with `torch.index_select` over a stacked
  weight and applies `torch.einsum`. There is no NKI kernel. `examples/slora.md` is the example.
- Contrib PR 166 (open) updates LoRA buffers at runtime with `input_output_aliases`, reported at under 1 ms for a
  diffusion model, without recompiling. This is a pattern for in-place device buffer writes. Not opened by us.
- No RL, no weight sync loop, no merge-into-base. Adapters are applied per request, not merged.

For us: a different design (apply per request, keep base untouched) from ours (merge into base shard). It is the only
public Neuron multi-LoRA path we found, and it uses plain torch ops.

---

## 3. Kernel libraries and examples

### 3.1 aws-neuron/nki-library (earlier reading, not re-opened)

https://github.com/aws-neuron/nki-library. From the saved upstream reading of its git tree:

- Attention: `attention_tkg` (decode), `attention_tkg_seq_packed`, `attention_cte`, `attention_segmented_cte`,
  `attention_kv_parallel_segmented_cte`, `gen_mask_tkg`; experimental `attention_const_max`, `swa_fused_cte`,
  `attention_mxfp8_tkg`.
- Other core kernels: `qkv`, `mlp`, `moe` (cte, tkg), `moe_block_tkg`, `output_projection`, `rmsnorm`, `rope`,
  `cumsum`, `topk` / `rotational_topk`, `router_topk`, `cascaded_max`.
- Losses: `experimental/loss/cross_entropy.py` has forward and backward kernels with an online log-sum-exp over vocab
  chunks (default 32768, up to 65535). It is a template for a logprob-of-sampled-token kernel, not a sampler.
- Others: `experimental/foreach` (elementwise add/sub/mul/div), `experimental/optimizer/fused_adam`.
- **No LoRA, sampler, Gumbel or LM-head kernel.** No file name in the tree contains those words (checked by file name
  only, not by content).
- GDN: not named in the earlier reading's summary, so we cannot say. Not checked. PR 31 (open) is an `ssd_block` kernel.
- Dynamic flash attention (decode) needs a block KV cache, `use_pos_id=True`, and a prior length that is a multiple of
  the tile size (`dynamic_fa_design_spec.md`). It is the code PR 4 steps around.
- Activity: last mainline sync seen 2026-10-19. Issue 30 asks whether `dge_mode.none` DMAs are ordered against later
  DMAs on the same memory. This matters for any kernel that writes and then reads HBM.

Where the plugin uses it: `sglang_neuron` main wraps nkilib kernels (for example `rmsnorm_residual_prefill`, and
`rotational_topk` in `sampling_topk.py`). The wrapper pattern (`wrap_nki`, a torch fallback on CPU) is the one our kernel
code should follow so that the CPU path stays testable. In `sampling_topk.py`, per the same reading, the rotational
top-k kernel is used on trn2.

### 3.2 aws-neuron/nki-samples (earlier reading, not re-opened)

https://github.com/aws-neuron/nki-samples.

- `contributed/decode_attention.py` (merged August 2026): single-query GQA decode with KV tiling and online softmax.
  Validated on the simulator only, and says it carries no compatibility guarantees. Split-KV for long context is
  marked planned. A readable reference, not a production kernel.
- PR 136 (closed, not merged) proposed an exact top-k mask by radix selection over the fp32 bit pattern. A sort-free
  method that could be tested in the simulator.

### 3.3 AWS neuron-agentic-development skill pack (earlier reading, second-hand)

Public source: https://github.com/aws-neuron/neuron-agentic-development (Apache-2.0, confirmed by `gh api`). Our reading
was of the copy that VibeSys vendors, pinned to commit `e63853627271bcb98d941c242641e698d210d5a0`, which VibeSys's
README ties to AWS Neuron 2.33. We did not compare that copy with the AWS repo.

What it holds: skills for writing NKI kernels, profiling, profile querying, debugging and an NKI docs set. Points that
matter to us:

- The kernel-writing constraint file says its rules were validated against **NKI 0.7.0**. That is the version of our
  CPU simulator. Treat it as the syntax reference for our simulator-tested kernels. The pack contradicts itself in
  places (another file still calls NKI 0.4.0 the latest), so check signatures against the installed stubs.
- A separate VibeSys-written safeguards file in the vendored pack says a CPU test and the simulator do not prove device
  correctness, because the simulator misses some out-of-bounds DMA. That rule is not in the AWS text.
- Trn2 constants it states: 8 NeuronCore-v3 per device, 96 GiB HBM, 3 TB/s; per core SBUF 28 MiB (128 partitions of
  224 KiB), PSUM 2 MiB; tensor engine peak 78.6 TFLOPS bf16 per core (fp32 peak as stated there, not re-checked). A worked matmul example uses 435 GB/s DMA
  bandwidth per core.
- DMA rule: best throughput needs transfers with 4 KiB or more per partition. A doc example went from 259 GB/s at
  1024-byte descriptors to 354 GB/s at 4096 bytes.
- A profile-based "performance bounds" method (memory bound, compute bound, perfect pipeline) for reading
  `neuron-explorer` output. It does not diagnose kernels bound by the vector, scalar or GpSimd engines.
- Limits: no LoRA, delta-merge, sampler or Gumbel example. Nothing on `neuronx-cc` -O levels. The profiling scripts
  assume the old `torch_xla` flow.

### 3.4 One third-party engine's notes (earlier reading, not opened by us)

`RunanywhereAI/trnrt` (https://github.com/RunanywhereAI/trnrt, `docs/MEGAKERNELS.md`). It reports: a minimal NKI call in
a dependent chain costs about 6.3 us; one NKI program streams 340-400 GB/s of weights on trn2 and the two programs of a
logical core together 672-708 GB/s; left to the compiler, 98% of one kernel's bytes went through the GpSimd software
DGE and static descriptors cut a call from 64 to 54.8 us. Those are the source's numbers for its own model and tensor
parallel size. For us they are leads to check on a profile of our own decode graph, not numbers for us.

### 3.5 What we know about our own kernel

Our LoRA shard-merge kernel (`sglang_neuron/models/rlkit_lora_merge.py` on branch `feat/lora-merge-kernel`; tests in
`tests/rlkit/`; details in [KERNELS.md](KERNELS.md)) computes
`out = bf16(float32(base) + a_stack.T @ bt)`.

- Tested only in the **NKI 0.7.0 CPU simulator** (trn2 target override). All simulator tests pass; the kernel is
  bit-identical to a numpy float32 reference (`bf16(float32 sum)`) in every case, including all four dense kinds for
  Qwen3-0.6B and Qwen3-8B at TP4 (over about 52 million output entries). Against the exact float64 sum rounded once,
  between 1e-5 and 1e-4 of entries differ, mostly by one unit (`test_merge_kernel_sim.py` in `tests/rlkit/`; [KERNELS.md](KERNELS.md) section 1).
- Not compiled with `neuronx-cc`. Not run on a NeuronCore. No timing. No test inside the serving stack.
- Simulator evidence is not device evidence (see 3.3).

---

## 4. Practice on Google TPU that follows the same static-shape logic

Source of this section: our saved notes on TPU practice (earlier reading, not published). The URLs listed were opened
then. We did not re-open them for this file. The "reported" column is what the source says, not what we measured. "Status" compares with
SGLang-Neuron v0.1.0 (`main`, commit `4e24326`). Where we say "read", we opened the file on `main` ourselves in this
pass. Where we say "not checked against the code", we did not.

| Practice | Source URL | What was reported there (reported by the source) | Transfers to Trainium? | Status in SGLang-Neuron v0.1.0 |
|---|---|---|---|---|
| Precompile and persistent compile cache | https://docs.vllm.ai/en/v0.11.1/configuration/tpu/ ; https://docs.jax.dev/en/latest/persistent_compilation_cache.html | vLLM on TPU compiles common shapes at first launch and stores them on disk or remote storage for later launches. JAX's disk cache is keyed on HLO, jaxlib version, flags and device config; only rank 0 writes, so it needs shared storage. | Yes. Same idea as the NEFF cache. The shared-storage point matters for several workers. | Partly done. Disk NEFF and HLO caches exist (`backend/compile_cache.py` on main, read), and the warm-up compiles each shape. Traces are not kept across launches: that is PR 10. Our restart: 455-460 s at 21 decode shapes. |
| Bucket padding policy and bucket gap | https://docs.vllm.ai/en/v0.11.1/configuration/tpu/ ; https://tunix.readthedocs.io/en/stable/rollout.html | Two padding modes: power of 2 (default) and linear buckets with a configurable gap. Too fine a gap means more buckets, more warm-up and graph memory. Too coarse gives no gain over power of 2. Tunix says to keep `max_prompt_length` fixed because longer prompts round up and may recompile. | Yes. The same trade-off applies to Trainium bucket ladders. | Not checked against the code. v0.1.0 has per-bucket graphs (`graph_runner.BucketConfig` is named in `models/host.py` comments). The policy and whether the gap is tunable were not read. Our measurement: with one bucket, a step costs the same for 1 or 32 live rows (9.3-10.4 ms); with default buckets a one-row step is 4.6 ms. |
| Multi-step decode inside one compiled device loop | https://vllm.ai/blog/2026-11-12-rl-small-dense-tpu-v7 | "Continue Decode" moved many decode iterations into one compiled XLA loop. The write-up reports that the cross-stack Continue Decode path cut a 616 s clean step to 369 s (-40.1 %) in its controlled MaxText ladder (page opened 2026-11-23). | Unknown. Needs neuronx-cc to compile a loop that carries KV state. Unverified for Neuron. | Not done, as far as a search goes. A search of `sglang_neuron` on main for multi-step decode terms found nothing relevant. Not read in depth. |
| KV reuse across a weight update | https://vllm.ai/blog/2026-11-12-rl-small-dense-tpu-v7 | After each policy update the prefix mappings are reset while the physical KV memory stays allocated (when HBM headroom allows it), so no re-allocation. KV computed under the old policy must never be visible to the new one. | Yes, directly, for an RL rollout server with a prefix cache. | Not checked against the code. Our patch leaves the prefix-cache flush to upstream's scheduler. We verified only that answers after an update match a fresh start. |
| Over-padding | https://vllm.ai/blog/2026-11-12-rl-small-dense-tpu-v7 | They fixed a runner that padded past the token budget. | Yes, as a check. | Not checked against the code. Our own measure: row fill of one group of 32 answers is 0.58-0.63 (tokens divided by 32 times the longest answer), Qwen3-0.6B. |
| Vocabulary-sharded sampling | https://vllm.ai/blog/2026-11-12-rl-small-dense-tpu-v7 | Sharded sampling removed a full-vocabulary logits all-gather (about 78 MB per data-parallel rank, as the write-up states). | Yes. | Already done. `sample_sharded_logits` in `sampling.py` (read) takes each shard's best Gumbel score, token, raw logit, log-normalizer and optional top-logprob candidates and sends one small all-gather, not the full logits. Done as torch ops, not as a kernel (see section 6). |
| Overlapped logprob extraction and detokenization | https://vllm.ai/blog/2026-11-12-rl-small-dense-tpu-v7 | Logprob extraction and detokenization were overlapped with decode. | Relevant: an RL rollout server must return logprobs. | Not checked against the code. The sampler computes logprobs on the device (read); whether host-side extraction overlaps with the next step was not checked. |
| Separate tuning of fetch tiles and compute tiles in the attention kernel | https://vllm.ai/blog/2026-11-12-rl-small-dense-tpu-v7 | A change to RPA v3 decoupled HBM fetch geometry from VMEM compute tiling and raised decode throughput from 64,929 to 96,273 tok/s (+48%) at TP4 (reported by the source, on TPU; page opened 2026-11-23). | Idea transfers: tune DMA tile and SBUF compute tile separately. The 48% does not. | Not checked against the code. Decode attention comes from nkilib `attention_tkg`; whether its tile parameters can be set from the plugin was not read. |
| Pallas as the analogue of NKI | https://docs.jax.dev/en/latest/pallas/index.html ; https://docs.jax.dev/en/latest/pallas/tpu/details.html ; https://vllm.ai/blog/2026-10-20-vllm-torchtpu | Pallas gives explicit control of memory movement and pipelining. Inputs sit in HBM, pipelined buffers in VMEM, scalar work in SMEM. Block shapes must be divisible by 8 and 128. TorchTPU registers Pallas kernels as custom ops inside the compiled graph. | Partly. Same job as NKI, but Pallas pipelines for you while NKI exposes SBUF, PSUM and engines more directly. Memory sizes differ. | Already done in kind: the plugin calls NKI kernels from compiled graphs (for example `rmsnorm_residual_prefill`, `rotational_topk`). Our own kernels are not yet in a serving graph. |
| Ragged Paged Attention (RPA) paper | https://arxiv.org/html/2604.15464 | Irregular paged-KV access fits badly with a compiler built for static shapes. RPA pads to fixed upper bounds, picks kernel variants for decode, prefill and mixed batches, overlaps the KV write with attention, and keeps four block-size parameters in an offline-tuned table. | Partly. The static-shape constraint is the same. Ideas worth borrowing: a kernel variant per workload, a tile table per shape bucket, a KV write fused into attention. | Not checked against the code. nkilib has separate decode (`attention_tkg`) and prefill kernels; a fused KV-write kernel exists in PR 8 for MiMo (earlier reading). Not read for Qwen3. |

Other things the TPU reading found:

- SGLang-JAX is the closest sibling of SGLang-Neuron: it precompiles graphs per batch size and overlaps CPU batch
  preparation with TPU compute (https://lmsys.org/blog/2025-10-29-sglang-jax). Whether SGLang-Neuron has an equivalent
  overlap scheduler was not checked.
- Tunix syncs weights in memory, but its vLLM LoRA path is documented as work in progress
  (https://tunix.readthedocs.io/en/stable/rollout.html). The reading found no proven TPU LoRA hot swap for RL.
- A search snippet describes a Pallas LoRA kernel with padded inputs in vLLM. **Not opened by us; unverified.**
- The official PyTorch/XLA recompilation pages are now deprecation stubs, so the evidence for a torch-neuronx
  comparison is thinner than for JAX.
- JetStream is archived and moved to tpu-inference. The pages we have did not state its bucketing logic.

---

## 5. Agent-built engines on Trainium (VibeSys)

Sources: framework https://github.com/uw-syfi/vibesys (MIT, confirmed by `gh api`). Public run record
https://github.com/vibesys-playground/qwen3.8-trainium2-vibesys. **That record has no licence** (`gh api` shows none), so
we read it and do not copy its code. Everything below is from the saved readings (earlier reading), not re-opened.

### 5.1 What was shown

| Engine | Result (reported by the source) | Condition |
|---|---|---|
| Qwen3.8-27B | 194.8 tok/s (round 1) to 860.7 tok/s (round 37) over 37 rounds | 64 concurrent requests, 1k input and 1k output tokens, one trn2.3xlarge chip, TP=4. Own first version as baseline. The record's own noise bar is 2%, and several rounds it marks as accepted are logged as within noise, so read 861 as about 857 +/- 2%. |
| Qwen3.8-Flash-Next | 403.69 tok/s (round 1) to 1,242.90 tok/s (official round-12 measure); two re-measures 1,245.1 and 1,246.6 | Profile A: burst of 64 requests, 8,192 input and 1,024 output tokens, 2 replicas of DP8 x TP4, MTP with one draft token, one trn2.48xlarge (16 chips). Identical code spans about 1,080 to 1,264 tok/s on the headline, so the authors decided most rounds on TPOT and TTFT bands. Final A TPOT p50 35.89 ms. |

Our arithmetic, not the source's: 1,245 divided by 16 chips is about 78 tok/s per chip. The source publishes no per-chip
figure and no comparison with other accelerators. This is a throughput burst of random-token prompts, not an RL rollout.

Where the 27B gain came from (reading of the per-round evidence): about +240 of the +666 tok/s is NKI kernel work
(a single-pass GDN decode kernel, a decode attention kernel, a small prefill GEMM for one projection). The larger early
steps were graph and data-layout changes: GDN state updates as a masked `where`, a dense triangular inverse in prefill,
bit-exact data movement in prefill. For Flash-Next, only the round-12 active-expert decode kernel is new hand-written
NKI (TPOT 37.6 to 33.3 ms, -11.5%, on profile B); other gains came from calling existing nkilib kernels, torch-level
changes, expert placement and masks.

### 5.2 What they do not have

- Flash-Next: greedy argmax only. The `/generate` API has no temperature, top-p or logprob field (search found none).
  No weight update path; changing weights means restarting about 64 rank processes. Hard-coded to two profiles with
  static buckets. It cannot serve a dense Qwen3 model without a rewrite.
- 27B: a bespoke TorchNeuron server, not SGLang. A search found no weight-update path, no LoRA, no sampler kernel, no
  RL hooks. Kernels are tied to Qwen3.8 shapes (head dim 256, 6 query heads and 1 KV head per rank, GDN state 128 x 128,
  TP=4, 64 slots).
- The Flash-Next handover is a record, not a runnable package: compiled artifacts, weights and the compile cache are
  not included.
- Full-model accuracy for Flash-Next is not established: the original exact replay test already fails on the unchanged
  seed engine (reported by the source).
- VibeSys publishes no TPU numbers, and its AMD tasks have no stored measured table for the 9B model. Its one headline
  AMD result (a 397B model on 4x MI300A) is a blog figure we know only through a relayed summary: **not opened by us**.

### 5.3 Lessons, in our own words

1. **A kernel boundary changes how the compiler lowers the ops next to it.** After they added an NKI call, XLA lowered
   neighbouring sums in a different order and flipped single bits; in one case 26% of elements of a gate product
   differed. A kernel that is correct alone can still change tokens. So the full-model accuracy gate runs after every
   kernel, not only the kernel's own test. One of their GDN kernels passed numeric checks and failed that gate.
2. **The simulator does not prove the device.** The VibeSys safeguards file in the vendored pack says the simulator misses some
   out-of-bounds DMA. They check each new kernel on the device at start-up, at the largest shapes, against the
   reference path, and exit with an error on mismatch. We should do the same for our two kernels. Their 27B record used
   no simulator at all; every kernel was compared with the compiler's own path on the device.
3. **Measure A against B in a paired way.** Run baseline and candidate alternately on the same node, and compare the
   difference against measured noise. A single before/after pair is not evidence: the same Flash-Next code varied
   by about +/- 8% on its headline.
4. **Most gains came from graph and data-layout changes, not new kernels.** The biggest NKI win (decode attention)
   removed compiler-inserted layout churn (about 4,816 identity transposes and 290 MB read against 134 MB needed), not
   slow arithmetic. Several kernel ideas did not pay in the served graph: a skinny decode GEMM, folding two transposes
   into a kernel, a prefill GEMM on all projections (it caused accuracy flips).
5. **Measure existing library kernels before writing a new one.** Flash-Next rounds 2 and 3 gained +19% and +70% by
   calling existing nkilib kernels the right way. A hand-written kernel needed a DMA microbenchmark first, which
   chose whole experts per program (553 GB/s) over splitting rows (256 GB/s).
6. **Unverified: NKI trace caching and closure constants.** An earlier note says a per-bucket kernel written as a
   closure reused the first bucket's trace and gave wrong output. We could not find this in our saved readings, so treat
   it as unverified until someone finds the round that shows it. Passing sizes as kernel arguments is cheap either way.
7. **Write the falsifier first.** Their rounds name the result that would make them drop a change. Several of
   their changes were dropped that way.

---

## 6. Gaps nobody covers (as far as this reading goes)

"Covers" means we found nothing public that does it. We searched the sources named in sections 1 to 5. We did not search
everything. Treat each line as "not found", not as "does not exist".

1. **LoRA adapter sync on device.**
   - What exists: our host-side adapter path (trn2, 0.6B, hardware-tested; 8B shard-only form CPU-tested). Upstream PR
     20 and 21 move full checkpoints and full deltas on the host. Miles rejects disk-delta together with LoRA. NxD
     Inference serves adapters per request with torch ops.
   - What does not exist: applying the adapter delta to the live parameters on the device. Our LoRA shard-merge kernel
     is written for this. It runs only in the CPU simulator. Whether it brings the 8B adapter swap (13.9 s today, of
     which 13.5 s is forming full matrices on the host) toward the proposed 1.5 s is unknown; it needs hardware.
2. **Update under load.**
   - What exists: our 40 updates with 8 threads of short requests (0.6B). PR 20 and 21 report 30+ syncs in a Miles GRPO
     run but no in-flight test.
   - What does not exist: a test with long requests in flight, with `weight_version`, and with a prefix-cache flush
     checked after the update. Nobody reports KV reuse across an update on Neuron either.
3. **Sampling with exact logprobs as one kernel.**
   - What exists: the plugin's sharded sampler in torch ops (exact Gumbel-max over the full vocabulary plus logprobs
     with one small all-gather). Our measure on Qwen3-0.6B (256 requests of up to 256 tokens, 32 in flight, three
     passes): `topk_topp` 2,713-2,730 tok/s, `temperature` 2,183-2,218 tok/s, both with 20 top logprobs. We do not know why
     the second is slower; the noise hash running as XLA ops over `[rows, vocab]` is a candidate and not measured. The
     RL loop's configuration (no top logprobs) was not timed ([KERNELS.md](KERNELS.md) section 2). The MiMo sampler kernel
     (PR 8) is one NKI kernel but for TP64, caps top-k at 128 and returns no full-vocabulary log-normalizer.
     A Kimi-K3 full-vocabulary sampler kernel is on main. The nkilib cross-entropy kernel has the online
     log-sum-exp pattern. (The kernel facts in this item are from the earlier reading.)
   - What does not exist: a kernel for Qwen3 dense at TP4 that samples and returns exact logprobs in one pass. This is
     a design idea in our earlier reading, not built, not measured.
4. **Like-for-like comparison with TPU and AMD (G2).**
   - What exists: TPU practice and numbers from vLLM and Tunix write-ups (section 4), Flash-Next and 27B numbers on
     Trainium (section 5), and Miles blog entries for AMD (listed in the Miles README, not opened by us).
   - What does not exist: one protocol (model, precision, context, concurrency, sampler, metric) run on all three. The
     numbers above use different models, different workloads and different metrics. VibeSys's own AMD records
     say that only paired same-node runs count as evidence, because identical code differed by tens of percent across nodes
     (an AMD example README, second-hand reading). A per-node spread figure was not found in our readings. A Trainium versus AMD table built from separate nodes would not meet that standard.

Smaller items others own, which we only watch:

- Decode attention reads every tile of the compiled context. Our step time on Qwen3-0.6B at 32 rows is 10.4 / 14.8 /
  26.5 ms at compiled context 1024 / 4096 / 8192, and the same for 1 or 32 live rows (context ladder off, one bucket).
  PR 4 removed the early exit because it is wrong past 8K on SDK 2.33. PR 8 carries a fix for the block-table load that
  the author keeps off because it was slower on their mix. A decode attention that reads only the live prior is planned
  in PR 8's design note and not built (earlier reading). Check with the author of PR 4 and PR 8 before starting a kernel.
- DMA and GpSimd load in our Qwen3 decode graph: profile of 4 decode steps (39.5 ms of engine time) shows DMA 21.7 ms,
  GpSimd 14.0 ms, scalar 13.2 ms, vector 10.4 ms, tensor engine 8.9 ms. The profile had no per-operation attribution
  (no DGE notifications, no compiler metrics). Why the GpSimd share is large is not known.
- Cross-layer fusion is not a promising kernel target for dense decode: PR 8 reports putting two layers in one kernel
  measured no gain (earlier reading). Look for compiler-generated ops between kernels instead.

---

## Links index

- Upstream: https://github.com/sgl-project/sglang-neuron (PRs 4, 6, 10, 11, 12, 16, 20, 21; issue 15)
- vllm-neuron: https://github.com/vllm-project/vllm-neuron/pull/98 ; https://github.com/susmit-holbox/vllm-neuron
- Miles: https://github.com/radixark/miles (docs/advanced/disaggregated-rollout.md, docs/advanced/lora.md)
- Our public repo: https://github.com/pengchenglu1997/trainium-sglang-rl
- nki-library: https://github.com/aws-neuron/nki-library ; nki-samples: https://github.com/aws-neuron/nki-samples
- NxD Inference: https://github.com/aws-neuron/neuronx-distributed-inference
- Agent skill pack: https://github.com/aws-neuron/neuron-agentic-development
- VibeSys: https://github.com/uw-syfi/vibesys ; run record (no licence): https://github.com/vibesys-playground/qwen3.8-trainium2-vibesys
- TPU sources: listed per row in section 4
- Files behind our own numbers: `docs/NIGHT_RECORD_2026-11-21.md`, `docs/experiment-plan-2026-11-22.md`,
  `patches/PR_DRAFT_inplace_weight_update.md` (public repo); [KERNELS.md](KERNELS.md), [RESULTS.md](RESULTS.md) and
  `tests/rlkit/README.md` (this fork)
