# Roadmap: tasks you can hand out

This file lists work, in order, with enough detail to give one task to one person. Read [VISION.md](VISION.md) for the goals and KPIs, [METHOD.md](METHOD.md) for how a result is accepted, [UPSTREAM_PRS.md](UPSTREAM_PRS.md) for the pull requests, [KERNELS.md](KERNELS.md) for the kernels, [RESULTS.md](RESULTS.md) for measured results, and [LANDSCAPE.md](LANDSCAPE.md) for what others do.

## Goals

- **G1.** Run RL post-training end to end on a larger model on Trainium2. Ladder: Qwen3-0.6B (done on hardware), Qwen3-8B dense (serving side measured, no loop yet), then a 27 to 32B class model (Qwen3-30B-A3B or Qwen3.8-27B; not chosen).
- **G2.** Performance at least comparable to Google TPU and AMD accelerators, on a like-for-like protocol.
- **G3.** One or two NKI kernel optimisations with before and after numbers.
- **G4.** Work with upstream instead of beside it.

## Status of what we hold

| Item | Evidence |
|---|---|
| In-place full update and LoRA adapter sync, Qwen3-0.6B and Qwen3-8B (our patch) | tested on trn2 hardware |
| RL loop at 0.6B, 150 steps, 3 seeds | run on trn2 hardware |
| 8B RL loop, 8B trainer | not run. The trainer passes CPU checks against a random 3-layer model only |
| LoRA shard-merge NKI kernel | NKI 0.7.0 CPU simulator only. Never run on a NeuronCore |
| Sampler kernel | designed, not built |
| Upstream PRs 20, 4, 9, 16, 17, 18 | not run by us. Authors report results, many on trn3 |
| Qwen3-30B-A3B or 27B on trn2 | not run by us |

## How to read a task

Each task has: **id**, **goal**, **what to do**, **deliverable**, **done when**, **needs hardware** (yes or no), **depends on**, **size** (S about a day, M a few days, L more than a week; sizes are rough guesses, not measured), **owner** (blank until assigned).

Rules for every task: write the four lines from METHOD.md section 1 (question, expected result with basis, decision either way, machine-time limit) in the task or its pull request before a hardware run. Stop rules are written in advance. A number is reported with its script, commit and result file, or marked unverified.

Hardware tasks need the researcher's go-ahead before launch. Nothing in this file has been posted or sent anywhere.

---

## 1. Now, no hardware needed

### N1. Fork skeleton
- **Goal:** G4
- **What to do:** the layout is built as follows. Branch `main`: upstream v0.1.0 (4e24326) plus these documents under `docs/rl/` and a short section at the top of README.md; no code change. Branch `integration`: `main` plus upstream PRs 20, 4, 9, 16, 17, 18 merged in that order (see [UPSTREAM_PRS.md](UPSTREAM_PRS.md)); none of them has been run by us. Branch `feat/inplace-adapter-sync`: our in-place weight update and LoRA adapter sync patch on top of v0.1.0 (files `sglang_neuron/general_plugin.py`, `sglang_neuron/models/host.py`, `sglang_neuron/models/qwen3/sglang_model.py`); it is not on `integration`, because it and PR 20 hook the same function. Branch `feat/lora-merge-kernel`: on top of `feat/inplace-adapter-sync`; adds `sglang_neuron/models/rlkit_lora_merge.py`, `sglang_neuron/models/rlkit_kernel_path.py`, the kernel branch in `host.py` and the CPU tests in `tests/rlkit/`. Branches `upstream-pr/<N>`: the unchanged heads of upstream PRs 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 20, 21 for reference (PRs 3, 14, 19 are not carried). The experiment scripts, launchers and session records stay in the public repo https://github.com/pengchenglu1997/trainium-sglang-rl.
- **Deliverable:** the branches above; the section at the top of README.md.
- **Done when:** done so far: the branches exist as listed, and the seven CPU test scripts in `tests/rlkit/` pass in the NKI 0.7.0 CPU simulator (`bash tests/rlkit/run_all.sh`; `test_host_kernel_path.py`: 47 passed, 0 failed; `test_lazy_delta.py`: 24 passed, 0 failed). Remaining: nothing has run on hardware from these branches. `integration` is not run by us, and `feat/lora-merge-kernel` is simulator only; the hardware tasks below produce the hardware results.
- **Needs hardware:** no
- **Depends on:** none
- **Size:** M
- **Owner:**

### N2. Paired-measurement function and its tests
- **Goal:** G3, G2
- **What to do:** write one function that runs interleaved A/B timing as in METHOD.md section 4: one warm-up, a set number of pairs (proposed default 7, the VibeSys script's default; not tuned for trn2), median of the per-pair deltas, count of pairs whose sign disagrees, and a drift check. It refuses a log with failed or missing runs. Test it on synthetic timing data: known gain, no gain, drift, one outlier, a missing run.
- **Deliverable:** `tools/paired_ab.py` and tests.
- **Done when:** tests pass on a laptop; the function reports "no decision" on pure noise and on drifting data; the noise-floor procedure (measure A against A first) is written in the docstring.
- **Needs hardware:** no
- **Depends on:** none
- **Size:** M
- **Owner:**

### N3. Per-operation profile attribution script
- **Goal:** G3
- **What to do:** write a script that reads a neuron-explorer profile of one decode step and prints a table: time per engine, DMA time, and time per operation scope (attention block, sampler, MLP, norm). State the denominator in every column (window time, or instruction time summed over engines). The earlier join script was not saved, and the two DMA shares in KERNELS.md (about 55 percent by division, 21.7 percent in our analysis notes, not published) do not agree, so this script must say which denominator it uses. Use the raw profile files of the 0.6B decode profile as test input. Check PR 8's `profile_step.py` for the method, after checking where its code came from.
- **Deliverable:** `tools/profile_attribution.py`, its output on the existing 0.6B profile, and a note that settles the denominator.
- **Done when:** the script reproduces or corrects the 2.62 ms sampler scope of 26.0 ms instruction time per step, with the command saved.
- **Needs hardware:** no (it reads existing profile files; new profiles come in B5)
- **Depends on:** the raw profile files of the 0.6B decode profile (they are in neither this fork nor the public repo, because of size)
- **Size:** M
- **Owner:**

### N4. Check each TPU practice in LANDSCAPE.md against the engine code
- **Goal:** G2, G3
- **What to do:** for each row of the TPU table in LANDSCAPE.md marked "not checked" (bucket gap policy, KV reuse across weight updates, over-padding, overlapped logprob extraction, attention tile tuning, multi-step decode), read the engine code at v0.1.0 and write: done, partly done, or not done, with file and function names. Do not run anything.
- **Deliverable:** updated status column with code references.
- **Done when:** no row says "not checked". Each claim names the file read.
- **Needs hardware:** no
- **Depends on:** N1
- **Size:** M
- **Owner:**

### N5. Task bundles for the decode step and the two kernels
- **Goal:** G3
- **What to do:** for each of the decode step, the LoRA shard-merge kernel and the sampler kernel, write a bundle with: a correctness checker (reference implementation, tolerances, the CPU tests that exist), a benchmark script that uses N2, the baseline numbers with conditions, and the stop rules from KERNELS.md. The bundle lets someone who has not read the rest of the fork run or improve the target.
- **Deliverable:** `tasks/decode_step/`, `tasks/lora_merge/`, `tasks/sampler/`, each with a README.
- **Done when:** the checker of each bundle runs on CPU and fails on a planted fault; the benchmark script runs on CPU with a dummy workload and refuses a bad log.
- **Needs hardware:** no
- **Depends on:** N1, N2
- **Size:** L
- **Owner:**

### N6. Read PR 20, PR 21 and Miles end to end; write the run recipe for trn2
- **Goal:** G1, G4
- **What to do:** read both PRs and the Miles code paths they use (disk-delta transfer, LoRA push, the rejection of disk-delta with LoRA). Write the exact commands and settings to run a Miles GRPO loop against a Neuron rollout server on trn2: trainer, sync mode, ports, versions, what each side expects on disk. List every point where the upstream text says trn3 and the trn2 value is unknown.
- **Deliverable:** `docs/rl/recipes/rl-loop-trn2.md`.
- **Done when:** a person who did not write it can follow it up to the first launch command, and every unknown is marked.
- **Needs hardware:** no
- **Depends on:** N1
- **Size:** M
- **Owner:**

### N7. CPU-mode test for `reload_weights`, with the missing checks
- **Goal:** G4
- **What to do:** write a CPU-mode test of `TransformerForCausalLM.reload_weights` (PR 20) on a tiny random Qwen3 config. Cases: unknown tensor, shape mismatch, dtype mismatch, missing parameter. Each must raise before any `copy_`, and the served weights must stay untouched. Add the dtype and missing-parameter checks. Write the follow-up PR text for upstream (not posted).
- **Deliverable:** the test, the two checks, `docs/rl/upstream/pr20-followup.md`.
- **Done when:** the test passes with `SGLANG_NEURON_CPU_MODE=1`, fails on unpatched PR 20 for the dtype and missing-parameter cases, and the PR text is written.
- **Needs hardware:** no
- **Depends on:** N1
- **Size:** M
- **Owner:**

### N8. Re-base the adapter path on PR 20 (code and CPU tests)
- **Goal:** G1, G4
- **What to do:** on branch `feat/adapter-sync-on-pr20`, make the hook a thin wrapper. If `adapter_config.json` is in the folder, call our adapter function. Otherwise fall through to PR 20's reload. The adapter function uses the live first graph's `hf_weight_mappings()` and the loaders PR 20 keeps, not `build_graph`. Clear the base copies on every full reload. Check that the kept loaders accept `_LowRankSlice`. This is a hand rebase, because our hook overlaps PR 20's hunk line by line.
- **Deliverable:** the branch, and a CPU-mode test that a partial adapter is refused.
- **Done when:** CPU tests pass. The hardware re-test is task B1b.
- **Needs hardware:** no (the hardware test is B1b)
- **Depends on:** N1, N7
- **Size:** L
- **Owner:**

### N9. Post-update canary in the RL loop
- **Goal:** G1
- **What to do:** after every weight update, run one fixed short prompt and compare with a stored expected output. Abort the step on mismatch. Reason: PR 5 shows a bad kernel can poison every later request and `flush_cache` does not recover.
- **Deliverable:** canary code in the trainer loop, and a CPU-mode test that forces a mismatch and sees the step abort.
- **Done when:** the test passes, and the canary is switched on in the next hardware RL run.
- **Needs hardware:** no
- **Depends on:** none
- **Size:** S
- **Owner:**

### N10. Comparison protocol with TPU and AMD
- **Goal:** G2
- **What to do:** write the protocol as a new section of [METHOD.md](METHOD.md) (section 5 states the rules): models (Qwen3-8B, Llama 3.1 8B, Qwen3-30B-A3B), input and output lengths, concurrency levels, sampler, warm-up exclusion, metrics (TPOT, TTFT, output tok/s per chip, and per instance-dollar), and the accuracy gate. Collect public TPU and AMD numbers only from pages someone opened, each with URL, date and conditions. Leave a row empty if it has no public source. Say how "per chip" is defined, because one TP4 server may not equal one chip (not confirmed).
- **Deliverable:** the protocol, and a table of external numbers with sources and a note where conditions differ.
- **Done when:** every number in the table has a link and conditions; empty rows are empty, not estimated.
- **Needs hardware:** no
- **Depends on:** none
- **Size:** M
- **Owner:**

### N11. Check that upstream's dependencies exist in the image
- **Goal:** G1
- **What to do:** PR 21 needs `zstandard`, `numpy`, and optionally `xxhash` or `blake3`. Read the v0.1.0 image's package list (from its public image description or Dockerfile if available) and say which are present. If this needs a running image, do it inside task L6 instead.
- **Deliverable:** a short dependency note in `docs/rl/recipes/`.
- **Done when:** each dependency is marked present, absent or unknown, with the source.
- **Needs hardware:** no
- **Depends on:** N1
- **Size:** S
- **Owner:**

---

## 2. First hardware session (planned checks only)

Run these on one trn2 instance, in this order, so earlier results can stop later steps. Each has a question, an expected result, and a stop rule. None has been run.

### B1a. Reproduce PR 20 on trn2
- **Goal:** G1, G4
- **Question:** does PR 20's in-place reload give the same logprobs as a fresh start on trn2 (SDK 2.33), and how long does it take?
- **Expected result:** match. Basis: the author reports identical greedy output on Qwen3-4B on trn3. Update time at 8B: unknown. Our own patch measured 4.7 to 6.2 s at 8B, so PR 20 is expected to be the same or faster (estimate).
- **What to do:** Qwen3-0.6B TP4 and Qwen3-8B TP4. Update to a different checkpoint and compare token logprobs on 20 prompts with a fresh start on that checkpoint. Update back and compare with the first start. Check the radix cache is flushed after the update.
- **Deliverable:** table with logprob differences and update times, commands and logs.
- **Done when:** logprobs match a fresh start within the tolerance our own record used (0.0 for our patch), and the update time sits next to 4.7 to 6.2 s for 8B on the same instance and SDK.
- **Stop rule:** a mismatch above the tolerance on either model: stop, keep the logs, and do not run B1b, B3 or the rest of PR 20 work until understood.
- **Needs hardware:** yes
- **Depends on:** N1
- **Size:** M
- **Owner:**

### B1b. Adapter path on PR 20
- **Goal:** G1
- **Question:** does the re-based adapter path still equal a fresh start on the host-merged checkpoint?
- **Expected result:** yes, as the old path did at 0.6B and 8B. Adapter swap time is expected near the old 13.9 s host-merge figure at 8B unless the lazy low-rank slice removes it (unknown).
- **What to do:** Qwen3-0.6B and Qwen3-8B. Adapter 1, adapter 2, adapter 1 again, zero adapter. Each compared with a fresh start on the host-merged checkpoint.
- **Deliverable:** table, with swap time next to our earlier numbers.
- **Done when:** all four states equal a fresh start on both sizes.
- **Stop rule:** any inequality: the branch stays unmerged; report the first state that differs.
- **Needs hardware:** yes
- **Depends on:** B1a, N8
- **Size:** M
- **Owner:**

### B2. LoRA shard-merge kernel, first hardware run (H1 to H5)
- **Goal:** G3
- **Question:** can the kernel run on a NeuronCore, is it correct, and does it bring the 8B adapter swap from 13.9 s toward 1.5 s?
- **Expected result:** proposed target at most 1.5 s per swap at 8B (VISION K2). This is an estimate from one measured number (16.4 GB written to the device in 1.19 s), not from the kernel. Most of the drop from 13.9 s may come from not forming full matrices on the host, not from the kernel.
- **Steps (from KERNELS.md; each needs the researcher's go-ahead):**

| Step | Check | Stop rule |
|---|---|---|
| H1 | `tests/rlkit/hw_merge_kernel_bench.py`, one core. Can the kernel be called on device tensors outside a serving graph? Time per 8B shard, read-back against the reference, integer factors bit-exact, time of the eager torch device merge and the host merge at the same shapes | exit code 3 means it cannot be called: stop the kernel part, report the torch-device and host numbers only |
| H2 | `50_adapter_server_test.py`, 0.6B TP4, kernel mode, starting with `RLKIT_KERNEL_CHECK=all`. Swap time split, adapter 1, 2, 1 identical, zero adapter equals base | any check fails: kernel mode stays off |
| H3 | same test at 8B TP4. Headline number, device memory after the base copy, KV pool size | swap above 1.5 s: try one launch per kind over stacked layers, then report what was measured |
| H4 | restart on the kernel-merged checkpoint, logprob difference against the swapped server | not 0.0: report the difference as unexplained (the earlier eager device merge also did not equal a restart, and the cause was not found) |
| H5 | compile-cache keys and decode step time before and after a swap | none |

- **Deliverable:** before and after table (host merge vs kernel) with max absolute error, marked as hardware; [RESULTS.md](RESULTS.md) updated so simulator and hardware rows are separate.
- **Done when:** H1 to H5 each have a result or a stop-rule outcome written, with script, commit and log named; KERNELS.md section 1 moves the kernel from "simulator only" to "hardware" only for the steps that passed.
- **Needs hardware:** yes
- **Depends on:** N1 (branch `feat/lora-merge-kernel`), N5
- **Size:** L
- **Owner:**

### B3. Update under load, 100 updates, and async launch
- **Goal:** G1
- **Question:** can PR 20 and our adapter path update while requests run, with every answer on one weight set and no crash?
- **Expected result:** yes. Basis: our own patch ran 40 updates with 8 sender threads at 0.6B and 100 updates in a row without error (short requests). Long requests (2,000 tokens or more) and an update during a queued decode launch were not tested.
- **What to do:** 100 in-place updates in a row, then the 8-thread concurrent-request test from our record (`49_sync_stress.py`, `68_update_under_load.py`) with 8 requests of 2,000+ tokens in flight. Then send an update while a decode launch is queued under async execution, with and without `drain_launches` added.
- **Deliverable:** counts of HTTP errors and of answers produced on neither weight set; wait or pause time; a line saying whether `drain_launches` is needed, with the log.
- **Done when:** zero errors and every answer on one weight set, or the failure is written with its log.
- **Stop rule:** any answer on neither weight set, or any crash: stop, keep the log, no further updates on that server.
- **Needs hardware:** yes
- **Depends on:** B1a
- **Size:** M
- **Owner:**

### B4. Sampler baseline and compile-flag test (step 0, before any sampler kernel)
- **Goal:** G3
- **Question:** how fast is `temperature` mode in the configuration the RL loop runs (top logprobs 0, decode at `-O2` by the code), and does the 20 percent gap measured by the probe (2,183 to 2,218 against 2,713 to 2,730 tok/s, both servers with 20 top logprobs) come from the sampler operations, the top-logprob path or the compiler flags?
- **Expected result:** unknown. The earlier probe compared two servers that, by the code, kept the same default flags, so the `-O2` flag does not explain that gap. The loop's own configuration was never timed.
- **What to do:** Qwen3-0.6B TP4, `58_sampler_probe.py`, three passes, three servers: `topk_topp` with top logprobs 20 (as before); `temperature` with `--top-logprobs 0`; `temperature` with top logprobs 0 and the default flags (a one-line change in our patched `host.py`). Read the compiler arguments back from each server log.
- **Deliverable:** three-pass table with the compiler arguments of each server, and a note on which sampler mode and top-logprob setting the 0.6B RL loop used, read back from a loop server log (today it is known only from the launch scripts).
- **Done when:** the table has all three servers with the compiler arguments read from the logs, and the decision below is written.
- **Decision rule (proposed, from KERNELS.md):** if `temperature` at top logprobs 0 reaches 2,600 tok/s or more in either flag setting, the probe gap was not the sampler operations; build the kernel only if a device profile of that configuration still shows the sampler scope at 1.0 ms or more per step. Near 2,200 tok/s means go on to the kernel.
- **Needs hardware:** yes
- **Depends on:** N1
- **Size:** S
- **Owner:**

### B5. Device profile of one decode step at context 1024 and 8192
- **Goal:** G3
- **Question:** where does one decode step spend its time, and what does a dead tile cost at compiled context 8192?
- **Expected result:** unknown. Today's reading: one live row at compiled context 1024 / 4096 / 8192 takes 9.3 / 13.7 / 25.2 ms on Qwen3-0.6B, and the cause of the slope is unexplained.
- **What to do:** capture a neuron-explorer profile of one decode step, ladder off, at compiled context 1024 and 8192, first on Qwen3-0.6B (to compare with earlier numbers), then on Qwen3-8B TP4. Run N3's script on each. Use PR 4's needle grid as the correctness gate for any decode kernel change that follows.
- **Deliverable:** baseline table in `docs/rl/` with the exact command, a ranked list of kernel candidates with the measurement behind each.
- **Done when:** both contexts and both models have a table, and the candidate list is written.
- **Stop rule:** if profiling changes step time by more than the noise floor, the timing pass is run without the profiler (METHOD.md).
- **Needs hardware:** yes
- **Depends on:** N3, B4 (same session)
- **Size:** M
- **Owner:**

### B6. Context ladder and decode buckets
- **Goal:** G2
- **Question:** how much of the context slope does the context ladder (`RLKIT_DECODE_LADDER`) remove, and what bucket policy is best for an RL rollout?
- **Expected result:** the ladder is expected to remove much of the long-context cost with no kernel. That is an estimate. It has not been run (plan item W1.2). With one bucket a step costs the same for 1 or 32 live rows (9.3 to 10.4 ms); with default buckets a one-row step is 4.6 ms.
- **What to do:** Qwen3-0.6B TP4, ladder on vs off, using N2 for the comparison, at contexts 1024, 4096, 8192 and live rows 1, 8, 32. Then the same at 8B.
- **Deliverable:** paired A/B table with median deltas.
- **Done when:** the table exists for 0.6B and 8B, with the noise floor stated.
- **Stop rule:** if the A/A noise floor is larger than the effect, report "no decision" and do not claim a gain.
- **Needs hardware:** yes
- **Depends on:** N2, B4
- **Size:** M
- **Owner:**

### B7. PR 4: needle test and decode timing on Qwen3-8B
- **Goal:** G2
- **Question:** does PR 4 keep output correct past 8K, and what does it cost below and above 8K?
- **What to do:** needle at 8,204 / 9,644 / 24,044 tokens on `main` and on the `integration` branch. TPOT at 2K, 8K and 32K context with and without `max_context_len`. Record the NEFF recompile time after restamping.
- **Expected result:** author reports a fix for wrong output past 8K on Qwen3-30B-A3B. Below 8K no change is expected, but nobody has measured it.
- **Deliverable:** pass/fail per length per build, three TPOT numbers per build, and equal greedy tokens to main for a prompt under 8K.
- **Done when:** the table has a pass/fail entry for every length on both builds, three TPOT numbers per build, the recompile time, and the greedy-token comparison below 8K, each with its command and log.
- **Stop rule:** greedy tokens differ below 8K: stop and report.
- **Needs hardware:** yes
- **Depends on:** N1
- **Size:** M
- **Owner:**

### B8. Warm-up decision run: PR 11 vs PR 12, then PR 10
- **Goal:** G1
- **Question:** which warm-up change do we promote, and does PR 10 follow?
- **What to do:** Qwen3-8B TP4, empty NEFF cache. Arms: main, `upstream-pr/11`, `upstream-pr/12`, then `upstream-pr/10` (on 11) with cache off, first launch and restored. Record ready time, decode and prefill warm-up, GSM8K-200, TPOT, NEFF count, SPMD check result, host RSS and number of neuronx-cc processes.
- **Expected result:** author reports 1467 s to 540 s ready for PR 11 on this configuration, and 540 s to 92 s restored for PR 10. PR 12's evidence is trn3 only.
- **Deliverable:** table in `docs/rl/`, and a written decision.
- **Done when:** the decision rests on SPMD passing on every launch and on ready time.
- **Stop rule:** an SPMD mismatch in any arm: that arm is dropped from promotion.
- **Needs hardware:** yes
- **Depends on:** N1
- **Size:** M
- **Owner:**

### B9. PR 16 measurement and NEFF restamp procedure
- **Goal:** G1
- **Question:** does PR 16 shorten warm launches on trn2 without changing output?
- **What to do:** Qwen3-8B TP4, warm cache. Ready time with and without the PR, greedy tokens equal, zero neuronx-cc compiles, and whether the torch_mlir stub installs in our image. Write the NEFF restamp steps for PRs 4, 16 and 17 in the README, and have someone other than the author run them once.
- **Deliverable:** two ready times with logs, the stub log line, the restamp steps.
- **Done when:** the restamp was run once by a second person.
- **Needs hardware:** yes
- **Depends on:** N1
- **Size:** S
- **Owner:**

### B10. PR 9 Llama 3.1 8B, then reload and adapter on Llama
- **Goal:** G1, G2
- **Question:** does Llama 3.1 8B match Hugging Face on trn2, and does weight sync work outside Qwen?
- **What to do:** TP4. Token parity with Hugging Face on a 4096-token prompt and GSM8K first 100. Then PR 20 reload from checkpoint A to B and back, with logprobs against a fresh start. Then our adapter sync on Llama.
- **Expected result:** the PR reports 151 of 151 identical and 0.78.
- **Deliverable:** our numbers beside the PR's; a line saying whether PEFT key names map without edits.
- **Done when:** reload logprobs match a fresh start, and the adapter result is written.
- **Stop rule:** parity fails: reload and adapter tests are not run.
- **Needs hardware:** yes
- **Depends on:** B1a, B1b
- **Size:** M
- **Owner:**

---

## 3. Larger models

### L1. Qwen3-8B trainer feasibility (memory on one core)
- **Goal:** G1
- **Question:** does one 8B LoRA step (rank 16, seven projections, micro-batch 1) fit and run on trn2?
- **Expected result:** weights alone take 16.38 GB of 25.77 GB on one core (measured). Peak 17.3 to 18.5 GB and under 2 s per compiled micro-batch are estimates only.
- **What to do:** run `70_mem_probe.py` and the 8B trainer probes. The Trainium-side trainer route (the TorchTitan Neuron probe, listed as beta in SDK 2.33.1) has not been run; this task also decides whether it works.
- **Deliverable:** peak device memory, step time, and the chosen design.
- **Done when:** one step completes without out-of-memory, or the failure is recorded with the log.
- **Stop rule:** out-of-memory after the design options in the probe file are used: stop and write what does not fit.
- **Needs hardware:** yes
- **Depends on:** none
- **Size:** L
- **Owner:**

### L2. Qwen3-8B RL loop
- **Goal:** G1
- **What to do:** run the 0.6B loop's recipe at 8B: 32 questions x 8 answers, GSM8K, temperature 0.7, up to 256 new tokens, adapter sync, 3 seeds x 150 steps. Record the step split (rollout, train, export, sync), trainer-vs-server logprob difference, and accuracy on 500 test questions.
- **Deliverable:** KPI K1, K4, K10 tables for 8B (VISION.md).
- **Done when:** step time is within 20 percent of the sum of parts measured earlier (a proposed target, to be confirmed by the team) and accuracy rises above the step-0 spread in 3 of 3 seeds. No size of rise is promised.
- **Needs hardware:** yes
- **Depends on:** L1, B1b, B3, N9
- **Size:** L
- **Owner:**

### L3. Qwen3-30B-A3B bring-up and reproduction of PR 17 and 18
- **Goal:** G2, G1
- **What to do:** bring up Qwen3-30B-A3B TP4 EP4 on one trn2.3xlarge from the `integration` branch. Reproduce PR 17 (2048/128 at c=32 and c=64, 128/512 at c=64, block vs hybrid on the same build) and PR 18 (c=1 TPOT, env 0 vs default). Run logprob difference checks for hybrid vs block prefill and selective vs all-expert decode on 20 prompts.
- **Deliverable:** our tok/s and TPOT beside the PRs' (220.2 to 371.0 tok/s; 45.8 to 17.1 ms), max and mean logprob difference, cold compile time. Note: the recipe's `--max-total-tokens 266240` does not fit a trn2.3xlarge, per the PRs; use 131072.
- **Done when:** every PR 17 and PR 18 point listed above has our number beside the author's, from one build, with the logprob differences and the cold compile time, each with its command and log.
- **Needs hardware:** yes
- **Depends on:** N1
- **Size:** L
- **Owner:**

### L4. PR 20 on Qwen3-30B-A3B, and adapter coverage of experts
- **Goal:** G1
- **Question:** does PR 20 reload work on this MoE model on trn2 (the author validated on trn3 only), and does our adapter mapping cover expert weights?
- **What to do:** TP4 EP4 with hybrid prefill on. Update to a second checkpoint and compare logprobs with a fresh start. Decide whether the adapter path must refuse expert targets.
- **Deliverable:** logprob comparison and update time; either "attention-only adapters pass and expert targets are refused with a clear message", or the gap written as a named task.
- **Done when:** one of those two is true.
- **Needs hardware:** yes
- **Depends on:** L3, B1b
- **Size:** M
- **Owner:**

### L5. RL loop with Miles and PR 20 / 21: Qwen3-4B, then Qwen3-30B-A3B
- **Goal:** G1
- **Question:** does the upstream author's setup (Miles GRPO, disk-delta sync) run on trn2? The PR text validates it on trn3; trn2 is not validated.
- **What to do:** follow N6's recipe on Qwen3-4B first, then Qwen3-30B-A3B. Record trainer-vs-rollout logprob difference per sync (the author reports 0.007 to 0.015) and the sync time. Which trainer runs on Trainium is open (see L1).
- **Deliverable:** per-sync table and a list of what changed from the trn3 report.
- **Done when:** 30 syncs on the 4B model with logprob differences recorded, then the same for 30B, or a written reason for stopping.
- **Stop rule (proposed):** mean logprob difference above 0.017 (our 0.6B value after an adapter update; the PR 20 author reports 0.007 to 0.015 on trn3) with no explanation: stop and report.
- **Needs hardware:** yes
- **Depends on:** N6, L3, L4, and the trainer decision in L1
- **Size:** L
- **Owner:**

### L6. Delta pull (PR 21) on 8B
- **Goal:** G1
- **What to do:** check the dependencies (N11). Decide the publishing format (full every K steps, deltas between). Measure a Qwen3-8B delta size and pull time against a full copy with `upstream-pr/21` on top of PR 20.
- **Deliverable:** dependency list, format decision in this file, one measured row (delta bytes, pull seconds, reload seconds).
- **Done when:** one measured row (delta bytes, pull seconds, reload seconds, checksum result) and one full-copy row on the same instance exist with command and log, and the format decision is written in this file.
- **Needs hardware:** yes
- **Depends on:** B1a, N11
- **Size:** M
- **Owner:**

### L7. Choose the 27 to 32B rung
- **Goal:** G1
- **What to do:** compare Qwen3-30B-A3B (reloads through PR 20 per its author), PR 6 Qwen3.8-27B (has `load_weights` and `hf_weight_mappings(graph)` but no `reload_weights`, so it needs a reload path, and a trainer reference for Gated DeltaNet layers; backward kernels were not found), and PR 2 Gemma 4 31B (needs `reload_weights` and a `k_eq_v` check). Base the choice on reload readiness, trainer availability and measured throughput.
- **Deliverable:** a one-page decision here, naming the rung, its blocking tasks, and an owner for each.
- **Done when:** the page is written. Our current guess is Qwen3-30B-A3B first, because PR 20 covers Qwen3-MoE and PRs 17 and 18 serve it on trn2. This is an assessment, not a measurement.
- **Needs hardware:** no
- **Depends on:** L4
- **Size:** S
- **Owner:**

### L8. Reload path for the chosen non-Qwen3 rung (only if chosen)
- **Goal:** G1
- **What to do:** if PR 6 is chosen: PR 6 (Qwen3.8-27B) has a host `load_weights` and a module-level `hf_weight_mappings(graph)` but no `reload_weights`, so PR 20 refuses it. The task is to add a reload path (`reload_weights`) on top of those two, not to write `load_weights`. If PR 2 is chosen, add `reload_weights` and run the `k_eq_v` check on the adapter path.
- **Deliverable:** branch and CPU-mode test, then a hardware test of the reload as in B1a.
- **Done when:** the logprob comparison with a fresh start passes.
- **Needs hardware:** yes
- **Depends on:** L7
- **Size:** L
- **Owner:**

---

## 4. Comparison

### C1. The like-for-like table against TPU and AMD
- **Goal:** G2
- **What to do:** run the protocol from N10 on our hardware for Qwen3-8B, Llama 3.1 8B and Qwen3-30B-A3B, at every concurrency level. Fill our rows beside the external rows. Report per chip and per instance-dollar. Same checkpoint, precision, workload and accuracy gate as the external source, or the row says it differs.
- **Deliverable:** `docs/rl/comparison.md` with the full concurrency table (not a peak).
- **Done when:** every row has a source and conditions; rows without a public source stay empty.
- **Needs hardware:** yes
- **Depends on:** N10, B6, B10, L3
- **Size:** L
- **Owner:**

---

## 5. Second kernel

### X1. Choose the second kernel from the first session's profile

(Task ids X1 and X2 are tasks; K1 to K12 in VISION.md are KPIs.)
- **Goal:** G3
- **What to do:** use B4 and B5 to pick one candidate. Candidates in KERNELS.md and elsewhere in this fork: the sampler kernel (only if B4 says go); live-length decode attention (only if B5 shows a large cost per dead tile); separate fetch and compute tiles in decode attention; batching small tiles per DMA (our merge kernel moves 1 KiB per partition, below the 4 KiB guidance); multi-step decode in one compiled loop with sampling and logprobs; a mid-batch MoE decode mode (PR 18 leaves 8 to 32 rows reading all experts); the nkilib flash-attention early-exit repro from PR 4. Apply the METHOD.md rule first: measure the library kernel before writing a new one.
- **Deliverable:** a one-page choice with the measurement behind it.
- **Done when:** the page names one kernel, its acceptance numbers set before building, and its stop rule.
- **Needs hardware:** no
- **Depends on:** B4, B5, B6
- **Size:** S
- **Owner:**

### X2. Build and test the second kernel on CPU, then on hardware
- **Goal:** G3
- **What to do:** use the bundle from N5 or write one. Simulator tests with two references and planted faults, then a hardware run with N2's paired A/B.
- **Deliverable:** before and after table, marked simulator or hardware.
- **Done when:** the acceptance numbers from X1 are met or the stop rule is reported.
- **Needs hardware:** yes
- **Depends on:** X1, N2, N5
- **Size:** L
- **Owner:**

---

## 6. Upstream

### U1. Comment on PR 20 pointing to issue 15
- **Goal:** G4
- **What to do:** a comment on PR 20 that points to our issue 15, reports what B1a measured, and offers the CPU test (N7) and the adapter path. **Needs the researcher's explicit approval before posting.** Post only after B1a has results.
- **Deliverable:** comment text in `docs/rl/upstream/`.
- **Done when:** text is approved and posted, or held.
- **Needs hardware:** no
- **Depends on:** B1a, N7
- **Size:** S
- **Owner:**

### U2. Follow-up PR to PR 20 (dtype and refreshed-parameter checks)
- **Goal:** G4
- **What to do:** open a small PR after PR 20 lands, or a PR against its branch if the author agrees, with the checks and test from N7. Propose `drain_launches` only if B3 shows it is needed.
- **Done when:** PR opened (needs approval) or reason to hold written.
- **Needs hardware:** no
- **Depends on:** N7, B3, U1
- **Size:** S
- **Owner:**

### U3. Propose the adapter route upstream
- **Goal:** G4
- **What to do:** after B1b passes, open a PR with the adapter route, linked from issue 15. Port the kernel into the adapter write path only after B2 shows it works on hardware.
- **Done when:** PR opened (needs approval).
- **Needs hardware:** no
- **Depends on:** B1b, U2
- **Size:** M
- **Owner:**

### U4. Other PRs we could send
- **Goal:** G4
- **What to do:** from our own work, candidates are: the one-rank gloo-group test pattern (borrowed from PR 14 with credit) for weight-update tests, the needle grid as a CI check for decode kernels, the restart acceptance table as a template, and a measured trn2 result for any PR in UPSTREAM_PRS.md that we have run (as a data comment, with the command and log). Each goes through the researcher before posting.
- **Done when:** a list of ready texts exists.
- **Needs hardware:** no
- **Depends on:** the tasks that produce the data
- **Size:** S
- **Owner:**

---

## Dependency sketch

- N1 fork skeleton
  - N4 TPU practice check
  - N6 PR 20 / 21 / Miles recipe
    - L5 RL loop with Miles on trn2
  - N7 CPU reload test
    - N8 adapter re-base (CPU)
      - B1b adapter on PR 20 (hardware)
  - N11 dependency check
    - L6 delta pull
  - B1a PR 20 reproduction (hardware)
    - B1b adapter on PR 20
    - B3 update under load
    - B10 Llama (also needs B1b)
    - L6 delta pull
    - U1 comment on PR 20 (also needs N7)
      - U2 follow-up PR (also needs B3)
        - U3 adapter PR (also needs B1b)
  - B2 LoRA kernel first run (also needs N5)
  - B4 sampler compile-flag test
    - B5 device profile (also needs N3)
    - B6 context ladder (also needs N2)
      - X1 choose second kernel (also needs B4, B5)
        - X2 build second kernel (also needs N2, N5)
  - B7 PR 4 needle test
  - B8 warm-up decision
  - B9 PR 16 and restamp
  - L3 Qwen3-30B-A3B bring-up
    - L4 PR 20 on 30B (also needs B1b)
      - L7 choose 27 to 32B rung
        - L8 reload path for the chosen rung
      - L5 RL loop with Miles (also needs N6, L1)
- N2 paired-measurement function
  - N5 task bundles (also needs N1)
  - B6, X2 use it
- N3 profile attribution script
  - B5 device profile
- N9 canary
  - L2 8B RL loop (also needs L1, B1b, B3)
- N10 comparison protocol
  - C1 comparison table (also needs B6, B10, L3)
- L1 8B trainer feasibility
  - L2 8B RL loop
  - L5 RL loop with Miles (trainer decision)

## Open points for the team

- One TP4 server may not equal one chip. K5 per-chip numbers depend on it. Confirm first.
- Several KPI targets are proposals with no stronger basis than plan estimates (seven pairs for A/B, step time within 20 percent). Confirm before the first runs.
- The 26.0 ms instruction-time split of one decode step comes from a join script that was not saved. N3 replaces it.
- The upstream PR count: 20 open PRs plus our issue 15. VISION.md and LANDSCAPE.md say the same.
- Sizes and the order of hardware steps are rough. Dates are not set on purpose.
