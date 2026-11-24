# Method: how work in this fork is measured and accepted

These are rules, not advice. A change is accepted when its pull request shows that each rule below was followed, or says which rule was not and why. Numbers in this file are examples with their source. None is a target for your work unless a task says so.

## 1. Before any experiment

Write four lines in the task or the pull request, before the run:

1. **Question.** One question the run can answer.
2. **Expected result, with basis.** A number or a range, and why (an earlier measurement, or "estimate"). Mark estimates as estimates.
3. **Decision either way.** What we do if the expectation is met, and what we do if it is not.
4. **Machine-time limit.** Maximum device minutes and number of launches. Stop at the limit and report what was measured.

Our experiment plan already uses this shape as a table (check, expected and why, if met, if not, minutes): `trainium-sglang-rl/docs/experiment-plan-2026-11-22.md`. Each hardware step also has a stop rule written in advance. Example from [KERNELS.md](KERNELS.md) (hardware steps): if the kernel cannot be called on device tensors outside a serving graph, stop the kernel part and report only the other timings.

Raise a limit only with a new written line, never in the middle of a run.

## 2. Before building: survey, then measure what exists

Before writing a new kernel or patch, do three searches and list what you found. "Nothing found" counts, with the places you looked.

- **Upstream.** Open and merged pull requests and issues in sgl-project/sglang-neuron, and the sibling Trainium projects (for example the vLLM plugin).
- **Kernel libraries.** The NKI kernel library and the vendor's sample kernels. Read the source of the kernel closest to yours.
- **Other accelerators.** How the same operation is done on TPU and on AMD in published code and write-ups.

**Rule: measure an existing library kernel before writing a new one.** Run it on your shape, with your correctness gate, and put the result in the pull request. Write a new kernel only if the library kernel is missing, wrong for your layout or precision, or measured slower under the same gate. A new kernel without that table is not accepted.

One reported case shows the reason. In a public Qwen3.8 Trainium project, a library attention kernel was offered to the agents, and no measurement of it is recorded, while a hand-written attention kernel was built (second-hand reading of its logs; see Sources). The opposite case in the same reading: another run got its largest single gain from switching an existing library kernel to a different mode.

## 3. Correctness gates, in this order

A later gate does not replace an earlier one. State the highest gate each change reached.

1. **Unit level, in the simulator.** Compare against two references: a float64 sum rounded once to the output type, and the engine's own code path (its own weight loaders, not a copy we wrote). The result must be bit-exact, or within a stated tolerance with the count of differing entries. Use integer-valued inputs where possible, because the result must then be exact. Example from our merge kernel: equal to the float32 reference on every entry, and between 255 of 6,291,456 entries (8B qkv shard) and 735 of 25,165,824 entries (8B gate_up shard) differ from the float64 sum rounded once, by single units.
2. **Mutation tests and negative controls.** Inject single faults (wrong rank, forgotten scale, summed segments, wrong tile edge) and require a test to fail. Tabulate which check caught which fault. A fault no test catches is a hole: add a test, or write down that it changes no result. Our wiring test: 21 single faults, 15 caught at first, 3 more after new tests, 3 changed no result in any scenario built ([KERNELS.md](KERNELS.md) section 1; CPU tests in `tests/rlkit/`).
3. **Device self-test at start-up.** The simulator does not show that a kernel is right on the device. Each new kernel runs at server start on fixed inputs at the largest shapes, compares with the reference path, and exits non-zero on mismatch. Put a stall timeout on the first warm-up, because a new kernel can warm up for a long time (a public run reports 50 to 54 minutes, second-hand; set our own timeout from our first measured warm-up). Compile from a fresh temporary directory.
4. **Whole-model gate.** Greedy tokens and teacher-forced logprobs against a restart on the same weights, on a fixed prompt set. Fix the thresholds before the run. When the arithmetic is unchanged the threshold is logprob difference 0.0 (the in-place full-checkpoint update gave 0.0 on Qwen3-8B, GLM-4-9B and DeepSeek-V2-Lite: `trainium-sglang-rl/docs/NIGHT_RECORD_2026-11-21.md`). When the arithmetic is reordered, measure the noise floor of the unchanged engine first and set the threshold from it. Another project's thresholds are not ours until recalibrated on our model. A kernel boundary can change how the compiler lowers the neighbouring operations, so a correct kernel can still change model output. Only this gate shows it.
5. **Weight-swap identities.** Adapter 1, then 2, then 1 gives identical weights and outputs. The zero adapter equals the base. A swapped server equals a restart on the checkpoint merged by the same code (logprob difference 0.0).

## 4. Performance claims

- **Noise floor first.** Run the unchanged code against itself, interleaved, and record the spread. The acceptance threshold comes from that spread, not from a default.
- **Paired, interleaved A/B.** Alternate baseline and candidate in one session. Use at least 7 pairs (proposed; 7 is the default pair count of the VibeSys statistics script) after 1 discarded warm-up pair (our proposal). Report the median of the per-pair percent deltas. If the pair signs disagree, there is no verdict. If the series drifts steadily in one direction, it has not converged: repeat.
- **Refuse bad logs.** Reject timing logs that are cut off, lack a closing count line, or hold a different number of samples than the count says. Never read numbers from a file a job is still writing.
- **Table, not a point.** Report a table over model sizes and shapes (for us: 0.6B and 8B, the tensor-parallel degrees and adapter ranks in scope). No-regression rule: no row may be slower than baseline by more than the noise floor. A win on one row and a loss on another is reported as it is.
- **Show that the kernel ran.** The new kernel must appear in a device profile, and the compiled program must differ from the baseline. A speedup with the same kernel name and program is noise.
- **Timing pass without the profiler.** Time in a run with no profiler attached. Use a separate profiled run for attribution. If the profiled headline differs from the unprofiled one by more than 10% (our proposed threshold; VibeSys's profiler notes say to classify such a profile as perturbed, and the 10% figure was not found in their text), mark the profile perturbed and use it only to show what ran, not for time shares. Session profiling also reserves device memory (4.75 GiB per logical core on trn2, as reported by VibeSys for their stack), so an 8B run with tensor parallel 4 may need system-only capture. Unchecked on our stack.
- **Credit honestly.** Do not credit a kernel with a gain that came from removing other work. Example: our adapter swap at 8B is 13.9 s today (host merge, measured 2026-11-21) against a proposed target of 1.5 s. Most of the drop is expected from not forming full matrices on the host. The 1.5 s is an estimate from one measured copy (16.4 GB overwritten in 1.19 s, plain TorchNeuron process, no server), not a measurement of the kernel.

## 5. Comparison with other accelerators (goal G2)

A comparison row is ours only if we ran it. Required on both sides:

- the same checkpoint, precision and prompt set;
- the same fixed workload: fixed request count, fixed input and output lengths, temperature 0, prompts without a shared cache prefix unless caching is the test, and any failed request fails the point;
- the full concurrency table, not the best row;
- results per chip and per instance-dollar, with the price and its date;
- the same accuracy gate (section 3, gate 4) passed on both sides;
- repeats on the same node, because identical software on different nodes differed by tens of percent in a public AMD example (see Sources).

No number from a blog post or a vendor page goes into our column. It may appear in a separate "as published" column with its link, marked not comparable. If the other accelerator cannot run the model we run, say so and compare on the nearest model both can run.

## 6. What to write down for every reported number

- commit hash of this fork, and of upstream if it differs;
- script name and exact command;
- instance type and count;
- image and SDK version (Neuron SDK, NKI);
- checkpoint and precision;
- date;
- the raw output file, kept in the repository or linked.

A number missing any item is a draft and is marked unverified. Label each number "measured on hardware", "simulator only" or "estimate". Do not mix labels in one table cell.

## 7. Fallbacks

Every new path sits behind a switch that is off by default. It falls back to the previous path when any check fails: start-up self-test, read-back of the written values, or a refused configuration. On fallback it logs the reason once and stays off for that server. A pull request states the switch name, the default, and what a fallback looks like in the log. Example: `RLKIT_DEVICE_WRITE=kernel` in our patch (branch `feat/lora-merge-kernel`), with the host path as the default.

## Status of our own work, for scale

- Hardware-tested on trn2: in-place full-checkpoint weight update and adapter sync, Qwen3-0.6B and Qwen3-8B.
- NKI 0.7.0 CPU simulator only: the LoRA shard-merge kernel and its wiring. Not run on hardware, not compiled with the Neuron compiler, not called through torch. Any hardware claim for it waits for the hardware steps in [KERNELS.md](KERNELS.md).

## Sources

Ideas are borrowed and written in our own words. We copy no code.

- VibeSys repository (MIT), github.com/uw-syfi/vibesys: paired A/B statistics script, measurement protocol, accuracy-gate design with injected-fault calibration, Neuron safeguards (device self-test, warm-up stall timeout). Their thresholds are for their models and hardware. We do not adopt them.
- AWS neuron-agentic-development skills, github.com/aws-neuron/neuron-agentic-development (vendored in VibeSys): NKI writing and debugging guidance, profiling notes.
- Public Qwen3.8 Trainium2 project, github.com/vibesys-playground/qwen3.8-trainium2-vibesys: the survey observation (section 2), the kernel-boundary effect (gate 4) and the long warm-up (gate 3) are second-hand readings of its records, not checked by us against the raw files. The repository has no licence.
- Public AMD example for the node-to-node spread (section 5): the `qwen3.5-397b-a17b-mi300a-bespoke` example README in the VibeSys repository, second-hand.
