# Method: when a measurement counts, and what may be claimed

These are rules. A change is accepted when its pull request shows that each rule was followed, or says which one was not
and why. The numbers here are settings and thresholds, not results: results sit in files under [`rl/data/`](data/), and the
headline ones are listed with their tier in [`best.tsv`](best.tsv) (`python rl/tools/best.py`).

## 1. Tiers: every public number carries one
- **`screen`**: at least 2 passes on one server start per arm, in one session. This is the usual practice here.
- **`paired(n)`**: n interleaved server-start pairs of baseline and candidate, after 1 discarded warm-up pair; the verdict
  is the median of the per-pair percent deltas. If the signs of the pairs disagree there is no verdict. A series that
  drifts in one direction has not converged: repeat. n = 7 is proposed (VibeSys default), until our A/A run sets it.
- **What a claim may rest on.** "Faster" or "slower" by X percent may rest on `screen` only when |X| is 10 or more; the 10
  is provisional and is replaced by 5 times the start-to-start spread of the first public A/A run (the unchanged code
  against itself, in pairs). The tier is written next to the number. A smaller difference, and every "no difference" or
  "no change", needs `paired(n)`; until then the text says "not distinguished at screen tier" (the same reading applies
  to older text that says "no effect"). A claim from one pass, or from a file a job is still writing, is not a claim.
- `rl/tools/best.py` prints the tier of each headline row and marks one within 10 percent of its baseline as not distinguished.

## 2. Before any experiment
Write four lines in the task or the pull request, before the run: (1) **Question**, one the run can answer; (2) **expected
result, with basis**, a number or range and why (an earlier measurement, or "estimate"); (3) **decision either way**; (4)
**machine-time limit**, maximum device minutes and launches: stop at the limit and report what was measured, and raise it
only with a new written line, never mid-run. Each hardware step also has a stop rule written in advance (for example: if
the kernel cannot be called on device tensors outside a serving graph, stop the kernel part).

## 3. Before building: survey, then measure what exists
List what you found ("nothing found" counts, with the places looked at): upstream pull requests and issues, sibling Trainium
projects (for example the vLLM plugin), the NKI kernel library and vendor samples (read the source of the kernel closest to
yours), other accelerators. **Measure an existing library kernel before writing a new one**, on your shape, under your
correctness gate, and put the result in the pull request. A new kernel is accepted only when the library kernel is missing,
wrong for the layout or precision, or measured slower under the same gate.

## 4. Correctness gates, in this order
A later gate does not replace an earlier one. State the highest gate each change reached.

1. **Unit level, off the device.** Compare with two references: a float64 sum rounded once to the output type, and the
   engine's own code path (its own weight loaders). Bit-exact, or within a stated tolerance with the count of differing
   entries. Use integer-valued inputs where possible.
2. **Mutation tests and negative controls.** Inject single faults (wrong rank, forgotten scale, wrong tile edge) and require
   a test to fail. Tabulate which check caught which fault. A fault no test catches is a hole: add a test, or write down
   that it changes no result.
3. **Device self-test at start-up.** A run off the device does not show that a kernel is right on it. Each new kernel runs at
   server start on fixed inputs at the largest shapes, compares with the reference path, and exits non-zero on a mismatch.
   Put a stall timeout on the first warm-up, set from our own first measured warm-up. Compile from a fresh temporary directory.
4. **Whole-model gate.** Greedy tokens and teacher-forced logprobs against a restart on the same weights, on a fixed prompt
   set, thresholds fixed before the run. When the arithmetic is unchanged the threshold is a logprob difference of 0.0 (the
   in-place full-checkpoint update: [`docs/NIGHT_RECORD_2026-11-21.md`](docs/NIGHT_RECORD_2026-11-21.md)). When it is
   reordered, measure the noise floor of the unchanged engine first and set the threshold from it. Another project's
   thresholds are not ours until recalibrated on our model. A kernel boundary can change how the compiler lowers the
   neighbouring operations, so a correct kernel can still change the model output; only this gate shows it.
5. **Weight-swap identities.** Adapter 1, then 2, then 1 gives identical weights and outputs. The zero adapter equals the
   base. A swapped server equals a restart on the checkpoint merged by the same code.

A difference from the released engine is not called a kernel's fault until the kernel has been compared with a pair of
servers of the same configuration (the A/A control) and with the nearest released mode. **The one-ulp control**: serve base
weights with a few elements of each matrix moved by one unit in the last place and compare with the base on the same
prompts; it sizes the difference such changes make, not the cause of a kernel's.

## 5. Performance claims
- **Noise floor first.** Run the unchanged code against itself, interleaved; the acceptance threshold comes from that spread.
- **Refuse bad logs.** Reject timing logs that are cut off, lack a closing count line, or hold a different number of samples
  than the count says. Never read a file a job is still writing: wait for its completion line, then check the row count.
- **Table, not a point.** Report a table over the model sizes and shapes in scope. No row may be slower than baseline by more
  than the noise floor; a win on one row and a loss on another is reported as it is.
- **Show that the kernel ran.** It must appear in a device profile, and the compiled program must differ from the baseline;
  a speedup with the same kernel name and program is noise. Time without the profiler: a profiled run is for attribution
  only, and if its headline differs from the unprofiled one by more than 10 percent (provisional) mark it perturbed.
- **Credit honestly.** Do not credit a kernel with a gain that came from removing other work. Say whether other runs shared
  the machine; a quieter repeat is better.

## 6. Other accelerators
We run public protocols on Trainium with their client and flags, put the published number in a separate column ("not
reproduced by us") and list every mismatch (precision, engine, chip count, context length, chip generation). The two kinds
of comparison are never mixed in one table. **Like for like** needs the same checkpoint, precision and prompt set; the same
fixed workload (fixed request count and lengths, temperature 0, any failed request fails the point); the full concurrency
table; the same accuracy gate (gate 4) on both sides; our own run on both sides, repeated on the same node. **Indicative,
same protocol** is our run with the named protocol's client and flags. Wording:
no percentage across platforms unless model, precision, workload and metric definition all match, else "indicative only"
with each mismatch named; divide by chips only when both sides are per chip, and mark a value we derive "our arithmetic"; a
ratio found only in text is not a baseline; say "run with the <name> protocol", never "an MLPerf result"; a number from an
open upstream pull request is "reported by the PR author, unmerged, unreproduced"; no number from a blog post or a vendor
page goes into our column; if no published row uses our model, say so; Trainium2 and Trainium3 rows are never mixed, and
the generation is named on every row.

## 7. The standard benchmark setting (llmbench-1024)
One setting for the serving-throughput numbers; a run with other settings is allowed, its data README says so, and it is
not compared with these. These values are in the `.yaml` next to each result `.csv`: client llmbench at commit
`08f416347db19bb45ff2489b8a69eacef62b6875`, run by [`22b_llmbench_grid.sh`](22b_llmbench_grid.sh) (its header names the
repository); prompt 1,024 tokens (the script's grid also has 128 and 3,072); output 256 tokens with `ignore_eos`; 64
requests per cell after 4 warm-up requests; concurrency 1, 8 and 32; temperature 0 (`t0`, other passes are named in the
file name); metrics `output_token_throughput_tps` (output tokens over the wall time of the cell) and `ttft_p50_s`. Report
the full table, not the best cell. A cell with a failed request, or fewer than 256 output tokens per request, is marked
and not used for comparison.

## 8. What to write down for every reported number
The commit of the fork (and of upstream if it differs), the script name and exact command; Trainium generation, instance
type, chips, cores, logical NeuronCore setting; image and SDK versions; engine, version, and every flag and environment
variable that differs from the released defaults; checkpoint and precision; for a public-protocol run, the client commit
and flags; the date, the number of passes (repeat the 1,024 row at least once and give both values), and the raw output
file in `rl/data/<session>/` with its sha256 in that folder's `SHA256SUMS`. A number missing an item is marked unverified
and is not used as a result. Label each number "measured on hardware", "simulator only" or "estimate", never mixed in one
cell. Public text may say that a CPU or simulator test exists and give its command; it quotes no quantity from it.

## 9. Fallbacks
Every new path sits behind a switch that is off by default (`python rl/tools/switches.py` lists them from the code). It
falls back to the previous path when a check fails: start-up self-test, read-back of the written values, or a refused
configuration. On fallback it logs the reason once and stays off for that server. A pull request names the switch, its
default, and what a fallback looks like in the log. A switch that defaults to on needs a line, with its reason, in
`rl/tools/switches_exceptions.txt`.

## Sources
Ideas are borrowed and written in our own words; no code is copied. Their thresholds are for their models and hardware.
[uw-syfi/vibesys](https://github.com/uw-syfi/vibesys) (MIT): paired A/B statistics and pair count, measurement protocol,
injected-fault gate design, device self-test, warm-up stall timeout; its `qwen3.5-397b-a17b-mi300a-bespoke` example README
(node-to-node spread, section 6) is second-hand. [aws-neuron/neuron-agentic-development](https://github.com/aws-neuron/neuron-agentic-development):
NKI and profiling guidance, read through vibesys (second-hand). [vibesys-playground/qwen3.8-trainium2-vibesys](https://github.com/vibesys-playground/qwen3.8-trainium2-vibesys)
(no licence stated): the survey observation (section 3), kernel-boundary effect (gate 4) and long warm-up (gate 3), second-hand.
