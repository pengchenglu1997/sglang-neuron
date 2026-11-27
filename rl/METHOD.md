# Method: when a measurement counts, and what may be claimed

These are rules. A change is accepted when its pull request shows that each rule was followed, or says which one was not
and why. The numbers here are settings and thresholds, not results: results sit in files under [`rl/data/`](data/), and the
headline ones are listed with their tier in [`best.tsv`](best.tsv) (`python rl/tools/best.py`).

## 1. Tiers: every public number carries one

- **`screen`**: at least 2 passes on one server start per arm, in one session. This is the usual practice here.
- **`paired(n)`**: n interleaved server-start pairs of baseline and candidate, after 1 discarded warm-up pair; the
  verdict is the median of the per-pair percent deltas. If the signs of the pairs disagree there is no verdict. A series
  that drifts in one direction has not converged: repeat. n = 7 is the proposed default, until an A/A run sets it.
- **What a claim may rest on.**
  - "Faster" or "slower" by X percent may rest on `screen` only when |X| is 10 or more. The 10 is provisional: it is
    replaced by 5 times the start-to-start spread of the first public A/A run (the unchanged code against itself, in
    pairs). The tier is written next to the number.
  - A smaller difference, and every "no difference" or "no change", needs `paired(n)`. Until then the text says "not
    distinguished at screen tier". The same reading applies to older text that says "no effect".
  - A claim from one pass, or from a file a job is still writing, has no tier and is not a claim.
- `rl/tools/best.py` prints the tier of each headline row and marks one within 10 percent of its baseline as not distinguished.

## 2. Before any experiment

Write four lines in the task or the pull request, before the run:

1. **Question.** One question the run can answer.
2. **Expected result, with basis.** A number or a range and why (an earlier measurement, or "estimate").
3. **Decision either way.** What we do if the expectation is met, and what we do if it is not.
4. **Machine-time limit.** Maximum device minutes and launches. Stop at the limit and report what was measured; raise it
   only with a new written line, never in the middle of a run.

## 3. Before building: survey, then measure what exists

Search and list what you found ("nothing found" counts, with the places looked at): upstream pull requests and issues,
the NKI kernel library and vendor samples, other accelerators. **Measure an existing library kernel before writing a new
one**, on your shape, under your correctness gate, and put the result in the pull request. A new kernel is accepted only
when the library kernel is missing, wrong for the layout or precision, or measured slower under the same gate.

## 4. Correctness gates, in this order

A later gate does not replace an earlier one. State the highest gate each change reached.

1. **Unit level, off the device.** Compare with two references: a float64 sum rounded once to the output type, and the
   engine's own code path (its own weight loaders). Bit-exact, or within a stated tolerance with the count of differing
   entries. Use integer-valued inputs where possible.
2. **Mutation tests and negative controls.** Inject single faults (wrong rank, forgotten scale, wrong tile edge) and
   require a test to fail. A fault no test catches is a hole: add a test, or write down that it changes no result.
3. **Device self-test at start-up.** A run off the device does not show that a kernel is right on the device. Each new kernel runs at
   server start on fixed inputs at the largest shapes, compares with the reference path, and exits non-zero on a mismatch.
   Put a stall timeout on the first warm-up, set from our own first measured warm-up.
4. **Whole-model gate.** Greedy tokens and teacher-forced logprobs against a restart on the same weights, on a fixed prompt
   set, thresholds fixed before the run. When the arithmetic is unchanged the threshold is a logprob difference of 0.0
   (the in-place full-checkpoint update: [`docs/NIGHT_RECORD_2026-11-21.md`](docs/NIGHT_RECORD_2026-11-21.md)). When it is
   reordered, measure the noise floor of the unchanged engine first and set the threshold from it. A kernel boundary can
   change how the compiler lowers the neighbouring operations, so a correct kernel can still change the model output; only
   this gate shows it.
5. **Weight-swap identities.** Adapter 1, then 2, then 1 gives identical weights and outputs. The zero adapter equals the
   base. A swapped server equals a restart on the checkpoint merged by the same code.

**What a kernel is compared against.** A difference from the released engine is not called the kernel's fault until the
kernel has been compared with a pair of servers of the same configuration (the A/A control) and with the nearest released mode.

**The one-ulp control.** Serve base weights with a few elements of each matrix moved by one unit in the last place and
compare with the base on the same prompts: it sizes the difference such changes make, not the cause of a kernel's.

## 5. Performance claims

- **Noise floor first.** Run the unchanged code against itself, interleaved; the acceptance threshold comes from that spread.
- **Refuse bad logs.** Reject timing logs that are cut off, lack a closing count line, or hold a different number of samples
  than the count says. Never read a file a job is still writing: wait for its completion line, then check the row count.
- **Table, not a point.** Report a table over the model sizes and shapes in scope; a win on one row and a loss on another
  is reported as it is.
- **Show that the kernel ran.** It must appear in a device profile, and the compiled program must differ from the baseline.
- **Time without the profiler.** A profiled run is for attribution only: if its headline differs from the unprofiled one by
  more than 10 percent (provisional), mark the profile perturbed and use it only to show what ran.
- **Credit honestly.** Do not credit a kernel with a gain that came from removing other work.
- **Other accelerators.** Only a run by us under a named public protocol, with its client and flags; the published number
  goes in a separate column ("not reproduced by us") beside every mismatch (precision, engine, chip count, context length,
  chip generation). No percentage across platforms unless model, precision, workload and metric definition all match;
  otherwise "indicative only". Trainium2 and Trainium3 rows are never mixed.

## 6. The standard benchmark setting (llmbench-1024)

One setting for the serving-throughput numbers. A run with other settings is allowed; its data README says so and it is
not compared with these.

| Item | Value (as in the `.yaml` next to each result `.csv`) |
|---|---|
| Client | llmbench at commit `08f416347db19bb45ff2489b8a69eacef62b6875`, run by [`22b_llmbench_grid.sh`](22b_llmbench_grid.sh) (its header names the repository) |
| Prompt length | 1,024 tokens (the script's own grid also has 128 and 3,072) |
| Output | 256 tokens, `ignore_eos` on |
| Requests per cell | 64, after 4 warm-up requests |
| Concurrency | 1, 8 and 32 |
| Temperature | 0 (`t0`); other passes are named in the file name |
| Throughput, latency | `output_token_throughput_tps` (output tokens over the wall time of the cell), `ttft_p50_s` |

Report the full table, not the best cell. A cell with a failed request, or with fewer than 256 output tokens per request,
is marked and not used for comparison.

## 7. What to write down for every reported number

- the commit of the fork (and of upstream if it differs), the script name and the exact command;
- Trainium generation, instance type, chips, cores, logical NeuronCore setting; whether other runs shared the machine;
- image and SDK versions; engine, version, and every flag and environment variable that differs from the released
  defaults; checkpoint and precision; for a public-protocol run, the client commit and flags;
- the date, the number of passes (repeat the 1,024 row at least once and give both values), and the raw output file in
  `rl/data/<session>/`, with its sha256 in that folder's `SHA256SUMS`.

A number missing an item is a draft and is marked unverified. Label each number "measured on hardware", "simulator only" or
"estimate", never mixed in one cell. Public text may say that a CPU or simulator test exists and give its command; it
quotes no quantity from it.

## 8. Fallbacks

Every new path sits behind a switch that is off by default (`python rl/tools/switches.py` lists them from the code). It
falls back to the previous path when a check fails: start-up self-test, read-back of the written values, or a refused
configuration. On fallback it logs the reason once and stays off for that server. A pull request names the switch, its
default, and what a fallback looks like in the log. A switch that defaults to on needs a line, with its reason, in
`rl/tools/switches_exceptions.txt`.
