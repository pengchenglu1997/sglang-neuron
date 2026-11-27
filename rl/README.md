# rl/: team tooling for RL post-training on Trainium2

This folder is team tooling. It is not part of sglang-neuron and never travels upstream (`rl/tools/upstream_ready.sh`
fails a branch that carries it). It holds the scripts that measure the engine changes of this fork, the raw files they
wrote, and the checks that keep the public text honest. Paths below are relative to this folder; scripts sit at the top
level because the records, `backends.json` and every script's `sys.path` name them.

- Rules for a measurement and what a public number may claim: [`METHOD.md`](METHOD.md). Headline numbers and their tier:
  [`best.tsv`](best.tsv), printed by `python rl/tools/best.py`.
- Rebuild one number from a single checkout: [`REPRODUCE.md`](REPRODUCE.md).
- Per-probe index (question, hardware, result files): [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Script groups

| Group | Files | Role |
|---|---|---|
| Core library | `rl_common.py`, `rollout_client.py`, `server_ctl.py`, `backends.json`, `mk_alt.py` | result format, HTTP client, server start and stop from `backends.json` |
| Trainers and loop | `trainer.py`, `ptrainer.py`, `trainer8b.py`, `30_rl_loop.py` | LoRA policy-gradient trainers and the GRPO-style loop (`--backend`, `--sync-payload`, `--overlap`, `--seed`) |
| Server launchers | `srv_sglang_neuron.sh`, `srv_vllm_neuron_docker.sh`, `srv_sglang_cuda_docker.sh`, `srv_other.sh`, `nki_index_check.py` | one script per engine |
| Probes | `20_` to `82_*.py` and `*.sh` | one script per timing or correctness question; see `docs/EXPERIMENTS.md` |
| Session scripts | `50_run_trn2_session.sh`, `run_night_1121.sh`, `run_shape_1122.sh`, `run_w1w2.sh`, `r9_run.sh`, `s4_*.sh`, `s4_*.py`, `scripts/` | the order in which probes ran on one machine, as run |
| Benchmark client | `22_llmbench.sh`, `22b_llmbench_grid.sh` | llmbench grid (the setting is in `METHOD.md` section 6) |
| Patch folders | `make_patch_dir.py`, `patch_manifest.py`, `patches/` | build the folder a server mounts from a fork commit; `patches/` is the frozen as-run in-place patch |
| Weight and checkpoint tools | `lora_merge.py`, `mk_ulp.py`, `untie_ckpt.py` | merge an adapter, perturb or untie a checkpoint |
| Report helpers | `60_report_figures.py`, `59b_` to `59d_`, `63_padding_share.py`, `77_pair_table.py`, `81_opt_table.py` | tables and figures read from result files |
| Selftests | `selftest_*.py` | no device, no server; see below |
| Data | `data/<date>_<tag>/` | raw result files of each session, with its README and sha256 list; frozen once merged |
| Records | `docs/` | session records and plans, frozen |
| Checks | `tools/` | the team checks and the generators below |

## Tools (`rl/tools/`)

| Command | What it answers |
|---|---|
| `python rl/tools/best.py` | the headline numbers with tier and baseline; `--check` reads every file |
| `python rl/tools/switches.py` | the environment variables this fork's engine changes add, read from the code |
| `bash rl/tools/fork_state.sh` | branches, open pull requests, our upstream pull requests and issues, what main has that integration lacks |
| `bash rl/tools/upstream_ready.sh <branch>` | whether a branch carries only engine paths |
| `python rl/tools/rewrite_links.py` | the links that still point into the former run repository (dry run) |
| `python rl/tools/scrub_check.py <paths>` | machine, account and name leaks, before every push |

## Selftests

From the repository root:

```bash
for f in rl/selftest_*.py; do python "$f"; done       # needs git and bash; each prints its own pass line
python -m unittest discover -s rl/tools/tests -p 'test_*.py'
python rl/tools/best.py --check
python rl/tools/switches.py --check
python rl/tools/check_tables.py
```

`rl/selftest_*.py` and the unittest run are what the required CI check F9 runs. The tests that need a checkout of
another project (`62_lazy_delta_test.py`, `66_pull_weights_shim_test.py`) are not in that loop; their headers say how
to point them at it.

## License

Apache-2.0, as the repository. The files under `patches/sglang_neuron_v0.1.0_inplace/` are modified copies of
sgl-project/sglang-neuron at tag v0.1.0 (Apache-2.0); `orig/` holds the unmodified v0.1.0 files.
