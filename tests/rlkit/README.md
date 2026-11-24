# CPU tests of the RL kit

These tests cover the LoRA adapter path of `sglang_neuron/models/host.py` and the LoRA shard-merge NKI kernel
(`sglang_neuron/models/rlkit_lora_merge.py`, wired in by `sglang_neuron/models/rlkit_kernel_path.py`).

They run on a CPU. They need Python with `nki` 0.7.0 (its CPU simulator), `torch` (CPU build), `safetensors` and
`numpy`. They do not need `sglang`, a Neuron device or a compiler.

```bash
PY=/path/to/python bash tests/rlkit/run_all.sh
```

| test | what it checks |
|---|---|
| `test_merge_kernel_sim.py` | the kernel in the NKI simulator against a float32 reference: zero adapter, integer factors, random factors, 20 alternating swaps, K 16 / 32 / 48, partial tiles, every per-rank shape of Qwen3-0.6B (and of Qwen3-8B with `RUN_8B=1`), LNC 1 and 2 |
| `test_k_chunks.py` | contraction sizes above 128 (two accumulating matmuls) and the shape guards |
| `test_factor_layout.py` | the per-rank LoRA factors, built through the engine's own weight loaders, against `loader(scale * B @ A)` with integer factors: bit-exact for qkv, o, gate_up and down at TP 1, 2, 4, 8, 16 |
| `negative_control_factor_layout.py` | three deliberate layout mistakes (wrong rank, forgotten scale, summed segments); the script passes only if the correct factors pass and every mistake fails on every rank |
| `test_end_to_end_review.py` | loader -> factors -> kernel against `loader(base + full delta in float64)` rounded once |
| `test_lazy_delta.py` | the lazy low-rank delta of the host path against the full-matrix path, through the engine's loaders |
| `test_host_kernel_path.py` | the patched `host.py` imported whole and driven through `update_weights_in_place` on adapter folders on disk, with the kernel in the simulator (Qwen3-0.6B TP4 shapes) and as torch arithmetic elsewhere; refusals, fallbacks, failures in the middle of an update, adapter 1 -> 2 -> 1 |

## What these tests do not show

- Nothing here ran on a Neuron device. The kernel has not been compiled by `neuronx-cc` and has not been called
  through `wrap_nki` on hardware.
- `test_host_kernel_path.py` is not a server run: the engine modules that `host.py` imports at the top are stubs and the
  graph is a stand-in that declares the Qwen3 layer parameters. The engine's weight loaders are the real ones.
- The simulator shows that the arithmetic is right. It says nothing about speed.

The kernel mode is off by default (`RLKIT_DEVICE_WRITE=kernel` turns it on). On a device, start with
`RLKIT_KERNEL_CHECK=all`, which reads every written parameter back and compares it with the host result.
