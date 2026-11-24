#!/usr/bin/env python3
"""CPU test of the kernel mode of the patched host.py (RLKIT_DEVICE_WRITE=kernel) from the adapter folder to the
live parameters. NOT a server run and NOT hardware.

What is real here: the patched models/host.py itself (imported whole, its update_weights_in_place called), the patch's
rlkit_kernel_path.py and rlkit_lora_merge.py, the engine's own weight loaders (v0.1.0 source), safetensors adapter
and checkpoint files on disk, and the NKI kernel in the NKI CPU simulator (RLKIT_KERNEL_IMPL=sim).
What is a stand-in: the engine modules host.py imports at the top (envs, sampling, sglang_tree, runtime_init,
models.base, models.phase, parallel.state) are empty stubs, and the model graph is a small module that declares the
Qwen3 parameters of each layer with the engine's loaders and the engine's checkpoint-key mapping. No serving graph,
no compile, no device.

    ~/nki_sim_venv/bin/python test_host_kernel_path.py        (exit code 1 on any failure)
"""
import hashlib
import importlib.util
import json
import logging
import os
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PATCH = HERE.parents[1] / "sglang_neuron"
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
sys.path.insert(0, str(HERE))

import factor_builders as fb  # noqa: E402  (puts the engine source on sys.path)
from lora_merge import BF16, lora_merge_reference  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from sglang_neuron.utils.weight_loader import set_weight_loader  # noqa: E402

fails = []
n_pass = 0


def check(name, ok, detail=""):
    global n_pass
    print(("PASS " if ok else "FAILED ") + name + (" | " + detail if detail else ""))
    if ok:
        n_pass += 1
    else:
        fails.append(name)


# ---- the patched host.py, with stubs for the engine modules it imports at the top ---------------------------------
def stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


import sglang_neuron.models  # noqa: E402,F401  (the real package, from the engine source)


class _NeuronModel(nn.Module):
    pass


pkg = sys.modules["sglang_neuron"]
for nm, attrs in (("envs", {}), ("sampling", {"SAMPLER_MODES": ("greedy",), "HOST_SAMPLER": "host"}),
                  ("sglang_tree", {})):
    setattr(pkg, nm, stub("sglang_neuron." + nm, **attrs))
stub("sglang_neuron.model_executor", __path__=[])
stub("sglang_neuron.model_executor.runtime_init", neuron_device=lambda: torch.device("cpu"))
stub("sglang_neuron.models.base", NeuronModel=_NeuronModel, segmented_prefill_cache_policy=lambda phase: None)
stub("sglang_neuron.models.phase", model_phase=lambda: "all")
stub("sglang_neuron.parallel", __path__=[])
stub("sglang_neuron.parallel.state", initialize_neuron_parallel_state=lambda **kw: None)


def load_patch(module_name, rel):
    spec = importlib.util.spec_from_file_location(module_name, PATCH / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


lm_patch = load_patch("sglang_neuron.models.rlkit_lora_merge", "models/rlkit_lora_merge.py")
kp = load_patch("sglang_neuron.models.rlkit_kernel_path", "models/rlkit_kernel_path.py")
setattr(sys.modules["sglang_neuron.models"], "rlkit_kernel_path", kp)
hostmod = load_patch("sglang_neuron.models.host_patched", "models/host.py")


class LogTap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


tap = LogTap()
for lg in (hostmod.logger, kp.logger):
    lg.addHandler(tap)
    lg.setLevel(logging.INFO)
    lg.propagate = False

import lora_merge as _kernel_under_test  # noqa: E402
check("the tests load the kernel from sglang_neuron/models/rlkit_lora_merge.py (one source, no copy)",
      Path(_kernel_under_test._SRC).resolve() == (PATCH / "models" / "rlkit_lora_merge.py").resolve())

# ---- a stand-in graph: the Qwen3 layer parameters, the engine's loaders and checkpoint keys -------------------------
KIND_OF = {"qkv_weight": "qkv", "o_weight": "o", "gate_up_weight": "gate_up", "down_weight": "down"}
MODULE_PATH = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
               "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}


class _Attn(nn.Module):
    def __init__(self, dims, tp, loaders):
        super().__init__()
        self.qkv_weight = nn.Parameter(torch.empty(*dims.live_shape("qkv", tp), dtype=torch.bfloat16))
        self.o_weight = nn.Parameter(torch.empty(*dims.live_shape("o", tp), dtype=torch.bfloat16))
        set_weight_loader(self.qkv_weight, loaders["qkv"])
        set_weight_loader(self.o_weight, loaders["o"])


class _MLP(nn.Module):
    def __init__(self, dims, tp, loaders):
        super().__init__()
        self.gate_up_weight = nn.Parameter(torch.empty(*dims.live_shape("gate_up", tp), dtype=torch.bfloat16))
        self.down_weight = nn.Parameter(torch.empty(*dims.live_shape("down", tp), dtype=torch.bfloat16))
        set_weight_loader(self.gate_up_weight, loaders["gate_up"])
        set_weight_loader(self.down_weight, loaders["down"])


class _Layer(nn.Module):
    def __init__(self, dims, tp, loaders):
        super().__init__()
        self.self_attn = _Attn(dims, tp, loaders)
        self.mlp = _MLP(dims, tp, loaders)


class FakeGraph(nn.Module):
    def __init__(self, dims, tp, rank, n_layers):
        super().__init__()
        loaders = fb.engine_loaders(dims, tp)
        self.layers = nn.ModuleList([_Layer(dims, tp, loaders) for _ in range(n_layers)])
        self.norm_weight = nn.Parameter(torch.empty(dims.hidden, dtype=torch.bfloat16))    # never adapted
        self.tp_group = SimpleNamespace(rank_in_group=rank)

    def hf_weight_mappings(self):
        m = {"norm_weight": "model.norm.weight"}
        for i in range(len(self.layers)):
            src, dst = f"model.layers.{i}", f"layers.{i}"
            m[f"{dst}.self_attn.qkv_weight"] = [f"{src}.self_attn.{n}_proj.weight" for n in "qkv"]
            m[f"{dst}.self_attn.o_weight"] = f"{src}.self_attn.o_proj.weight"
            m[f"{dst}.mlp.gate_up_weight"] = [f"{src}.mlp.gate_proj.weight", f"{src}.mlp.up_proj.weight"]
            m[f"{dst}.mlp.down_weight"] = f"{src}.mlp.down_proj.weight"
        return m

    def load_weights(self, path, device):
        from sglang_neuron.utils.weight_loader import get_weight_loader
        rank = self.tp_group.rank_in_group
        params = dict(self.named_parameters())
        with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as f:
            for name, keys in self.hf_weight_mappings().items():
                keys = [keys] if isinstance(keys, str) else keys
                t = get_weight_loader(params[name]).load([f.get_slice(k) for k in keys], rank)
                params[name].data = t.to(torch.bfloat16).to(device)


class THost(hostmod.ModelHost):
    name = "stand-in"

    def __init__(self, dims, tp, rank, n_layers, ckpt):
        nn.Module.__init__(self)
        self.dims, self.tp, self.rank, self.n_layers = dims, tp, rank, n_layers
        self.hf_config = SimpleNamespace(_name_or_path=ckpt)
        g = self.build_graph("decode", self.hf_config)
        g.load_weights(ckpt, torch.device("cpu"))
        self.model_graphs = {"decode": g}
        self._weights_loaded = True

    def build_graph(self, phase, config):
        return FakeGraph(self.dims, self.tp, self.rank, self.n_layers)

    def live(self):
        return {n: p.detach().clone() for n, p in self.model_graphs["decode"].named_parameters()}


# ---- files ---------------------------------------------------------------------------------------------------------
TMP = Path(tempfile.mkdtemp(prefix="hostkernel_"))


def bf16_valued(t):
    return t.to(torch.bfloat16).to(torch.float32)


def make_checkpoint(tag, dims, n_layers, seed):
    g = torch.Generator().manual_seed(seed)
    d = TMP / tag
    d.mkdir()
    t = {"model.norm.weight": bf16_valued(torch.randn(dims.hidden, generator=g))}
    for i in range(n_layers):
        for kind in fb.KINDS:
            for mod, out in zip(fb.HF_MODULES[kind], dims.hf_out_dims(kind)):
                t[f"model.layers.{i}.{MODULE_PATH[mod]}.{mod}.weight"] = bf16_valued(
                    torch.randn(out, dims.hf_in_dim(kind), generator=g) * 0.02)
    save_file(t, str(d / "model.safetensors"))
    return str(d), t


def make_adapter(tag, dims, n_layers, seed, r=16, alpha=32, modules=None, zero_b=False):
    """A PEFT-format adapter folder; returns (path, {hf key: (B, A)}, scale)."""
    g = torch.Generator().manual_seed(seed)
    d = TMP / tag
    d.mkdir()
    t, pairs = {}, {}
    for i in range(n_layers):
        for kind in fb.KINDS:
            for mod, out in zip(fb.HF_MODULES[kind], dims.hf_out_dims(kind)):
                if modules is not None and mod not in modules:
                    continue
                a = torch.randn(r, dims.hf_in_dim(kind), generator=g) * 0.05
                b = torch.zeros(out, r) if zero_b else torch.randn(out, r, generator=g) * 0.05
                stem = f"model.layers.{i}.{MODULE_PATH[mod]}.{mod}"
                t[f"base_model.model.{stem}.lora_A.weight"] = a
                t[f"base_model.model.{stem}.lora_B.weight"] = b
                pairs[stem + ".weight"] = (b, a)
    save_file(t, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha}))
    return str(d), pairs, alpha / r


def env(**kv):
    for k in ("RLKIT_DEVICE_WRITE", "RLKIT_KERNEL_IMPL", "RLKIT_KERNEL_FACTORS", "RLKIT_KERNEL_CHECK",
              "RLKIT_ADAPTER_ON_DEVICE", "RLKIT_LAZY_DELTA"):
        os.environ.pop(k, None)
    os.environ.update(kv)


def same(x, y):
    return all(torch.equal(x[n], y[n]) for n in x)


def n_diff(x, y):
    return sum(int((x[n].float() != y[n].float()).sum()) for n in x), sum(x[n].numel() for n in x)


def reference(host, base_ckpt, pairs, scale):
    """Per live parameter, by code that shares nothing with rlkit_kernel_path: the engine loader on the checkpoint
    for the base shard, builder B's kind-based factor builders, and the numpy reference of the kernel
    (float32 sum, one cast). Also the float64 sum rounded once. Parameters with no pair are the base."""
    loaders = fb.engine_loaders(host.dims, host.tp)
    out32, out64 = {}, {}
    for name, p in host.model_graphs["decode"].named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in KIND_OF:
            out32[name] = out64[name] = p.detach().clone()
            continue
        kind = KIND_OF[leaf]
        i = int(name.split(".")[1])
        keys = [f"model.layers.{i}.{MODULE_PATH[m]}.{m}.weight" for m in fb.HF_MODULES[kind]]
        base = loaders[kind].load([fb.TensorSlice(base_ckpt[k]) for k in keys], host.rank)
        got = [pairs.get(k) for k in keys]
        if all(g is None for g in got):
            out32[name] = out64[name] = base.to(torch.bfloat16)
            continue
        a_stack, bt = fb.build_factors(kind, got, scale, host.rank, host.tp, host.dims)
        b_np = base.numpy().astype(BF16)
        ref = lora_merge_reference(b_np, a_stack.numpy(), bt.numpy()).astype(np.float32)
        out32[name] = torch.from_numpy(ref).to(torch.bfloat16)
        full = [fb.TensorSlice(base_ckpt[k].double() + scale * (pairs[k][0].double() @ pairs[k][1].double()))
                for k in keys]
        exact = loaders[kind].load(full, host.rank)
        out64[name] = torch.from_numpy(exact.numpy().astype(np.float32).astype(BF16).astype(np.float32)).to(torch.bfloat16)
    return out32, out64


def kernel_log():
    return [ln for ln in tap.lines if "by the kernel (" in ln]


# ---- 1. the real path at Qwen3-0.6B shapes, TP4, the kernel in the simulator ------------------------------------------
D06 = fb.QWEN3_0_6B
ckpt06, base06 = make_checkpoint("base06", D06, 1, seed=1)
ad1, pairs1, scale1 = make_adapter("ad1", D06, 1, seed=11)
ad2, pairs2, scale2 = make_adapter("ad2", D06, 1, seed=12, alpha=5)          # scale 0.3125
ad_zero, pairs0, _ = make_adapter("ad_zero", D06, 1, seed=13, zero_b=True)
ad_attn, pairs_attn, scale_attn = make_adapter("ad_attn", D06, 1, seed=14, modules=("q_proj", "k_proj", "v_proj", "o_proj"))

for rank in (0, 3):
    tag = f"0.6B TP4 rank {rank} sim"
    env(RLKIT_DEVICE_WRITE="kernel", RLKIT_KERNEL_IMPL="sim")
    h = THost(D06, 4, rank, 1, ckpt06)
    base_live = h.live()
    tap.lines.clear()
    h.update_weights_in_place(ad1)
    first = h.live()
    logs = kernel_log()
    check(tag + ": the kernel mode ran (its log line is there, no fallback warning)",
          len(logs) == 1 and not any("failed" in ln for ln in tap.lines), (logs or tap.lines or [""])[0][:400])
    st = h._experiment_adapter_state
    check(tag + ": all 4 adapted parameters by the kernel, none on the host",
          len(st["base_dev"]) == 4 and not st["base"] and st.get("kernel_ok", True), f"base_dev {len(st['base_dev'])}, host base {len(st['base'])}")
    ref32, ref64 = reference(h, base06, pairs1, scale1)
    d, n = n_diff(first, ref32)
    check(tag + ": live parameters equal the independent float32 reference bit for bit", d == 0, f"{d} of {n} differ")
    d, n = n_diff(first, ref64)
    check(tag + ": close to the float64 sum rounded once", d <= n * 1e-3, f"{d} of {n} differ")
    check(tag + ": the unadapted parameter is untouched", torch.equal(first["norm_weight"], base_live["norm_weight"]))
    # host path on a twin
    env()
    twin = THost(D06, 4, rank, 1, ckpt06)
    twin.update_weights_in_place(ad1)
    d, n = n_diff(first, twin.live())
    check(tag + ": vs the host path of the same patch", d <= n * 1e-3, f"{d} of {n} differ")
    # swaps
    env(RLKIT_DEVICE_WRITE="kernel", RLKIT_KERNEL_IMPL="sim")
    h.update_weights_in_place(ad2)
    second = h.live()
    r32, _ = reference(h, base06, pairs2, scale2)
    d, n = n_diff(second, r32)
    check(tag + ": adapter 2 after adapter 1 equals adapter 2 on the base (no adding up)", d == 0, f"{d} of {n} differ")
    h.update_weights_in_place(ad1)
    check(tag + ": adapter 1 again is bit-identical to the first time", same(h.live(), first))
    h.update_weights_in_place(ad_attn)
    r32, _ = reference(h, base06, pairs_attn, scale_attn)
    got = h.live()
    d, n = n_diff(got, r32)
    check(tag + ": attention-only adapter: attention merged, MLP back to the base", d == 0 and
          torch.equal(got["layers.0.mlp.down_weight"], base_live["layers.0.mlp.down_weight"]) and
          torch.equal(got["layers.0.mlp.gate_up_weight"], base_live["layers.0.mlp.gate_up_weight"]), f"{d} of {n} differ")
    h.update_weights_in_place(ad_zero)
    check(tag + ": zero adapter (B = 0) gives the base exactly", same(h.live(), base_live))

# ---- 2. every rank at TP 1, 2, 4, two layers, torch arithmetic in place of the kernel ------------------------------
ckpt06b, base06b = make_checkpoint("base06b", D06, 2, seed=2)
ad1b, pairs1b, scale1b = make_adapter("ad1b", D06, 2, seed=21)
for tp in (1, 2, 4):
    for rank in range(tp):
        env(RLKIT_DEVICE_WRITE="kernel", RLKIT_KERNEL_IMPL="torch")
        h = THost(D06, tp, rank, 2, ckpt06b)
        tap.lines.clear()
        h.update_weights_in_place(ad1b)
        ref32, _ = reference(h, base06b, pairs1b, scale1b)
        d, n = n_diff(h.live(), ref32)
        st = h._experiment_adapter_state
        check(f"0.6B TP{tp} rank {rank} torch impl: 8 parameters by the kernel path, close to the reference",
              len(st["base_dev"]) == 8 and not st["base"] and d <= n * 1e-3 and len(kernel_log()) == 1,
              f"{d} of {n} differ; base_dev {len(st['base_dev'])}")

# ---- 3. refusals, mixed updates, failures ----------------------------------------------------------------------------
env()
host_twin = THost(D06, 4, 1, 2, ckpt06b)
host_twin.update_weights_in_place(ad1b)
host_result = host_twin.live()

# 3a. shards the serving guard does not cover (TP16: down IN = 192, qkv OUT = 384) go to the host
env(RLKIT_DEVICE_WRITE="kernel", RLKIT_KERNEL_IMPL="torch")
h = THost(D06, 16, 5, 2, ckpt06b)
tap.lines.clear()
h.update_weights_in_place(ad1b)
env()
t16 = THost(D06, 16, 5, 2, ckpt06b)
t16.update_weights_in_place(ad1b)
st = h._experiment_adapter_state
got, want = h.live(), t16.live()
refused = [n for n in got if n in st["base"]]
check("TP16 rank 5: unaligned shards are refused and merged on the host with the host path's exact values",
      len(refused) >= 4 and all(torch.equal(got[n], want[n]) for n in refused) and len(kernel_log()) == 1,
      f"{len(refused)} on the host, {len(st['base_dev'])} by the kernel; {(kernel_log() or [''])[0][:300]}")
d, n = n_diff(got, want)
check("TP16 rank 5: the whole result is close to the pure host path", d <= n * 1e-3, f"{d} of {n} differ")

# 3b. K above the guard (r = 64: qkv K = 192) on the host, then a rank-16 adapter takes the same parameter by the kernel
ad64, pairs64, scale64 = make_adapter("ad64", D06, 2, seed=31, r=64, alpha=64)
env(RLKIT_DEVICE_WRITE="kernel", RLKIT_KERNEL_IMPL="torch")
h = THost(D06, 4, 1, 2, ckpt06b)
h.update_weights_in_place(ad64)
st = h._experiment_adapter_state
env()
t64 = THost(D06, 4, 1, 2, ckpt06b)
t64.update_weights_in_place(ad64)
got, want = h.live(), t64.live()
qkv = "layers.0.self_attn.qkv_weight"
check("r = 64: qkv (K = 192) is refused and equals the host path exactly; o, gate_up, down by the kernel",
      qkv in st["base"] and qkv not in st["base_dev"] and torch.equal(got[qkv], want[qkv]) and
      "layers.0.mlp.gate_up_weight" in st["base_dev"], f"host {sorted(st['base'])[:2]}, kernel {len(st['base_dev'])}")
env(RLKIT_DEVICE_WRITE="kernel", RLKIT_KERNEL_IMPL="torch")
h.update_weights_in_place(ad1b)
fresh = THost(D06, 4, 1, 2, ckpt06b)
fresh.update_weights_in_place(ad1b)
check("r = 64 then r = 16: the parameter moved from the host path to the kernel starts from the true base",
      same(h.live(), fresh.live()) and qkv in h._experiment_adapter_state["base_dev"])
# and back: kernel first, then an adapter whose qkv is refused
h.update_weights_in_place(ad64)
check("r = 16 then r = 64: the parameter moved from the kernel to the host path starts from the true base",
      same(h.live(), got))

# 3c. the kernel call fails on its third call: the whole update is redone on the host, and stays there
real_make = kp.make_runner


class Failing:
    how = "failing"

    def __init__(self):
        self.calls = 0

    def __call__(self, base, a, bt):
        self.calls += 1
        if self.calls >= 3:
            raise RuntimeError("injected kernel failure")
        return kp._torch_merge(base, a, bt)


kp.make_runner = lambda device_type: Failing()
env(RLKIT_DEVICE_WRITE="kernel")
h = THost(D06, 4, 1, 2, ckpt06b)
tap.lines.clear()
h.update_weights_in_place(ad1b)
st = h._experiment_adapter_state
check("kernel failure on the third call: the result is the host path's, bit for bit",
      same(h.live(), host_result) and st["kernel_ok"] is False and not st["base_dev"] and
      any("by the kernel failed" in ln for ln in tap.lines))
kp.make_runner = real_make
env(RLKIT_DEVICE_WRITE="kernel", RLKIT_KERNEL_IMPL="torch")
ad2b, pairs2b, scale2b = make_adapter("ad2b", D06, 2, seed=22)
h.update_weights_in_place(ad2b)
after_fail = h.live()
env()
t2 = THost(D06, 4, 1, 2, ckpt06b)
t2.update_weights_in_place(ad2b)
check("after a kernel failure a later (different) adapter is merged on the host path, from the base",
      same(after_fail, t2.live()) and not kernel_log())

# 3d. a kernel that works for one update, then fails: parameters it wrote earlier restart from their device base copy


class FailsLater(Failing):
    def __call__(self, base, a, bt):
        self.calls += 1
        if self.calls >= 11:                  # 8 calls in the first update, fails on the third of the second
            raise RuntimeError("injected kernel failure")
        return kp._torch_merge(base, a, bt)


kp.make_runner = lambda device_type: FailsLater()
env(RLKIT_DEVICE_WRITE="kernel")
h = THost(D06, 4, 1, 2, ckpt06b)
h.update_weights_in_place(ad1b)
ok_first = h._experiment_adapter_state.get("kernel_ok", True) and len(h._experiment_adapter_state["base_dev"]) == 8
h.update_weights_in_place(ad2b)
check("kernel fails in the second update: the result is adapter 2 on the base (host path values), not on adapter 1",
      ok_first and same(h.live(), t2.live()) and h._experiment_adapter_state["kernel_ok"] is False)
kp.make_runner = real_make

# 3e. a kernel that returns wrong values is caught by the read-back check
class Wrong:
    how = "wrong"

    def __call__(self, base, a, bt):
        return (kp._torch_merge(base, a, bt).float() * 1.5).to(base.dtype)


kp.make_runner = lambda device_type: Wrong()
env(RLKIT_DEVICE_WRITE="kernel")
h = THost(D06, 4, 1, 2, ckpt06b)
tap.lines.clear()
h.update_weights_in_place(ad1b)
check("a kernel returning wrong values trips the read-back check of the live parameter; the update is redone on "
      "the host with the host path's values",
      same(h.live(), host_result) and any("differs from the host arithmetic" in ln for ln in tap.lines) and
      h._experiment_adapter_state["kernel_ok"] is False)
kp.make_runner = real_make

# 3e2. the read-back check runs once per kind of shard, not once per server
env(RLKIT_DEVICE_WRITE="kernel", RLKIT_KERNEL_IMPL="torch")
h = THost(D06, 4, 1, 2, ckpt06b)
h.update_weights_in_place(ad1b)
check("read-back check: one per (call, shard shape, K, factor dtype): 4 for the 4 kinds of Qwen3 shard",
      len(h._experiment_adapter_state["kernel_checked"]) == 4, str(sorted(h._experiment_adapter_state["kernel_checked"], key=str)))

# 3i. gaps found by the mutation test of 2026-11-23
ad_o, pairs_o, scale_o = make_adapter("ad_o", D06, 2, seed=41, modules=("o_proj",))
env()
t_o = THost(D06, 4, 1, 2, ckpt06b)
t_o.update_weights_in_place(ad_o)
only_o = t_o.live()
# kernel wrote qkv first (r = 16), then the guard refuses it (r = 64): the host part must start from the device base
env(RLKIT_DEVICE_WRITE="kernel", RLKIT_KERNEL_IMPL="torch")
h = THost(D06, 4, 1, 2, ckpt06b)
h.update_weights_in_place(ad1b)
h.update_weights_in_place(ad64)
d, n = n_diff(h.live(), t64.live())
check("fresh server, r = 16 by the kernel then r = 64: refused qkv equals the pure host path exactly",
      torch.equal(h.live()[qkv], t64.live()[qkv]) and d <= n * 1e-3, f"{d} of {n} differ in all")
# a parameter merged on the host (qkv at r = 64), then absent from the next adapter: back to the base
h = THost(D06, 4, 1, 2, ckpt06b)
h.update_weights_in_place(ad64)
h.update_weights_in_place(ad_o)
d, n = n_diff(h.live(), only_o)
check("kernel mode, r = 64 then an o_proj-only adapter: the host-merged qkv is put back to the base",
      torch.equal(h.live()[qkv], only_o[qkv]) and d <= n * 1e-3, f"{d} of {n} differ in all")
# the same in pure host mode (older code that the kernel mode's host part uses)
env()
h = THost(D06, 4, 1, 2, ckpt06b)
h.update_weights_in_place(ad1b)
h.update_weights_in_place(ad_o)
check("host mode, full adapter then an o_proj-only adapter: every other parameter is put back to the base",
      same(h.live(), only_o))

# 3j. an adapter covering only part of a fused parameter is refused before anything is written, in both modes
ad_part, _, _ = make_adapter("ad_part", D06, 2, seed=51, modules=("q_proj", "v_proj", "o_proj"))
for mode_env in ({}, {"RLKIT_DEVICE_WRITE": "kernel", "RLKIT_KERNEL_IMPL": "torch"}):
    env(**mode_env)
    h = THost(D06, 4, 1, 2, ckpt06b)
    h.update_weights_in_place(ad1b)
    before = h.live()
    try:
        h.update_weights_in_place(ad_part)
        raised = False
    except NotImplementedError:
        raised = True
    check(f"partial coverage (q and v without k) in {'kernel' if mode_env else 'host'} mode: refused, nothing written",
          raised and same(h.live(), before))

# 3k. each compiled helper of the device runner is a function with a code object of its own
f1, f2 = kp._own_function(lambda x, y, z: 1), kp._own_function(lambda x, y, z: 2)
check("the device runner's per-shape helpers have distinct code objects",
      f1.__code__ is not f2.__code__ and f1(0, 0, 0) == 1 and f2(0, 0, 0) == 2)

# 3f. a full checkpoint update is the new base for later adapters
ckpt_new, base_new = make_checkpoint("base06c", D06, 2, seed=3)
env(RLKIT_DEVICE_WRITE="kernel", RLKIT_KERNEL_IMPL="torch")
h = THost(D06, 4, 1, 2, ckpt06b)
h.update_weights_in_place(ad1b)
h.update_weights_in_place(ckpt_new)
st = h._experiment_adapter_state
cleared = not st["base"] and not st["base_dev"]
h.update_weights_in_place(ad1b)
fresh = THost(D06, 4, 1, 2, ckpt_new)
fresh.update_weights_in_place(ad1b)
check("full update then adapter: base copies dropped, the adapter merges into the new base",
      cleared and same(h.live(), fresh.live()))

# 3g. split factors (bf16 hi/lo, K' = 3K = 144 for qkv is inside the guard)
env(RLKIT_DEVICE_WRITE="kernel", RLKIT_KERNEL_IMPL="torch", RLKIT_KERNEL_FACTORS="split")
h = THost(D06, 4, 1, 2, ckpt06b)
h.update_weights_in_place(ad1b)
d, n = n_diff(h.live(), host_result)
# Bound 1 %: split mode keeps about 16 mantissa bits per factor and was measured in the simulator to differ from the
# float64 sum in 0.18 to 0.27 % of entries (RESULTS_kernel_sim.md). The first run of this test used the fp32 bound
# of 0.1 % here by mistake and failed with 22017 of 7865344 (0.28 %); the bound was corrected after that run.
check("split factors: all parameters by the kernel path, within the split-mode bound of the host path",
      len(h._experiment_adapter_state["base_dev"]) == 8 and d <= n * 1e-2, f"{d} of {n} differ")

# 3h. the layout check refuses a loader the factor builder does not understand
env(RLKIT_DEVICE_WRITE="kernel", RLKIT_KERNEL_IMPL="torch")
h = THost(D06, 4, 1, 2, ckpt06b)
g0 = h.model_graphs["decode"]
real_build = h.build_graph


def odd_build(phase, config):
    g = real_build(phase, config)
    for layer in g.layers:      # a loader that also flips the output order: shapes fit, values do not
        inner = fb.engine_loaders(D06, 4)["down"]
        from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader
        set_weight_loader(layer.mlp.down_weight, SafetensorsWeightLoader(
            transform=lambda s, rank, inner=inner: inner.load(s, rank).flip(1)))
    return g


h.build_graph = odd_build
for layer, lay0 in zip(odd_build("decode", None).layers, g0.layers):
    pass
# the live graph must hold what the odd loader would have loaded
odd = odd_build("decode", None)
odd.tp_group = g0.tp_group
odd.load_weights(ckpt06b, torch.device("cpu"))
h.model_graphs = {"decode": odd}
tap.lines.clear()
h.update_weights_in_place(ad1b)
st = h._experiment_adapter_state
env()
tw = THost(D06, 4, 1, 2, ckpt06b)
tw.build_graph = odd_build
tw.model_graphs = {"decode": odd_build("decode", None)}
tw.model_graphs["decode"].load_weights(ckpt06b, torch.device("cpu"))
tw.update_weights_in_place(ad1b)
dn = "layers.0.mlp.down_weight"
check("a loader with an unknown layout fails the probe and its parameters take the host path with exact values",
      dn in st["base"] and dn not in st["base_dev"] and torch.equal(h.live()[dn], tw.live()[dn]) and
      any("layout check failed" in ln for ln in tap.lines))

print(f"{n_pass} passed, {len(fails)} failed")
sys.exit(1 if fails else 0)
