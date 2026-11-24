# SPDX-License-Identifier: Apache-2.0
"""EXPERIMENT (2026-11-23), not upstream. NOT YET RUN ON HARDWARE.

The kernel mode of the adapter update in models/host.py (``RLKIT_DEVICE_WRITE=kernel``): for every live parameter an
adapter touches, only the LoRA factors of this rank's shard go to the device, and the NKI kernel
``rlkit_lora_merge.lora_merge`` writes ``bf16(float32(base) + a_stack.T @ bt)`` from a copy of the base shard that
stays on the device. The serving graphs are not touched.

This module does not import the engine at import time, so it can be tested on a CPU with the engine's loaders
(tests/rlkit/test_host_kernel_path.py). What it takes from the engine arrives as arguments: the live parameters,
their loaders and their checkpoint-key mappings.

Per-rank factors are built through the parameter's own loader, with no slicing arithmetic written here:
  column layout (the loader cuts and joins output rows: qkv, gate_up): a_stack = the A_i stacked on K; each
    scale * B_i is zero-padded to [out_i, K] and given to the loader as if it were checkpoint tensor i, which returns
    the block-diagonal bt [K, OUT_shard];
  row layout (the loader cuts input columns: o, down): the loader cuts A; bt = (scale * B).T.
Which layout a parameter has is decided by shapes and then CHECKED once per kind of parameter: integer-valued probe
factors of the same shapes must give ``a_stack.T @ bt == loader(scale * B @ A)`` bit for bit. A kind that fails the
check, or a shard the kernel's serving guard does not cover, is returned as refused and the caller merges it on the
host.

RLKIT_KERNEL_IMPL: auto (default: the NKI kernel on a Neuron device, the torch arithmetic of the same formula on a
  CPU), nki, torch, sim (the NKI CPU simulator; tests only, slow).
RLKIT_KERNEL_FACTORS: fp32 (default) or split (bf16 hi/lo factors, K' = 3K).
RLKIT_KERNEL_CHECK: first (default: read the first kernel result back and compare it with the host arithmetic),
  all, none.
"""
import logging
import os
import re
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

# share of entries of a read-back kernel result allowed to differ from the host float32 arithmetic (the two may
# sum the K products in a different order); above it the kernel mode is refused
CHECK_MAX_SHARE = 1e-3
_NUMBER = re.compile(r"\.\d+(?=\.)")


def _lm():
    """The kernel module (it imports nki, so it is loaded on first use, not at import)."""
    try:
        from . import rlkit_lora_merge as lm
    except ImportError:
        import rlkit_lora_merge as lm
    return lm


class _Block:
    """A whole tensor behind the checkpoint-slice interface the loaders use (get_shape, indexing)."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def get_shape(self) -> list:
        return list(self.tensor.shape)

    def __getitem__(self, key):
        return self.tensor[key]


class _Like:
    """Shape and dtype of an array, for the guards of rlkit_lora_merge (they read ndim, shape and dtype only)."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype, self.ndim = tuple(shape), np.dtype(dtype), len(shape)


def build_factors(loader, pairs: list, scale: float, rank: int, live_shape: tuple) -> tuple:
    """(a_stack [K, IN], bt [K, OUT], layout) in float32 for one live parameter of shape [IN, OUT] on this rank.
    ``pairs`` holds one (B_i [out_i, r], A_i [r, in]) per checkpoint key of the parameter, in mapping order.
    Raises ValueError when neither layout fits the shapes."""
    if len(live_shape) != 2:
        raise ValueError(f"live parameter has {len(live_shape)} dimensions")
    in_dim, out_dim = int(live_shape[0]), int(live_shape[1])
    ranks = {int(a.shape[0]) for _, a in pairs}
    ins = {int(a.shape[1]) for _, a in pairs}
    if len(ranks) != 1 or len(ins) != 1:
        raise ValueError(f"the adapter pairs of one parameter differ in rank or input size: {ranks}, {ins}")
    (r,), (in_full,) = ranks, ins
    for b, a in pairs:
        if b.dim() != 2 or a.dim() != 2 or int(b.shape[1]) != r:
            raise ValueError(f"B {tuple(b.shape)} does not fit A {tuple(a.shape)}")
    n = len(pairs)
    k = n * r
    tried = []
    if in_full == in_dim:
        blocks = []
        for i, (b, _) in enumerate(pairs):
            z = torch.zeros(int(b.shape[0]), k, dtype=torch.float32)
            z[:, i * r:(i + 1) * r] = scale * b.to(torch.float32)
            blocks.append(_Block(z))
        try:
            bt = loader.load(blocks, rank)
        except Exception as e:  # noqa: BLE001
            tried.append(f"column: loader raised {type(e).__name__}: {str(e)[:80]}")
        else:
            if tuple(bt.shape) == (k, out_dim):
                a_stack = torch.cat([a.to(torch.float32) for _, a in pairs], dim=0).contiguous()
                return a_stack, bt.to(torch.float32).contiguous(), "column"
            tried.append(f"column: loader gave {tuple(bt.shape)}, wanted {(k, out_dim)}")
    if n == 1:
        (b, a), = pairs
        try:
            a_cut = loader.load([_Block(a.to(torch.float32))], rank)
        except Exception as e:  # noqa: BLE001
            tried.append(f"row: loader raised {type(e).__name__}: {str(e)[:80]}")
        else:
            if tuple(a_cut.shape) == (in_dim, r) and int(b.shape[0]) == out_dim:
                return (a_cut.t().contiguous(), (scale * b.to(torch.float32)).t().contiguous(), "row")
            tried.append(f"row: loader gave {tuple(a_cut.shape)}, wanted {(in_dim, r)}; B rows {int(b.shape[0])}, "
                         f"OUT {out_dim}")
    raise ValueError("no factor layout fits: " + "; ".join(tried or ["shapes fit neither layout"]))


def probe_layout(loader, pairs: list, rank: int, live_shape: tuple, seed: int = 0):
    """None if the factor layout is right for this loader, else a string saying what failed. Integer-valued probe
    factors with the shapes of ``pairs`` (not their values: a new adapter has B = 0, which would check nothing), a
    power-of-two scale, and the reference ``loader(scale * B @ A)`` formed in full; equality must be exact."""
    g = torch.Generator().manual_seed(1234 + seed)
    probe = [(torch.randint(-3, 4, tuple(b.shape), generator=g).to(torch.float32),
              torch.randint(-3, 4, tuple(a.shape), generator=g).to(torch.float32)) for b, a in pairs]
    scale = 2.0
    try:
        a_stack, bt, _ = build_factors(loader, probe, scale, rank, live_shape)
        ref = loader.load([_Block(scale * (b @ a)) for b, a in probe], rank)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:200]}"
    got = a_stack.t() @ bt
    if tuple(ref.shape) != tuple(got.shape):
        return f"probe shapes differ: factors {tuple(got.shape)}, loader {tuple(ref.shape)}"
    if not bool(ref.abs().max() > 0):
        return "the probe delta is all zero on this rank; nothing was checked"
    if not torch.equal(got, ref.to(torch.float32)):
        bad = int((got != ref).sum())
        return f"probe factors differ from the loader's delta in {bad} of {got.numel()} entries"
    return None


# ---- how the merge itself is run ---------------------------------------------------------------------------------

def _torch_merge(base, a_stack, bt):
    """The kernel's formula with torch operations, on whatever device the tensors are on."""
    return (base.float() + a_stack.float().t() @ bt.float()).to(base.dtype)


class _SimRunner:
    """The NKI kernel in the NKI CPU simulator (tests only)."""

    how = "sim"

    def __call__(self, base, a_stack, bt):
        import nki

        lm = _lm()
        b_np = base.detach().float().cpu().numpy().astype(lm.BF16)
        fac = lm.BF16 if a_stack.dtype == torch.bfloat16 else np.float32
        a_np = np.ascontiguousarray(a_stack.detach().float().cpu().numpy().astype(fac))
        bt_np = np.ascontiguousarray(bt.detach().float().cpu().numpy().astype(fac))
        out = np.asarray(nki.simulate(lm.lora_merge)(b_np, a_np, bt_np)).astype(np.float32)
        return torch.from_numpy(out).to(base.dtype).to(base.device)


class _TorchRunner:
    how = "torch"

    def __call__(self, base, a_stack, bt):
        return _torch_merge(base, a_stack, bt)


def _own_function(call):
    """A new function with a code object of its own that calls ``call``."""
    ns = {"call": call}
    exec("def helper(x, y, z):\n    return call(x, y, z)\n", ns)  # noqa: S102
    return ns["helper"]


class _NkiRunner:
    """The NKI kernel on the device. Whether it can be called outside a serving graph is not known before step H1 of
    the plan: an eager call is tried first, then one small compiled helper per shard shape. Each helper is defined
    by its own exec, so it has a code object of its own and a second shape is a first compile of a new function, not
    a recompile of the first (nested defs of one source line would share a code object; checked on CPU with the
    eager backend only). Each new shape still costs a Neuron compile during the first update that meets it."""

    def __init__(self):
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        self.call = wrap_nki(_lm().lora_merge)[get_logical_nc_config()]
        self.how = None
        self.eager_error = None
        self.compiled = {}

    def __call__(self, base, a_stack, bt):
        if self.how in (None, "eager"):
            try:
                out = self.call(base, a_stack, bt)
                self.how = "eager"
                return out
            except Exception as e:  # noqa: BLE001
                if self.how == "eager":
                    raise
                self.eager_error = f"{type(e).__name__}: {str(e)[:300]}"
                logger.warning("[neuron] EXPERIMENT lora_merge eager call failed (%s); trying a compiled helper",
                               self.eager_error)
        key = (tuple(base.shape), int(a_stack.shape[0]), str(a_stack.dtype))
        fn = self.compiled.get(key)
        if fn is None:
            fn = self.compiled[key] = torch.compile(_own_function(self.call), backend="neuron", fullgraph=True,
                                                    dynamic=False)
        out = fn(base, a_stack, bt)
        self.how = "compiled"
        return out


def make_runner(device_type: str):
    impl = os.environ.get("RLKIT_KERNEL_IMPL", "auto")
    if impl == "auto":
        impl = "torch" if device_type == "cpu" else "nki"
    if impl == "nki":
        return _NkiRunner()
    if impl == "sim":
        return _SimRunner()
    if impl == "torch":
        return _TorchRunner()
    raise ValueError(f"RLKIT_KERNEL_IMPL={impl!r} is not auto, nki, torch or sim")


# ---- one adapter update ------------------------------------------------------------------------------------------

def apply_adapter(*, live: dict, mappings: dict, loaders: dict, a_of: dict, b_of: dict, scale: float, rank: int,
                  state: dict, sync=None) -> dict:
    """One pass over the adapted parameters in kernel mode. Writes, in place, every parameter the kernel takes, and
    puts back the base of a parameter adapted before and absent from this adapter when its base copy is on the
    device. Returns a report; ``report["host"]`` is the list of parameter names left for the caller's host path
    (refused by a guard or the layout check, or with their base copy on the host only).

    ``state`` keeps, across updates: "base" (host float32 copies, the caller's), "base_dev" (device copies in the
    parameter's dtype), "kernel_layout" (checked kinds), "kernel_runner", "kernel_checked".
    Raises on a failure of the kernel call itself; the caller then takes the host path for the whole update."""
    lm = _lm()
    mode = os.environ.get("RLKIT_KERNEL_FACTORS", "fp32")
    if mode not in lm.K_PRODUCTION:
        raise ValueError(f"RLKIT_KERNEL_FACTORS={mode!r} is not one of {sorted(lm.K_PRODUCTION)}")
    check = os.environ.get("RLKIT_KERNEL_CHECK", "first")
    base, base_dev = state["base"], state.setdefault("base_dev", {})
    layout_ok = state.setdefault("kernel_layout", {})
    sync = sync or (lambda: None)
    rep = {"kernel": 0, "restored": 0, "host": [], "refused": {}, "bytes": 0, "t_factors": 0.0, "t_probe": 0.0,
           "t_upload": 0.0, "t_kernel": 0.0, "t_copy": 0.0, "t_check": 0.0, "check": [], "mode": mode}

    def refuse(name, why):
        rep["host"].append(name)
        rep["refused"][why] = rep["refused"].get(why, 0) + 1

    with torch.no_grad():
        for name, keys in mappings.items():
            keys = [keys] if isinstance(keys, str) else list(keys)
            hit = [k in a_of for k in keys]
            if not any(hit):
                if name in base_dev:                # adapted before, not in this adapter: back to the base
                    t = time.time()
                    live[name].data.copy_(base_dev[name])
                    rep["t_copy"] += time.time() - t
                    rep["restored"] += 1
                    rep["bytes"] += live[name].numel() * live[name].element_size()
                elif name in base:
                    rep["host"].append(name)
                continue
            if not all(hit):
                raise NotImplementedError(f"adapter covers only part of {name}: {keys}")
            p = live[name]
            if p.dtype != torch.bfloat16 or p.dim() != 2:
                refuse(name, f"parameter is {p.dtype} with {p.dim()} dimensions")
                continue
            pairs = [(b_of[k], a_of[k]) for k in keys]
            # the parameter's path with numbers (layer index) replaced: parameters of one kind share a loader
            kind = (_NUMBER.sub(".N.", "." + name + "."), tuple(p.shape), tuple(tuple(b.shape) for b, _ in pairs),
                    tuple(tuple(a.shape) for _, a in pairs), tuple(k.rsplit(".", 2)[-2] for k in keys))
            if kind not in layout_ok:
                t = time.time()
                layout_ok[kind] = probe_layout(loaders[name], pairs, rank, tuple(p.shape), seed=len(layout_ok))
                rep["t_probe"] += time.time() - t
                if layout_ok[kind] is not None:
                    logger.warning("[neuron] EXPERIMENT kernel adapter path: layout check failed for %s (%s); such "
                                   "parameters are merged on the host", name, layout_ok[kind])
            if layout_ok[kind] is not None:
                refuse(name, "layout check: " + layout_ok[kind][:80])
                continue
            t = time.time()
            a_stack, bt, _ = build_factors(loaders[name], pairs, scale, rank, tuple(p.shape))
            a_np, bt_np = lm.prepare_factors(a_stack.numpy(), bt.numpy(), mode)
            rep["t_factors"] += time.time() - t
            why = lm.production_guard(_Like(p.shape, lm.BF16), a_np, bt_np, mode)
            if why is not None:
                refuse(name, why)
                continue
            t = time.time()
            if name not in base_dev:                # before the first kernel write to this parameter
                base_dev[name] = base[name].to(p.dtype).to(p.device) if name in base else p.detach().clone()
            fdtype = torch.float32 if mode == "fp32" else torch.bfloat16
            a_dev = torch.from_numpy(a_np.astype(np.float32)).to(fdtype).to(p.device)
            bt_dev = torch.from_numpy(bt_np.astype(np.float32)).to(fdtype).to(p.device)
            sync()
            rep["t_upload"] += time.time() - t
            runner = state.get("kernel_runner")
            if runner is None:
                runner = state["kernel_runner"] = make_runner(p.device.type)
            t = time.time()
            out = runner(base_dev[name], a_dev, bt_dev)
            sync()
            rep["t_kernel"] += time.time() - t
            if tuple(out.shape) != tuple(p.shape) or out.dtype != p.dtype:
                raise RuntimeError(f"kernel result for {name} is {tuple(out.shape)} {out.dtype}, "
                                   f"live {tuple(p.shape)} {p.dtype}")
            t = time.time()
            p.data.copy_(out)
            sync()
            rep["t_copy"] += time.time() - t
            rep["kernel"] += 1
            rep["bytes"] += p.numel() * p.element_size()
            # Read-back check of the LIVE parameter after the write (not of the kernel's output: on 2026-11-20 a
            # device tensor copied into a parameter was the form whose served output did not match). "first" checks
            # the first write of every (way of calling, shard shape, K, factor dtype), so each compiled helper and
            # each kind of shard is checked once per server. A failure raises; the caller redoes the whole update
            # on the host from the base copies.
            ckey = (getattr(runner, "how", None), tuple(p.shape), int(a_dev.shape[0]), str(a_dev.dtype))
            checked = state.setdefault("kernel_checked", set())
            if check == "all" or (check == "first" and ckey not in checked):
                t = time.time()
                want = _torch_merge(base_dev[name].detach().cpu(), torch.from_numpy(a_np.astype(np.float32)),
                                    torch.from_numpy(bt_np.astype(np.float32)))
                diff = (p.detach().cpu().float() - want.float()).abs()
                share, worst = float((diff > 0).float().mean()), float(diff.max())
                rep["check"].append((name, share, worst))
                rep["t_check"] += time.time() - t
                if not share <= CHECK_MAX_SHARE:        # also catches NaN
                    raise RuntimeError(f"live {name} after the kernel write differs from the host arithmetic in "
                                       f"{100 * share:.3f}% of entries (largest {worst:.3e})")
                checked.add(ckey)
    sync()
    rep["how"] = getattr(state.get("kernel_runner"), "how", None)
    rep["base_dev_bytes"] = sum(t.numel() * t.element_size() for t in base_dev.values())
    return rep
