"""Per-rank LoRA factor builders for the sglang-neuron dense transformer (Qwen3), built through the engine's OWN
weight loaders (sglang_neuron/models/transformer/weights.py, engine tag v0.1.0).

For a live parameter P (a TP shard stored as [IN_shard, OUT_shard]) and a PEFT adapter with, for each HF tensor
W_i [out_i, in] that feeds P, a pair (B_i [out_i, r], A_i [r, in]) and a scale, we need per rank

    a_stack [K, IN_shard], bt [K, OUT_shard]   (float32)   with   a_stack.T @ bt == loader(scale * B_i @ A_i for all i)

Column-parallel fused parameters (qkv, gate_up): a_stack = the A_i stacked on dim 0 (K = c * r). For bt each B_i is
zero-padded into a [out_i, K] block (scale * B_i in columns i*r:(i+1)*r) and handed to the engine loader as if it were
the checkpoint tensor i. The loader row-slices, concatenates and transposes, and returns [K, OUT_shard]: the
block-diagonal bt, with no slicing arithmetic written here.

Row-parallel parameters (o, down): A goes through the engine loader (it cuts this rank's input columns and transposes
to [IN_shard, r]); a_stack is that, transposed. bt = (scale * B).T, which is NOT loader work (B is not sliced for a
row-parallel parameter; the only hand-written step is the transpose).

Nothing in this file cuts a tensor by rank. The engine source is imported from ENGINE_SRC (a checkout of
sglang-neuron v0.1.0); it is only read, never modified.
"""

import os
import sys
from dataclasses import dataclass

import torch

ENGINE_SRC = os.environ.get("SGLANG_NEURON_SRC", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ENGINE_SRC not in sys.path:
    sys.path.insert(0, ENGINE_SRC)

from sglang_neuron.models.transformer import weights as engine_weights  # noqa: E402

KINDS = ("qkv", "o", "gate_up", "down")
# number of HF tensors (and so K segments, each of rank r) that feed each kind
SEGMENTS = {"qkv": 3, "o": 1, "gate_up": 2, "down": 1}
# PEFT module names feeding each kind, in K-segment order (the engine's hf_weight_mappings order)
HF_MODULES = {"qkv": ("q_proj", "k_proj", "v_proj"), "o": ("o_proj",), "gate_up": ("gate_proj", "up_proj"),
              "down": ("down_proj",)}


@dataclass(frozen=True)
class DenseDims:
    hidden: int
    n_heads: int
    n_kv: int
    head_dim: int
    inter: int

    def check_tp(self, tp: int) -> None:
        """The refusals of model.py:69-73 and 163-164."""
        if self.n_heads % tp:
            raise ValueError(f"{self.n_heads} attention heads do not shard over {tp}")
        if self.n_kv % tp and tp % self.n_kv:
            raise ValueError(f"{self.n_kv} KV heads neither shard over nor replicate across {tp}")
        if self.inter % tp:
            raise ValueError(f"intermediate size {self.inter} does not shard over {tp}")

    def hf_out_dims(self, kind: str) -> tuple:
        """Output rows of each HF tensor that feeds ``kind`` (full, unsharded)."""
        d = self.head_dim
        return {"qkv": (self.n_heads * d, self.n_kv * d, self.n_kv * d), "o": (self.hidden,),
                "gate_up": (self.inter, self.inter), "down": (self.hidden,)}[kind]

    def hf_in_dim(self, kind: str) -> int:
        return {"qkv": self.hidden, "o": self.n_heads * self.head_dim, "gate_up": self.hidden,
                "down": self.inter}[kind]

    def live_shape(self, kind: str, tp: int) -> tuple:
        """[IN_shard, OUT_shard] of the live parameter on one rank (model.py:75-86 and 165-167)."""
        self.check_tp(tp)
        d = self.head_dim
        nh, nkv = self.n_heads // tp, max(1, self.n_kv // tp)
        return {"qkv": (self.hidden, (nh + 2 * nkv) * d), "o": (nh * d, self.hidden),
                "gate_up": (self.hidden, 2 * (self.inter // tp)), "down": (self.inter // tp, self.hidden)}[kind]


# Hugging Face config.json values, fetched 2026-11-23 (see RESULTS_factor_layout.md)
QWEN3_0_6B = DenseDims(hidden=1024, n_heads=16, n_kv=8, head_dim=128, inter=3072)
QWEN3_8B = DenseDims(hidden=4096, n_heads=32, n_kv=8, head_dim=128, inter=12288)


class TensorSlice:
    """Stands in for a safetensors PySafeSlice: the loaders only call ``get_shape()`` and index with slices."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def get_shape(self) -> list:
        return list(self.tensor.shape)

    def __getitem__(self, idx):
        return self.tensor[idx]


def engine_loaders(dims: DenseDims, tp: int) -> dict:
    """The loader of each kind, built with the arguments Attention.__init__ and MLP.__init__ pass
    (model.py:89-90, 93, 165-169). The argument arithmetic is mirrored here; the loaders are the engine's."""
    dims.check_tp(tp)
    d = dims.head_dim
    nh, nkv = dims.n_heads // tp, max(1, dims.n_kv // tp)
    rows = dims.inter // tp
    return {
        "qkv": engine_weights.fused_qkv_weight_loader(tp, nh * d, nkv * d, dims.n_kv),
        "o": engine_weights.row_parallel_loader(nh * d),
        "gate_up": engine_weights.gate_up_loader(rows, False),   # Qwen3: separate gate_proj and up_proj
        "down": engine_weights.row_parallel_loader(rows),
    }


def live_delta(kind: str, pairs: list, scale: float, rank: int, loader, dtype=torch.float32) -> torch.Tensor:
    """Reference: the loader applied to the full ``scale * B @ A`` of each HF tensor, as the host path does
    (host.py _experiment_apply_adapter: ``loaders[name].load([scale * (b @ a) ...], rank)``). Shape [IN_shard, OUT_shard]."""
    full = [TensorSlice(scale * (b.to(dtype) @ a.to(dtype))) for b, a in pairs]
    return loader.load(full, rank)


def build_factors(kind: str, pairs: list, scale: float, rank: int, tp: int, dims: DenseDims, loaders: dict = None,
                  lora_r: int = None) -> tuple:
    """(a_stack [K, IN_shard], bt [K, OUT_shard]) in float32 for this rank. ``pairs`` is one (B [out_i, r], A [r, in])
    per HF tensor of ``kind`` (see HF_MODULES); an entry may be None (that tensor has no adapter: its K segment is
    zero, which equals the base). ``scale`` (lora_alpha / r) is folded into bt."""
    loaders = loaders or engine_loaders(dims, tp)
    loader = loaders[kind]
    n = SEGMENTS[kind]
    if len(pairs) != n:
        raise ValueError(f"{kind} takes {n} (B, A) pairs, got {len(pairs)}")
    out_dims, in_dim = dims.hf_out_dims(kind), dims.hf_in_dim(kind)
    r = lora_r or next((p[1].shape[0] for p in pairs if p is not None), None)
    if r is None:
        raise ValueError("all pairs are None and lora_r is not given")
    for p, out in zip(pairs, out_dims):
        if p is not None and (tuple(p[0].shape) != (out, r) or tuple(p[1].shape) != (r, in_dim)):
            raise ValueError(f"{kind}: B {tuple(p[0].shape)} / A {tuple(p[1].shape)} do not fit [{out}, {r}] / [{r}, {in_dim}]")

    if kind in ("qkv", "gate_up"):
        k_total = n * r
        a_parts, blocks = [], []
        for i, (p, out) in enumerate(zip(pairs, out_dims)):
            z = torch.zeros(out, k_total, dtype=torch.float32)
            if p is None:
                a_parts.append(torch.zeros(r, in_dim, dtype=torch.float32))
            else:
                a_parts.append(p[1].to(torch.float32))
                z[:, i * r:(i + 1) * r] = scale * p[0].to(torch.float32)
            blocks.append(TensorSlice(z))
        a_stack = torch.cat(a_parts, dim=0).contiguous()
        bt = loader.load(blocks, rank)                       # [K, OUT_shard]: the engine's cut, concat, transpose
    else:
        (p,) = pairs
        if p is None:
            raise ValueError(f"{kind} has no adapter pair; there is nothing to build")
        a_stack = loader.load([TensorSlice(p[1].to(torch.float32))], rank).t().contiguous()   # [r, IN_shard]
        bt = (scale * p[0].to(torch.float32)).t().contiguous()                                # [r, H], hand-written
    expect_in, expect_out = dims.live_shape(kind, tp)
    assert a_stack.shape == (n * r if kind in ("qkv", "gate_up") else r, expect_in), (kind, a_stack.shape)
    assert bt.shape == (a_stack.shape[0], expect_out), (kind, bt.shape)
    return a_stack, bt
