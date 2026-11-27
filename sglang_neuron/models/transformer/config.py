# SPDX-License-Identifier: Apache-2.0
"""The shape of a dense decoder-only transformer, and the switches the models built on it differ by."""

import os
from dataclasses import dataclass

import torch

from sglang_neuron import env_registry

# the graphs compute in bfloat16
DTYPE = torch.bfloat16


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    # RoPE rotates channel pairs (j, j + head_dim / 2) for j < rotary_dim / 2 and passes the others through; a
    # checkpoint that rotates its leading channels as interleaved pairs is permuted into this layout at load
    rotary_dim: int
    tie_word_embeddings: bool = False
    qkv_bias: bool = False
    # a per-head RMSNorm of Q and K before RoPE
    qk_norm: bool = False
    # RMSNorms of the attention and MLP outputs before each residual add
    post_attention_norm: bool = False
    post_mlp_norm: bool = False

    def __post_init__(self):
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(f"{self.num_attention_heads} attention heads do not group over "
                             f"{self.num_key_value_heads} KV heads")
        if not 0 < self.rotary_dim <= self.head_dim or self.rotary_dim % 2:
            raise ValueError(f"a rotary dimension of {self.rotary_dim} does not fit a head of {self.head_dim}")


LM_HEAD_PRETRANSPOSED_ENV = "SGLANG_NEURON_LM_HEAD_PRETRANSPOSED"


def lm_head_pretransposed(environ=os.environ) -> bool:
    """SGLANG_NEURON_LM_HEAD_PRETRANSPOSED: 1 gives the LM head its own [H, V / tp] weight, the layout its matmul reads,
    filled from the checkpoint tensor at load; a tied model then holds the embedding's [V / tp, H] shard and this one
    (one extra vocabulary shard per rank). Off (0, the default) the LM head reads the [V / tp, H] shard transposed, and
    a tied model shares the embedding's device copy. Any other value is refused."""
    return env_registry.zero_or_one(LM_HEAD_PRETRANSPOSED_ENV, environ)


def check_bf16_launch(hf_config, architecture: str) -> None:
    """Refuse a launch whose model dtype (--dtype, or the checkpoint's with auto) is not the graphs' bfloat16."""
    from sglang.srt.runtime_context import get_model

    requested = str(get_model().dtype)
    dtype = (getattr(hf_config, "dtype", None) or getattr(hf_config, "torch_dtype", None)) if requested == "auto" \
        else requested
    if str(dtype).removeprefix("torch.") not in ("bfloat16", "bf16"):
        raise ValueError(f"{architecture}: the graphs compute in bfloat16, not {dtype}; launch with --dtype bfloat16")


def rope_theta_of(hf_config, architecture: str) -> float:
    """The checkpoint's RoPE base, refusing a scaled RoPE: the graphs rotate by plain frequencies."""
    params = getattr(hf_config, "rope_parameters", None) or {}
    scaling = getattr(hf_config, "rope_scaling", None) or {}
    kind = params.get("rope_type") or scaling.get("rope_type") or scaling.get("type") or "default"
    if kind != "default":
        raise ValueError(f"{architecture}: a {kind!r} RoPE scaling is not served; the graphs rotate by plain "
                         "frequencies")
    return float(params.get("rope_theta", getattr(hf_config, "rope_theta", 10000.0)))


def check_no_expert_parallel(architecture: str) -> None:
    """Refuse --ep-size above 1 for a model whose graphs split no experts over EP: it would be silently ignored."""
    from sglang_neuron import sglang_tree

    ep = int(getattr(sglang_tree.parallel_config(), "ep_size", None) or 1)
    if ep != 1:
        raise ValueError(f"--ep-size {ep}: {architecture}'s graphs do not split experts over EP, so the flag would be "
                         "ignored; launch without --ep-size")
