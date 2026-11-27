# SPDX-License-Identifier: Apache-2.0
"""SGLANG_NEURON_LM_HEAD_PRETRANSPOSED: the LM head's logits with the switch on equal its logits with it off, tied and
untied, and the switch's two layouts hold the same checkpoint tensor.

Runs on the CPU with torch and safetensors only. The NKI attention kernels and SGLang's parallel state do not exist
off the device, so they are replaced by stand-ins before the model module is imported; the test builds the smallest
dense transformer (one layer, tensor parallel size 1), loads a small checkpoint through ``load_weights`` and runs the
model's tail (final norm, LM head), which is the part the switch changes. Run it from the repository root with
``python -m unittest tests.cpu.test_lm_head_pretransposed`` or with pytest."""

import importlib
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
import torch.distributed as dist
from safetensors.torch import save_file

SWITCH = "SGLANG_NEURON_LM_HEAD_PRETRANSPOSED"
VOCAB, HIDDEN, INTER, HEADS, KV_HEADS, HEAD_DIM = 64, 16, 32, 2, 1, 8


class FakeGroup:
    """A tensor parallel group of one rank, as SGLang's GroupCoordinator and the plugin's groups are used here."""

    world_size = 1
    rank_in_group = 0
    device_group = None


def _install_stand_ins() -> list:
    """Modules the model imports that need SGLang or the device; returns the names put into ``sys.modules``."""
    added = []

    def put(name, **attrs):
        if name in sys.modules:
            return
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module
        added.append(name)

    for name in ("sglang", "sglang.srt", "sglang.srt.distributed"):
        put(name)
    put("sglang.srt.distributed.parallel_state", get_attn_tp_group=lambda: FakeGroup(), GroupCoordinator=FakeGroup)
    put("sglang_neuron.ops.attention.attention_decode_paged", attention_decode_paged=None, paged_decode_inputs=None)
    put("sglang_neuron.ops.attention.qkv_paged", qkv_paged=None)
    return added


def _checkpoint(directory: str, tied: bool, seed: int) -> dict:
    """A safetensors checkpoint of the one-layer model, random in bfloat16; returns its tensors by key."""
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return (torch.randn(*shape, generator=gen) * 0.5).to(torch.bfloat16)

    layer = "model.layers.0"
    tensors = {
        "model.embed_tokens.weight": rand(VOCAB, HIDDEN),
        "model.norm.weight": rand(HIDDEN),
        f"{layer}.self_attn.q_proj.weight": rand(HEADS * HEAD_DIM, HIDDEN),
        f"{layer}.self_attn.k_proj.weight": rand(KV_HEADS * HEAD_DIM, HIDDEN),
        f"{layer}.self_attn.v_proj.weight": rand(KV_HEADS * HEAD_DIM, HIDDEN),
        f"{layer}.self_attn.o_proj.weight": rand(HIDDEN, HEADS * HEAD_DIM),
        f"{layer}.self_attn.q_norm.weight": rand(HEAD_DIM),
        f"{layer}.self_attn.k_norm.weight": rand(HEAD_DIM),
        f"{layer}.input_layernorm.weight": rand(HIDDEN),
        f"{layer}.post_attention_layernorm.weight": rand(HIDDEN),
        f"{layer}.mlp.gate_proj.weight": rand(INTER, HIDDEN),
        f"{layer}.mlp.up_proj.weight": rand(INTER, HIDDEN),
        f"{layer}.mlp.down_proj.weight": rand(HIDDEN, INTER),
    }
    if not tied:
        tensors["lm_head.weight"] = rand(VOCAB, HIDDEN)
    save_file(tensors, os.path.join(directory, "model.safetensors"))
    return tensors


class LmHeadPretransposedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._added = _install_stand_ins()
        cls.groups = importlib.import_module("sglang_neuron.parallel.groups")
        cls.model_module = importlib.import_module("sglang_neuron.models.transformer.model")
        cls.config_module = importlib.import_module("sglang_neuron.models.transformer.config")
        # the checkpoint loader shares which files are cached through the default store
        cls._store_dir = tempfile.TemporaryDirectory()
        store = dist.FileStore(os.path.join(cls._store_dir.name, "store"), 1)
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        cls._store_dir.cleanup()
        for name in cls._added:
            sys.modules.pop(name, None)

    def build(self, tied: bool, switch: str | None, directory: str):
        """The model with the switch ``switch`` (None: unset), loaded from the checkpoint in ``directory``."""
        config = self.config_module.TransformerConfig(
            vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTER, num_hidden_layers=1,
            num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM, rms_norm_eps=1e-6,
            rope_theta=10000.0, rotary_dim=HEAD_DIM, tie_word_embeddings=tied, qk_norm=True)
        environ = {k: v for k, v in os.environ.items() if k != SWITCH}
        if switch is not None:
            environ[SWITCH] = switch
        with mock.patch.dict(os.environ, environ, clear=True), \
                mock.patch.object(self.groups, "get_neuron_attn_tp_group", lambda: FakeGroup()):
            model = self.model_module.TransformerForCausalLM(config, "decode")
        model.load_weights(directory, torch.device("cpu"))
        return model

    def tail(self, model, hidden):
        """The model's forward after its layers: the final norm and the LM head."""
        return model.lm_head(model.norm(hidden))

    def check_case(self, tied: bool):
        with tempfile.TemporaryDirectory() as directory:
            tensors = _checkpoint(directory, tied, seed=7)
            head_key = "model.embed_tokens.weight" if tied else "lm_head.weight"
            off = self.build(tied, None, directory)
            on = self.build(tied, "1", directory)

            # layouts: off reads the [V, H] shard, on its own [H, V] copy of the same checkpoint tensor
            self.assertFalse(off.lm_head_pretransposed)
            self.assertTrue(on.lm_head_pretransposed)
            self.assertEqual(tuple(off.lm_head.weight.shape), (VOCAB, HIDDEN))
            self.assertEqual(tuple(on.lm_head.weight.shape), (HIDDEN, VOCAB))
            self.assertTrue(on.lm_head.weight.is_contiguous())
            self.assertTrue(torch.equal(on.lm_head.weight, tensors[head_key].t()))
            self.assertTrue(torch.equal(off.lm_head.weight, tensors[head_key]))
            # the embedding is untouched by the switch
            self.assertTrue(torch.equal(on.embed_tokens.weight, tensors["model.embed_tokens.weight"]))
            self.assertTrue(torch.equal(off.embed_tokens.weight, on.embed_tokens.weight))
            # tied off shares one device copy; tied on holds a second one; untied holds one LM head weight either way
            self.assertEqual(off.lm_head.weight is off.embed_tokens.weight, tied)
            self.assertIsNot(on.lm_head.weight, on.embed_tokens.weight)
            extra = on.lm_head.weight.numel() * on.lm_head.weight.element_size() if tied else 0
            self.assertEqual(sum(p.numel() * p.element_size() for p in on.parameters())
                             - sum(p.numel() * p.element_size() for p in off.parameters()), extra)
            if tied:
                self.assertEqual(extra, VOCAB * HIDDEN * 2)

            # logits
            hidden = (torch.randn(5, HIDDEN, generator=torch.Generator().manual_seed(11)) * 2).to(torch.bfloat16)
            logits_off, logits_on = self.tail(off, hidden), self.tail(on, hidden)
            self.assertEqual(tuple(logits_off.shape), (5, VOCAB))
            self.assertEqual(logits_off.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(logits_off, logits_on), f"max diff {(logits_off - logits_on).abs().max()}")
            # and both are the plain product with the checkpoint's table
            normed = off.norm(hidden)
            expected = torch.nn.functional.linear(normed, tensors[head_key])
            self.assertTrue(torch.equal(logits_off, expected))

    def test_tied_logits_equal(self):
        self.check_case(tied=True)

    def test_untied_logits_equal(self):
        self.check_case(tied=False)

    def test_mappings_name_the_same_checkpoint_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            _checkpoint(directory, True, seed=3)
            tied_off = self.build(True, "0", directory).hf_weight_mappings()
            tied_on = self.build(True, "1", directory).hf_weight_mappings()
        self.assertNotIn("lm_head.weight", tied_off)
        self.assertEqual(tied_on["lm_head.weight"], tied_on["embed_tokens.weight"])

    def test_default_layout_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            _checkpoint(directory, False, seed=5)
            self.assertEqual(self.build(False, None, directory).lm_head.weight_layout, "out_in")
            self.assertEqual(self.build(False, "1", directory).lm_head.weight_layout, "in_out")

    def test_vocab_shards_match_rows_loader(self):
        """At TP 2 and 4, each rank's [H, V / tp] shard is the transpose of the rows loader's [V / tp, H] shard, the
        shards side by side are the whole table transposed, and a checkpoint table shorter than the padded vocabulary
        gives the same zero padding in both layouts."""
        from safetensors import safe_open

        weights = importlib.import_module("sglang_neuron.models.transformer.weights")
        for stored_rows in (VOCAB, VOCAB - 3):
            table = (torch.randn(stored_rows, HIDDEN, generator=torch.Generator().manual_seed(stored_rows))
                     ).to(torch.bfloat16)
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "t.safetensors")
                save_file({"w": table}, path)
                with safe_open(path, framework="pt") as f:
                    for tp in (2, 4):
                        rows = VOCAB // tp
                        by_rows = weights.vocab_rows_loader(rows, HIDDEN)
                        by_columns = weights.vocab_columns_loader(rows, HIDDEN)
                        shards = []
                        for rank in range(tp):
                            r = by_rows.load([f.get_slice("w")], rank)
                            c = by_columns.load([f.get_slice("w")], rank)
                            self.assertEqual(tuple(c.shape), (HIDDEN, rows))
                            self.assertTrue(c.is_contiguous())
                            self.assertTrue(torch.equal(c, r.t()))
                            shards.append(c)
                        whole = torch.cat(shards, dim=1)
                        self.assertTrue(torch.equal(whole[:, :stored_rows], table.t()))
                        self.assertTrue(torch.equal(whole[:, stored_rows:], torch.zeros_like(whole[:, stored_rows:])))

    def test_switch_value_is_checked_and_registered(self):
        envs = self.config_module
        registry = importlib.import_module("sglang_neuron.env_registry")
        self.assertFalse(envs.lm_head_pretransposed({}))
        self.assertFalse(envs.lm_head_pretransposed({SWITCH: "0"}))
        self.assertTrue(envs.lm_head_pretransposed({SWITCH: "1"}))
        with self.assertRaises(ValueError):
            envs.lm_head_pretransposed({SWITCH: "true"})
        registry.refuse_unknown({SWITCH: "1"})  # a name the launch accepts


if __name__ == "__main__":
    unittest.main()
