"""The LoRA shard-merge kernel under its test name. One source: sglang_neuron/models/rlkit_lora_merge.py, loaded by
path so that the tests need neither sglang nor the sglang_neuron package import."""
import importlib.util
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "sglang_neuron" / "models" / "rlkit_lora_merge.py"
_spec = importlib.util.spec_from_file_location("rlkit_lora_merge", _SRC)
_mod = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("rlkit_lora_merge", _mod)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
