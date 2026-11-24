"""Extra simulator tests asked for by the review of 2026-11-23.

1. K above one matmul chunk (K = 144 and K = 200, two accumulating matmuls) on a shape with partial
   tiles on both axes, under LNC=1 and LNC=2, against the numpy reference (bit for bit) and against
   an independent float64 sum (bound on the difference).
2. production_guard and prepare_factors: what the serving path accepts and refuses.

Run: ~/nki_sim_venv/bin/python test_k_chunks.py      (exit code 1 on any failure)
"""
import os
import sys

import ml_dtypes
import numpy as np

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import nki  # noqa: E402

from lora_merge import (BF16, lora_merge, lora_merge_reference, prepare_factors,  # noqa: E402
                        production_guard, shape_guard)

fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAILED ") + name + (" | " + detail if detail else ""))
    if not ok:
        fails.append(name)


def run(base, a, b, lnc):
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(lnc)
    return np.asarray(nki.simulate(lora_merge)(base, a, b))


rng = np.random.default_rng(11)
for (in_dim, out_dim) in ((200, 700), (384, 1024)):
    for k in (144, 200):
        base = (rng.standard_normal((in_dim, out_dim)) * 0.02).astype(np.float32).astype(BF16)
        a = (rng.standard_normal((k, in_dim)) * 0.05).astype(np.float32).astype(BF16)
        b = (rng.standard_normal((k, out_dim)) * 0.05).astype(np.float32).astype(BF16)
        assert shape_guard(base, a, b) is None
        # The kernel sums K in chunks of 128 (one matmul each, accumulated in PSUM). A float32 sum
        # depends on its order, so the bit-level reference must use the same chunks; the unchunked
        # numpy reference differs from it on a few entries and is reported, not asserted.
        prod = np.zeros((in_dim, out_dim), dtype=np.float32)
        for k0 in range(0, k, 128):
            prod = prod + a[k0:k0 + 128].astype(np.float32).T @ b[k0:k0 + 128].astype(np.float32)
        ref = (base.astype(np.float32) + prod).astype(BF16)
        unchunked = lora_merge_reference(base, a, b)
        exact = base.astype(np.float64) + a.astype(np.float64).T @ b.astype(np.float64)
        first_chunk_only = lora_merge_reference(base, a[:128], b[:128])
        outs = {}
        for lnc in (1, 2):
            got = run(base, a, b, lnc)
            outs[lnc] = got
            tag = f"K={k} {in_dim}x{out_dim} LNC={lnc}"
            n_ref = int((got.astype(np.float32) != ref.astype(np.float32)).sum())
            n_un = int((got.astype(np.float32) != unchunked.astype(np.float32)).sum())
            check(tag + " equals chunk-ordered float32 reference", n_ref == 0,
                  f"{n_ref} of {got.size} differ; vs unchunked numpy sum {n_un} differ (order of the sum)")
            check(tag + " has no NaN", not np.isnan(got.astype(np.float32)).any())
            err = np.abs(got.astype(np.float64) - exact)
            # one bf16 rounding of a value near |exact| is at most 2^-8 relative; fp32 sums add far less
            bound = np.abs(exact) * 2.0 ** -8 + 1e-6
            n_out = int((err > bound).sum())
            check(tag + " within one-rounding bound of the float64 sum", n_out == 0, f"{n_out} outside")
            # the test must be able to see a lost second chunk
            n_vs_first = int((got.astype(np.float32) != first_chunk_only.astype(np.float32)).sum())
            check(tag + " differs from first-chunk-only result", n_vs_first > got.size // 2,
                  f"{n_vs_first} of {got.size} differ")
        same = int((outs[1].astype(np.float32) != outs[2].astype(np.float32)).sum())
        check(f"K={k} {in_dim}x{out_dim} LNC=1 equals LNC=2", same == 0, f"{same} differ")

# guards
base = np.zeros((256, 1024), dtype=BF16)
a32 = np.ones((48, 256), dtype=np.float32)
b32 = np.ones((48, 1024), dtype=np.float32)
check("production_guard accepts aligned fp32 K=48", production_guard(base, a32, b32, "fp32") is None)
check("production_guard refuses partial tiles",
      production_guard(np.zeros((200, 700), dtype=BF16), np.ones((16, 200), np.float32),
                       np.ones((16, 700), np.float32)) is not None)
check("production_guard refuses K=144 in fp32 mode",
      production_guard(base, np.ones((144, 256), np.float32), np.ones((144, 1024), np.float32)) is not None)
check("production_guard refuses bf16 factors in fp32 mode",
      production_guard(base, a32.astype(BF16), b32.astype(BF16), "fp32") is not None)
a_s, b_s = prepare_factors(a32, b32, "split")
check("prepare_factors split gives bf16 with K'=3K", a_s.dtype == BF16 and a_s.shape[0] == 144)
check("production_guard accepts split K'=144", production_guard(base, a_s, b_s, "split") is None)
try:
    prepare_factors(a32.astype(BF16), b32.astype(BF16))
    check("prepare_factors refuses bf16 input", False)
except TypeError:
    check("prepare_factors refuses bf16 input", True)

print(f"{len(fails)} failed")
sys.exit(1 if fails else 0)
