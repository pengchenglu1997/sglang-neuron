"""Reviewer's end-to-end test: factor builders -> lora_merge kernel (NKI simulator) -> compare with the engine loader
applied to the FULL base tensors plus scale * B @ A in float64, rounded once to bf16, in the live [IN_shard, OUT_shard]
orientation.

Written by the reviewer, independent of the builders' tests. Run (WSL):
  ~/nki_sim_venv/bin/python test_end_to_end_review.py      (exit 1 if any check fails)

Reference path (does not use a_stack or bt at all):
  W_i  full HF tensors [out_i, in] in bf16 (random)
  ref = round_bf16_once( loader.load([W_i.double() + scale * B_i.double() @ A_i.double() ...], rank) )
Kernel path:
  base = loader.load([W_i (bf16)], rank)                      (live shard as the engine stores it)
  a_stack, bt = build_factors(kind, [(B_i, A_i)...], scale, rank, tp, dims)
  out = nki.simulate(lora_merge)(base, a_stack, bt)
"""
import os
import sys

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"

import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import nki  # noqa: E402
from factor_builders import QWEN3_0_6B, DenseDims, build_factors, engine_loaders  # noqa: E402
from factor_builders import TensorSlice  # noqa: E402
from lora_merge import lora_merge, lora_merge_reference, shape_guard, split_hi_lo  # noqa: E402

BF16 = ml_dtypes.bfloat16
FAIL = []


def round_once_bf16(x64):
    """float64 -> bf16, round to nearest even, a single rounding (no float32 step)."""
    x64 = np.asarray(x64, dtype=np.float64)
    m, e = np.frexp(x64)
    r = np.ldexp(np.rint(m * 256.0) / 256.0, e)
    return r.astype(np.float32).astype(BF16)   # r is a bf16 value already, cast is exact


def ulp(x_bf16):
    x = np.abs(x_bf16.astype(np.float64))
    _, e = np.frexp(x)
    u = np.ldexp(1.0, e - 8)
    u[x == 0] = np.ldexp(1.0, -133)
    return u


def bits(x):
    return np.ascontiguousarray(x).view(np.uint16)


def to_np_bf16(t):
    return t.to(torch.float32).numpy().astype(BF16)


def check(name, ok, msg):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}", flush=True)
    if not ok:
        FAIL.append(name)


def sim(base, a, bt, lnc=1):
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(lnc)
    try:
        return np.asarray(nki.simulate(lora_merge)(base, a, bt))
    finally:
        os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"


def run_case(label, dims, kind, tp, rank, r, scale, seed, lnc2=False):
    g = torch.Generator().manual_seed(seed)
    loaders = engine_loaders(dims, tp)
    loader = loaders[kind]
    in_full = dims.hf_in_dim(kind)
    W = [(torch.randn(o, in_full, generator=g) * 0.02).to(torch.bfloat16) for o in dims.hf_out_dims(kind)]
    pairs = [(torch.randn(o, r, generator=g) * 0.1, torch.randn(r, in_full, generator=g) * 0.1)
             for o in dims.hf_out_dims(kind)]

    # reference: loader on full (base + scale * B @ A) in float64, one rounding
    full64 = [TensorSlice(w.double() + scale * (b.double() @ a.double())) for w, (b, a) in zip(W, pairs)]
    s64 = loader.load(full64, rank).numpy()
    ref = round_once_bf16(s64)
    # magnitude sum for the error bound, same loader (it only slices / concatenates / transposes)
    abs64 = [TensorSlice((abs(scale) * b.double().abs()) @ a.double().abs()) for b, a in pairs]
    sabs = loader.load(abs64, rank).numpy()

    # kernel path
    base = to_np_bf16(loader.load([TensorSlice(w) for w in W], rank))
    a_t, bt_t = build_factors(kind, pairs, scale, rank, tp, dims, loaders)
    a = np.ascontiguousarray(a_t.numpy().astype(np.float32))
    bt = np.ascontiguousarray(bt_t.numpy().astype(np.float32))
    k = a.shape[0]
    in_dim, out_dim = base.shape
    assert base.shape == s64.shape == tuple(dims.live_shape(kind, tp)), (base.shape, s64.shape)
    g_err = shape_guard(base, a, bt)
    check(f"{label} shape_guard", g_err is None, f"base {base.shape} K={k} -> {g_err}")
    if g_err is not None:
        return
    n_in_t, n_out_t = -(-in_dim // 128), -(-out_dim // 512)
    print(f"  {label}: kind={kind} tp={tp} rank={rank} r={r} scale={scale} base[IN,OUT]={base.shape} K={k}"
          f" tiles IN={n_in_t} (last {in_dim - 128 * (n_in_t - 1)}) OUT={n_out_t} (last {out_dim - 512 * (n_out_t - 1)})")

    changed = int((bits(ref) != bits(base)).sum())
    print(f"  delta moves {changed}/{ref.size} entries of the base (the test is not vacuous if this is large)")

    def compare(tag, got, kfac, extra_rel):
        nd = int((bits(got) != bits(ref)).sum())
        dif = np.abs(got.astype(np.float64) - ref.astype(np.float64)) / ulp(ref)
        mx = float(dif.max())
        n_gt1 = int((dif > 1).sum())
        bound = (0.5 * ulp(ref) + (kfac + 2) * 2.0 ** -24 * (sabs + np.abs(base.astype(np.float64)))
                 + extra_rel * sabs)
        err = np.abs(got.astype(np.float64) - s64)
        nviol = int((err > bound * (1 + 1e-12)).sum())
        print(f"    {tag}: differ from float64-once {nd}/{got.size} ({nd / got.size:.2e}), max {mx:.0f} ulp,"
              f" {n_gt1} entries > 1 ulp, outside bound {nviol}, max err/bound {float((err / bound).max()):.2f}")
        return nd, mx, nviol

    # 1) native fp32 factors (what build_factors returns)
    got = sim(base, a, bt)
    nref = int((bits(got) != bits(lora_merge_reference(base, a, bt))).sum())
    nd, mx, nviol = compare("fp32 factors", got, k, 2.0 ** -23)
    check(f"{label} fp32 kernel == numpy reference(kernel inputs)", nref == 0, f"{nref} differ")
    check(f"{label} fp32 kernel vs loader(full base+delta) float64-once", nviol == 0 and nd / got.size < 1e-3,
          f"{nd} differ, max {mx:.0f} ulp, {nviol} outside bound")

    # 2) scale folded exactly once: the output must be far from scale^0 and scale^2 references
    for p, nm in ((0, "scale omitted"), (2, "scale twice")):
        if scale == 1.0:
            continue
        alt = round_once_bf16(loader.load([TensorSlice(w.double() + scale ** p * (b.double() @ a_.double()))
                                           for w, (b, a_) in zip(W, pairs)], rank).numpy())
        nalt = int((bits(got) != bits(alt)).sum())
        check(f"{label} kernel differs from '{nm}' reference", nalt > got.size // 4, f"{nalt}/{got.size} differ")

    # 3) split hi/lo from the builders' fp32 factors
    a2, b2 = split_hi_lo(a, bt)
    if shape_guard(base, a2, b2) is None:
        gs = sim(base, a2, b2)
        nds, mxs, nvs = compare("split hi/lo", gs, 3 * k, 2.0 ** -16 + 2.0 ** -23)
        check(f"{label} split vs float64-once inside bound", nvs == 0, f"{nds} differ, max {mxs:.0f} ulp, {nvs} outside")

    # 4) bf16-cast factors: informative only (lossy by design, not asserted)
    gb = sim(base, a.astype(BF16), bt.astype(BF16))
    compare("bf16-cast factors (info only)", gb, k, 2.0 ** -8)

    # 5) LNC=2 equals LNC=1
    if lnc2:
        g2 = sim(base, a, bt, lnc=2)
        n2 = int((bits(g2) != bits(got)).sum())
        check(f"{label} LNC=2 == LNC=1", n2 == 0 and not np.isnan(g2.astype(np.float32)).any(), f"{n2} differ")


def orientation_control(dims, tp, rank, r, seed):
    """Negative control: if bt were built in the HF [OUT, K] orientation (transposed) or a_stack not transposed,
    the kernel shapes do not fit; and if a square shard hid it, the values would be wrong. Use a square-ish shard to
    check the second case for the row-parallel 'down' kind with IN==OUT impossible here, so only the shape check runs."""
    loaders = engine_loaders(dims, tp)
    g = torch.Generator().manual_seed(seed)
    kind = "qkv"
    pairs = [(torch.randn(o, r, generator=g), torch.randn(r, dims.hf_in_dim(kind), generator=g))
             for o in dims.hf_out_dims(kind)]
    a_t, bt_t = build_factors(kind, pairs, 1.0, rank, tp, dims, loaders)
    base = np.zeros(dims.live_shape(kind, tp), dtype=BF16)
    wrong = shape_guard(base.T.copy(), a_t.numpy(), bt_t.numpy())
    check("orientation: [OUT,IN] base is rejected by shape_guard", wrong is not None, str(wrong))


def main():
    # rounding helper sanity against ml_dtypes on float32 values
    x = (np.random.default_rng(0).standard_normal(100000) * 5).astype(np.float32)
    check("own float64->bf16 rounding == ml_dtypes on float32 values",
          np.array_equal(bits(round_once_bf16(x.astype(np.float64))), bits(x.astype(BF16))), "100000 values")

    # small, non-symmetric synthetic config: every live shard spans > 1 tile on both axes except gate_up OUT
    synth = DenseDims(hidden=600, n_heads=8, n_kv=4, head_dim=144, inter=800)
    synth_rep = DenseDims(hidden=600, n_heads=8, n_kv=2, head_dim=144, inter=800)   # kv heads < tp: replicated
    tp, rank = 4, 1
    run_case("synth qkv", synth, "qkv", tp, rank, r=16, scale=0.3, seed=11, lnc2=True)
    run_case("synth qkv scale2", synth, "qkv", tp, rank, r=16, scale=2.0, seed=12)
    run_case("synth-repKV qkv", synth_rep, "qkv", tp, rank, r=16, scale=0.3, seed=13)
    run_case("synth o", synth, "o", tp, rank, r=16, scale=0.3, seed=14, lnc2=True)
    run_case("synth down", synth, "down", tp, rank, r=16, scale=0.3, seed=15)
    run_case("synth gate_up", synth, "gate_up", tp, rank, r=16, scale=0.3, seed=16)
    # real Qwen3-0.6B per-rank shapes at TP4 rank 1
    run_case("0.6B qkv", QWEN3_0_6B, "qkv", tp, rank, r=16, scale=2.0, seed=21)
    run_case("0.6B down", QWEN3_0_6B, "down", tp, rank, r=16, scale=2.0, seed=22)
    orientation_control(synth, tp, rank, 16, 31)

    print(f"\n{len(FAIL)} failed")
    if FAIL:
        print("FAILED:", *FAIL, sep="\n  ")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
