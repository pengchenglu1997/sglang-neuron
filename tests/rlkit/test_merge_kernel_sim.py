"""Simulator tests for lora_merge (nki 0.7.0, CPU simulator, target trn2).

Run:  python test_merge_kernel_sim.py        (exit code 0 only if every assert held)
Options (environment): RUN_8B=1 also runs Qwen3-8B per-rank shapes.

Every test compares the simulated kernel with
  (R) lora_merge_reference            (numpy, float32 accumulate, one cast)
  (X) an independent float64 sum rounded ONCE to bf16 (own round-to-nearest-even, not via float32)
and the independent computation is itself checked against ml_dtypes on float32-representable values.
"""
import os
import sys
import time
import traceback

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"

import ml_dtypes
import numpy as np

import nki
from lora_merge import (BF16, efficient_shape, lora_merge, lora_merge_reference, shape_guard, split_hi_lo)

RESULTS = []      # (test name, "PASS"/"FAIL", numbers text)
TIMINGS = []      # (shape text, seconds)


# ----------------------------------------------------------------------------- helpers
def f64_to_bf16_bits(x64):
    """float64 -> bf16, round to nearest even, ONE rounding, no float32 in between."""
    x64 = np.asarray(x64, dtype=np.float64)
    m, e = np.frexp(x64)                      # x = m * 2**e, 0.5 <= |m| < 1
    r = np.ldexp(np.rint(m * 256.0) / 256.0, e)   # 8 significant bits; np.rint = ties to even
    return r.astype(np.float32).astype(BF16)  # r is exactly a bf16 value, so this cast is lossless


def exact_reference(base, a, bt):
    """Independent: float64 everything, one rounding to bf16."""
    s = base.astype(np.float64) + a.astype(np.float64).T @ bt.astype(np.float64)
    return f64_to_bf16_bits(s)


def bits(x):
    return np.ascontiguousarray(x).view(np.uint16)


def ulp_of(x_bf16):
    """bf16 spacing at each value (value of one last-place unit), as float64."""
    x = x_bf16.astype(np.float64)
    _, e = np.frexp(x)
    u = np.ldexp(1.0, e - 8)
    u[x == 0] = np.ldexp(1.0, -133)           # smallest bf16 subnormal spacing, irrelevant here
    return u


def run_kernel(base, a, bt, lnc=None):
    if lnc is not None:
        os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(lnc)
    try:
        t = time.time()
        out = np.asarray(nki.simulate(lora_merge)(base, a, bt))
        dt = time.time() - t
    finally:
        os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
    TIMINGS.append((f"base{tuple(base.shape)} K={a.shape[0]} {a.dtype} lnc={lnc or 1}", dt))
    assert out.dtype == base.dtype and out.shape == base.shape, (out.dtype, out.shape)
    return out


def diff_stats(got, other):
    """Number of differing entries and max |diff| in units of the last place of `other`."""
    g, o = got.astype(np.float64), other.astype(np.float64)
    nd = int((bits(got) != bits(other)).sum())
    mx = float((np.abs(g - o) / ulp_of(other)).max()) if nd else 0.0
    return nd, mx


def within_bound(got, base, a, bt, factor_rel_err=0.0):
    """Principled accuracy check against the float64 sum s.

    Allowed error per entry:  0.5 ulp(rounded exact)  [one rounding]
                            + (K+2) * 2^-24 * (sum_k |a b| + |base|)   [fp32 accumulation / fp32 add]
                            + factor_rel_err * sum_k |a b|             [split mode: hi/lo truncation]
    Returns (number of entries outside the bound, max error / bound).
    A bound in ulps of the OUTPUT cannot work: when base and delta nearly cancel the output is tiny, its ulp
    is tiny, and a fixed fp32 error is many output ulps.
    """
    k = a.shape[0]
    a64, b64 = a.astype(np.float64), bt.astype(np.float64)
    s = base.astype(np.float64) + a64.T @ b64
    sabs = np.abs(a64).T @ np.abs(b64)
    rounded = f64_to_bf16_bits(s)
    bound = 0.5 * ulp_of(rounded) + (k + 2) * 2.0 ** -24 * (sabs + np.abs(base.astype(np.float64))) + factor_rel_err * sabs
    err = np.abs(got.astype(np.float64) - s)
    return int((err > bound * (1 + 1e-12)).sum()), float((err / bound).max())


def bf(x):
    return np.asarray(x, dtype=np.float32).astype(BF16)


def record(name, ok, numbers):
    RESULTS.append((name, "PASS" if ok else "FAIL", numbers))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {numbers}", flush=True)


def test(fn):
    def wrapper():
        try:
            fn()
        except AssertionError as e:
            record(fn.__name__, False, f"assertion failed: {e}")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            record(fn.__name__, False, f"error {type(e).__name__}: {str(e)[:300]}")
    wrapper.__name__ = fn.__name__
    return wrapper


def make_case(rng, in_dim, out_dim, k, base_std=0.02, f_std=0.05, factor_dtype="bf16"):
    base = bf(rng.standard_normal((in_dim, out_dim)) * base_std)
    a = rng.standard_normal((k, in_dim)) * f_std
    bt = rng.standard_normal((k, out_dim)) * f_std
    if factor_dtype == "bf16":
        return base, bf(a), bf(bt)
    return base, a.astype(np.float32), bt.astype(np.float32)


# ----------------------------------------------------------------------------- tests
@test
def test_00_checker_sanity():
    """The independent float64->bf16 rounding agrees with ml_dtypes on float32 inputs, and is a single rounding."""
    rng = np.random.default_rng(100)
    x32 = (rng.standard_normal(200000) * 3).astype(np.float32)
    mine = f64_to_bf16_bits(x32.astype(np.float64))
    theirs = x32.astype(BF16)
    assert np.array_equal(bits(mine), bits(theirs)), "own rounding differs from ml_dtypes on float32 values"
    # a double-rounding trap: a float64 value just above a bf16 tie that float32 would round onto the tie
    tie = 1.0 + 2.0 ** -8                     # exact tie between bf16 1.0 and 1.0078125 (ties-to-even -> 1.0)
    v = tie + 2.0 ** -40                      # slightly above the tie: correct single rounding is UP
    assert float(f64_to_bf16_bits(np.array([v]))[0]) == 1.0078125
    assert float(np.float32(v).astype(BF16)) == 1.0, "float32 path should lose the tie-break (shows why double rounding matters)"
    record("00 checker sanity", True, "own float64->bf16 equals ml_dtypes on 200000 float32 values; tie trap resolved correctly")


@test
def test_01_zero_adapter():
    rng = np.random.default_rng(1)
    parts = []
    for fd in ("bf16", "fp32"):
        for (i, o, k) in [(256, 512, 16), (200, 700, 48)]:
            base, a, bt = make_case(rng, i, o, k, factor_dtype=fd)
            bt = np.zeros_like(bt)                     # zero adapter: B = 0
            got = run_kernel(base, a, bt)
            assert np.array_equal(bits(got), bits(base)), f"B=0 changed the base ({fd}, {(i, o, k)})"
            a0 = np.zeros_like(a)
            bt2 = make_case(rng, i, o, k, factor_dtype=fd)[2]
            got2 = run_kernel(base, a0, bt2)           # A = 0
            assert np.array_equal(bits(got2), bits(base)), f"A=0 changed the base ({fd}, {(i, o, k)})"
            parts.append(f"{fd}{(i, o, k)}")
    record("(i) zero adapter returns base bit for bit", True, "B=0 and A=0, bf16 and fp32 factors: " + ", ".join(parts) + "; 0 differing entries")


@test
def test_02_integer_exact():
    rng = np.random.default_rng(2)
    rows = []
    for fd in ("bf16", "fp32"):
        for k in (16, 32, 48):
            i, o = 256, 1024
            base = bf(rng.integers(-32, 33, (i, o)))
            a = rng.integers(-2, 3, (k, i)).astype(np.float32)
            bt = rng.integers(-2, 3, (k, o)).astype(np.float32)
            if fd == "bf16":
                a, bt = bf(a), bf(bt)
            got = run_kernel(base, a, bt)
            ref = lora_merge_reference(base, a, bt)
            exact = exact_reference(base, a, bt)
            assert np.array_equal(bits(got), bits(ref)), f"kernel != reference ({fd}, K={k})"
            assert np.array_equal(bits(ref), bits(exact)), f"reference != float64 ({fd}, K={k})"
            assert np.array_equal(got.astype(np.float64), base.astype(np.float64) + a.astype(np.float64).T @ bt.astype(np.float64)), "not the exact integer"
            rows.append(f"{fd} K={k}: 0 differing of {got.size}")
    record("(ii) integer-valued factors bit-exact", True, "; ".join(rows) + " (kernel == reference == float64 == exact integers)")


@test
def test_03_random_vs_float64():
    rng = np.random.default_rng(3)
    lines = []
    worst_frac = 0.0
    worst_ulp = 0.0
    for fd in ("bf16", "fp32"):
        for k in (16, 32, 48):
            i, o = 384, 1024
            base, a, bt = make_case(rng, i, o, k, factor_dtype=fd)
            got = run_kernel(base, a, bt)
            ref = lora_merge_reference(base, a, bt)
            exact = exact_reference(base, a, bt)
            n_ref, u_ref = diff_stats(got, ref)
            n_x, u_x = diff_stats(got, exact)
            n_rx, u_rx = diff_stats(ref, exact)
            lines.append(f"{fd} K={k}: kernel vs float64-once: {n_x}/{got.size} differ (max {u_x:.0f} ulp); "
                         f"kernel vs numpy reference: {n_ref} differ (max {u_ref:.0f} ulp); "
                         f"reference vs float64-once: {n_rx} differ (max {u_rx:.0f} ulp)")
            nviol, ratio = within_bound(got, base, a, bt)
            assert nviol == 0, f"{nviol} entries outside the error bound ({fd}, K={k}), max err/bound {ratio:.2f}"
            lines[-1] += f"; max err/bound {ratio:.2f}"
            worst_frac = max(worst_frac, n_x / got.size)
            worst_ulp = max(worst_ulp, u_x)
    for l in lines:
        print("    " + l)
    # entries differing from float64-once come from the fp32 accumulation; each stays inside within_bound()
    assert worst_frac < 1e-3, f"differing fraction {worst_frac}"
    record("(iii)+(v) random factors std 0.05, K=16/32/48, base std 0.02", True,
           f"worst differing fraction vs float64-once {worst_frac:.2e}, worst diff {worst_ulp:.0f} ulp; per-case numbers:\n    " + "\n    ".join(lines))


@test
def test_04_swap_rounds():
    rng = np.random.default_rng(4)
    i, o, k = 256, 1024, 32
    base, a1, b1 = make_case(rng, i, o, k)
    _, a2, b2 = make_case(rng, i, o, k)
    base_copy = base.copy()
    first1 = None
    first2 = None
    for r in range(20):
        if r % 2 == 0:
            out = run_kernel(base, a1, b1)
            first1 = out if first1 is None else first1
            assert np.array_equal(bits(out), bits(first1)), f"adapter-1 output drifted at round {r}"
        else:
            out = run_kernel(base, a2, b2)
            first2 = out if first2 is None else first2
            assert np.array_equal(bits(out), bits(first2)), f"adapter-2 output drifted at round {r}"
    assert np.array_equal(bits(base), bits(base_copy)), "kernel modified its base input"
    assert not np.array_equal(bits(first1), bits(first2)), "adapters 1 and 2 gave the same output"
    assert np.array_equal(bits(first1), bits(exact_reference(base, a1, b1))) or diff_stats(first1, exact_reference(base, a1, b1))[0] / first1.size < 1e-3
    record("(iv) 20 rounds adapter 1 / 2 from the same base", True,
           "10 adapter-1 outputs identical to the first, 10 adapter-2 outputs identical to the first, base unchanged, 1 != 2 "
           f"({diff_stats(first1, first2)[0]}/{first1.size} entries differ between the adapters); by construction, asserted anyway")


@test
def test_05_shapes():
    rng = np.random.default_rng(5)
    rows = []
    for (i, o, k) in [(256, 512, 16), (384, 1024, 16), (384, 1024, 48), (200, 700, 16), (130, 513, 32), (128, 512, 48), (100, 300, 16)]:
        base, a, bt = make_case(rng, i, o, k)
        assert shape_guard(base, a, bt) is None
        got = run_kernel(base, a, bt)
        ref = lora_merge_reference(base, a, bt)
        exact = exact_reference(base, a, bt)
        n_ref, _ = diff_stats(got, ref)
        n_x, u_x = diff_stats(got, exact)
        nviol, ratio = within_bound(got, base, a, bt)
        assert nviol == 0 and n_x / got.size < 1e-3, (i, o, k, n_x, u_x, nviol)
        rows.append(f"{(i, o, k)}{'' if efficient_shape(i, o) else ' (partial tiles)'}: vs float64-once {n_x}/{got.size} differ, vs reference {n_ref}")
    record("(vi) shapes incl. remainders", True, "; ".join(rows))


# Per-rank shapes at TP=4 (all [IN, OUT]), from the shard formulas in the reader findings and the configs
# qwen3_0.6b_config.json / qwen3_8b_config.json in this folder (0.6B: hidden 1024, 16 heads, 8 kv heads, head_dim 128,
# intermediate 3072; 8B: hidden 4096, 32 heads, 8 kv heads, head_dim 128, intermediate 12288).
# K per kind: qkv 3r = 48, o 1r = 16, gate_up 2r = 32, down 1r = 16 (r = 16).
SHAPES_06B = {
    "qkv_weight  [H, (4+2*2)*128]": (1024, 1024, 48),
    "o_weight    [4*128, H]": (512, 1024, 16),
    "gate_up     [H, 2*3072/4]": (1024, 1536, 32),
    "down_weight [3072/4, H]": (768, 1024, 16),
}
SHAPES_8B = {   # hidden 4096, MLP 12288, heads 32, kv 8, head_dim 128 (readers: hidden 4096, MLP 12288, kv rows 1024)
    "qkv_weight  [4096, (8+2*2)*128]": (4096, 1536, 48),
    "o_weight    [8*128, 4096]": (1024, 4096, 16),
    "gate_up     [4096, 2*12288/4]": (4096, 6144, 32),
    "down_weight [12288/4, 4096]": (3072, 4096, 16),
}


def run_realistic(shapes, tag, fd="bf16"):
    rng = np.random.default_rng(6)
    rows = []
    for name, (i, o, k) in shapes.items():
        base, a, bt = make_case(rng, i, o, k, factor_dtype=fd)
        assert shape_guard(base, a, bt) is None and efficient_shape(i, o)
        t = time.time()
        got = run_kernel(base, a, bt)
        dt = time.time() - t
        ref = lora_merge_reference(base, a, bt)
        exact = exact_reference(base, a, bt)
        n_ref, _ = diff_stats(got, ref)
        n_x, u_x = diff_stats(got, exact)
        nviol, ratio = within_bound(got, base, a, bt)
        assert nviol == 0 and n_x / got.size < 1e-3, (name, n_x, u_x, nviol)
        rows.append(f"{name} base {(i, o)} K={k}: {dt:.1f}s sim, vs float64-once {n_x}/{got.size} differ (max {u_x:.0f} ulp), vs reference {n_ref}")
    return rows


@test
def test_06_realistic_06b():
    rows = run_realistic(SHAPES_06B, "0.6B")
    record("(vi) Qwen3-0.6B per-rank shapes TP4 (bf16 factors)", True, "\n    " + "\n    ".join(rows))


@test
def test_06b_realistic_8b():
    if os.environ.get("RUN_8B") != "1":
        record("(vi) Qwen3-8B per-rank shapes TP4", True, "SKIPPED (set RUN_8B=1)")
        return
    rows = run_realistic(SHAPES_8B, "8B")
    record("(vi) Qwen3-8B per-rank shapes TP4 (bf16 factors)", True, "\n    " + "\n    ".join(rows))


@test
def test_07_lnc():
    rng = np.random.default_rng(7)
    rows = []
    for (i, o, k, fd) in [(384, 1024, 16, "bf16"), (200, 700, 48, "bf16"), (640, 512, 32, "fp32"), (128, 512, 16, "bf16"), (1024, 1536, 32, "bf16")]:
        base, a, bt = make_case(rng, i, o, k, factor_dtype=fd)
        o1 = run_kernel(base, a, bt, lnc=1)
        o2 = run_kernel(base, a, bt, lnc=2)
        assert not np.isnan(o2.astype(np.float32)).any(), "LNC=2 left unwritten (NaN) entries"
        assert np.array_equal(bits(o1), bits(o2)), f"LNC=1 and LNC=2 differ at {(i, o, k, fd)}"
        rows.append(f"{(i, o, k)} {fd}: identical ({o1.size} entries)")
    record("(vii) LNC=1 vs LNC=2 (NEURON_LOGICAL_NC_CONFIG, body runs per program)", True, "; ".join(rows))


@test
def test_08_split_hi_lo():
    """Alternative path: host splits fp32 factors into bf16 hi/lo, kernel runs bf16 with K' = 3K."""
    rng = np.random.default_rng(8)
    rows = []
    for k in (16, 32, 48):
        i, o = 256, 1024
        base, a, bt = make_case(rng, i, o, k, factor_dtype="fp32")
        a_cat, bt_cat = split_hi_lo(a, bt)
        assert a_cat.dtype == BF16 and a_cat.shape[0] == 3 * k and shape_guard(base, a_cat, bt_cat) is None
        got = run_kernel(base, a_cat, bt_cat)
        exact = exact_reference(base, a, bt)                 # float64 on the ORIGINAL fp32 factors
        native = run_kernel(base, a, bt)                     # native fp32 operands
        n_s, u_s = diff_stats(got, exact)
        n_n, u_n = diff_stats(native, exact)
        nviol_s, ratio_s = within_bound(got, base, a, bt, factor_rel_err=2.0 ** -16)
        nviol_n, ratio_n = within_bound(native, base, a, bt)
        assert nviol_s == 0 and nviol_n == 0, (k, nviol_s, nviol_n)
        rows.append(f"K={k} (K'={3 * k}, {-(-3 * k // 128)} matmuls): split vs float64-once {n_s}/{got.size} differ (max {u_s:.0f} ulp); native fp32 {n_n}/{got.size} differ (max {u_n:.0f} ulp); max err/bound split {ratio_s:.2f}, native {ratio_n:.2f}")
    record("split hi/lo path (fp32 factors as bf16 pairs)", True, "\n    " + "\n    ".join(rows))


@test
def test_09_shape_guard():
    rng = np.random.default_rng(9)
    base, a, bt = make_case(rng, 256, 512, 16)
    assert shape_guard(base, a, bt) is None
    cases = {
        "K too large": (base, bf(np.zeros((300, 256))), bf(np.zeros((300, 512)))),
        "mixed dtype": (base, a, bt.astype(np.float32)),
        "base fp32": (base.astype(np.float32), a, bt),
        "IN mismatch": (base, a[:, :200], bt),
        "OUT mismatch": (base, a, bt[:, :400]),
        "K mismatch": (base, a, bt[:8]),
    }
    bad = [n for n, args in cases.items() if shape_guard(*args) is None]
    assert not bad, f"guard accepted: {bad}"
    record("shape guard", True, f"accepts a valid case, rejects {len(cases)} invalid ones: {list(cases)}")


if __name__ == "__main__":
    t0 = time.time()
    for fn in (test_00_checker_sanity, test_01_zero_adapter, test_02_integer_exact, test_03_random_vs_float64,
               test_04_swap_rounds, test_05_shapes, test_06_realistic_06b, test_06b_realistic_8b, test_07_lnc,
               test_08_split_hi_lo, test_09_shape_guard):
        fn()
    print("\n================ SUMMARY ================")
    for name, status, nums in RESULTS:
        print(f"{status}  {name}")
    print(f"kernel simulator calls: {len(TIMINGS)}, total kernel sim time {sum(t for _, t in TIMINGS):.1f}s, wall {time.time() - t0:.1f}s")
    slow = sorted(TIMINGS, key=lambda x: -x[1])[:6]
    for s, t in slow:
        print(f"   slowest: {s}: {t:.2f}s")
    failed = [r for r in RESULTS if r[1] != "PASS"]
    print("FAILED:" if failed else "ALL PASSED", len(failed) if failed else "")
    sys.exit(1 if failed else 0)
