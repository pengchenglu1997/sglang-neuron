"""CPU layout test of the per-rank LoRA factor builders against the engine's own loaders.

Run (WSL): ~/nki_sim_venv/bin/python test_factor_layout.py
Exit code 1 if any case fails. A "case" is (model, kind, tp) with every rank 0..tp-1 checked inside it.
"""

import sys
import time

import torch

from factor_builders import (KINDS, SEGMENTS, QWEN3_0_6B, QWEN3_8B, DenseDims, build_factors, engine_loaders,
                             live_delta)

R = 16
ALPHA = 32
SCALE = ALPHA / R            # 2.0, a power of two: exact in float32
TPS = (1, 2, 4, 8, 16)       # 8 and 16 go past the KV head count (8) of Qwen3: 16 replicates each KV head twice
SYNTH = DenseDims(hidden=256, n_heads=8, n_kv=2, head_dim=64, inter=512)   # kv heads < tp at tp = 4 and 8
failures = []


def report(name: str, ok: bool, extra: str = "") -> None:
    print(f"{'PASS' if ok else 'FAILED'}  {name}  {extra}".rstrip(), flush=True)
    if not ok:
        failures.append(name)


def make_pairs(kind, dims, seed, integer=True, none_at=()):
    g = torch.Generator().manual_seed(seed)
    pairs = []
    for i, out in enumerate(dims.hf_out_dims(kind)):
        if i in none_at:
            pairs.append(None)
            continue
        if integer:
            b = torch.randint(-3, 4, (out, R), generator=g).float()
            a = torch.randint(-3, 4, (R, dims.hf_in_dim(kind)), generator=g).float()
        else:
            b = torch.randn(out, R, generator=g)
            a = torch.randn(R, dims.hf_in_dim(kind), generator=g)
        pairs.append((b, a))
    return pairs


def valid_tps(dims):
    out = []
    for tp in TPS:
        try:
            dims.check_tp(tp)
            out.append(tp)
        except ValueError:
            pass
    return out


def integer_cases(model, dims, tps):
    """a_stack.T @ bt == loader(full scale*B@A), bit for bit, in float32 and float64."""
    for kind in KINDS:
        pairs = make_pairs(kind, dims, seed=sum(map(ord, model + kind)))
        for tp in tps:
            loaders = engine_loaders(dims, tp)
            problems = []
            for rank in range(tp):
                a, bt = build_factors(kind, pairs, SCALE, rank, tp, dims, loaders)
                ref32 = live_delta(kind, pairs, SCALE, rank, loaders[kind], torch.float32)
                ref64 = live_delta(kind, pairs, SCALE, rank, loaders[kind], torch.float64)
                got32 = a.t() @ bt
                got64 = a.double().t() @ bt.double()
                if got32.shape != ref32.shape or not torch.equal(got32, ref32):
                    problems.append(f"rank{rank}:fp32")
                if not torch.equal(got64, ref64):
                    problems.append(f"rank{rank}:fp64")
                if a.dtype != torch.float32 or bt.dtype != torch.float32:
                    problems.append(f"rank{rank}:dtype")
                if ref32.shape != dims.live_shape(kind, tp):
                    problems.append(f"rank{rank}:live_shape {tuple(ref32.shape)}")
            report(f"integer {model} {kind:8s} tp={tp:2d} ranks=0..{tp - 1} K={SEGMENTS[kind] * R if kind in ('qkv', 'gate_up') else R}"
                   f" shard[IN,OUT]={list(dims.live_shape(kind, tp))}", not problems, ",".join(problems))


def random_cases(model, dims, tp=4):
    """Random float adapters: max abs difference in float64 between a_stack.T@bt and loader(float64 full delta)."""
    for scale in (SCALE, 0.3):
        for kind in KINDS:
            pairs = make_pairs(kind, dims, seed=7, integer=False)
            loaders = engine_loaders(dims, tp)
            worst, ref_max, worst32 = 0.0, 0.0, 0.0
            for rank in range(tp):
                a, bt = build_factors(kind, pairs, scale, rank, tp, dims, loaders)
                ref64 = live_delta(kind, pairs, scale, rank, loaders[kind], torch.float64)
                ref32 = live_delta(kind, pairs, scale, rank, loaders[kind], torch.float32)
                worst = max(worst, (a.double().t() @ bt.double() - ref64).abs().max().item())
                worst32 = max(worst32, (a.t() @ bt - ref32).abs().max().item())
                ref_max = max(ref_max, ref64.abs().max().item())
            # thresholds fixed from dtype reasoning before the run: scale 2.0 is exact in float32, so only the fp64
            # summation order differs (~1e-15 relative); scale 0.3 is rounded once in float32 (<= 6e-8 relative per
            # entry of B, sum over K <= 48 terms).
            tol = 1e-12 * ref_max if scale == SCALE else 1e-6 * ref_max
            report(f"random  {model} {kind:8s} tp={tp} scale={scale}", worst <= tol,
                   f"max|diff|_fp64={worst:.3e} (max|ref|={ref_max:.3e}, tol={tol:.1e}) fp32-pipeline max|diff|={worst32:.3e}")


def replicated_kv_cases():
    """kv heads < tp: ranks that share a KV head must get the same K and V block of bt, and a different Q block."""
    for model, dims, tp in (("synth(h8,kv2)", SYNTH, 4), ("synth(h8,kv2)", SYNTH, 8), ("0.6B", QWEN3_0_6B, 16),
                            ("8B", QWEN3_8B, 16)):
        loaders = engine_loaders(dims, tp)
        pairs = make_pairs("qkv", dims, seed=3)
        d = dims.head_dim
        q_w, kv_w = (dims.n_heads // tp) * d, d     # kv heads per rank is 1 when replicated
        group = tp // dims.n_kv
        bts = [build_factors("qkv", pairs, SCALE, rank, tp, dims, loaders)[1] for rank in range(tp)]
        problems = []
        for rank in range(tp):
            lead = rank - rank % group
            k_rows, v_rows = slice(R, 2 * R), slice(2 * R, 3 * R)
            k_cols, v_cols = slice(q_w, q_w + kv_w), slice(q_w + kv_w, q_w + 2 * kv_w)
            if not torch.equal(bts[rank][k_rows, k_cols], bts[lead][k_rows, k_cols]):
                problems.append(f"rank{rank}:K!=rank{lead}")
            if not torch.equal(bts[rank][v_rows, v_cols], bts[lead][v_rows, v_cols]):
                problems.append(f"rank{rank}:V!=rank{lead}")
            if rank != lead and torch.equal(bts[rank][0:R, 0:q_w], bts[lead][0:R, 0:q_w]):
                problems.append(f"rank{rank}:Q equals rank{lead}")
            # the K block of the group must be the HF k_proj rows of KV head rank // group, from B directly
            kv = rank // group
            want = (SCALE * pairs[1][0][kv * d:(kv + 1) * d]).t()
            if not torch.equal(bts[rank][k_rows, k_cols], want):
                problems.append(f"rank{rank}:K block is not KV head {kv}")
        report(f"replicated-KV {model} qkv tp={tp} (n_kv={dims.n_kv}, each KV head on {group} ranks)", not problems,
               ",".join(problems))


def partial_adapter_case():
    """q and v adapted, k not (pairs[1] = None): equals the loader applied with a zero delta for k."""
    dims, tp = QWEN3_0_6B, 4
    loaders = engine_loaders(dims, tp)
    pairs = make_pairs("qkv", dims, seed=5, none_at=(1,))
    full = [p if p is not None else (torch.zeros(dims.n_kv * dims.head_dim, R), torch.zeros(R, dims.hidden))
            for p in pairs]
    problems = []
    for rank in range(tp):
        a, bt = build_factors("qkv", pairs, SCALE, rank, tp, dims, loaders, lora_r=R)
        ref = live_delta("qkv", full, SCALE, rank, loaders["qkv"])
        if not torch.equal(a.t() @ bt, ref):
            problems.append(f"rank{rank}")
    report("partial adapter 0.6B qkv tp=4 (k_proj not adapted)", not problems, ",".join(problems))


def kinds_are_independent_of_r_case():
    """A different LoRA rank (r=8, 64) must work too: K = c*r."""
    dims, tp = QWEN3_0_6B, 4
    loaders = engine_loaders(dims, tp)
    problems = []
    for r in (8, 64):
        g = torch.Generator().manual_seed(r)
        for kind in KINDS:
            pairs = [(torch.randint(-3, 4, (o, r), generator=g).float(),
                      torch.randint(-3, 4, (r, dims.hf_in_dim(kind)), generator=g).float())
                     for o in dims.hf_out_dims(kind)]
            for rank in range(tp):
                a, bt = build_factors(kind, pairs, SCALE, rank, tp, dims, loaders)
                ref = live_delta(kind, pairs, SCALE, rank, loaders[kind])
                if not torch.equal(a.t() @ bt, ref):
                    problems.append(f"r{r}:{kind}:rank{rank}")
    report("other LoRA ranks r=8,64 on 0.6B tp=4, all kinds", not problems, ",".join(problems))


def shape_table():
    print("\nPer-rank shard shapes of the adapted live parameters (TP=4, rank-16 adapter), IN x OUT = a_stack cols x bt cols")
    print(f"{'model':6s} {'kind':8s} {'IN':>6s} {'OUT':>6s} {'K':>4s}  IN%128 OUT%128 OUT%512  (IN/128, OUT/512)")
    for model, dims in (("0.6B", QWEN3_0_6B), ("8B", QWEN3_8B)):
        for kind in KINDS:
            i, o = dims.live_shape(kind, 4)
            k = SEGMENTS[kind] * R if kind in ("qkv", "gate_up") else R
            print(f"{model:6s} {kind:8s} {i:6d} {o:6d} {k:4d}  {str(i % 128 == 0):6s} {str(o % 128 == 0):6s} {str(o % 512 == 0):6s}  ({i / 128:.2f}, {o / 512:.2f})")


def main():
    t0 = time.time()
    print(f"torch {torch.__version__}; r={R} alpha={ALPHA} scale={SCALE}")
    for model, dims in (("0.6B", QWEN3_0_6B), ("8B", QWEN3_8B)):
        integer_cases(model, dims, valid_tps(dims))
    integer_cases("synth(h8,kv2)", SYNTH, [tp for tp in (1, 2, 4, 8)])
    replicated_kv_cases()
    partial_adapter_case()
    kinds_are_independent_of_r_case()
    for model, dims in (("0.6B", QWEN3_0_6B), ("8B", QWEN3_8B)):
        random_cases(model, dims)
    shape_table()
    print(f"\n{len(failures)} failed, took {time.time() - t0:.1f}s")
    if failures:
        print("FAILED:", *failures, sep="\n  ")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
