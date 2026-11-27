#!/usr/bin/env python3
"""Selftest of the topk_topp sampler kernel path (sglang_neuron/rlkit_sampler_topk.py and its guarded branch in
sglang_neuron/sampling.py), CPU only.

    python tests/rlkit_sampler/selftest_sampler_topk.py [--repo REPO] [--scratch DIR] [--seeds N] [--sim]

Builds the scratch package of selftest_sampler_kernel.py (the RELEASED v0.1.0 sampling.py, a collectives stub over a
simulated TP4 group, a copy of rlkit_sampler_kernel.py) and adds: a stub of sglang_neuron.sampling_topk whose topk is
torch.topk (what the released _pool_topk runs on a CPU tensor; the real module imports nkilib, which the simulator venv
does not have), a copy of this repository's rlkit_sampler_topk.py, and this repository's sampling.py under other module
names (for the switch tests). Then, at the TP4 shard width 37,984, against the released sample_filtered_sharded_logits:
  K0  switch: unset and "0" run the released function body (outputs bit-equal, rlkit_sampler_topk never imported);
      "1" routes to the kernel path, whose output is bit-equal to the released output (CPU reference stage)
  K1  whole output dict bit-equal (token ids, logprobs, top logprobs) on every rank, kernel stage =
      gumbel_partials_reference: top-k {1, 50, 256, off} x top-p {0.5, 0.95, 1} x min-p {0, 0.1}, temperatures
      0 / 0.6 / 0.8 / 1.0, rows 1 / 8 / 32
  K2  mixed per-row settings (filters and temperatures, temperature-0 rows included) in one batch, top logprobs 0 and 20,
      SGLANG_RETURN_ORIGINAL_LOGPROB false and true
  K3  ties: equal logits across ranks inside the pool and straddling the top-k cut, equal maxima of greedy rows across
      ranks, constant rows, -inf columns
  K4  restricted sampled rows do not depend on the kernel stage: garbage partials leave their outputs bit-equal, and
      change the outputs of unrestricted and greedy rows (so the comparison can see the kernel stage at all)
  K5  planted faults that MUST be caught: wrong noise column, wrong row key, no tie rule, filters ignored
  K6  source checks: this repository's sampling.py = released + the two guarded branches only; every other function of
      it has the released bytecode; the tail of the kernel path is the released text verbatim
  K7  backend/compile_cache.py counts rlkit_sampler_topk.py among the kernel sources (so the cache digest changes)
  K8  padded vocabulary (width * world > vocab, large values in the padded columns) and non-finite logits (-inf
      columns, a whole -inf shard, +inf, NaN): whole output bit-equal
  K9  128 rows run; 129 rows raise an error that names RLKIT_SAMPLER_KERNEL_TOPK
  K10 the vocabulary mask is applied on the kernel path also when no column is padded (width * world == vocab), as the
      released path does: the same number of vocabulary-mask masked_fill calls per rank in both paths, and not zero
  --sim  (WSL with nki 0.7.0): the kernel stage is the NKI kernel in nki.simulate; restricted rows must be bit-equal,
         greedy rows must have equal tokens and logprobs (also on the K8 inputs), unrestricted sampled rows are counted (the simulator's log is not torch.log)
Exit 0 when every test passes. No device, no docker, no network.
"""
from __future__ import annotations

import argparse
import difflib
import importlib
import importlib.util
import os
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEF_REPO = HERE.parents[1]
PKG = DEF_REPO / "sglang_neuron"
RESULTS = []
SWITCH = "RLKIT_SAMPLER_KERNEL_TOPK"

SAMPLING_TOPK_STUB = '''"""Selftest stub of sglang_neuron.sampling_topk: the CPU branch of its topk (torch.topk)."""
import torch


def select_topk_method(tensor, k, dim):
    return None


def topk(tensor, k, dim=-1, gather_dim=None):
    return torch.topk(tensor, k, dim=dim)
'''


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load("selftest_sampler_kernel", HERE / "selftest_sampler_kernel.py")
WIDTH, WORLD, VOCAB = B.WIDTH, B.WORLD, B.VOCAB


def check(name, ok, info=""):
    RESULTS.append((name, ok, info))
    print(("PASS " if ok else "FAIL ") + name + (" | " + info if info else ""), flush=True)
    return ok


def build_scratch(repo: Path, scratch: Path):
    src = B.build_scratch(repo, scratch)
    pkg = scratch / "sglang_neuron"
    (pkg / "sampling_topk.py").write_text(SAMPLING_TOPK_STUB, encoding="utf-8", newline="\n")
    shutil.copyfile(PKG / "rlkit_sampler_topk.py", pkg / "rlkit_sampler_topk.py")
    shutil.copyfile(PKG / "sampling.py", pkg / "sampling_this_repo.py")
    return src


# ----------------------------------------------------------------------------------------------- helpers
def make_filters(S, rows, top_k, top_p, min_p):
    k = float(S.TOP_K_ALL) if top_k in (-1, None) else float(top_k)
    return torch.tensor([[k, top_p, min_p]] * rows, dtype=torch.float32)


def make_logits(rows, gen):
    """bf16-rounded logits (many exact ties, as the engine's bf16 LM head gives) with per-row spread 1 to 8."""
    spread = torch.tensor([1.0, 2.0, 4.0, 8.0])[torch.randint(0, 4, (rows, 1), generator=gen)]
    return (torch.randn(rows, VOCAB, generator=gen) * spread).to(torch.bfloat16).float()


def make_keys(S, rows, gen):
    return S.noise_row_keys(torch.randint(0, 2**31 - 1, (rows,), generator=gen, dtype=torch.int64),
                            torch.randint(0, 4096, (rows,), generator=gen))


def released_fn(S, temps, keys, filters, top_logprobs, vocab=VOCAB):
    return lambda sh, g, r: S.sample_filtered_sharded_logits(sh, g, r, vocab, temps, keys, filters,
                                                             top_logprobs=top_logprobs)


def kernel_fn(S, K, T, temps, keys, filters, top_logprobs, partials=None, vocab=VOCAB):
    partials = partials or K.gumbel_partials_reference
    return lambda sh, g, r: T.sample_filtered_sharded_logits_kernel(sh, g, r, vocab, temps, keys, filters,
                                                                    top_logprobs, S.TOP_K_MAX, partials_fn=partials)


def bits(t):
    return t.contiguous().view(torch.int32) if t.dtype == torch.float32 else t


def same_dict(a, b):
    return a.keys() == b.keys() and all(torch.equal(bits(a[k]), bits(b[k])) for k in a)


def row_mismatch(e, k):
    """[R] bool: a row differs in any output on any rank."""
    bad = None
    for a, b in zip(e, k):
        d = (a["token_ids"] != b["token_ids"]) | (bits(a["logprobs"]) != bits(b["logprobs"]))
        for key in ("top_vals", "top_ids"):
            if key in a:
                d |= (bits(a[key]) != bits(b[key])).any(dim=-1)
        bad = d if bad is None else bad | d
    return bad


def row_classes(S, temps, filters):
    sampled = temps > 0
    restricted = sampled & ((filters[:, 0] < S.TOP_K_ALL) | (filters[:, 1] < 1) | (filters[:, 2] > 0))
    return restricted, sampled & ~restricted, ~sampled


def compare(S, K, T, logits, temps, keys, filters, top_logprobs=0, partials=None, kernel_filters=None, vocab=VOCAB):
    e = B.run_tp(released_fn(S, temps, keys, filters, top_logprobs, vocab), logits, temps, keys)
    kf = filters if kernel_filters is None else kernel_filters
    k = B.run_tp(kernel_fn(S, K, T, temps, keys, kf, top_logprobs, partials, vocab), logits, temps, keys)
    return e, k, row_mismatch(e, k)


class Tally:
    def __init__(self):
        self.cases = self.rows = self.bad = 0
        self.kinds = [0, 0, 0]

    def add(self, S, temps, filters, bad):
        self.cases += 1
        self.rows += int(temps.numel())
        self.bad += int(bad.sum())
        for i, m in enumerate(row_classes(S, temps, filters)):
            self.kinds[i] += int(m.sum())

    def info(self):
        return (f"{self.cases} cases x {WORLD} ranks, {self.rows} rows (restricted sampled {self.kinds[0]}, "
                f"unrestricted sampled {self.kinds[1]}, temperature 0 {self.kinds[2]}); rows differing {self.bad}")


# ----------------------------------------------------------------------------------------------- K0 switch
def k0(S, scratch: Path):
    gen = torch.Generator().manual_seed(11)
    rows = 8
    logits, keys = make_logits(rows, gen), make_keys(S, rows, gen)
    temps = torch.tensor([0.8, 0.8, 0.0, 1.0, 0.6, 0.8, 0.0, 0.8])
    filters = torch.tensor([[50, 0.95, 0], [S.TOP_K_ALL, 1, 0], [50, 0.95, 0], [S.TOP_K_ALL, 0.5, 0.1],
                            [1, 1, 0], [256, 1, 0.1], [S.TOP_K_ALL, 1, 0], [50, 0.95, 0]], dtype=torch.float32)
    want = B.run_tp(released_fn(S, temps, keys, filters, 0), logits, temps, keys)
    path = scratch / "sglang_neuron" / "sampling_this_repo.py"
    ok_all, info = True, []
    saved = os.environ.get(SWITCH)
    try:
        for value in (None, "0"):
            if value is None:
                os.environ.pop(SWITCH, None)
            else:
                os.environ[SWITCH] = value
            sys.modules.pop("sglang_neuron.rlkit_sampler_topk", None)
            P = _load(f"sglang_neuron.sampling_switch_{value}", path)
            got = B.run_tp(released_fn(P, temps, keys, filters, 0), logits, temps, keys)
            imported = "sglang_neuron.rlkit_sampler_topk" in sys.modules
            ok = (P._RLKIT_SAMPLER_KERNEL_TOPK is False and not imported
                  and all(same_dict(a, b) for a, b in zip(want, got)))
            info.append(f"{SWITCH}={value!s}: flag {P._RLKIT_SAMPLER_KERNEL_TOPK}, kernel module imported {imported}, "
                        f"bit-equal {ok}")
            ok_all &= ok
        os.environ[SWITCH] = "1"
        P = _load("sglang_neuron.sampling_switch_1", path)
        calls = []
        T = importlib.import_module("sglang_neuron.rlkit_sampler_topk")
        real = T.sample_filtered_sharded_logits_kernel
        K = importlib.import_module("sglang_neuron.rlkit_sampler_kernel")

        def spy(*args, **kw):
            calls.append(1)
            return real(*args, partials_fn=K.gumbel_partials_reference, **kw)
        T.sample_filtered_sharded_logits_kernel = spy
        try:
            got = B.run_tp(released_fn(P, temps, keys, filters, 0), logits, temps, keys)
        finally:
            T.sample_filtered_sharded_logits_kernel = real
        routed_equal = all(same_dict(a, b) for a, b in zip(want, got))   # routed output = released output
        ok = P._RLKIT_SAMPLER_KERNEL_TOPK is True and len(calls) == 2 * WORLD and routed_equal
        info.append(f"{SWITCH}=1: flag {P._RLKIT_SAMPLER_KERNEL_TOPK}, kernel path calls {len(calls)}, routed output "
                    f"bit-equal to the released output {routed_equal}")
        ok_all &= ok
    finally:
        if saved is None:
            os.environ.pop(SWITCH, None)
        else:
            os.environ[SWITCH] = saved
    return check("K0 switch unset / 0 runs the released body (bit-equal, kernel module not imported); 1 routes to "
                 "the kernel path", ok_all, "; ".join(info))


# ----------------------------------------------------------------------------------------------- K1 settings matrix
def k1(S, K, T, seeds):
    gen = torch.Generator().manual_seed(1234)
    tally = Tally()
    temps_cycle = [0.8, 0.6, 1.0, 0.0]
    i = 0
    for top_k in (1, 50, 256, -1):
        for top_p in (0.5, 0.95, 1.0):
            for min_p in (0.0, 0.1):
                for rows in (1, 8, 32):
                    for _ in range(seeds):
                        temp = temps_cycle[i % len(temps_cycle)]
                        i += 1
                        logits, keys = make_logits(rows, gen), make_keys(S, rows, gen)
                        temps = torch.full((rows,), temp)
                        filters = make_filters(S, rows, top_k, top_p, min_p)
                        _, _, bad = compare(S, K, T, logits, temps, keys, filters)
                        tally.add(S, temps, filters, bad)
                        if bad.any():
                            check(f"K1 top_k={top_k} top_p={top_p} min_p={min_p} rows={rows} T={temp}", False,
                                  f"{int(bad.sum())} rows differ")
    return check("K1 settings matrix: outputs bit-equal to the released path (CPU reference stage)", tally.bad == 0,
                 tally.info())


# ----------------------------------------------------------------------------------------------- K2 mixed batches
def random_settings(S, rows, gen):
    temps = torch.tensor([0.0, 0.6, 0.8, 1.0])[torch.randint(0, 4, (rows,), generator=gen)]
    ks = torch.tensor([1.0, 50.0, 256.0, float(S.TOP_K_ALL)])[torch.randint(0, 4, (rows,), generator=gen)]
    ps = torch.tensor([0.5, 0.95, 1.0])[torch.randint(0, 3, (rows,), generator=gen)]
    ms = torch.tensor([0.0, 0.1])[torch.randint(0, 2, (rows,), generator=gen)]
    return temps, torch.stack([ks, ps, ms], dim=1)


def k2(S, K, T, seeds):
    gen = torch.Generator().manual_seed(4321)
    tally = Tally()
    saved = os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB")
    try:
        for original in ("false", "true"):
            os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] = original
            for top_logprobs in (0, 20):
                for rows in (1, 8, 32):
                    for _ in range(seeds):
                        logits, keys = make_logits(rows, gen), make_keys(S, rows, gen)
                        temps, filters = random_settings(S, rows, gen)
                        _, _, bad = compare(S, K, T, logits, temps, keys, filters, top_logprobs)
                        tally.add(S, temps, filters, bad)
                        if bad.any():
                            check(f"K2 original={original} top_logprobs={top_logprobs} rows={rows}", False,
                                  f"{int(bad.sum())} rows differ")
    finally:
        if saved is None:
            os.environ.pop("SGLANG_RETURN_ORIGINAL_LOGPROB", None)
        else:
            os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] = saved
    return check("K2 mixed per-row settings in one batch, top logprobs 0 / 20, original logprob off / on: bit-equal",
                 tally.bad == 0, tally.info())


# ----------------------------------------------------------------------------------------------- K3 ties
def k3(S, K, T):
    gen = torch.Generator().manual_seed(99)
    ok_all, info = True, []
    for rows in (1, 8, 32):
        keys = make_keys(S, rows, gen)
        temps, filters = random_settings(S, rows, gen)
        temps[0] = 0.0                                    # a greedy row always
        # equal maxima in several ranks (greedy rows must take the smallest id) and a 4-way tie at the top of the pool
        lg = make_logits(rows, gen)
        for col in (WIDTH * 3 + 5, WIDTH + 1200, 2400, 2 * WIDTH + 7):
            lg[:, col] = 30.0
        e, _, bad = compare(S, K, T, lg, temps, keys, filters)
        ok = not bad.any() and int(e[0]["token_ids"][0]) == 2400
        info.append(f"top ties r{rows} {ok}")
        ok_all &= ok
        # ties straddling the top-k cut: 6 equal values ranked 48..53 (top_k 50 cuts inside them)
        lg = make_logits(rows, gen)
        lg[:, :48 * 97:97] = 40.0 + torch.arange(48, dtype=torch.float32).flip(0) * 0.25
        lg[:, [3, WIDTH + 3, 2 * WIDTH + 3, 3 * WIDTH + 3, 9000, WIDTH + 9000]] = 39.0
        f50 = make_filters(S, rows, 50, 1.0, 0.0)
        _, _, bad = compare(S, K, T, lg, torch.full((rows,), 1.0), keys, f50)
        info.append(f"top-k cut ties r{rows} {not bad.any()}")
        ok_all &= not bad.any()
        # constant rows (every column ties), mixed settings
        _, _, bad = compare(S, K, T, torch.zeros(rows, VOCAB), temps, keys, filters)
        info.append(f"const r{rows} {not bad.any()}")
        ok_all &= not bad.any()
        # -inf columns
        lg = make_logits(rows, gen)
        lg[:, ::7] = float("-inf")
        _, _, bad = compare(S, K, T, lg, temps, keys, filters, top_logprobs=20)
        info.append(f"-inf r{rows} {not bad.any()}")
        ok_all &= not bad.any()
    return check("K3 ties (pool top, top-k cut, greedy across ranks), constant rows, -inf columns: bit-equal", ok_all,
                 ", ".join(info))


# ----------------------------------------------------------------------------------------------- K4 independence
def k4(S, K, T):
    gen = torch.Generator().manual_seed(8)
    ref = K.gumbel_partials_reference
    noise_gen = torch.Generator().manual_seed(77)

    def garbage(x, k, i, g, s, b):
        out = ref(x, k, i, g, s, b).clone()
        n = out[:, 0::K.NPART].shape
        out[:, 0::K.NPART] = torch.randn(n, generator=noise_gen) * 100.0    # some other chunk wins every row
        return out
    restricted_same, other_changed, counts = True, 0, [0, 0]
    for rows in (8, 32):
        for _ in range(3):
            logits, keys = make_logits(rows, gen), make_keys(S, rows, gen)
            temps, filters = random_settings(S, rows, gen)
            restricted, unrestricted, greedy = row_classes(S, temps, filters)
            _, _, bad = compare(S, K, T, logits, temps, keys, filters, partials=garbage)
            restricted_same &= not bool(bad[restricted].any())
            other_changed += int(bad[unrestricted | greedy].sum())
            counts[0] += int(restricted.sum())
            counts[1] += int((unrestricted | greedy).sum())
    ok = restricted_same and other_changed > 0
    return check("K4 garbage kernel stage: restricted sampled rows bit-equal, other rows change", ok,
                 f"restricted rows {counts[0]} all equal {restricted_same}; unrestricted or greedy rows changed "
                 f"{other_changed} of {counts[1]}")


# ----------------------------------------------------------------------------------------------- K5 faults
def k5(S, K, T):
    ref = K.gumbel_partials_reference
    faults = {
        "wrong noise column (base + 1)": (lambda x, k, i, g, s, b: ref(x, k, i, g, s, b + 1.0), None),
        "wrong row key (key ^ 1)": (lambda x, k, i, g, s, b: ref(x, k ^ 1, i, g, s, b), None),
        "no tie rule (largest id)": (lambda x, k, i, g, s, b: B._largest_id(ref(x, k, i, g, s, b), b, x.shape[1], K),
                                     None),
        "filters ignored": (None, "off"),
    }
    gen = torch.Generator().manual_seed(5)
    ok_all = True
    for name, (fn, kfilters) in faults.items():
        caught = 0
        for rows in (1, 8, 32):
            logits, keys = make_logits(rows, gen), make_keys(S, rows, gen)
            if "tie" in name:
                logits[:, 100] = 60.0
                logits[:, 90000] = 60.0
                temps = torch.zeros(rows)
                filters = make_filters(S, rows, -1, 1.0, 0.0)
            elif kfilters:
                temps = torch.full((rows,), 1.0)
                filters = make_filters(S, rows, 50, 0.95, 0.0)
            else:                                           # unrestricted sampled rows: the noise is visible there
                temps = torch.full((rows,), 1.0)
                filters = make_filters(S, rows, -1, 1.0, 0.0)
            kf = make_filters(S, rows, -1, 1.0, 0.0) if kfilters else None
            _, _, bad = compare(S, K, T, logits, temps, keys, filters, partials=fn, kernel_filters=kf)
            caught += int(bad.sum())
        ok_all &= check(f"K5 planted fault caught: {name}", caught > 0, f"{caught} rows differ over rows 1/8/32")
    return ok_all


# ----------------------------------------------------------------------------------------------- K6 source checks
EXPECTED_CONTINUATIONS = ("row_keys, top_logprobs)",
                          "logits, group, rank, vocabulary_size, temperatures, row_keys, filters, top_logprobs, candidates)")


def k6(S, src_orig: bytes, scratch: Path):
    new = (PKG / "sampling.py").read_bytes()
    a, b = src_orig.decode().splitlines(), new.decode().splitlines()
    diff = [d for d in difflib.unified_diff(a, b, lineterm="", n=0) if d[:1] in "+-" and d[:3] not in ("+++", "---")]
    removed = [d for d in diff if d.startswith("-")]
    added = [d[1:] for d in diff if d.startswith("+")]
    guarded = all("RLKIT" in x or "rlkit" in x or not x.strip() or x.strip() in EXPECTED_CONTINUATIONS for x in added)
    ok1 = not removed and guarded and len(added) <= 14 and b"\r" not in new
    for d in diff:
        print("   diff " + d)
    check("K6a this repository's sampling.py = released + the two guarded branches only", ok1,
          f"removed {len(removed)}, added {len(added)}")

    # every function but the two patched ones keeps the released bytecode
    os.environ.pop(SWITCH, None)
    P = sys.modules.get("sglang_neuron.sampling_switch_None") or _load("sglang_neuron.sampling_switch_None",
                                                                      scratch / "sglang_neuron" / "sampling_this_repo.py")
    patched = {"sample_sharded_logits", "sample_filtered_sharded_logits"}
    differ = [n for n, f in vars(S).items() if isinstance(f, types.FunctionType) and f.__module__ == S.__name__
              and n not in patched and getattr(P, n).__code__.co_code != f.__code__.co_code]
    check("K6b every other function of sampling.py has the released bytecode", not differ, f"differ: {differ}")

    # the kernel path's tail is the released text verbatim
    released = src_orig.decode()
    start = released.index("    payload = [value, token.float(), raw, lse, tempered_lse, pool_vals,")
    end = released.index("    return result\n", start) + len("    return result\n")
    tail = released[start:end]
    body = (PKG / "rlkit_sampler_topk.py").read_text(encoding="utf-8")
    return check("K6c kernel path tail = released (v0.1.0) sample_filtered_sharded_logits lines 379-408, verbatim",
                 tail in body,
                 f"{tail.count(chr(10))} lines")


# ----------------------------------------------------------------------------------------------- K7 cache digest
def k7(repo: Path):
    cc = _load("rlkit_compile_cache_probe", repo / "sglang_neuron" / "backend" / "compile_cache.py")
    names = {str(p.relative_to(repo)).replace(os.sep, "/") for p in cc.kernel_sources(repo)}
    want = {"sglang_neuron/rlkit_sampler_topk.py", "sglang_neuron/rlkit_sampler_kernel.py", "sglang_neuron/sampling.py"}
    return check("K7 compile_cache.kernel_sources counts rlkit_sampler_topk.py, rlkit_sampler_kernel.py, sampling.py",
                 want <= names, f"missing {sorted(want - names)}; digest {cc.kernel_sources_digest(repo, ['probe'])}")


# ----------------------------------------------------------------------------------------------- K8 K9 edge inputs
def edge_batches(S, gen, rows_list):
    """(name, logits, temps, keys, filters, vocab): a padded vocabulary (width * world > vocab; the padded columns hold
    a large value, so a missing mask would show) and non-finite logits (-inf columns, a whole -inf shard, +inf, NaN).
    Row 0 is temperature 0, row 1 is an unrestricted sampled row, row 2 a restricted sampled row; the rest are random."""
    for rows in rows_list:
        keys = make_keys(S, rows, gen)
        temps, filters = random_settings(S, rows, gen)
        for i, (t, f) in enumerate(((0.0, (50.0, 0.95, 0.0)), (0.8, (float(S.TOP_K_ALL), 1.0, 0.0)),
                                    (0.8, (50.0, 0.95, 0.0)))):
            if i < rows:
                temps[i], filters[i] = t, torch.tensor(f)
        for vocab in (VOCAB - 1000, 3 * WIDTH + 100):          # last rank: 36,984 valid columns; 100 valid columns
            lg = make_logits(rows, gen)
            lg[:, vocab:] = 50.0
            yield f"padded vocabulary {vocab} (width*world {VOCAB}) rows={rows}", lg, temps, keys, filters, vocab
        lg = make_logits(rows, gen)
        lg[:, ::7] = float("-inf")
        yield f"-inf in every 7th column rows={rows}", lg, temps, keys, filters, VOCAB
        lg = make_logits(rows, gen)
        lg[:, WIDTH:2 * WIDTH] = float("-inf")
        yield f"a whole shard (rank 1) -inf rows={rows}", lg, temps, keys, filters, VOCAB
        lg = make_logits(rows, gen)
        lg[:, 100:140] = float("-inf")
        lg[0, 5000] = 20.0                                       # the greedy row's maximum is a finite value
        yield f"-inf block next to a finite maximum rows={rows}", lg, temps, keys, filters, VOCAB
        lg = make_logits(rows, gen)
        lg[3 % rows, 5000] = float("inf")
        yield f"+inf in one column of one row rows={rows}", lg, temps, keys, filters, VOCAB
        lg = make_logits(rows, gen)
        lg[3 % rows, 777] = float("nan")
        yield f"NaN in one column of one row rows={rows}", lg, temps, keys, filters, VOCAB


def k8(S, K, T):
    """CPU reference stage, whole output bit-equal (NaN compared by bits), for every edge batch."""
    gen = torch.Generator().manual_seed(2718)
    ok_all, info = True, []
    for name, logits, temps, keys, filters, vocab in edge_batches(S, gen, (1, 8, 32)):
        _, _, bad = compare(S, K, T, logits, temps, keys, filters, top_logprobs=20, vocab=vocab)
        info.append(f"{name}: {int(bad.sum())} rows differ")
        ok_all &= not bad.any()
        if vocab != VOCAB and logits.shape[0] == 8:       # the padded columns are visible: without the mask, tokens move
            e = B.run_tp(released_fn(S, temps, keys, filters, 0, vocab), logits, temps, keys)
            u = B.run_tp(released_fn(S, temps, keys, filters, 0, VOCAB), logits, temps, keys)
            moved = int(row_mismatch(e, u).sum())
            info.append(f"{name}: unmasked run differs in {moved} rows")
            ok_all &= moved > 0 and all(int(x["token_ids"].max()) < vocab for x in e)
    return check("K8 padded vocabulary and non-finite logits: bit-equal to the released path (CPU reference stage)",
                 ok_all, "; ".join(i for i in info if not i.endswith(" 0 rows differ")) or f"{len(info)} batches")


def k9(S, K, T):
    """128 rows run; 129 rows raise a RuntimeError that names RLKIT_SAMPLER_KERNEL_TOPK (not RLKIT_SAMPLER_KERNEL)."""
    gen = torch.Generator().manual_seed(129)
    out = []
    for rows in (128, 129):
        logits, keys = make_logits(rows, gen), make_keys(S, rows, gen)
        temps, filters = torch.full((rows,), 0.8), make_filters(S, rows, 50, 0.95, 0.0)
        try:
            B.run_tp(kernel_fn(S, K, T, temps, keys, filters, 0), logits, temps, keys)
            out.append((rows, None))
        except RuntimeError as exc:
            out.append((rows, str(exc)))
    msg = out[1][1] or ""
    ok = (out[0][1] is None and "RLKIT_SAMPLER_KERNEL_TOPK" in msg and "128" in msg
          and "unset RLKIT_SAMPLER_KERNEL)" not in msg)
    return check("K9 128 rows run; 129 rows raise an error that names RLKIT_SAMPLER_KERNEL_TOPK", ok,
                 f"128 rows: {'ran' if out[0][1] is None else out[0][1]}; 129 rows: {msg or 'no error'}")


def k10(S, K, T):
    """Unpadded vocabulary (width * world == vocab): the kernel path applies the vocabulary mask, as the released path
    does. The mask changes no value there (all-false), so the outputs cannot show it; count the masked_fill calls whose
    mask is the [1, width] vocabulary mask instead."""
    gen = torch.Generator().manual_seed(1010)
    rows = 8
    logits, keys = make_logits(rows, gen), make_keys(S, rows, gen)
    temps, filters = torch.full((rows,), 0.8), make_filters(S, rows, 50, 0.95, 0.0)
    width = VOCAB // 4
    real, calls = torch.Tensor.masked_fill, []

    def counting(self, mask, value):
        if tuple(mask.shape) == (1, width):
            calls.append(1)
        return real(self, mask, value)

    counts = {}
    torch.Tensor.masked_fill = counting
    try:
        for label, fn in (("released", released_fn(S, temps, keys, filters, 0)),
                          ("kernel", kernel_fn(S, K, T, temps, keys, filters, 0))):
            calls.clear()
            B.run_tp(fn, logits, temps, keys)
            counts[label] = len(calls)
    finally:
        torch.Tensor.masked_fill = real
    ok = counts["kernel"] > 0 and counts["kernel"] == counts["released"]
    return check("K10 the vocabulary mask is applied on the kernel path with an unpadded vocabulary", ok,
                 f"vocabulary-mask masked_fill calls over the 4 ranks: released {counts['released']}, "
                 f"kernel {counts['kernel']}")


# ----------------------------------------------------------------------------------------------- SIM
def sim_tests(S, K, T):
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
    import nki
    import numpy as np

    kern = nki.simulate(nki.jit(K.gumbel_partials))
    memo = {}
    best_cols = [0, 0]                       # [sim best partials not bit-equal to the CPU reference, all compared]

    def sim_partials(x, keys, inv, gate, scale, base):
        args = [t.contiguous() for t in (x, keys, inv, gate, scale, base)]
        key = tuple(t.numpy().tobytes() for t in args)
        if key not in memo:
            got = torch.from_numpy(np.array(kern(*[t.numpy() for t in args]), dtype=np.float32))
            ref = K.gumbel_partials_reference(*args)
            best_cols[0] += int((bits(got[:, 0::K.NPART]) != bits(ref[:, 0::K.NPART])).sum())
            best_cols[1] += int(got[:, 0::K.NPART].numel())
            memo[key] = got
        return memo[key]

    gen = torch.Generator().manual_seed(21)
    cases = [("RL T0.8 k50 p0.95", 0.8, (50, 0.95, 0.0)), ("unrestricted T0.8", 0.8, (-1, 1.0, 0.0)),
             ("greedy", 0.0, (-1, 1.0, 0.0)), ("mixed", None, None)]
    ok_all = True
    for rows in (1, 8, 32):
        for name, temp, filt in cases:
            t0 = time.time()
            n = [0] * 6                      # restricted equal / all, greedy equal / all, unrestricted equal / all
            ok = True
            for _ in range(3):
                logits, keys = make_logits(rows, gen), make_keys(S, rows, gen)
                if temp is None:
                    temps, filters = random_settings(S, rows, gen)
                else:
                    temps, filters = torch.full((rows,), temp), make_filters(S, rows, *filt)
                e, k, bad = compare(S, K, T, logits, temps, keys, filters, partials=sim_partials)
                restricted, unrestricted, greedy = row_classes(S, temps, filters)
                tok_bad = None
                for a, b in zip(e, k):
                    d = a["token_ids"] != b["token_ids"]
                    tok_bad = d if tok_bad is None else tok_bad | d
                ok &= not bad[restricted].any() and not bad[greedy].any()      # greedy: token and logprob
                for i, (m, wrong) in enumerate(((restricted, bad), (greedy, bad), (unrestricted, tok_bad))):
                    n[2 * i] += int((m & ~wrong).sum())
                    n[2 * i + 1] += int(m.sum())
            ok_all &= ok
            check(f"SIM rows={rows} {name}, 3 batches", ok,
                  f"restricted rows bit-equal {n[0]}/{n[1]}, temperature-0 rows same token and logprob {n[2]}/{n[3]}, unrestricted "
                  f"sampled rows same token {n[4]}/{n[5]} (counted, not required), {time.time() - t0:.1f}s")
    for name, logits, temps, keys, filters, vocab in edge_batches(S, gen, (8,)):
        t0 = time.time()
        e, k, bad = compare(S, K, T, logits, temps, keys, filters, partials=sim_partials, vocab=vocab)
        restricted, unrestricted, greedy = row_classes(S, temps, filters)
        ok = not bad[restricted].any() and not bad[greedy].any()
        ok_all &= ok
        check(f"SIM edge: {name}", ok,
              f"restricted rows bit-equal {int((restricted & ~bad).sum())}/{int(restricted.sum())}, temperature-0 rows "
              f"same token and logprob {int((greedy & ~bad).sum())}/{int(greedy.sum())}, unrestricted sampled rows "
              f"equal {int((unrestricted & ~bad).sum())}/{int(unrestricted.sum())} (counted, not required), "
              f"{time.time() - t0:.1f}s")
    check("SIM kernel stage really is the simulator (informational)", True,
          f"best partials not bit-equal to the CPU reference: {best_cols[0]} of {best_cols[1]}")
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(DEF_REPO))
    ap.add_argument("--scratch", default=os.environ.get("RLKIT_SCRATCH",
                                                       str(Path(tempfile.gettempdir()) / "selftest_topk_scratch")))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--sim-only", action="store_true")
    a = ap.parse_args()
    global torch
    import torch  # noqa: F811
    B.torch = torch
    scratch = Path(a.scratch)
    src = build_scratch(Path(a.repo), scratch)
    S = importlib.import_module("sglang_neuron.sampling")
    K = importlib.import_module("sglang_neuron.rlkit_sampler_kernel")
    t0 = time.time()
    if not a.sim_only:
        k0(S, scratch)
    T = importlib.import_module("sglang_neuron.rlkit_sampler_topk")
    if not a.sim_only:
        k6(S, src, scratch)
        k7(Path(a.repo))
        k3(S, K, T)
        k8(S, K, T)
        k9(S, K, T)
        k10(S, K, T)
        k4(S, K, T)
        k5(S, K, T)
        k2(S, K, T, a.seeds)
        k1(S, K, T, a.seeds)
    if a.sim or a.sim_only:
        sim_tests(S, K, T)
    bad = [r for r in RESULTS if not r[1]]
    print(f"SELFTEST_SAMPLER_TOPK {'OK' if not bad else 'FAILED'}: {len(RESULTS) - len(bad)} pass, {len(bad)} fail, "
          f"{time.time() - t0:.0f}s")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
