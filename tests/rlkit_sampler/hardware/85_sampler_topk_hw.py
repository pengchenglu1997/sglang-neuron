#!/usr/bin/env python3
"""Hardware check of the topk_topp sampler kernel path (sglang_neuron/rlkit_sampler_topk.py, switch
RLKIT_SAMPLER_KERNEL_TOPK=1) against the released topk_topp sampler, Qwen3-0.6B TP4. NOT RUN YET: written with the
CPU and simulator tests (selftest_sampler_topk.py), for the paired hardware issue. Needs the launcher kit of the
companion repository, as 82_sampler_kernel_hw.py (see README.md in the parent folder); it imports that script.

    python 85_sampler_topk_hw.py --results DIR --tag TAG [--model Qwen3-0.6B] [--only-correctness] [--skip A2]
                                 [--ready-cap 3000] [--dry-run]

Servers (srv_sglang_neuron.sh through server_ctl; containers samplertk_<tag>_A/A2/B), SGLANG_NEURON_SAMPLER=topk_topp,
top logprobs 0, flags FLAGS of 82_sampler_kernel_hw.py:
  A   release image, cores 20-23, port 30095, cache $DATA_ROOT/sgl_cache_v0.1.0_samplertk_A
  A2  release image, cores 32-35, port 30097, cache $DATA_ROOT/sgl_cache_v0.1.0_samplertk_A2 (control: two released
      servers that should agree)
  B   + SGL_PATCH_DIR=patches/sglang_neuron_v0.1.0_sampler_kernel_topk (sampling.py, rlkit_sampler_kernel.py,
      rlkit_sampler_topk.py), SGL_ENV=RLKIT_SAMPLER_KERNEL_TOPK=1, cores 24-27, port 30096, cache
      $DATA_ROOT/sgl_cache_v0.1.0_sampler_kernel_topk (its own: the kernel sources are in the cache digest)
Watch first: whether B becomes ready at all. In topk_topp mode the decode graph also holds the rotational top-k, and
the engine records that another GpSimd top-k hung when co-resident with vector work in one NEFF; B's graph holds the
kernel and the rotational top-k together, which no run has tried. A B that is not ready is recorded with its log tail.
Checks (A and B up; each pair also A against A2 when A2 is up):
  (i)   temperature 0, 20 prompts, 48 tokens: same tokens (pass: B equals A on every prompt)
  (ii)  filtered sampled rows, the RL setting (T 0.8, top-k 50, top-p 0.95) and (T 0.8, top-p 0.5, min-p 0.1), 20
        prompts x 3 seeds: same tokens (pass: B equals A on every request). By the design these rows take the released
        code on the same inputs, so a difference here points at the device (another graph, other float bits), not at
        the kernel's noise
  (iii) unfiltered sampled rows (T 0.8, top-k -1, top-p 1, min-p 0), same seeds: counted only (the kernel's noise uses
        the device log; the temperature kernel gave 44 of 60 equal sequences on hardware)
  (iv)  "topk_topp sampler by the kernel" in B's log, absent from A's (pass)
  (v)   8 and 32 concurrent requests in the RL setting, same seeds, information only (batching may differ)
  (vi)  unless --only-correctness: llmbench as 82 (A, B, A, B); its sampling parameters are the bench script's own and
        were not read for this script, so which rows it makes restricted is UNKNOWN
Result: <results>/85_sampler_topk_<tag>.jsonl ending {"done": true}, and 85_sampler_topk_<tag>.md for the bench table.
Last stdout line SAMPLERTK_DONE ... or SAMPLERTK_FAILED .... Every server is stopped in finally.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(modname, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load("hw82", "82_sampler_kernel_hw.py")
O, rc, server_ctl = H.O, H.rc, H.server_ctl

PATCH = HERE / "patches" / "sglang_neuron_v0.1.0_sampler_kernel_topk"
PATCH_FILES = ("sampling.py", "rlkit_sampler_kernel.py", "rlkit_sampler_topk.py")
SWITCH = "RLKIT_SAMPLER_KERNEL_TOPK=1"
SERVERS = {
    "A": {"cores": "20-23", "port": 30095, "patch": False, "cache": "sgl_cache_v0.1.0_samplertk_A"},
    "B": {"cores": "24-27", "port": 30096, "patch": True, "cache": "sgl_cache_v0.1.0_sampler_kernel_topk"},
    "A2": {"cores": "32-35", "port": 30097, "patch": False, "cache": "sgl_cache_v0.1.0_samplertk_A2"},
}
H.SERVERS.update(SERVERS)                      # H.bench and the dry-run print read ports from here
KERNEL_LINE = "topk_topp sampler by the kernel"    # rlkit_sampler_topk._log_first_use
SEEDS = [11, 222, 3333]
FILTERED = {"rl": {"temperature": 0.8, "top_k": 50, "top_p": 0.95, "min_p": 0.0},
            "minp": {"temperature": 0.8, "top_k": -1, "top_p": 0.5, "min_p": 0.1}}
UNFILTERED = {"temperature": 0.8, "top_k": -1, "top_p": 1.0, "min_p": 0.0}
CONC_SIZES = (8, 32)
BENCH_ORDER = ["A", "B", "A", "B"]


def server_env(key, data_root):
    s = SERVERS[key]
    return {"SGL_CACHE": str(Path(data_root) / s["cache"]), "SGL_CORES": s["cores"], "SGL_SAMPLER": "topk_topp",
            "SGL_EXTRA": H.FLAGS, "SGL_SCRIPT": O.SCRIPTS["Qwen3-0.6B"], "SGL_TOP_LOGPROBS": "",
            "SGL_PATCH_DIR": str(PATCH) if s["patch"] else "", "SGL_ENV": SWITCH if s["patch"] else ""}


def chat(srv, prompt, params, seed=None, max_tokens=H.MAX_TOKENS):
    body = {"model": srv.served_name(), "messages": O.C.messages_for(prompt), "max_tokens": max_tokens,
            "logprobs": True, "chat_template_kwargs": rc.THINK_OFF, **params}
    if seed is not None:
        body["seed"] = seed
    code, out, _ = rc.post(srv.base_url + "/v1/chat/completions", body, 300)
    if code != 200 or not isinstance(out, dict) or not out.get("choices"):
        return {"ok": False, "status": code, "error": str(out)[:300]}
    ch = out["choices"][0]
    content = ((ch.get("logprobs") or {}).get("content")) or []
    return {"ok": True, "text": (ch.get("message") or {}).get("content"),
            "tokens": [c.get("token") for c in content], "logprobs": [c.get("logprob") for c in content]}


def pairs(ev, event, up, params, seeds):
    """Every prompt (x seed) on A, B and A2 (when up); writes B vs A and A2 vs A. Returns (same, of) of B vs A."""
    jobs = [(q, None) for q in H.PROMPTS] if seeds is None else [(q, s) for q in H.PROMPTS for s in seeds]
    outs = {k: [chat(up[k], q, params, s) for q, s in jobs] for k in ("A", "B", "A2") if k in up}
    result = None
    for k in ("B", "A2"):
        if k not in outs:
            continue
        cmp_ = [H.compare(a, b) for a, b in zip(outs["A"], outs[k])]
        same = sum(1 for c in cmp_ if c["same_tokens"])
        d = [c["max_abs_dlogprob"] for c in cmp_ if c["max_abs_dlogprob"] is not None]
        ev.write(event=event, pair=f"A vs {k}", params=params, same_tokens=same, of=len(cmp_),
                 errors=sum(1 for c in cmp_ if not c["n"]), first_diffs=[c["first_diff"] for c in cmp_
                                                                         if not c["same_tokens"]][:30],
                 max_abs_dlogprob_before_fork=max(d) if d else None)
        if k == "B":
            result = (same, len(cmp_), all(c["n"] for c in cmp_))
    return result


def correctness(ev, up):
    ok = True
    s, n, full = pairs(ev, "check_i_temperature0", up, {"temperature": 0.0}, None)
    p1 = full and s == n
    ev.write(event="check_i", ok=p1)
    ok &= p1
    # seed effective on A?
    rl = FILTERED["rl"]
    x1, x2, x3 = (chat(up["A"], H.PROMPTS[5], rl, seed=sd) for sd in (SEEDS[0], SEEDS[0], SEEDS[1]))
    effective = all(x["ok"] for x in (x1, x2, x3)) and x1["tokens"] == x2["tokens"] and x1["tokens"] != x3["tokens"]
    ev.write(event="seed_check", effective=effective)
    p2 = effective
    for name, params in FILTERED.items():
        s, n, full = pairs(ev, f"check_ii_filtered_{name}", up, params, SEEDS)
        p2 &= full and s == n
    ev.write(event="check_ii", ok=p2, inconclusive=not effective)
    ok &= p2
    pairs(ev, "check_iii_unfiltered", up, UNFILTERED, SEEDS)           # counted only
    la, lb = H.docker_logs(up["A"]), H.docker_logs(up["B"])
    lines = [x for x in lb.splitlines() if KERNEL_LINE in x]
    p4 = len(lines) >= 1 and KERNEL_LINE not in la
    ev.write(event="check_iv", ok=p4, b_lines=len(lines), a_has_line=KERNEL_LINE in la,
             b_distinct_pids=len(set(re.findall(r"pid=(\d+)", "\n".join(lines)))),
             b_rows=sorted(set(re.findall(r"rows=(\d+)", "\n".join(lines)))), b_examples=[x[-300:] for x in lines[:4]])
    ok &= p4
    conc = {}
    for size in CONC_SIZES:
        prompts = [f"Write the numbers from {100 + i} to {110 + i} separated by commas." for i in range(size)]
        outs = {}
        for key in ("A", "B"):
            with ThreadPoolExecutor(max_workers=size) as ex:
                outs[key] = list(ex.map(lambda q, s=up[key]: chat(s, q, rl, SEEDS[0]), prompts))
        cmp_ = [H.compare(a, b) for a, b in zip(outs["A"], outs["B"])]
        conc[size] = sum(1 for c in cmp_ if c["same_tokens"])
        ev.write(event="check_v", n=size, same_tokens=conc[size], of=size, note="information only")
    return ok, {"seed_inconclusive": not effective, "conc_same": conc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="Qwen3-0.6B")
    ap.add_argument("--ready-cap", type=float, default=3000.0)
    ap.add_argument("--only-correctness", action="store_true")
    ap.add_argument("--skip", default="", help="comma list of servers not to start (A2 only makes sense)")
    ap.add_argument("--backends-json", default=str(HERE / "backends.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not re.match(r"^[A-Za-z0-9_.-]+$", a.tag):
        print("SAMPLERTK_FAILED bad tag")
        return 2
    data_root = os.environ.get("DATA_ROOT") or str(Path.home() / "w")
    models_root = os.environ.get("MODELS_ROOT") or str(Path(data_root) / "models")
    os.environ["MODELS_ROOT"] = models_root
    a.model_dir = str(Path(models_root) / a.model)
    a.results = Path(a.results)
    keys = [k for k in ("A", "B", "A2") if k not in a.skip.split(",")]
    names = {k: f"samplertk_{a.tag}_{k}" for k in keys}
    backend = dict(server_ctl.load_backend(a.backends_json, O.BACKEND))
    backend["ready_timeout_s"] = a.ready_cap
    if a.dry_run:
        print(f"DRY RUN: nothing is started. model {a.model_dir} results {a.results} patch {PATCH}")
        for k in keys:
            argv = [str(x).format(model=a.model_dir, port=SERVERS[k]["port"], name=names[k]) for x in backend["start"]]
            print(f"[{k}] " + " ".join(f"{n}='{v}'" for n, v in server_env(k, data_root).items()) + " " + " ".join(argv))
        print("SAMPLERTK_DONE dry-run")
        return 0
    if not (Path(a.model_dir) / "config.json").is_file():
        print(f"SAMPLERTK_FAILED no {a.model_dir}/config.json")
        return 2
    missing = [f for f in PATCH_FILES if not (PATCH / f).is_file()]
    if missing:
        print(f"SAMPLERTK_FAILED patch folder {PATCH} lacks {missing}")
        return 2
    a.results.mkdir(parents=True, exist_ok=True)
    ev = H.Events(a.results / f"85_sampler_topk_{a.tag}.jsonl")
    ev.write(event="open", tag=a.tag, servers={k: dict(SERVERS[k], env=server_env(k, data_root)) for k in keys},
             flags=H.FLAGS, patch=str(PATCH))
    servers = {k: server_ctl.Server(backend, SERVERS[k]["port"], a.results / "server_logs", name=names[k],
                                    env=server_env(k, data_root)) for k in keys}
    up = {}
    status, code = "error", 1
    info = {"seed_inconclusive": False, "conc_same": {}}

    def on_signal(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is not None:
            signal.signal(sig, on_signal)
    try:
        O.force_remove(list(names.values()))
        res = O.parallel([lambda s=servers[k]: s.start(a.model_dir) for k in keys])
        for k, r in zip(keys, res):
            if isinstance(r, dict) and r.get("ready_s") is not None:
                up[k] = servers[k]
                ev.write(event="server_ready", server=k, **r)
            else:
                log = r.get("log") if isinstance(r, dict) else None
                ev.write(event="not_ready", server=k, result=str(r)[:300], last_lines=O.tail_lines(log, 25) if log else [])
        ok = False
        if "A" in up and "B" in up:
            ok, info = correctness(ev, up)
        else:
            ev.write(event="correctness_skipped", reason="A or B not up", up=sorted(up))
        table = []
        if not a.only_correctness:
            count = {}
            for k in BENCH_ORDER:
                if k in up:
                    count[k] = count.get(k, 0) + 1
                    table.append((k, count[k], H.bench(ev, a, k, up[k], count[k])))
            H.write_table(a.results / f"85_sampler_topk_{a.tag}.md", table)
            ev.write(event="table", rows=[{"server": k, "pass": n, "tps": d} for k, n, d in table])
        code = 0 if ok and (a.only_correctness or (table and all(d for _, _, d in table))) else 1
        status = "ok" if code == 0 else "failed"
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as e:  # noqa: BLE001
        status = "error"
        ev.write(event="exception", error=f"{type(e).__name__}: {e}"[:300])
    finally:
        for k, s in servers.items():
            try:
                ev.write(event="stop", server=k, seconds=s.stop())
            except Exception as e:  # noqa: BLE001
                ev.write(event="stop_error", server=k, error=str(e)[:200])
        O.force_remove(list(names.values()))
        ev.write(event="end", status=status, up=sorted(up))
        ev.write(done=True)
    print(("SAMPLERTK_DONE " if code == 0 else "SAMPLERTK_FAILED ") + f"status={status} up={','.join(sorted(up))} "
          f"seed_inconclusive={info['seed_inconclusive']} concurrent_same_tokens={info['conc_same']}")
    return code


if __name__ == "__main__":
    sys.exit(main())
