#!/usr/bin/env python3
"""Pull request checks F1 (label from base and paths) and F2 (hardware evidence).

    check_pr.py f1 --base BASE (--labels a,b | --labels-json '["a","b"]') --paths-file FILE|- [--head REV --main REV [--repo DIR]]
    check_pr.py f2 (--labels a,b | --labels-json JSON) --paths-file FILE|- --body-file FILE --head REV [--main REV] [--repo DIR]
    check_pr.py classify --kind team|engine [-z] < paths  (prints the paths of that kind, one per line or NUL-separated)
    check_pr.py replay --pr N [--repo OWNER/NAME]        (F1 on a real pull request, read with gh)

F1 (fails the job):
  - a TEAM-only pull request needs no label and must target main (a label, if present, must not be both),
    or target integration as a sync from main (next point);
  - sync rule: a pull request into integration that changes a TEAM path passes only when it is a sync from main, that is,
    every TEAM path at --head has the same mode and blob id as at --main (git ls-tree of the two commits, whole trees,
    so an extra, a missing or a different team file fails). A sync that also carries ENGINE paths still needs exactly
    one label (cpu-only for the sync of the layout migration). Without --head and --main nothing counts as a sync;
  - a pull request with any ENGINE path carries exactly one of cpu-only (base integration only) or
    hardware-verified (base main or integration);
  - never both labels; a pull request with no changed path needs one label (the old rule);
  - a base other than main or integration keeps the old rule only: exactly one of the two labels.
F2 (report mode): for a hardware-verified pull request that changes ENGINE paths, the body cites
  rl/data/<session>/..., which exists at the head or on main, and every changed sglang_neuron/**/*.py has its
  sha256 in a PATCH_MANIFEST.txt under that folder.
Paths come NUL- or newline-separated. Exit status: 0 pass, 1 fail, 2 bad input.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

HW, CPU = "hardware-verified", "cpu-only"


def team_tree(rev, repo="."):
    """{TEAM path: (mode, blob id)} of every TEAM path at rev (git ls-tree -r of the whole tree)."""
    out = common.git(["ls-tree", "-r", "-z", rev], cwd=repo)
    res = {}
    for rec in out.split("\0"):
        if not rec:
            continue
        meta, path = rec.split("\t", 1)
        mode, _typ, oid = meta.split()
        if common.is_team(path):
            res[path] = (mode, oid)
    return res


def sync_differences(head, main, repo="."):
    """Sorted TEAM paths whose mode or blob id differ between head and main, or that exist on one side only.
    An empty list means that head is a sync from main."""
    a, b = team_tree(head, repo), team_tree(main, repo)
    return sorted(p for p in set(a) | set(b) if a.get(p) != b.get(p))


def f1(base, labels, paths, sync=None):
    """(ok, messages). sync: None when it was not computed; otherwise the list of TEAM paths on which the head differs
    from main (an empty list: the pull request is a sync from main)."""
    ev = [l for l in labels if l in common.EVIDENCE_LABELS]
    ev_set = sorted(set(ev))
    msgs = []
    if len(ev_set) == 2:
        return False, ["F1: both cpu-only and hardware-verified are set; a pull request carries one of them."]
    if base not in ("main", "integration"):
        # Old rule only: exactly one evidence label (count duplicates the way the old check did).
        if len(ev) != 1:
            return False, ["F1: base '%s': exactly one of the labels cpu-only or hardware-verified is required (found %d)." % (base, len(ev))]
        return True, ["F1: base '%s' is outside the team rule; the label is present." % base]
    if not paths:
        if len(ev) != 1:
            return False, ["F1: no changed path and %d evidence labels; exactly one is required." % len(ev)]
        return True, ["F1: no changed path; one label present."]
    engine = [p for p in paths if not common.is_team(p)]
    team = [p for p in paths if common.is_team(p)]
    is_sync = False
    if base == "integration" and team and ev != [HW]:
        # A hardware-verified pull request keeps the old allowance: it may carry its own rl/data and docs/rl lines
        # (pull request 15 did). Every other pull request that changes a team path into integration must be a sync.
        if sync is None:
            return False, ["F1: a pull request into integration changes a team path (%s) and was not compared with main; only a sync from main may do that." % team[0]]
        if sync:
            return False, ["F1: a pull request into integration changes team paths and is not a sync from main: %d team path(s) differ from main (%s%s)."
                           % (len(sync), ", ".join(sync[:3]), ", ..." if len(sync) > 3 else "")]
        is_sync = True
    if not engine:
        if is_sync:
            return True, ["F1: team-only pull request into integration; every team path equals main (sync from main); no label needed (%s)." % (ev_set[0] if ev_set else "none given")]
        if base != "main":
            return False, ["F1: a team-only pull request (rl/, docs/rl/, README.md, AGENTS.md, CLAUDE.md, .github/, .gitattributes) must target main, not %s." % base]
        return True, ["F1: team-only pull request into main; no label needed (%s)." % (ev_set[0] if ev_set else "none given")]
    if len(ev) != 1:
        return False, ["F1: an engine path is changed (%s%s); exactly one of the labels cpu-only or hardware-verified is required (found %d)."
                       % (engine[0], " and %d more" % (len(engine) - 1) if len(engine) > 1 else "", len(ev))]
    if ev[0] == CPU and base != "integration":
        return False, ["F1: cpu-only engine changes target integration, never %s." % base]
    return True, ["F1: engine pull request into %s with label %s%s." % (base, ev[0], "; its team paths equal main (sync from main)" if is_sync else "")]


def parse_manifest(text):
    """{package-relative path: set of sha256} from PATCH_MANIFEST.txt lines '<sha256>  <path>'."""
    res = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        mo = re.match(r"^([0-9a-fA-F]{64})\s+\*?(\S.*?)\s*$", line)
        if mo:
            res.setdefault(mo.group(2), set()).add(mo.group(1).lower())
    return res


def cited_sessions(body):
    return sorted(set(re.findall(r"rl/data/([A-Za-z0-9][A-Za-z0-9_.-]*)", body or "")))


def f2(labels, paths, body, head, main, repo="."):
    """(ok, messages). Only a hardware-verified pull request with engine paths is checked."""
    if HW not in labels:
        return True, ["F2: not labelled hardware-verified; not checked."]
    engine = [p for p in paths if not common.is_team(p)]
    if not engine:
        return True, ["F2: no engine path changed; not checked."]
    sessions = cited_sessions(body)
    if not sessions:
        return False, ["F2: the body cites no rl/data/<session>/ folder; the merged text was not run; label cpu-only, base integration."]
    manifest, found = {}, []
    for s in sessions:
        prefix = "rl/data/%s" % s
        rev = None
        for cand in (head, main):
            if cand and common.ls_tree(cand, prefix, cwd=repo):
                rev = cand
                break
        if rev is None:
            return False, ["F2: cited folder %s exists neither at the head nor on main." % prefix]
        for p, _ in common.ls_tree(rev, prefix, cwd=repo):
            if p.endswith("/PATCH_MANIFEST.txt"):
                found.append(p)
                data = common.show(rev, p, cwd=repo) or b""
                for k, v in parse_manifest(data.decode("utf-8", "replace")).items():
                    manifest.setdefault(k, set()).update(v)
    msgs, ok = [], True
    pyfiles = [p for p in engine if p.startswith("sglang_neuron/") and p.endswith(".py")]
    if pyfiles and not found:
        return False, ["F2: no PATCH_MANIFEST.txt under %s; the merged text was not run; label cpu-only, base integration."
                       % ", ".join("rl/data/" + s for s in sessions)]
    for p in pyfiles:
        data = common.show(head, p, cwd=repo)
        if data is None:
            continue  # deleted by the pull request: a patch folder cannot express it
        digest = hashlib.sha256(data).hexdigest()
        key = p[len("sglang_neuron/"):]
        if digest not in manifest.get(key, ()):
            ok = False
            msgs.append("F2: %s sha256 %s is not in the cited manifest(s); the merged text was not run; label cpu-only, base integration." % (p, digest[:12]))
    if ok:
        msgs.append("F2: %d changed sglang_neuron file(s) match %d manifest(s)." % (len(pyfiles), len(found)))
    return ok, msgs


def gh_json(args):
    r = subprocess.run(["gh"] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError("gh %s failed: %s" % (" ".join(args), r.stderr.decode("utf-8", "replace").strip()))
    return json.loads(r.stdout.decode("utf-8"))


def replay(pr, repo):
    info = gh_json(["pr", "view", str(pr), "-R", repo, "--json", "baseRefName,labels,state,title"])
    files = gh_json(["api", "--paginate", "repos/%s/pulls/%s/files" % (repo, pr)])
    paths = []
    for f in files:
        paths.append(f["filename"])
        if f.get("previous_filename"):
            paths.append(f["previous_filename"])
    labels = [l["name"] for l in info["labels"]]
    return info["baseRefName"], labels, paths, info


def read_paths_arg(arg):
    if arg == "-":
        raw = sys.stdin.buffer.read()
    else:
        with open(arg, "rb") as fh:
            raw = fh.read()
    return common.split_paths(raw)


def labels_arg(s, as_json=None):
    """Label names. CI passes --labels-json (a JSON list of names): a name that holds a comma stays one name.
    --labels (comma-separated) is for typing by hand."""
    if as_json is not None:
        try:
            names = json.loads(as_json or "[]")
        except ValueError as e:
            raise RuntimeError("--labels-json is not JSON: %s" % e)
        if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
            raise RuntimeError("--labels-json must be a JSON list of strings")
        return names
    return [x.strip() for x in (s or "").replace("\n", ",").split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("f1")
    p1.add_argument("--base", required=True)
    p1.add_argument("--labels", default="")
    p1.add_argument("--labels-json")
    p1.add_argument("--paths-file", required=True)
    p1.add_argument("--head", help="the pull request head; with --main, it enables the sync rule for integration")
    p1.add_argument("--main", help="the fork's main at the time of the check (a ref or a sha)")
    p1.add_argument("--repo", default=".")
    p2 = sub.add_parser("f2")
    p2.add_argument("--labels", default="")
    p2.add_argument("--labels-json")
    p2.add_argument("--paths-file", required=True)
    p2.add_argument("--body-file", required=True)
    p2.add_argument("--head", required=True)
    p2.add_argument("--main", default="origin/main")
    p2.add_argument("--repo", default=".")
    p3 = sub.add_parser("classify")
    p3.add_argument("--kind", choices=["team", "engine"], required=True)
    p3.add_argument("-z", action="store_true", help="print NUL-separated (a name that holds a newline stays whole)")
    p4 = sub.add_parser("replay")
    p4.add_argument("--pr", type=int, required=True)
    p4.add_argument("--repo", default="pengchenglu1997/sglang-neuron")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "f1":
            paths = read_paths_arg(a.paths_file)
            sync = None
            if a.base == "integration" and a.head and a.main and any(common.is_team(p) for p in paths):
                sync = sync_differences(a.head, a.main, a.repo)
            ok, msgs = f1(a.base, labels_arg(a.labels, a.labels_json), paths, sync)
        elif a.cmd == "f2":
            body = open(a.body_file, encoding="utf-8", errors="replace").read()
            ok, msgs = f2(labels_arg(a.labels, a.labels_json), read_paths_arg(a.paths_file), body, a.head, a.main, a.repo)
        elif a.cmd == "classify":
            want_team = a.kind == "team"
            out = sys.stdout.buffer
            for p in common.split_paths(sys.stdin.buffer.read()):
                if common.is_team(p) == want_team:
                    out.write(os.fsencode(p) + (b"\0" if a.z else b"\n"))
            out.flush()
            return 0
        else:
            base, labels, paths, info = replay(a.pr, a.repo)
            ok, msgs = f1(base, labels, paths)
            msgs.insert(0, "replay #%d (%s): base=%s labels=%s paths=%d (%d team, %d engine)"
                        % (a.pr, info["state"], base, ",".join(labels) or "-", len(paths),
                           sum(common.is_team(p) for p in paths), sum(not common.is_team(p) for p in paths)))
    except (RuntimeError, OSError) as e:
        print("check_pr: %s" % e, file=sys.stderr)
        return 2
    for m in msgs:
        print(m)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
