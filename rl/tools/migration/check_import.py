#!/usr/bin/env python3
"""Checks of the lean import of the former run repository (see select_import.py).

  python check_import.py --run-repo RUN --ref F --repo FORKCHECKOUT --import-commit C [--import-commit C3]
                         --dropped-list FILE [--scan-commit HEAD] [--allow-changed PATH ...]

The import now has two commits that add files under rl/ (the first one, then the one that adds the files the lean
selection had left out); the dropped list is then empty. (f) reports imported files that differ from F at the scan commit.

(a) every imported path has the same blob id under rl/ at C as in F;
(b) kept + dropped = all paths of F, with no overlap (kept = the paths imported at C);
(c) every dropped path lies under data/ and has one of the five extensions;
(d) no imported text file names a dropped path: its repository-relative path, its path relative to its
    session folder, or its base name when that base name is unique in F. Files that are raw measurement data
    (extensions .csv, .jsonl, .json, .diff) are scanned too but listed apart, because they cannot be edited;
(e) files and bytes kept and dropped, per session folder and outside data/.
Exit status 1 when (a), (b), (c) or the text part of (d) fails. Standard library only.
"""
import argparse
import subprocess
import sys

EXT = (".yaml", ".out", ".log", ".sh", ".py")
RAW = (".csv", ".jsonl", ".json", ".diff")
LEFT = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
RIGHT = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
ALNUM = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def git(repo, *args, stdin=None):
    p = subprocess.run(["git", "-C", repo] + list(args), input=stdin, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if p.returncode != 0:
        sys.exit("git %s failed: %s" % (" ".join(args), p.stderr.decode("utf-8", "replace")))
    return p.stdout


def ls(repo, ref, prefix=None):
    args = ["ls-tree", "-r", "-l", "-z", ref] + ([prefix] if prefix else [])
    res = {}
    for rec in git(repo, *args).split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, typ, sha, size = meta.split()
        res[path.decode("utf-8", "surrogateescape")] = (sha.decode(), int(size))
    return res


def blobs(repo, shas):
    shas = sorted(set(shas))
    out = git(repo, "cat-file", "--batch", stdin=("\n".join(shas) + "\n").encode())
    res, pos = {}, 0
    for _ in shas:
        nl = out.index(b"\n", pos)
        head = out[pos:nl].split()
        size = int(head[2])
        res[head[0].decode()] = out[nl + 1:nl + 1 + size]
        pos = nl + 1 + size + 1
    return res


def has_name(text, needle):
    start = 0
    while True:
        i = text.find(needle, start)
        if i < 0:
            return False
        start = i + 1
        if i > 0 and text[i - 1] in LEFT:
            continue
        j = i + len(needle)
        if j < len(text):
            c = text[j]
            if c in RIGHT:
                continue
            if c == "." and j + 1 < len(text) and text[j + 1] in ALNUM:
                continue
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-repo", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--import-commit", required=True, action="append",
                    help="a commit that adds imported files under rl/ (repeat for a second one)")
    ap.add_argument("--allow-changed", action="append", default=[],
                    help="a path relative to rl/ that is meant to differ from F at the scan commit (repeat)")
    ap.add_argument("--dropped-list", required=True)
    ap.add_argument("--show-hits", type=int, default=40)
    ap.add_argument("--scan-commit", help="commit whose files check (d) reads (default: the import commit)")
    a = ap.parse_args()
    bad = 0

    f = ls(a.run_repo, a.ref)
    # the files the import commit added under rl/ (the first parent has no rl/ import)
    imp = {}
    for commit in a.import_commit:
        added = git(a.repo, "diff-tree", "-r", "--name-only", "--diff-filter=A", "--no-commit-id", "-z",
                    commit + "~1", commit).split(b"\0")
        after = ls(a.repo, commit, "rl")
        for raw in added:
            if not raw:
                continue
            p = raw.decode("utf-8", "surrogateescape")
            assert p.startswith("rl/"), p
            imp[p[3:]] = after[p]
    dropped = [l for l in open(a.dropped_list, "rb").read().decode("utf-8").split("\n") if l]

    # (a)
    diff = [p for p in imp if p not in f or imp[p][0] != f[p][0]]
    print("(a) imported paths: %d, blob id differs from F or path not in F: %d" % (len(imp), len(diff)))
    bad += len(diff)
    # (b)
    kept, dset = set(imp), set(dropped)
    overlap = kept & dset
    union = kept | dset
    missing = set(f) - union
    extra = union - set(f)
    print("(b) kept %d + dropped %d = %d, paths of F %d; overlap %d, in F but neither %d, in neither F nor list %d"
          % (len(kept), len(dset), len(kept) + len(dset), len(f), len(overlap), len(missing), len(extra)))
    bad += len(overlap) + len(missing) + len(extra)
    # (c)
    wrong = [p for p in dropped if not (p.startswith("data/") and p.endswith(EXT))]
    print("(c) dropped paths not under data/ or without one of the five extensions: %d" % len(wrong))
    bad += len(wrong)

    # (d)
    # the files are read at --scan-commit (default: the import commit), so that edits of commit 2 are scanned too
    scan = ls(a.repo, a.scan_commit or a.import_commit[-1], "rl")
    # (f) an imported path whose blob at the scan commit is not the blob of F (apart from the named exceptions)
    changed = sorted(p for p in imp if "rl/" + p in scan and scan["rl/" + p][0] != f[p][0])
    unexpected = [p for p in changed if p not in a.allow_changed]
    print("(f) imported paths that differ from F at the scan commit: %d (allowed %d, unexpected %d)"
          % (len(changed), len(changed) - len(unexpected), len(unexpected)))
    for p in unexpected:
        print("    %s" % p)
    bad += len(unexpected)
    want = [p for p in imp if "rl/" + p in scan]
    gone = [p for p in imp if "rl/" + p not in scan]
    if gone:
        print("(d) imported paths missing at the scan commit: %d" % len(gone))
        bad += len(gone)
    bl = blobs(a.repo, [scan["rl/" + p][0] for p in want])
    imp = dict((p, scan["rl/" + p]) if p in want else (p, imp[p]) for p in imp)
    base_count = {}
    for p in f:
        b = p.rsplit("/", 1)[-1]
        base_count[b] = base_count.get(b, 0) + 1
    needles = []
    for d in sorted(dropped):
        parts = d.split("/")
        forms = [d, "/".join(parts[2:])]
        if base_count[parts[-1]] == 1:
            forms.append(parts[-1])
        needles.append((d, sorted(set(forms), key=len, reverse=True)))
    hits_text, hits_raw = [], []
    for p in sorted(want):
        data = bl[imp[p][0]]
        if b"\0" in data[:4096]:
            continue
        try:
            t = data.decode("utf-8")
        except UnicodeDecodeError:
            t = data.decode("utf-8", "replace")
        for d, forms in needles:
            for form in forms:
                if has_name(t, form):
                    (hits_raw if p.endswith(RAW) else hits_text).append((p, d, form))
                    break
    print("(d) text files naming a dropped path: %d hits" % len(hits_text))
    for p, d, form in hits_text[:a.show_hits]:
        print("    %s names %s (as %s)" % (p, d, form))
    print("(d) raw measurement files (csv, jsonl, json, diff; not editable) naming a dropped path: %d hits in %d files"
          % (len(hits_raw), len(set(h[0] for h in hits_raw))))
    for p, d, form in hits_raw[:a.show_hits]:
        print("    %s names %s (as %s)" % (p, d, form))
    bad += len(hits_text)

    # (e)
    def bucket(p):
        parts = p.split("/")
        return parts[1] if p.startswith("data/") and len(parts) > 2 else "(outside data/)"
    rows = {}
    for p in kept:
        r = rows.setdefault(bucket(p), [0, 0, 0, 0])
        r[0] += 1
        r[1] += f[p][1]
    for p in dset:
        r = rows.setdefault(bucket(p), [0, 0, 0, 0])
        r[2] += 1
        r[3] += f[p][1]
    print("(e) bucket\tkept files\tkept bytes\tdropped files\tdropped bytes")
    tot = [0, 0, 0, 0]
    for k in sorted(rows):
        print("    %s\t%d\t%d\t%d\t%d" % ((k,) + tuple(rows[k])))
        tot = [x + y for x, y in zip(tot, rows[k])]
    print("    total\t%d\t%d\t%d\t%d" % tuple(tot))
    print("RESULT %s" % ("FAIL" if bad else "OK"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
