#!/usr/bin/env python3
"""Write the two generated lists of the lean import (rl/data/NOT_IMPORTED.tsv and rl/data/LEGACY.sha256).

  python make_lists.py not-imported --run-repo RUN --ref F --dropped-list FILE --out rl/data/NOT_IMPORTED.tsv
  python make_lists.py legacy [--repo .] --out rl/data/LEGACY.sha256

not-imported: one line per dropped path, tab-separated: path, git blob id, sha256, size in bytes. The hashes are those of
  the bytes of the blob in the run repository at F (never of a working-tree file). The header line starts with "#".
legacy: "<sha256>  <path>" (readable by `sha256sum -c`, run from the repository root) for every file under rl/data/ that is
  tracked in the index, except the two generated lists and rl/data/README.md. The bytes are read from the index, so
  line-ending settings of the working tree cannot change the result.
Both outputs are sorted by path, use LF, and depend only on their inputs. Standard library only.
"""
import argparse
import hashlib
import subprocess
import sys

NOT_IMPORTED_HEADER = "# path in the former run repository (rl/<path> here would be the same file)\tgit blob id\tsha256\tsize (bytes)\n"
SKIP_LEGACY = ("rl/data/LEGACY.sha256",)


def git(repo, *args, stdin=None):
    p = subprocess.run(["git", "-C", repo] + list(args), input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        sys.exit("git %s failed: %s" % (" ".join(args), p.stderr.decode("utf-8", "replace")))
    return p.stdout


def blob_bytes(repo, shas):
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


def not_imported(a):
    tree = {}
    for rec in git(a.run_repo, "ls-tree", "-r", "-z", a.ref).split(b"\0"):
        if rec:
            meta, path = rec.split(b"\t", 1)
            tree[path.decode("utf-8")] = meta.split()[2].decode()
    dropped = sorted(l for l in open(a.dropped_list, encoding="utf-8").read().split("\n") if l)
    data = blob_bytes(a.run_repo, [tree[p] for p in dropped])
    lines = [NOT_IMPORTED_HEADER]
    for p in dropped:
        b = data[tree[p]]
        lines.append("%s\t%s\t%s\t%d\n" % (p, tree[p], hashlib.sha256(b).hexdigest(), len(b)))
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(lines))
    print("wrote %d paths to %s" % (len(dropped), a.out))


def legacy(a):
    entries = {}
    for rec in git(a.repo, "ls-files", "-s", "-z", "rl/data").split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, sha, stage = meta.split()
        p = path.decode("utf-8")
        if p not in SKIP_LEGACY:
            entries[p] = sha.decode()
    data = blob_bytes(a.repo, entries.values())
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for p in sorted(entries):
            assert " " not in p and "\\" not in p, p
            f.write("%s  %s\n" % (hashlib.sha256(data[entries[p]]).hexdigest(), p))
    print("wrote %d files to %s" % (len(entries), a.out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("not-imported")
    n.add_argument("--run-repo", required=True)
    n.add_argument("--ref", required=True)
    n.add_argument("--dropped-list", required=True)
    n.add_argument("--out", required=True)
    n.set_defaults(fn=not_imported)
    l = sub.add_parser("legacy")
    l.add_argument("--repo", default=".")
    l.add_argument("--out", required=True)
    l.set_defaults(fn=legacy)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
