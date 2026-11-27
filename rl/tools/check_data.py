#!/usr/bin/env python3
"""F6: session data under rl/data and the frozen patch folder under rl/patches.

    check_data.py --base REV --head REV [--repo DIR]

All bytes are read from git objects (git show), never from the working tree.
  - a new folder rl/data/<session>/ has README.md and SHA256SUMS (every line verified, every other file listed);
  - its README.md has a line 'Served: <fork commit>[, <commit> ...]' or 'Served: none'. A folder "served an engine
    patch" when that line names anything other than none; it then needs a PATCH_MANIFEST.txt (fails without one);
  - files added to a session folder that already exists at the base must be listed, with their sha256, in a sums file
    that the same pull request adds to that folder, named SHA256SUMS.<tag> (the folder's own SHA256SUMS is never
    changed: it would fail as modified); the sums file is verified like SHA256SUMS;
  - a file that exists at the base under rl/data/ or rl/patches/ is never modified or deleted;
  - when rl/data/LEGACY.sha256 exists at the head, every file it lists matches (paths relative to the
    repository root or to rl/data/);
  - a file is at most 2 MB, and a session folder that is new or gets added files is at most 10 MB in total at the head.
Exit status: 0 pass, 1 fail, 2 bad input.
"""
import argparse
import hashlib
import re
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

MAX_FILE = 2 * 1024 * 1024
MAX_FOLDER = 10 * 1024 * 1024
LINE = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(\S.*?)\s*$")
EXTRA_SUMS = re.compile(r"^SHA256SUMS\.[A-Za-z0-9_.-]+$")


def parse_sums(data):
    res = []
    for line in data.decode("utf-8", "replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        mo = LINE.match(line)
        if mo:
            res.append((mo.group(1).lower(), mo.group(2)))
    return res


def sha(data):
    return hashlib.sha256(data).hexdigest()


SERVED = re.compile(r"^\s*(?:[-*]\s+)?Served:\s*(.*?)\s*$")


def served_line(readme):
    """The text after 'Served:' on the first such line of a session README (optionally a list item), or None."""
    for line in readme.decode("utf-8", "replace").splitlines():
        mo = SERVED.match(line)
        if mo:
            return mo.group(1)
    return None


def check(base, head, repo="."):
    ok, msgs, notes = True, [], []

    def bad(m):
        nonlocal ok
        ok = False
        msgs.append(m)

    ch = common.changed(base, head, cwd=repo)
    for st, p in ch:
        if p.startswith(("rl/data/", "rl/patches/")) and st[0] != "A":
            bad("F6: %s is %s; files under rl/data and rl/patches are never modified or deleted." % (p, "deleted" if st[0] == "D" else "modified"))
    base_tree = {p for p, _ in common.ls_tree(base, "rl/data", cwd=repo)}
    base_sessions = {p.split("/")[2] for p in base_tree if p.count("/") >= 3}
    head_files = common.ls_tree(head, "rl/data", cwd=repo)
    head_sessions = sorted({p.split("/")[2] for p, _ in head_files if p.count("/") >= 3})
    new = [s for s in head_sessions if s not in base_sessions]
    added = [p for st, p in ch if st[0] == "A" and p.startswith("rl/data/")]
    for p, size in head_files:
        if p in added and size > MAX_FILE:
            bad("F6: %s is %d bytes, over the 2 MB file cap." % (p, size))
    def verify(pre, sums_name):
        """Check every line of one sums file of a folder; the set of files it lists."""
        listed = set()
        for digest, rel in parse_sums(common.show(head, pre + sums_name, cwd=repo) or b""):
            listed.add(rel)
            data = common.show(head, pre + rel, cwd=repo)
            if data is None:
                bad("F6: %s%s lists %s, which does not exist." % (pre, sums_name, rel))
            elif sha(data) != digest:
                bad("F6: %s%s does not match its %s line." % (pre, rel, sums_name))
        return listed

    def folder_cap(pre):
        total = sum(n for p, n in head_files if p.startswith(pre))
        if total > MAX_FOLDER:
            bad("F6: %s is %d bytes, over the 10 MB folder cap." % (pre, total))

    touched = sorted({p.split("/")[2] for p in added if p.count("/") >= 3 and p.split("/")[2] in base_sessions})
    for s in touched:
        pre = "rl/data/%s/" % s
        folder_cap(pre)
        new_files = sorted(p[len(pre):] for p in added if p.startswith(pre))
        sums = [r for r in new_files if EXTRA_SUMS.match(r)]
        if not sums:
            bad("F6: %s already exists at the base; the added file(s) %s need a new sums file %sSHA256SUMS.<tag> that lists them."
                % (pre, ", ".join(new_files), pre))
            continue
        listed = set()
        for r in sums:
            listed |= verify(pre, r)
        for rel in new_files:
            if rel not in listed and rel not in sums:
                bad("F6: %s%s is added to an existing folder and is not listed in %s." % (pre, rel, ", ".join(sums)))
    for s in new:
        pre = "rl/data/%s/" % s
        files = [(p, n) for p, n in head_files if p.startswith(pre)]
        folder_cap(pre)
        names = {p[len(pre):]: p for p, _ in files}
        if "README.md" not in names:
            bad("F6: new folder %s has no README.md." % pre)
        if "SHA256SUMS" not in names:
            bad("F6: new folder %s has no SHA256SUMS." % pre)
        else:
            listed = verify(pre, "SHA256SUMS")
            for rel in sorted(set(names) - listed - {"SHA256SUMS"}):
                bad("F6: %s%s is not listed in SHA256SUMS." % (pre, rel))
        if "README.md" in names:
            served = served_line(common.show(head, pre + "README.md", cwd=repo) or b"")
            if served is None:
                bad("F6: %sREADME.md has no 'Served:' line; write 'Served: <fork commit>[, <commit> ...]' (the engine "
                    "patches its servers ran) or 'Served: none'." % pre)
            elif served.lower() != "none" and not any(r.endswith("PATCH_MANIFEST.txt") for r in names):
                bad("F6: %sREADME.md says 'Served: %s', so the folder served an engine patch and needs the "
                    "PATCH_MANIFEST.txt of each patch folder (python rl/make_patch_dir.py writes it)." % (pre, served))
    legacy = common.show(head, "rl/data/LEGACY.sha256", cwd=repo)
    if legacy is None:
        notes.append("F6 note: rl/data/LEGACY.sha256 is absent at the head; imported files not compared.")
    else:
        n = 0
        for digest, rel in parse_sums(legacy):
            data = common.show(head, rel, cwd=repo)
            if data is None:
                data = common.show(head, "rl/data/" + rel, cwd=repo)
            n += 1
            if data is None:
                bad("F6: LEGACY.sha256 lists %s, which does not exist." % rel)
            elif sha(data) != digest:
                bad("F6: %s differs from LEGACY.sha256." % rel)
        notes.append("F6 note: %d LEGACY.sha256 lines checked." % n)
    if ok:
        msgs.append("F6: %d new session folder(s), %d existing folder(s) with added files, %d added data file(s)."
                    % (len(new), len(touched), len(added)))
    return ok, msgs + notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--repo", default=".")
    a = ap.parse_args(argv)
    try:
        ok, msgs = check(a.base, a.head, a.repo)
    except RuntimeError as e:
        print("check_data: %s" % e, file=sys.stderr)
        return 2
    for m in msgs:
        print(m)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
