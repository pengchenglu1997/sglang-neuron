#!/usr/bin/env python3
"""Rewrite every link into the former run repository to the relative rl/ path. Prints "old -> new" for each hit.

    rewrite_links.py [--apply] [--exclude PREFIX]... [--repo DIR]

The former run repository was imported complete under rl/: https://github.com/<owner>/trainium-sglang-rl/tree/main/X
(or blob/main/X) becomes rl/X. Inside a markdown link target the new path is relative to the file that holds it; in prose
it is written from the repository root. Only that form is rewritten. Three kinds of hit are printed and left alone:
  MANUAL    the bare repository address (a sentence that says "the public repo" is false once the scripts are in this
            repository: edit it by hand), and a link pinned to a commit (tree/<sha>/X would silently lose its pin)
  UNMAPPED  a target that does not exist in the checkout, and a clone address (.git)
Not touched: tests/ (it travels upstream), the files of rl/data/LEGACY.sha256, and the frozen records under rl/data/,
rl/patches/ and rl/docs/, and rl/tools/ (its fixtures name the address). Without --apply nothing is written. --exclude
PREFIX skips the files under a repo-relative prefix (repeatable). Line endings and every other byte are kept. Exit
status: 0 nothing is left for a person (everything was mapped), 1 a MANUAL or UNMAPPED hit remains, 2 bad input.
"""
import argparse
import os
import posixpath
import re
import subprocess
import sys

sys.dont_write_bytecode = True

OLD = re.compile(
    r"https://github\.com/[A-Za-z0-9-]+/trainium-sglang-rl(\.git)?"
    r"(?:/(?:tree|blob)/([A-Za-z0-9._-]+)/((?:[^\s)>\]`\"'#.,;:]|[.,;:](?=[^\s)>\]`\"'#.,;:]))*))?/?")
FROZEN_PREFIXES = ("rl/data/", "rl/patches/", "rl/docs/", "rl/tools/")   # rl/tools/: its fixtures name the old address


def tracked(repo):
    r = subprocess.run(["git", "-C", repo, "ls-files", "-z"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise SystemExit("rewrite_links: git ls-files failed: " + r.stderr.decode("utf-8", "replace"))
    return [p for p in r.stdout.decode("utf-8", "surrogateescape").split("\0") if p]


def legacy_paths(repo):
    out = set()
    try:
        with open(os.path.join(repo, "rl", "data", "LEGACY.sha256"), "r", encoding="utf-8") as fh:
            for line in fh:
                m = re.match(r"^[0-9a-fA-F]{64}\s+\*?(\S.*?)\s*$", line)
                if m:
                    out.add(m.group(1))
    except OSError:
        pass
    return out


def skipped(path, legacy, excludes):
    return (path.startswith("tests/") or path in legacy or path.startswith(FROZEN_PREFIXES)
            or any(path.startswith(x) for x in excludes))


def map_match(m, text, path, repo):
    """(new text or None, kind, reason); kind is MANUAL or UNMAPPED when the new text is None."""
    if m.group(1):
        return None, "UNMAPPED", "a clone address; the checkout that holds rl/ is the repository"
    sub = (m.group(3) or "").strip("/")
    if not sub:
        return None, "MANUAL", "the bare repository address in running text: rewrite the sentence, the scripts are in this repository now"
    if m.group(2) != "main":
        return None, "MANUAL", "a link pinned to %s: choose between rl/%s and a permalink of this repository" % (m.group(2), sub)
    target = "rl/" + sub
    if not os.path.exists(os.path.join(repo, *target.rstrip("/").split("/"))):
        return None, "UNMAPPED", "%s does not exist in this checkout" % target
    in_link = text[max(0, m.start() - 2):m.start()] == "]("
    if in_link:
        rel = posixpath.relpath(target.rstrip("/"), posixpath.dirname(path) or ".")
        return rel + ("/" if target.endswith("/") else ""), "", ""
    return target, "", ""


def process(repo, apply, excludes):
    legacy = legacy_paths(repo)
    hits, unmapped = [], 0
    for path in tracked(repo):
        if skipped(path, legacy, excludes):
            continue
        full = os.path.join(repo, *path.split("/"))
        try:
            with open(full, "rb") as fh:
                raw = fh.read()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "trainium-sglang-rl" not in text:
            continue
        out, last = [], 0
        for m in OLD.finditer(text):
            new, kind, why = map_match(m, text, path, repo)
            line = text.count("\n", 0, m.start()) + 1
            if new is None:
                unmapped += 1
                hits.append("%s:%d: %s -> %s (%s)" % (path, line, m.group(0), kind, why))
                continue
            hits.append("%s:%d: %s -> %s" % (path, line, m.group(0), new))
            out.append(text[last:m.start()])
            out.append(new)
            last = m.end()
        if apply and out:
            out.append(text[last:])
            with open(full, "wb") as fh:
                fh.write("".join(out).encode("utf-8"))
    return hits, unmapped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--repo", default=".")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.repo):
        print("rewrite_links: no such folder %s" % a.repo, file=sys.stderr)
        return 2
    hits, unmapped = process(os.path.abspath(a.repo), a.apply, a.exclude)
    for h in hits:
        print(h)
    print("rewrite_links.py: %d hit(s), %d left for a person (MANUAL or UNMAPPED), %s" % (len(hits), unmapped, "written" if a.apply else "dry run"))
    return 1 if unmapped else 0


if __name__ == "__main__":
    sys.exit(main())
