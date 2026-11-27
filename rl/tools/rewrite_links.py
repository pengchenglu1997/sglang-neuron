#!/usr/bin/env python3
"""Rewrite every link into the former run repository to the relative rl/ path. Prints "old -> new" for each hit.

    rewrite_links.py [--apply] [--exclude PREFIX]... [--repo DIR]

The former run repository was imported complete under rl/: https://github.com/<owner>/trainium-sglang-rl/tree/main/X
(or blob/main/X, or a pinned tree/<sha>/X) becomes rl/X, and the bare repository address becomes rl/. Inside a markdown
link target the new path is relative to the file that holds it; in prose it is written from the repository root. A hit
whose target does not exist in the checkout, and a clone address (.git), is printed as UNMAPPED and left alone.
Not touched: tests/ (it travels upstream), the files of rl/data/LEGACY.sha256, and the frozen records under rl/data/,
rl/patches/ and rl/docs/, and rl/tools/ (its fixtures name the address). Without --apply nothing is written. --exclude PREFIX skips the files under a repo-relative
prefix (repeatable). Line endings and every other byte are kept. Exit status: 0 nothing left to map (or all mapped),
1 an UNMAPPED hit, 2 bad input.
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
    """(new text or None, reason)."""
    if m.group(1):
        return None, "a clone address; the checkout that holds rl/ is the repository"
    sub = (m.group(3) or "").strip("/")
    target = "rl/" + sub if sub else "rl/"
    if not os.path.exists(os.path.join(repo, *target.rstrip("/").split("/"))):
        return None, "%s does not exist in this checkout" % target
    in_link = text[max(0, m.start() - 2):m.start()] == "]("
    if in_link:
        rel = posixpath.relpath(target.rstrip("/"), posixpath.dirname(path) or ".")
        return rel + ("/" if target.endswith("/") else ""), ""
    return target, ""


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
            new, why = map_match(m, text, path, repo)
            line = text.count("\n", 0, m.start()) + 1
            if new is None:
                unmapped += 1
                hits.append("%s:%d: %s -> UNMAPPED (%s)" % (path, line, m.group(0), why))
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
    print("rewrite_links.py: %d hit(s), %d unmapped, %s" % (len(hits), unmapped, "written" if a.apply else "dry run"))
    return 1 if unmapped else 0


if __name__ == "__main__":
    sys.exit(main())
