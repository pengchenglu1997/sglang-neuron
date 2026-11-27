#!/usr/bin/env python3
"""Shared helpers of the team checks (standard library only; runs on Linux CI and on Windows Git Bash).

TEAM paths (fork) are rl/**, docs/rl/**, README.md, AGENTS.md, CLAUDE.md, .github/** and .gitattributes.
ENGINE paths are every other path, including tests/, docs/ outside docs/rl, scripts/, tools/, patches/, docker/,
examples/ and pyproject.toml. Only the root-level README.md, AGENTS.md and CLAUDE.md are TEAM files.
"""
import os
import subprocess

TEAM_PREFIXES = ("rl/", "docs/rl/", ".github/")
TEAM_FILES = frozenset(["README.md", "AGENTS.md", "CLAUDE.md", ".gitattributes"])
# Frozen records: the checks that read prose skip them, because their text cannot be edited any more.
FROZEN_PREFIXES = ("rl/data/", "rl/patches/", "rl/docs/")
EVIDENCE_LABELS = ("cpu-only", "hardware-verified")


def norm_user(path):
    """Tidy a path that a person typed (a command line, a pasted list). Never use it on a name that git reports:
    a name with a backslash, a leading or a trailing space is a different file from the one without."""
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def is_team(path):
    return path in TEAM_FILES or path.startswith(TEAM_PREFIXES)


def is_frozen(path):
    return path.startswith(FROZEN_PREFIXES)


def split_paths(raw):
    """Parse a path list exactly as given: NUL-separated when it holds a NUL byte (git diff -z, classify -z),
    otherwise one path per line. Nothing is stripped; empty entries are dropped. A name that holds a newline
    can only be listed with NUL separators."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "surrogateescape")
    parts = raw.split("\0") if "\0" in raw else raw.split("\n")
    return [p for p in parts if p != ""]


def git(args, cwd=".", binary=False, check=True):
    cmd = ["git", "-c", "core.quotepath=false"] + list(args)
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and r.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), r.stderr.decode("utf-8", "replace").strip()))
    if binary:
        return r.stdout if r.returncode == 0 else None
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None


def show(rev, path, cwd="."):
    """Bytes of path at rev, or None when it does not exist there."""
    return git(["show", "%s:%s" % (rev, path)], cwd=cwd, binary=True, check=False)


def exists_at(rev, path, cwd="."):
    return git(["cat-file", "-e", "%s:%s" % (rev, path)], cwd=cwd, check=False) is not None


def ls_tree(rev, prefix=None, cwd="."):
    """[(path, size)] of every blob at rev under prefix ([] when there is none)."""
    args = ["ls-tree", "-r", "-l", "-z", rev]
    if prefix:
        args += ["--", prefix]
    out = git(args, cwd=cwd, check=False)
    res = []
    for rec in (out or "").split("\0"):
        if not rec:
            continue
        meta, path = rec.split("\t", 1)
        size = meta.split()[-1]
        res.append((path, int(size) if size.isdigit() else 0))
    return res


def changed(base, head, cwd="."):
    """[(status, path)] between base...head, renames split into a delete and an add."""
    out = git(["diff", "--name-status", "-z", "--no-renames", "%s...%s" % (base, head)], cwd=cwd)
    toks = [t for t in out.split("\0") if t != ""]
    return [(toks[i], toks[i + 1]) for i in range(0, len(toks) - 1, 2)]


def is_text(data):
    return b"\0" not in data[:8192]


def report(name, ok, msgs, out):
    out.write("%s %s\n" % (name, "ok" if ok else "FAIL"))
    for m in msgs:
        out.write("  %s\n" % m)


def root_of(path_arg):
    return os.path.abspath(path_arg or ".")
