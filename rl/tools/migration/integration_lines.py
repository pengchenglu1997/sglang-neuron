#!/usr/bin/env python3
"""Every line that exists only on integration in docs/rl: is it on main now, or listed as dropped?

    integration_lines.py [--main REV] [--integration REV] [--head REV] [--scope PATH]... [--repo DIR]

A line is "integration only" when it is added by integration against main in a file of the scope (git diff of the two
commits). It is "on main" when its text, with white space collapsed, occurs in the text of the same file at --head (the
working tree when --head is not given), so a re-wrapped line still counts. Every other line is listed as dropped, with a
reason read from its text: a state paragraph ("Changed <date>", "as of", "not yet"), a quantity of a CPU or simulator
test (cut by the public-text rule), "reworded in place" (a line of the same file shares most of its words), or "cut". The script reads; it never writes.
Exit status: 0 listed, 2 bad input.
"""
import argparse
import os
import re
import subprocess
import sys

sys.dont_write_bytecode = True

STATE_RX = re.compile(r"(?i)Changed 20\d\d-\d\d-\d\d|\bas of 20\d\d-|not yet in any repo|\bcurrently\b|\bfor now\b")
CPU_RX = re.compile(r"(?i)\b(?:cpu|simulator|laptop)")


def git(repo, *args):
    r = subprocess.run(["git", "-c", "core.quotepath=false", "-C", repo] + list(args), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise SystemExit("integration_lines: git %s failed: %s" % (" ".join(args), r.stderr.decode("utf-8", "replace").strip()))
    return r.stdout


def collapse(text):
    return " ".join(text.split())


def head_text(repo, head, path):
    if head is None:
        full = os.path.join(repo, *path.split("/"))
        if not os.path.isfile(full):
            return None
        with open(full, "rb") as fh:
            data = fh.read()
    else:
        r = subprocess.run(["git", "-C", repo, "show", "%s:%s" % (head, path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            return None
        data = r.stdout
    return data.decode("utf-8", "replace")


def added_lines(repo, main, integration, scopes):
    """{path: [(line number on integration, text)]}"""
    out = git(repo, "diff", "--no-renames", "-U0", "%s..%s" % (main, integration), "--", *scopes).decode("utf-8", "replace")
    res, path, cur = {}, None, 0
    for line in out.split("\n"):
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
            continue
        mo = re.match(r"^@@ -\S+ \+(\d+)(?:,\d+)? @@", line)
        if mo:
            cur = int(mo.group(1))
            continue
        if line.startswith("+") and not line.startswith("+++") and path:
            res.setdefault(path, []).append((cur, line[1:]))
            cur += 1
    return res


def words_of(s):
    return set(re.findall(r"[A-Za-z0-9_.`-]+", s.lower()))


def nearest(line, text):
    """(similarity, line number) of the line of TEXT that shares most words with LINE (Jaccard over words)."""
    words = words_of(line)
    best = (0.0, 0)
    if not words or text is None:
        return best
    for n, other in enumerate(text.split("\n"), 1):
        ow = words_of(other)
        if ow:
            s = len(words & ow) / float(len(words | ow))
            if s > best[0]:
                best = (s, n)
    return best


def reason(line, text=None):
    if STATE_RX.search(line):
        return "state"
    if CPU_RX.search(line) and re.search(r"\d", re.sub(r"`[^`]*`", "", line)):
        return "cpu or simulator quantity"
    sim, n = nearest(line, text)
    if sim >= 0.5:
        return "reworded in place (nearest line %d)" % n
    return "cut"


def run(repo, main, integration, head, scopes, out=sys.stdout):
    added = added_lines(repo, main, integration, scopes)
    total = kept = 0
    dropped = []
    for path, lines in sorted(added.items()):
        text = head_text(repo, head, path)
        flat = collapse(text) if text is not None else None
        for ln, line in lines:
            if not line.strip():
                continue
            total += 1
            if flat is not None and collapse(line) in flat:
                kept += 1
            else:
                dropped.append((path, ln, line, text))
    out.write("integration_lines: %s..%s, head %s, scope %s\n" % (main, integration, head or "working tree", " ".join(scopes)))
    out.write("lines only on integration: %d; on main now: %d; dropped: %d\n" % (total, kept, len(dropped)))
    counts = {}
    reasons = [reason(line, text) for _p, _ln, line, text in dropped]
    for r in reasons:
        key = r.split(" (")[0]
        counts[key] = counts.get(key, 0) + 1
    out.write("dropped by reason: %s\n\n" % ", ".join("%s %d" % kv for kv in sorted(counts.items())))
    for (path, ln, line, _t), r in zip(dropped, reasons):
        out.write("- %s:%d [%s] %s\n" % (path, ln, r, collapse(line)[:260]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default="origin/main")
    ap.add_argument("--integration", default="origin/integration")
    ap.add_argument("--head", default=None)
    ap.add_argument("--scope", action="append")
    ap.add_argument("--repo", default=".")
    a = ap.parse_args(argv)
    return run(os.path.abspath(a.repo), a.main, a.integration, a.head, a.scope or ["docs/rl"])


if __name__ == "__main__":
    sys.exit(main())
