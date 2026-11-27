#!/usr/bin/env python3
"""Where did every numeral that a documentation pull request removed go?  (Migration safety net; standard library only.)

    numbers_moved.py [--base REV] [--head REV] [--scope PATH]... [--repo DIR] [--audit] [--list-moved]

SCOPE (default: README.md and docs/rl): the files whose text is compared, at --base and at --head (default: the
working tree of --repo, so a change that is not committed yet is read as it is). A numeral is REMOVED when it occurs
more often in the scope at the base than in the scope at the head. Everything is compared as text, so a number that
moved from one file of the scope to another is not removed.
What counts as a numeral: 1,234.5  0.0252  2e-6  and plain integers. Not counted: an ISO date as a whole (2026-11-24),
a digit that touches a letter or an underscore (Qwen3-0.6B, TP4, 82_sampler_kernel_r12.md, p1), and a section
number (section 8.4, sections 8 and 9, a heading number).
HOME of a removed numeral: a line of a file that holds the same number (thousands commas ignored, a longer decimal
in the data file counts when it rounds to the written one), searched in this order: a README of a session folder, tables.txt,
rl/best.tsv, another .md or .txt file under rl/data, rl/docs or rl/patches, a .csv, a .tsv, a .json, .jsonl or .yaml file;
a figure rounded to an integer (3,337) is held by the decimal that rounds to it (3337.3). A small number (an
integer below 100) has a home only on a line that also holds another numeral of the removed sentence: a "20" alone proves nothing.
A home is a line that holds the same figure, not a proof that it means the same thing: a person reads the pairs that matter.
Output: the removed numerals with their homes, and "removed, no home": each sentence of the base text that holds a
removed numeral without a home, with those numerals. The last list is for a person to decide; nothing in it is dropped
silently. --audit lists, the other way round, every numeral that the head text holds and no file under rl/data holds
(a typo, or a number derived by arithmetic: both need a look). --list-moved prints the first home of every moved numeral.
Exit status: 0 listed, 2 bad input. The script reads; it never writes.
"""
import argparse
import collections
import decimal
import os
import re
import subprocess
import sys

sys.dont_write_bytecode = True

DATE_RX = re.compile(r"(?<![0-9A-Za-z])20\d\d-\d\d-\d\d(?!\d)")
SECTION_RX = re.compile(r"(?i)\b(?:sections?|pull requests?|issues?|PR)\s+(?:#?\d+(?:\.\d+)*(?:\s*(?:,|and|to|-)\s*#?\d+(?:\.\d+)*)*)")
HEADING_RX = re.compile(r"^\s*#{1,6}\s+\d+(?:\.\d+)*\.?\s")
NUM_RX = re.compile(r"(?<![A-Za-z0-9_.])(?<![A-Za-z_]-)\d[\d,]*(?:\.\d+)?(?:e-?\d{1,3})?(?![A-Za-z0-9_])")
TEXT_EXT = {".md": 3, ".txt": 3, ".tsv": 5, ".csv": 4, ".json": 6, ".jsonl": 6, ".yaml": 6, ".yml": 6}
KINDS = {1: "session README", 2: "tables.txt", 3: "best.tsv", 4: "text file", 5: "csv", 6: "tsv", 7: "json/yaml", 8: "file named by --elsewhere"}
CPU_RX = re.compile(r"(?i)\b(?:cpu|simulator|laptop)")


def git(repo, *args):
    r = subprocess.run(["git", "-c", "core.quotepath=false", "-C", repo] + list(args), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise SystemExit("numbers_moved: git %s failed: %s" % (" ".join(args), r.stderr.decode("utf-8", "replace").strip()))
    return r.stdout


def norm(tok):
    return tok.replace(",", "")


def numerals(line, data=False):
    """[(token, normalized)] of the numerals of one line, left out: dates, identifier digits, section numbers.
    data=True: a line of a csv, tsv or json file, where a comma separates two cells and is no thousands mark."""
    if HEADING_RX.match(line):
        line = HEADING_RX.sub("", line)
    if data:
        line = line.replace(",", " ")
    masked = DATE_RX.sub(lambda m: " " * len(m.group(0)), line)
    masked = SECTION_RX.sub(lambda m: " " * len(m.group(0)), masked)
    out = []
    for m in NUM_RX.finditer(masked):
        tok = m.group(0).rstrip(",")
        if tok:
            out.append((tok, norm(tok)))
    return out


def is_small(n):
    return re.fullmatch(r"\d{1,2}", n) is not None


def list_scope(repo, rev, scopes):
    """{path: text} of the tracked text files under the scopes: at REV, or in the working tree when REV is None."""
    names = []
    if rev is None:
        out = git(repo, "ls-files", "-z").decode("utf-8", "surrogateescape")
    else:
        out = git(repo, "ls-tree", "-r", "--name-only", "-z", rev).decode("utf-8", "surrogateescape")
    for p in out.split("\0"):
        if p and any(p == s or p.startswith(s.rstrip("/") + "/") for s in scopes) and p.endswith(".md"):
            names.append(p)
    res = {}
    for p in names:
        if rev is None:
            full = os.path.join(repo, *p.split("/"))
            if not os.path.isfile(full):
                continue
            with open(full, "rb") as fh:
                data = fh.read()
        else:
            data = git(repo, "show", "%s:%s" % (rev, p))
        res[p] = data.decode("utf-8", "replace").replace("\r\n", "\n")
    return res


def count_scope(files):
    c = collections.Counter()
    for text in files.values():
        for line in text.split("\n"):
            for _t, n in numerals(line):
                c[n] += 1
    return c


def decimals_of(n):
    return len(n.split(".", 1)[1]) if "." in n and "e" not in n else None


def kind_rank(rel):
    base = os.path.basename(rel)
    ext = os.path.splitext(base)[1].lower()
    if rel == "rl/best.tsv":
        return 3
    if base == "README.md":
        return 1
    if base == "tables.txt":
        return 2
    return {".md": 4, ".txt": 4, ".csv": 5, ".tsv": 6, ".json": 7, ".jsonl": 7, ".yaml": 7, ".yml": 7}.get(ext)


def data_files(repo):
    res = []
    for sub in ("data", "docs", "patches"):
        for d, _dirs, files in os.walk(os.path.join(repo, "rl", sub)):
            for f in files:
                rel = os.path.relpath(os.path.join(d, f), repo).replace("\\", "/")
                k = kind_rank(rel)
                if k is not None:
                    res.append((k, rel))
    if os.path.isfile(os.path.join(repo, "rl", "best.tsv")):
        res.append((3, "rl/best.tsv"))
    res.sort()
    return res


def round_to(tok, d):
    try:
        return str(decimal.Decimal(tok).quantize(decimal.Decimal(1).scaleb(-d), rounding=decimal.ROUND_HALF_UP))
    except (decimal.InvalidOperation, ValueError):
        return None


def find_homes(repo, wanted, line_context, elsewhere=()):
    """{normalized numeral: (rank, rel, line number, line text)} for the numerals in WANTED.
    line_context: {numeral: set of the other numerals of the removed sentences that hold it} (small numbers need them)."""
    homes = {}
    by_dec = collections.defaultdict(set)
    for n in wanted:
        d = decimals_of(n)
        if d is None and re.fullmatch(r"\d{4,}", n):
            d = 0   # a rounded figure: 3337 is held by 3337.3
        if d is not None:
            by_dec[d].add(n)
    files = data_files(repo) + [(8, p) for p in elsewhere]
    for rank, rel in files:
        try:
            with open(rel if os.path.isabs(rel) else os.path.join(repo, *rel.split("/")), "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        if b"\0" in data[:4096]:
            continue
        cells = kind_rank(rel) not in (1, 2, 3, 4) and not rel.endswith("README.md")   # csv, tsv, json: commas separate
        for ln, line in enumerate(data.decode("utf-8", "replace").split("\n"), 1):
            line_nums = None
            for tok, n in numerals(line, cells):
                cand = [(n, 0)]
                d = decimals_of(n)
                if d is not None:
                    for dd, ws in by_dec.items():
                        if dd < d:
                            r = round_to(n, dd)
                            if r is not None and r in ws:
                                cand.append((r, 10))   # a rounded match ranks below every exact one
                for c, extra in cand:
                    if c not in wanted:
                        continue
                    if c in homes and homes[c][0] <= rank + extra:
                        continue
                    if is_small(c):
                        if line_nums is None:
                            line_nums = set(x for _t, x in numerals(line, cells))
                        others = line_context.get(c, set()) - {c}
                        if not (others & line_nums):
                            continue
                    homes[c] = (rank + extra, rel, ln, line.strip())
    return homes


def sentences_with(files, tokens):
    """[(path, line number, line text, [numerals])] of the lines of FILES that hold one of TOKENS."""
    out = []
    for p, text in sorted(files.items()):
        for ln, line in enumerate(text.split("\n"), 1):
            ns = [n for _t, n in numerals(line)]
            hit = [n for n in ns if n in tokens]
            if hit:
                out.append((p, ln, line, hit, set(ns)))
    return out


def run(repo, base, head, scopes, audit=False, list_moved=False, out=sys.stdout, elsewhere=()):
    old = list_scope(repo, base, scopes)
    new = list_scope(repo, head, scopes)
    c_old, c_new = count_scope(old), count_scope(new)
    removed = {n: c_old[n] for n in c_old if c_new[n] == 0}
    fewer = sum(1 for n in c_old if 0 < c_new[n] < c_old[n])
    new_lines = set(l.strip() for t in new.values() for l in t.split("\n"))
    lines = [s for s in sentences_with(old, set(removed)) if s[2].strip() not in new_lines]
    ctx = collections.defaultdict(set)
    for _p, _ln, _line, hit, allns in lines:
        for n in hit:
            ctx[n] |= allns
    homes = find_homes(repo, set(removed), ctx, elsewhere)
    nohome = sorted(n for n in removed if n not in homes)
    out.write("numbers_moved: base %s, head %s, scope %s\n" % (base, head or "working tree", " ".join(scopes)))
    out.write("removed numerals: %d distinct, %d occurrences; with a home: %d; removed, no home: %d\n"
              % (len(removed), sum(removed.values()), len(removed) - len(nohome), len(nohome)))
    out.write("numerals that occur fewer times but still occur in the scope: %d (not itemised)\n" % fewer)
    if list_moved:
        out.write("\n## moved (first home of each removed numeral)\n")
        for n in sorted(removed, key=lambda x: (float(x) if re.fullmatch(r"[\d.]+", x) else 0, x)):
            if n in homes:
                r, rel, ln, _l = homes[n]
                shown = os.path.basename(rel) if os.path.isabs(rel) else rel
                out.write("%s -> %s:%d (%s%s)\n" % (n, shown, ln, KINDS[r % 10], ", rounded match" if r >= 10 else ""))
    out.write("\n## removed, no home\n")
    shown = set()
    for p, ln, line, hit, _all in lines:
        miss = [n for n in hit if n in nohome]
        if not miss:
            continue
        text = " ".join(line.split())
        if (p, text) in shown:
            continue
        shown.add((p, text))
        tag = " (cpu or simulator sentence: cut by the public-text rule)" if CPU_RX.search(line) else ""
        out.write("- %s:%d [%s]%s %s\n" % (p, ln, ", ".join(sorted(set(miss))), tag, text[:300]))
    # numerals whose removed lines were reworded (the line is still in the head) have no entry above: say so.
    unseen = [n for n in nohome if not any(n in s[3] for s in lines)]
    if unseen:
        out.write("- (the sentence of these was reworded in place, not removed; see the base text): %s\n" % ", ".join(sorted(unseen)))
    if audit:
        out.write("\n## audit: numerals of the head text that no file under rl/data holds\n")
        want = set(c_new)
        hmap = find_homes(repo, want, collections.defaultdict(set))
        for p, text in sorted(new.items()):
            for ln, line in enumerate(text.split("\n"), 1):
                miss = sorted(set(n for _t, n in numerals(line) if n in want and n not in hmap and not is_small(n)))
                if miss:
                    out.write("- %s:%d [%s] %s\n" % (p, ln, ", ".join(miss), " ".join(line.split())[:200]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--head", default=None)
    ap.add_argument("--scope", action="append")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--list-moved", action="store_true")
    ap.add_argument("--elsewhere", action="append", default=[], help="a file outside rl/data that is searched last for homes (repeatable)")
    a = ap.parse_args(argv)
    scopes = a.scope or ["README.md", "docs/rl"]
    return run(os.path.abspath(a.repo), a.base, a.head, scopes, a.audit, a.list_moved, elsewhere=[os.path.abspath(x) for x in a.elsewhere])


if __name__ == "__main__":
    sys.exit(main())
