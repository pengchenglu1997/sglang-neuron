#!/usr/bin/env python3
"""Document checks of the fork. Run from the repository root (a git checkout of the head).

    check_docs.py f7|f4 [--base REV] [--allow FILE]
    check_docs.py f10
    check_docs.py f8 --base REV --head REV

F7  every relative markdown link and every rl/... path in the prose resolves; a pinned
    sglang-neuron/tree/<sha>/rl/... link names a sha that contains that path; AGENTS.md has at most 80 lines.
F4  public-CPU lint: in docs/rl/**, README.md and rl/**/*.md, a line that matches cpu, simulator or laptop (also as
    the start of a longer word: CPUs, simulators) and holds a digit outside a code span, a link target or a URL fails.
    An allow entry "<path><TAB>public-cpu-lint<TAB><reason><TAB><line>" accepts that one line (white space collapsed).
F8  state lint on the lines added between base and head: phrases that state a changing fact. Lines of
    a quoted block (starting with >) are exempt.
F10 no ENGINE file mentions rl/ or imports from it; no symlink (git mode 120000) outside TEAM paths leads into a team
    path (following the links on the way), and no symlink under TEAM paths leads outside them.
The prose checks (F7, F4, F8) skip only frozen records that already exist: a file under rl/data/, rl/patches/ or
rl/docs/ that is listed in rl/data/LEGACY.sha256, or (with --base) that exists at the base. A file there that a pull
request adds is linted like any other.
Exit status: 0 pass, 1 fail, 2 bad input.
"""
import argparse
import posixpath
import re
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
import scrub_check  # noqa: E402

STATE_PATTERNS = [
    r"not yet in any repo",
    r"(still )?(is|stays|remains) (open|a draft|not merged)",
    r"open, not merged",
    r"nothing has been (sent|posted)",
    r"has not been (run|sent|started)",
    r"Changed 20\d\d-\d\d-\d\d",
    r"as of 20\d\d-",
    r"currently",
    r"for now",
]
STATE_RX = [re.compile(p, re.I) for p in STATE_PATTERNS]
CPU_RX = re.compile(r"\b(?:cpu|simulator|laptop)", re.I)
LINK_RX = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
RLPATH_RX = re.compile(r"(?<![A-Za-z0-9_./-])(rl/[A-Za-z0-9_./-]*[A-Za-z0-9_/-])")
PIN_RX = re.compile(r"sglang-neuron/(?:tree|blob)/([0-9a-f]{7,40})/(rl/[^\s)#>\"'`]*)")
FENCE_RX = re.compile(r"^\s*(```|~~~)")


def tracked(repo="."):
    out = common.git(["ls-files", "-z"], cwd=repo)
    return [p for p in out.split("\0") if p]


LEGACY_LINE = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(\S.*?)\s*$")


def exempt_files(legacy_text, base=None, repo="."):
    """Frozen records the prose checks skip: the paths in LEGACY.sha256 (relative to the repository root or to
    rl/data/) and, when a base is given, every frozen-prefix file that exists there."""
    res = set()
    for line in (legacy_text or "").splitlines():
        mo = LEGACY_LINE.match(line)
        if mo:
            res.add(mo.group(2))
            res.add("rl/data/" + mo.group(2))
    if base:
        res.update(p for p, _ in common.ls_tree(base, "rl", cwd=repo) if common.is_frozen(p))
    return res


def prose_files(files, exempt=frozenset()):
    return [p for p in files if p.endswith(".md") and common.is_team(p) and not p.startswith(".github/")
            and not (common.is_frozen(p) and p in exempt)]


def worktree_exempt(repo, base):
    legacy = read_text("rl/data/LEGACY.sha256", repo)
    return exempt_files(legacy, base, repo)


def read_text(path, repo="."):
    try:
        with open(os.path.join(repo, path), "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    return data.decode("utf-8", "replace") if common.is_text(data) else None


def prose_lines(text):
    """(line number, line, in_fence) for every line."""
    fenced = False
    for n, line in enumerate(text.split("\n"), 1):
        if FENCE_RX.match(line):
            fenced = not fenced
            yield n, line, True
            continue
        yield n, line, fenced


def f7(repo=".", base=None):
    files = tracked(repo)
    fileset = set(files)
    dirs = set()
    for p in files:
        parts = p.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    msgs = []
    for p in prose_files(files, worktree_exempt(repo, base)):
        text = read_text(p, repo)
        if text is None:
            continue
        folder = os.path.dirname(p)
        for n, line, fenced in prose_lines(text):
            if fenced:
                continue
            for target in LINK_RX.findall(line):
                if re.match(r"^(?:[a-z][a-z0-9+.-]*:|#|/)", target, re.I):
                    continue
                t = target.split("#", 1)[0].split("?", 1)[0]
                if not t:
                    continue
                full = os.path.normpath(os.path.join(folder, t)).replace("\\", "/")
                if full not in fileset and full not in dirs:
                    msgs.append("F7: %s:%d link %s does not resolve." % (p, n, target))
            for mo in PIN_RX.finditer(line):
                sha, path = mo.group(1), mo.group(2).rstrip(".,;:")
                if not common.exists_at(sha, path.rstrip("/"), cwd=repo):
                    msgs.append("F7: %s:%d pinned link %s/%s: that sha does not contain the path." % (p, n, sha, path))
            stripped = LINK_RX.sub("", line)
            stripped = re.sub(r"https?://\S+", "", stripped)
            for mo in RLPATH_RX.finditer(stripped):
                path = mo.group(1)
                nxt = stripped[mo.end():mo.end() + 1]
                if nxt in ("<", "*", "{", "$", "[") or re.search(r"[*<>{}$]", path) or path in ("rl/", "rl"):
                    continue   # a placeholder or a glob, not a path
                if path.rstrip("/") not in fileset and path.rstrip("/") not in dirs:
                    msgs.append("F7: %s:%d path %s does not exist." % (p, n, path))
    if "AGENTS.md" in fileset:
        text = read_text("AGENTS.md", repo) or ""
        nlines = len(text.splitlines())
        if nlines > 80:
            msgs.append("F7: AGENTS.md has %d lines, at most 80 are allowed." % nlines)
    return not msgs, msgs or ["F7: links and rl/ paths resolve."]


def strip_code(line):
    line = re.sub(r"`[^`]*`", "", line)
    line = LINK_RX.sub("", line)
    return re.sub(r"https?://\S+", "", line)


def f4(repo=".", allow=None, base=None):
    files = tracked(repo)
    allow = scrub_check.load_allow(allow) if allow else {}
    msgs, accepted = [], 0
    for p in prose_files(files, worktree_exempt(repo, base)):
        if not (p.startswith("docs/rl/") or p == "README.md" or p.startswith("rl/")):
            continue
        text = read_text(p, repo)
        if text is None:
            continue
        for n, line, fenced in prose_lines(text):
            if fenced or not CPU_RX.search(line):
                continue
            if re.search(r"[0-9]", strip_code(line)):
                key = " ".join(line.split())
                if any(t == key for t, _ in allow.get((p, "public-cpu-lint"), ()) if t is not None):
                    accepted += 1
                else:
                    msgs.append("F4: %s:%d a cpu/simulator/laptop line holds a digit outside a code span. "
                                "Allow entry text: %s" % (p, n, key))
    tail = ["F4: %d line(s) accepted by the allow file." % accepted] if accepted else []
    return not msgs, (msgs or ["F4: no public CPU quantity."]) + tail


def added_lines(base, head, repo="."):
    """{path: [(line number in head, text)]} for the added lines of the prose files."""
    res = {}
    names = [p for st, p in common.changed(base, head, cwd=repo) if st[0] != "D"]
    legacy = common.show(head, "rl/data/LEGACY.sha256", cwd=repo)
    for p in prose_files(names, exempt_files(legacy.decode("utf-8", "replace") if legacy else None, base, repo)):
        out = common.git(["diff", "-U0", "--no-renames", "%s...%s" % (base, head), "--", p], cwd=repo)
        cur = 0
        for line in out.split("\n"):
            mo = re.match(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", line)
            if mo:
                cur = int(mo.group(1))
                continue
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                res.setdefault(p, []).append((cur, line[1:]))
                cur += 1
    return res


def f8(base, head, repo="."):
    msgs = []
    for p, lines in sorted(added_lines(base, head, repo).items()):
        for n, line in lines:
            if line.lstrip().startswith(">"):
                continue
            for rx in STATE_RX:
                mo = rx.search(line)
                if mo:
                    msgs.append("F8: %s:%d state wording '%s'." % (p, n, mo.group(0)))
                    break
    return not msgs, msgs or ["F8: no state wording on added lines."]


def symlinks(repo="."):
    """{path: target text} of every tracked symlink (git mode 120000), read from the index so that a checkout without
    symlink support gives the same answer."""
    out = common.git(["ls-files", "-s", "-z"], cwd=repo)
    res = {}
    for rec in out.split("\0"):
        if not rec:
            continue
        meta, path = rec.split("\t", 1)
        mode, oid, _stage = meta.split()
        if mode == "120000":
            data = common.git(["cat-file", "blob", oid], cwd=repo, binary=True)
            res[path] = (data or b"").decode("utf-8", "replace")
    return res


def resolve_link(path, links, depth=0):
    """The repository-relative path that path leads to once every symlink on the way is followed; None when it
    leaves the repository (an absolute target, too many ..) or loops."""
    if depth > 20:
        return None
    path = posixpath.normpath(path)
    comps = path.split("/")
    for i in range(len(comps)):
        prefix = "/".join(comps[:i + 1])
        if prefix in links:
            target = links[prefix]
            if not target or target.startswith("/"):
                return None
            new = posixpath.normpath(posixpath.join(posixpath.dirname(prefix), target))
            if new == ".." or new.startswith("../"):
                return None
            return resolve_link("/".join([new] + comps[i + 1:]), links, depth + 1)
    return path


def f10_symlinks(repo="."):
    """Messages for a symlink outside TEAM paths that leads into a team path (or onto a folder that holds one), and for
    a symlink under TEAM paths that leads outside them."""
    links = symlinks(repo)
    if not links:
        return []
    paths = tracked(repo)
    team_paths = [p for p in paths if common.is_team(p)]
    msgs = []
    for link in sorted(links):
        dest = resolve_link(link, links)
        in_team = dest is not None and (common.is_team(dest) or common.is_team(dest + "/"))
        if common.is_team(link):
            if not in_team:
                msgs.append("F10: %s:1 is a symlink under team paths that points outside them (-> %s)." % (link, links[link]))
        elif dest is not None:
            holds_team = in_team or dest == "." or any(p.startswith(dest + "/") for p in team_paths)
            if holds_team:
                msgs.append("F10: %s:1 is a symlink outside team paths that points into a team path (-> %s)." % (link, links[link]))
    return msgs


def f10(repo="."):
    msgs = f10_symlinks(repo)
    imp = re.compile(r"^\s*(?:from|import)\s+rl(?:\.|\s|$)")
    ment = re.compile(r"(?<![A-Za-z0-9_.])rl/")
    for p in tracked(repo):
        if common.is_team(p):
            continue
        text = read_text(p, repo)
        if text is None:
            continue
        for n, line in enumerate(text.split("\n"), 1):
            if imp.match(line) or ment.search(line):
                msgs.append("F10: %s:%d mentions or imports rl/." % (p, n))
                break
    return not msgs, msgs or ["F10: no engine file mentions rl/, and no symlink crosses between team and engine paths."]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=["f7", "f4", "f8", "f10"])
    ap.add_argument("--base", help="f8: the base; f7, f4: frozen files that exist there are skipped")
    ap.add_argument("--head")
    ap.add_argument("--allow", default=os.path.join("rl", "tools", "scrub_allow.txt"))
    ap.add_argument("--repo", default=".")
    a = ap.parse_args(argv)
    try:
        if a.check == "f7":
            ok, msgs = f7(a.repo, a.base)
        elif a.check == "f4":
            allow = os.path.join(a.repo, a.allow)
            ok, msgs = f4(a.repo, allow if os.path.exists(allow) else None, a.base)
        elif a.check == "f10":
            ok, msgs = f10(a.repo)
        else:
            if not (a.base and a.head):
                print("check_docs: f8 needs --base and --head", file=sys.stderr)
                return 2
            ok, msgs = f8(a.base, a.head, a.repo)
    except RuntimeError as e:
        print("check_docs: %s" % e, file=sys.stderr)
        return 2
    for m in msgs[:300]:
        print(m)
    if len(msgs) > 300:
        print("... %d more" % (len(msgs) - 300))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
