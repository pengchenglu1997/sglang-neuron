#!/usr/bin/env python3
"""Document checks of the fork. Run from the repository root (a git checkout of the head).

    check_docs.py f7|f4|f10 [--allow FILE]
    check_docs.py f8 --base REV --head REV

F7  every relative markdown link and every rl/... path in the prose resolves; a pinned
    sglang-neuron/tree/<sha>/rl/... link names a sha that contains that path; AGENTS.md has at most 80 lines.
F4  public-CPU lint: in docs/rl/**, README.md and rl/**/*.md, a line that matches cpu, simulator, laptop or
    cpu-only and holds a digit outside a code span, a link target or a URL fails (an entry
    "<path><TAB>public-cpu-lint<TAB><reason>" in the allow file accepts a whole file).
F8  state lint on the lines added between base and head: phrases that state a changing fact. Lines of
    a quoted block (starting with >) are exempt.
F10 no ENGINE file mentions rl/ or imports from it.
The prose checks (F7, F4, F8) skip the frozen records: rl/data/, rl/patches/ and rl/docs/.
Exit status: 0 pass, 1 fail, 2 bad input.
"""
import argparse
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
CPU_RX = re.compile(r"\b(?:cpu(?:-only)?|simulator|laptop)\b", re.I)
LINK_RX = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
RLPATH_RX = re.compile(r"(?<![A-Za-z0-9_./-])(rl/[A-Za-z0-9_./-]*[A-Za-z0-9_/-])")
PIN_RX = re.compile(r"sglang-neuron/(?:tree|blob)/([0-9a-f]{7,40})/(rl/[^\s)#>\"'`]*)")
FENCE_RX = re.compile(r"^\s*(```|~~~)")


def tracked(repo="."):
    out = common.git(["ls-files", "-z"], cwd=repo)
    return [common.norm(p) for p in out.split("\0") if p]


def prose_files(files):
    return [p for p in files if p.endswith(".md") and common.is_team(p) and not common.is_frozen(p)
            and not p.startswith(".github/")]


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


def f7(repo="."):
    files = tracked(repo)
    fileset = set(files)
    dirs = set()
    for p in files:
        parts = p.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    msgs = []
    for p in prose_files(files):
        text = read_text(p, repo)
        if text is None:
            continue
        base = os.path.dirname(p)
        for n, line, fenced in prose_lines(text):
            if fenced:
                continue
            for target in LINK_RX.findall(line):
                if re.match(r"^(?:[a-z][a-z0-9+.-]*:|#|/)", target, re.I):
                    continue
                t = target.split("#", 1)[0].split("?", 1)[0]
                if not t:
                    continue
                full = os.path.normpath(os.path.join(base, t)).replace("\\", "/")
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


def f4(repo=".", allow=None):
    files = tracked(repo)
    allow = scrub_check.load_allow(allow) if allow else {}
    msgs, accepted = [], 0
    for p in prose_files(files):
        if not (p.startswith("docs/rl/") or p == "README.md" or p.startswith("rl/")):
            continue
        text = read_text(p, repo)
        if text is None:
            continue
        for n, line, fenced in prose_lines(text):
            if fenced or not CPU_RX.search(line):
                continue
            if re.search(r"[0-9]", strip_code(line)):
                if (p, "public-cpu-lint") in allow:
                    accepted += 1
                else:
                    msgs.append("F4: %s:%d a cpu/simulator/laptop line holds a digit outside a code span." % (p, n))
    tail = ["F4: %d line(s) accepted by the allow file." % accepted] if accepted else []
    return not msgs, (msgs or ["F4: no public CPU quantity."]) + tail


def added_lines(base, head, repo="."):
    """{path: [(line number in head, text)]} for the added lines of the prose files."""
    res = {}
    names = [p for st, p in common.changed(base, head, cwd=repo) if st[0] != "D"]
    for p in prose_files(names):
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


def f10(repo="."):
    msgs = []
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
    return not msgs, msgs or ["F10: no engine file mentions rl/."]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=["f7", "f4", "f8", "f10"])
    ap.add_argument("--base")
    ap.add_argument("--head")
    ap.add_argument("--allow", default=os.path.join("rl", "tools", "scrub_allow.txt"))
    ap.add_argument("--repo", default=".")
    a = ap.parse_args(argv)
    try:
        if a.check == "f7":
            ok, msgs = f7(a.repo)
        elif a.check == "f4":
            allow = os.path.join(a.repo, a.allow)
            ok, msgs = f4(a.repo, allow if os.path.exists(allow) else None)
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
