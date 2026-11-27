#!/usr/bin/env python3
"""Choose which files of the former run repository are imported under rl/ (lean import).

Rule, mechanical and by file type, under data/ only:
  * every path of the freeze commit F outside data/ is imported;
  * under data/, a file with the extension .yaml, .out, .log, .sh or .py is NOT imported,
    unless it is referenced.

"Referenced" means that one of these forms of the file's path occurs in one of the text sources:
  repo-path   the repository-relative path, e.g. data/<session>/dir/x.sh
  session-path the path relative to its session folder, e.g. dir/x.sh (or x.sh at the folder root)
  dir         the directory that contains the file (repository-relative, or relative to its session folder).
              A multi-part folder path counts with or without a trailing slash. A one-word folder name counts only
              when it is written with its slash AND it is the file's parent relative to the session folder
              ("serving/" for serving/x.log). A leaf folder at a deeper level ("driver_logs/" for
              serving/driver_logs/x.log, "scripts/" for host_step/scripts/x.py) is NOT a reference: its parent
              relative to the session is the multi-part path "serving/driver_logs", which must be written in full.
              Brace, glob and <placeholder> forms ("i34_{a,b}.out", "logs/*.out", "<container>.launch.out") are not
              expanded. These gaps are why the lean selection dropped files that kept text names (about 104 of 145);
              the import was therefore made complete (all 145 files are under rl/) and this script is kept as the
              record of the rule, not as the way the import is chosen.
  basename    the base name, when that base name is unique among all paths of F
Text sources:
  run   README.md, REPRODUCE.md, docs/**, every data/**/README.md, every data/**/tables.txt of F, and (one choice
        beyond the rule, so that the scrub records do not name a file that is not imported) every
        data/**/scrub_allow.txt and data/**/scrub_report.txt (these two count for file paths only),
        and every imported script (.py or .sh): all scripts outside data/, plus the scripts under data/ that
        are kept because they are referenced (the scan repeats until nothing new is kept)
  fork  README.md and docs/rl/** of the fork at --fork-ref
  hub   ledger/**, lessons/**, reference/**, analysis/** and the top-level .md files of the hub at --hub-ref
        (private text: the reason column names only "hub", never a hub file)

Usage:
  python select_import.py --run-repo RUN --ref F --fork-repo FORK --fork-ref REF --hub-repo HUB --hub-ref REF
                          [--no-dir-refs] [--kept-out FILE] [--dropped-out FILE]

Output (stdout, sorted by path): a header, then one line per path of F:
  K   path                 kept: outside data/ or not of a droppable type
  KX  path  reason         kept although of a droppable type: the reason names the first source and form found
  D   path                 dropped
and a summary. The result depends only on the contents of the three repositories at the given refs.
Standard library only.
"""
import argparse
import posixpath
import subprocess
import sys

DROP_EXT = (".yaml", ".out", ".log", ".sh", ".py")
SCRIPT_EXT = (".py", ".sh")
# text files under data/ that are sources: the folder READMEs and tables.txt of the rule, plus the two records of the
# scrub run, which list file paths by design (their entries must not point at a file that is not imported)
LISTS_ONLY_PATHS = ("/scrub_allow.txt", "/scrub_report.txt")
DATA_TEXT = ("/README.md", "/tables.txt", "/scrub_allow.txt", "/scrub_report.txt")
# characters that, next to a match, show that the match is only part of a longer name
LEFT_BLOCK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
RIGHT_BLOCK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
# characters that continue a path after a "/" (so "dir/file" is a file reference); "dir/*.sh" and "dir/{a,b}" name the folder
PATH_CONT = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-$<~")
ALNUM = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def git(repo, *args, stdin=None):
    p = subprocess.run(["git", "-C", repo] + list(args), input=stdin, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if p.returncode != 0:
        sys.exit("git %s failed in %s: %s" % (" ".join(args), repo, p.stderr.decode("utf-8", "replace")))
    return p.stdout


def ls_tree(repo, ref):
    out = git(repo, "ls-tree", "-r", "-z", ref)
    res = {}
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, typ, sha = meta.split()
        if typ == b"blob":
            res[path.decode("utf-8", "surrogateescape")] = sha.decode()
    return res


def read_blobs(repo, shas):
    """Return {sha: bytes} for the given blob ids, with one git process."""
    shas = sorted(set(shas))
    if not shas:
        return {}
    out = git(repo, "cat-file", "--batch", stdin=("\n".join(shas) + "\n").encode())
    res = {}
    pos = 0
    for _ in shas:
        nl = out.index(b"\n", pos)
        head = out[pos:nl].split()
        size = int(head[2])
        res[head[0].decode()] = out[nl + 1:nl + 1 + size]
        pos = nl + 1 + size + 1
    return res


def text_of(b):
    return b.decode("utf-8", "replace")


def find_ref(text, needle, is_dir, need_slash=False):
    """True when needle occurs in text as a whole name (see LEFT_BLOCK, RIGHT_BLOCK).

    need_slash: the needle must be followed by "/" (a bare one-word folder name such as "serving" is also
    an ordinary word, so it counts only when written as a folder: "serving/")."""
    start = 0
    n = len(needle)
    while True:
        i = text.find(needle, start)
        if i < 0:
            return False
        start = i + 1
        if i > 0 and text[i - 1] in LEFT_BLOCK:
            continue
        j = i + n
        c = text[j] if j < len(text) else ""
        if is_dir and c == "/":
            # "dir/" names the folder; "dir/file" names that file, not the folder (the file forms decide)
            nxt = text[j + 1] if j + 1 < len(text) else ""
            if nxt == "" or nxt not in PATH_CONT:
                return True
            continue
        if need_slash:
            continue
        if c == "":
            return True
        if c in RIGHT_BLOCK:
            continue
        if c == "." and j + 1 < len(text) and text[j + 1] in ALNUM:
            continue
        return True


def forms_of(path, basename_unique):
    """The (form, needle, is_dir) tuples that count as a reference to a path under data/<session>/."""
    parts = path.split("/")
    session = parts[1]
    rest = "/".join(parts[2:])
    forms = [("repo-path", path, False)]
    forms.append(("session-path", rest, False))
    parent = posixpath.dirname(path)
    forms.append(("dir", parent, True))
    parent_rel = posixpath.dirname(rest)
    if parent_rel:
        forms.append(("dir/", parent_rel, True) if "/" not in parent_rel else ("dir", parent_rel, True))
    base = parts[-1]
    if basename_unique and base != rest:
        forms.append(("basename", base, False))
    return forms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-repo", required=True)
    ap.add_argument("--ref", required=True, help="the freeze commit F of the run repository")
    ap.add_argument("--fork-repo", required=True)
    ap.add_argument("--fork-ref", required=True)
    ap.add_argument("--hub-repo", required=True)
    ap.add_argument("--hub-ref", required=True)
    ap.add_argument("--no-dir-refs", action="store_true",
                    help="do not count a mention of the containing directory (stricter: fewer files kept)")
    ap.add_argument("--kept-out")
    ap.add_argument("--dropped-out")
    a = ap.parse_args()

    run_sha = git(a.run_repo, "rev-parse", a.ref + "^{commit}").decode().strip()
    tree = ls_tree(a.run_repo, run_sha)
    paths = sorted(tree, key=lambda p: p.encode("utf-8", "surrogateescape"))

    base_count = {}
    for p in paths:
        b = p.rsplit("/", 1)[-1]
        base_count[b] = base_count.get(b, 0) + 1

    cand = [p for p in paths if p.startswith("data/") and p.endswith(DROP_EXT)]
    cand_set = set(cand)

    run_blobs = read_blobs(a.run_repo, [tree[p] for p in paths if not p.startswith("data/") or p in cand_set
                                        or p.endswith(DATA_TEXT)])

    def run_text(p):
        return text_of(run_blobs[tree[p]])

    # static sources, in a fixed order: (group, label, text)
    sources = []
    for p in paths:
        if p in ("README.md", "REPRODUCE.md") or p.startswith("docs/"):
            sources.append(("run", p, run_text(p)))
        elif p.startswith("data/") and p.endswith(DATA_TEXT):
            sources.append(("run", p, run_text(p)))
        elif not p.startswith("data/") and p.endswith(SCRIPT_EXT):
            sources.append(("run", p, run_text(p)))
    fork_tree = ls_tree(a.fork_repo, a.fork_ref)
    fork_paths = sorted(p for p in fork_tree if p == "README.md" or p.startswith("docs/rl/"))
    fb = read_blobs(a.fork_repo, [fork_tree[p] for p in fork_paths])
    for p in fork_paths:
        sources.append(("fork", p, text_of(fb[fork_tree[p]])))
    hub_tree = ls_tree(a.hub_repo, a.hub_ref)
    hub_paths = sorted(p for p in hub_tree if p.startswith(("ledger/", "lessons/", "reference/", "analysis/"))
                       or ("/" not in p and p.endswith(".md")))
    hb = read_blobs(a.hub_repo, [hub_tree[p] for p in hub_paths])
    for p in hub_paths:
        sources.append(("hub", p, text_of(hb[hub_tree[p]])))

    kept_x = {}  # path -> reason
    changed = True
    while changed:
        changed = False
        extra = [("run", p, run_text(p)) for p in sorted(kept_x)]
        # sources in play this round: static, then the imported candidates found so far
        for p in cand:
            if p in kept_x:
                continue
            forms = forms_of(p, base_count[p.rsplit("/", 1)[-1]] == 1)
            if a.no_dir_refs:
                forms = [f for f in forms if not f[0].startswith("dir")]
            hit = None
            for group, label, text in sources + extra:
                if label == p:
                    continue
                for form, needle, is_dir in forms:
                    if form.startswith("dir") and label.endswith(LISTS_ONLY_PATHS):
                        continue  # a scrub record names files, not folders
                    if needle and find_ref(text, needle, is_dir, need_slash=(form == "dir/")):
                        hit = (group, label, form, needle)
                        break
                if hit:
                    break
            if hit:
                group, label, form, needle = hit
                where = "hub text" if group == "hub" else "%s %s" % (group, label)
                kept_x[p] = "referenced by %s as %s (%s)" % (where, form.rstrip("/"), needle)
                changed = True

    kept = [p for p in paths if p not in cand_set or p in kept_x]
    dropped = [p for p in cand if p not in kept_x]

    out = []
    out.append("# select_import: F=%s" % run_sha)
    out.append("# run files: %d, droppable type under data/: %d, kept as referenced: %d, dropped: %d, kept: %d"
               % (len(paths), len(cand), len(kept_x), len(dropped), len(kept)))
    for p in paths:
        if p in cand_set and p not in kept_x:
            out.append("D\t%s" % p)
        elif p in kept_x:
            out.append("KX\t%s\t%s" % (p, kept_x[p]))
        else:
            out.append("K\t%s" % p)
    sys.stdout.buffer.write(("\n".join(out) + "\n").encode("utf-8", "surrogateescape"))
    if a.kept_out:
        with open(a.kept_out, "wb") as f:
            f.write(("\n".join(kept) + "\n").encode("utf-8", "surrogateescape"))
    if a.dropped_out:
        with open(a.dropped_out, "wb") as f:
            f.write(("\n".join(dropped) + "\n").encode("utf-8", "surrogateescape"))


if __name__ == "__main__":
    main()
