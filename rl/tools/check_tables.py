#!/usr/bin/env python3
"""F5, third item: where a data folder holds both make_tables.py and tables.txt, the script's output equals tables.txt.

    check_tables.py [--root DIR]

Each such folder is copied to a temporary folder and make_tables.py is run there, so nothing in the checkout changes.
The generated tables.txt is compared byte for byte with the committed one. Exit status: 0 pass (also when no folder
has both files), 1 a difference or a failed script, 2 bad input.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True


def default_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_folder(folder):
    """A list of problems for one folder (empty: equal)."""
    committed = os.path.join(folder, "tables.txt")
    with open(committed, "rb") as fh:
        want = fh.read()
    tmp = tempfile.mkdtemp(prefix="tables_")
    try:
        work = os.path.join(tmp, "d")
        shutil.copytree(folder, work)
        os.remove(os.path.join(work, "tables.txt"))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, "-I", "make_tables.py"], cwd=work, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        if r.returncode != 0:
            return ["make_tables.py failed: %s" % r.stderr.decode("utf-8", "replace").strip()[-300:]]
        out = os.path.join(work, "tables.txt")
        if not os.path.isfile(out):
            return ["make_tables.py wrote no tables.txt"]
        with open(out, "rb") as fh:
            got = fh.read()
        if got != want:
            a, b = want.decode("utf-8", "replace").split("\n"), got.decode("utf-8", "replace").split("\n")
            for i in range(max(len(a), len(b))):
                x = a[i] if i < len(a) else "<end>"
                y = b[i] if i < len(b) else "<end>"
                if x != y:
                    return ["tables.txt differs from the script's output at line %d: committed %r, generated %r"
                            % (i + 1, x[:100], y[:100])]
            return ["tables.txt differs from the script's output"]
        return []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=default_root())
    a = ap.parse_args(argv)
    data = os.path.join(a.root, "rl", "data")
    if not os.path.isdir(data):
        print("check_tables: no rl/data in %s" % a.root, file=sys.stderr)
        return 2
    bad, n = 0, 0
    for script in sorted(glob.glob(os.path.join(data, "*", "make_tables.py"))):
        folder = os.path.dirname(script)
        if not os.path.isfile(os.path.join(folder, "tables.txt")):
            continue
        n += 1
        rel = os.path.relpath(folder, a.root).replace(os.sep, "/")
        problems = check_folder(folder)
        print("%s: %s" % (rel, "ok" if not problems else "FAIL"))
        for p in problems:
            print("  " + p)
        bad += bool(problems)
    print("check_tables.py: %d folder(s), %s" % (n, "FAIL" if bad else "PASS"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
