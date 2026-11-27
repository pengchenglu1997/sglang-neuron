"""Fixtures of rl/tools/migration (the lean import): each case builds three small git repositories (run, fork, hub) and checks
one rule of select_import.py, then the generators and the checks of the import."""
import os
import subprocess
import sys
import tempfile
import unittest

import helpers  # noqa: F401  (puts rl/tools on sys.path)

MIG = os.path.join(helpers.TOOLS, "migration")
SELECT = os.path.join(MIG, "select_import.py")
LISTS = os.path.join(MIG, "make_lists.py")
CHECK = os.path.join(MIG, "check_import.py")


def git(repo, *args):
    r = subprocess.run(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=none", "-c", "core.autocrlf=false"] + list(args),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "replace"))
    return r.stdout.decode("utf-8").strip()


def make_repo(root, name, files):
    repo = os.path.join(root, name)
    os.makedirs(repo)
    git(repo, "init", "-q")
    for rel, data in files.items():
        p = os.path.join(repo, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data if isinstance(data, bytes) else data.encode("utf-8"))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "x")
    return repo


S = "data/2026-01-01_x/"
RUN = {
    "README.md": "The run notes are in " + S + "a/notes.log. The folder `b/` is described below.\n",
    "REPRODUCE.md": "run `bash c/run_c.sh` and read d/scripts/ for the rest; the word serving is plain prose.\n",
    "tool.py": "# reads e/only_here.out and calls helper_unique.sh\n",
    S + "README.md": "see f/cfg.yaml\n",
    S + "a/notes.log": "x\n",
    S + "b/inside.sh": "x\n",
    S + "b/inside2.py": "x\n",
    S + "c/run_c.sh": "bash chain_kept.sh\n",
    S + "c/chain_kept.sh": "x\n",
    S + "c/sibling.sh": "named by nobody; a sibling of a file that is named by path is not a folder reference\n",
    S + "d/scripts/x.py": "x\n",
    S + "e/only_here.out": "x\n",
    S + "f/cfg.yaml": "x\n",
    S + "g/lost.log": "never named\n",
    S + "serving/run.log": "never named; the folder word alone is not a folder reference\n",
    S + "h/helper_unique.sh": "x\n",
    S + "h/dup.sh": "x\n",
    S + "i/dup.sh": "x\n",
    S + "m/analysis_named.py": "x\n",
    S + "n/leaf/deep.log": "named only as a leaf folder with its slash: leaf/\n",
    S + "j/hub_named.py": "x\n",
    S + "j/data.csv": "1,2\n",
    S + "j/mutual_a.sh": "mutual_b.sh\n",
    S + "j/mutual_b.sh": "mutual_a.sh\n",
}
FORK = {"README.md": "see the fork readme\n", "docs/rl/RESULTS.md": "result file k/fork_named.log\n", S + "k/fork_named.log": "x\n"}
HUB = {"ledger/I1.md": "raw file hub_named.py was used\n", "OKR.md": "dup.sh is mentioned but is not unique\n",
       "analysis/2026-01-01_note.md": "the analysis reads analysis_named.py and the folder leaf/\n"}


class SelectImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = cls.tmp.name
        run = dict(RUN)
        run[S + "k/fork_named.log"] = "x\n"
        cls.runrepo = make_repo(t, "run", run)
        cls.fork = make_repo(t, "fork", FORK)
        cls.hub = make_repo(t, "hub", HUB)
        cls.f = git(cls.runrepo, "rev-parse", "HEAD")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def select(self, *extra):
        r = subprocess.run([sys.executable, "-I", SELECT, "--run-repo", self.runrepo, "--ref", self.f, "--fork-repo", self.fork,
                            "--fork-ref", "HEAD", "--hub-repo", self.hub, "--hub-ref", "HEAD"] + list(extra),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        res = {}
        for line in r.stdout.decode().splitlines():
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            res[parts[1]] = (parts[0], parts[2] if len(parts) > 2 else "")
        return res, r.stdout

    def test_rules(self):
        res, _ = self.select()
        for p in ("README.md", "tool.py", S + "j/data.csv"):
            self.assertEqual(res[p][0], "K", p)  # outside data/, or not of a droppable type
        cases = {
            S + "a/notes.log": "repo-path",
            S + "f/cfg.yaml": "session-path",
            S + "b/inside.sh": "dir",       # `b/` names the folder: every file that sits in it counts
            S + "b/inside2.py": "dir",
            S + "d/scripts/x.py": "dir",    # a folder reference written with its parent
            S + "e/only_here.out": "session-path",
            S + "h/helper_unique.sh": "basename",
            S + "k/fork_named.log": "fork",
            S + "j/hub_named.py": "hub text",
            S + "c/run_c.sh": "session-path",
            S + "c/chain_kept.sh": "run " + S + "c/run_c.sh",  # named by a kept script: the scan repeats
        }
        for p, why in cases.items():
            self.assertEqual(res[p][0], "KX", "%s: %s" % (p, res[p]))
            self.assertIn(why, res[p][1], p)
        for p in (S + "c/sibling.sh", S + "g/lost.log", S + "serving/run.log", S + "h/dup.sh", S + "i/dup.sh", S + "j/mutual_a.sh", S + "j/mutual_b.sh"):
            self.assertEqual(res[p][0], "D", "%s: %s" % (p, res[p]))

    def test_hub_analysis_is_a_source(self):
        res, _ = self.select()
        self.assertEqual(res[S + "m/analysis_named.py"][0], "KX")
        self.assertIn("hub text", res[S + "m/analysis_named.py"][1])

    def test_leaf_folder_with_slash_is_not_a_reference(self):
        # documented rule: a one-word folder counts only as the parent relative to the session ("serving/"),
        # not as a deeper leaf folder ("leaf/" for n/leaf/deep.log)
        res, _ = self.select()
        self.assertEqual(res[S + "n/leaf/deep.log"][0], "D")

    def test_hub_reason_names_no_hub_file(self):
        res, out = self.select()
        self.assertNotIn(b"ledger/", out)
        self.assertNotIn(b"OKR.md", out)

    def test_no_dir_refs_is_stricter(self):
        res, _ = self.select("--no-dir-refs")
        self.assertEqual(res[S + "b/inside.sh"][0], "D")
        self.assertEqual(res[S + "a/notes.log"][0], "KX")

    def test_deterministic(self):
        a = self.select()[1]
        b = self.select()[1]
        self.assertEqual(a, b)

    def test_lists_and_checks(self):
        res, _ = self.select()
        dropped = sorted(p for p, v in res.items() if v[0] == "D")
        dl = os.path.join(self.tmp.name, "dropped.txt")
        with open(dl, "w", newline="\n") as f:
            f.write("\n".join(dropped) + "\n")
        out = os.path.join(self.tmp.name, "NOT_IMPORTED.tsv")
        r = subprocess.run([sys.executable, "-I", LISTS, "not-imported", "--run-repo", self.runrepo, "--ref", self.f,
                            "--dropped-list", dl, "--out", out], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        rows = [l.split("\t") for l in open(out, encoding="utf-8").read().splitlines() if not l.startswith("#")]
        self.assertEqual([x[0] for x in rows], dropped)
        want = git(self.runrepo, "rev-parse", "%s:%s" % (self.f, dropped[0]))
        self.assertEqual(rows[0][1], want)
        self.assertEqual(rows[0][3], str(len(open(os.path.join(self.runrepo, *dropped[0].split("/")), "rb").read())))
        # build the "fork" import by hand, then run the checks
        target = make_repo(self.tmp.name, "target", {"keep.txt": "k\n"})
        git(target, "fetch", "-q", self.runrepo, self.f)
        idx = os.path.join(self.tmp.name, "idx")
        env = dict(os.environ, GIT_INDEX_FILE=idx)
        subprocess.check_call(["git", "-C", target, "read-tree", self.f], env=env)
        for p in dropped:
            subprocess.check_call(["git", "-C", target, "update-index", "--force-remove", "--", p], env=env)
        tree = subprocess.check_output(["git", "-C", target, "write-tree"], env=env).decode().strip()
        git(target, "read-tree", "--prefix=rl/", "-u", tree)
        git(target, "commit", "-q", "-m", "import")
        r = subprocess.run([sys.executable, "-I", CHECK, "--run-repo", self.runrepo, "--ref", self.f, "--repo", target,
                            "--import-commit", "HEAD", "--dropped-list", dl], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        text = r.stdout.decode()
        self.assertEqual(r.returncode, 0, text + r.stderr.decode())
        self.assertIn("RESULT OK", text)
        self.assertIn("kept %d + dropped %d" % (len(res) - len(dropped), len(dropped)), text)
        # a kept text file that names a dropped path makes check (d) fail
        with open(os.path.join(target, "rl", "README.md"), "a", newline="\n") as f:
            f.write("also see g/lost.log\n")
        git(target, "add", "-A")
        git(target, "commit", "-q", "-m", "dangling")
        r = subprocess.run([sys.executable, "-I", CHECK, "--run-repo", self.runrepo, "--ref", self.f, "--repo", target,
                            "--import-commit", "HEAD~1", "--scan-commit", "HEAD", "--dropped-list", dl],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        text = r.stdout.decode()
        self.assertEqual(r.returncode, 1, text)
        self.assertIn("rl/README.md".replace("rl/", "") + " names " + S + "g/lost.log", text)

    def test_complete_import_in_two_commits(self):
        res, _ = self.select()
        dropped = sorted(p for p, v in res.items() if v[0] == "D")
        self.assertTrue(dropped)
        empty = os.path.join(self.tmp.name, "empty.txt")
        open(empty, "w").close()
        target = make_repo(self.tmp.name, "target2", {"keep.txt": "k\n"})
        git(target, "fetch", "-q", self.runrepo, self.f)
        idx = os.path.join(self.tmp.name, "idx2")
        env = dict(os.environ, GIT_INDEX_FILE=idx)
        subprocess.check_call(["git", "-C", target, "read-tree", self.f], env=env)
        for p in dropped:
            subprocess.check_call(["git", "-C", target, "update-index", "--force-remove", "--", p], env=env)
        tree = subprocess.check_output(["git", "-C", target, "write-tree"], env=env).decode().strip()
        git(target, "read-tree", "--prefix=rl/", "-u", tree)
        git(target, "commit", "-q", "-m", "lean import")
        for p in dropped:  # the second commit adds the files the lean selection left out, byte for byte
            dest = os.path.join(target, "rl", *p.split("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(subprocess.check_output(["git", "-C", self.runrepo, "show", "%s:%s" % (self.f, p)]))
        git(target, "add", "-A")
        git(target, "commit", "-q", "-m", "the rest")
        c1, c2 = git(target, "rev-parse", "HEAD~1"), git(target, "rev-parse", "HEAD")

        def check(*extra):
            r = subprocess.run([sys.executable, "-I", CHECK, "--run-repo", self.runrepo, "--ref", self.f, "--repo", target,
                                "--import-commit", c1, "--import-commit", c2, "--scan-commit", "HEAD",
                                "--dropped-list", empty] + list(extra),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return r.returncode, r.stdout.decode()
        rc, text = check()
        self.assertEqual(rc, 0, text)
        self.assertIn("kept %d + dropped 0" % len(res), text)
        self.assertIn("(f) imported paths that differ from F at the scan commit: 0", text)
        # an imported file edited afterwards (a hedged sentence in a README) is reported, unless it is named as allowed
        with open(os.path.join(target, "rl", *(S + "README.md").split("/")), "ab") as f:
            f.write(b"edited\n")
        git(target, "add", "-A")
        git(target, "commit", "-q", "-m", "edit")
        rc, text = check()
        self.assertEqual(rc, 1, text)
        self.assertIn(S + "README.md", text)
        rc, text = check("--allow-changed", S + "README.md")
        self.assertEqual(rc, 0, text)


if __name__ == "__main__":
    unittest.main()
