"""fork_state.sh on a throw-away repository with refs under refs/remotes/origin and a stand-in gh on PATH."""
import os
import shutil
import subprocess
import tempfile
import unittest

import helpers

SH = os.path.join(helpers.TOOLS, "fork_state.sh")
FAKE_GH = r'''#!/usr/bin/env bash
# stand-in for gh: prints canned text chosen by the repository and the kind of call
[ "${FAKE_GH_FAIL:-0}" = 1 ] && { echo "gh: HTTP 401" >&2; exit 1; }
repo=""; kind="$1 $2"; heads=0; prev=""
for a in "$@"; do
  [ "$prev" = "-R" ] && repo="$a"
  case "$a" in headRefName,headRepositoryOwner) heads=1 ;; esac
  prev="$a"
done
case "$repo" in
  pengchenglu1997/sglang-neuron) cat "$FAKE_GH_DIR/fork_prs.txt" ;;
  sgl-project/sglang-neuron)
    case "$kind" in
      "issue list") cat "$FAKE_GH_DIR/up_issues.txt" ;;
      "pr list") if [ "$heads" = 1 ]; then cat "$FAKE_GH_DIR/up_heads.txt"; else cat "$FAKE_GH_DIR/up_prs.txt"; fi ;;
    esac ;;
esac
'''


@unittest.skipUnless(shutil.which("bash"), "bash is needed")
class ForkState(unittest.TestCase):
    def setUp(self):
        r = self.r = helpers.Repo()
        r.write("sglang_neuron/a.py", "a\n")
        r.commit("upstream root")
        base = r.git("rev-parse", "HEAD")
        r.git("update-ref", "refs/remotes/origin/upstream-main", base)
        r.git("update-ref", "refs/remotes/upstream/main", base)
        # integration: a carried upstream pull request and cpu-only work
        r.git("checkout", "-q", "-b", "side", base)
        r.write("sglang_neuron/carried.py", "c\n")
        r.commit("carried work")
        r.git("checkout", "-q", "-b", "integration", base)
        r.git("merge", "-q", "--no-ff", "-m", "Merge upstream PR #20 from x/y", "side")
        r.git("update-ref", "refs/remotes/origin/integration", r.git("rev-parse", "HEAD"))
        # main: engine change plus team files; feat/done is merged into it, feat/open is not
        r.git("checkout", "-q", "-b", "mainline", base)
        r.write("sglang_neuron/engine.py", "e\n")
        r.write("rl/README.md", "team\n")
        r.commit("engine and team")
        r.git("update-ref", "refs/remotes/origin/feat/done", r.git("rev-parse", "HEAD"))
        r.git("update-ref", "refs/remotes/origin/main", r.git("rev-parse", "HEAD"))
        r.git("checkout", "-q", "-b", "openwork", "mainline")
        r.write("sglang_neuron/more.py", "m\n")
        r.commit("open work")
        r.git("update-ref", "refs/remotes/origin/feat/open", r.git("rev-parse", "HEAD"))
        # an upstream head that is clean and one that carries a team path
        r.git("checkout", "-q", "-b", "clean", base)
        r.write("sglang_neuron/x.py", "x\n")
        r.commit("clean")
        r.git("update-ref", "refs/remotes/origin/fix/clean", r.git("rev-parse", "HEAD"))
        r.git("checkout", "-q", "-b", "dirty", base)
        r.write("sglang_neuron/y.py", "y\n")
        r.write("rl/leak.md", "team\n")
        r.commit("dirty")
        r.git("update-ref", "refs/remotes/origin/feat/dirty", r.git("rev-parse", "HEAD"))
        self.gh = tempfile.mkdtemp(prefix="fakegh_")
        with open(os.path.join(self.gh, "gh"), "w", newline="\n") as fh:
            fh.write(FAKE_GH)
        os.chmod(os.path.join(self.gh, "gh"), 0o755)
        self.canned("fork_prs.txt", "#30 -> main [] a team-only change\n")
        self.canned("up_prs.txt", "#22 OPEN fix/clean a fix\n#25 OPEN draft feat/dirty a feature\n#9 MERGED feat/old older\n")
        self.canned("up_issues.txt", "#15 OPEN a proposal\n")
        self.canned("up_heads.txt", "fix/clean\nfeat/dirty\n")

    def tearDown(self):
        self.r.close()
        shutil.rmtree(self.gh, ignore_errors=True)

    def canned(self, name, text):
        with open(os.path.join(self.gh, name), "w", newline="\n") as fh:
            fh.write(text)

    def run_sh(self, *args, fail=False):
        env = dict(os.environ, FAKE_GH_DIR=self.gh, PATH=self.gh + os.pathsep + os.environ["PATH"])
        if fail:
            env["FAKE_GH_FAIL"] = "1"
        p = subprocess.run(["bash", SH] + list(args), cwd=self.r.dir, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        return p.returncode, p.stdout.decode("utf-8", "replace") + p.stderr.decode("utf-8", "replace")

    def section(self, out, title):
        parts = out.split("== ")
        return [p for p in parts if p.startswith(title)][0]

    def test_branches_merged_state_and_delete_hint(self):
        code, out = self.run_sh()
        self.assertEqual(code, 0, out)
        sec = self.section(out, "branches")
        done = [l for l in sec.split("\n") if l.startswith("feat/done")][0]
        self.assertIn("main=yes", done)
        self.assertIn("merged: delete", done)
        opn = [l for l in sec.split("\n") if l.startswith("feat/open")][0]
        self.assertIn("main=no integration=no", opn)
        self.assertNotIn("delete", opn)
        main = [l for l in sec.split("\n") if l.startswith("main ")][0]
        self.assertNotIn("delete", main)

    def test_engine_paths_main_has_that_integration_lacks(self):
        code, out = self.run_sh()
        sec = self.section(out, "engine paths")
        self.assertIn("sglang_neuron/engine.py", sec)
        self.assertNotIn("rl/README.md", sec)

    def test_carried_upstream_pull_requests_and_mirror(self):
        code, out = self.run_sh()
        self.assertIn("Merge upstream PR #20", self.section(out, "carried"))
        self.assertIn("equal", self.section(out, "mirror"))
        self.r.git("update-ref", "refs/remotes/upstream/main", self.r.git("rev-parse", "refs/remotes/origin/main"))
        self.assertIn("DIFFERENT", self.section(self.run_sh()[1], "mirror"))

    def test_gh_sections_are_printed(self):
        code, out = self.run_sh()
        self.assertIn("#30 -> main", self.section(out, "open pull requests"))
        self.assertIn("#25 OPEN draft feat/dirty", self.section(out, "our pull requests"))
        self.assertIn("#15 OPEN", self.section(out, "our issues"))

    def test_upstream_cleanliness_per_head(self):
        code, out = self.run_sh()
        sec = self.section(out, "upstream cleanliness")
        clean = [l for l in sec.split("\n") if l.startswith("fix/clean")][0]
        self.assertIn("exit 0", clean)
        dirty = [l for l in sec.split("\n") if l.startswith("feat/dirty")][0]
        self.assertIn("exit 1", dirty)
        self.assertIn("rl/leak.md", sec)

    def test_gh_failure_is_said_and_exit_3(self):
        code, out = self.run_sh(fail=True)
        self.assertEqual(code, 3, out)
        self.assertIn("not read: gh failed", out)
        self.assertIn("HTTP 401", out)
        self.assertIn("feat/done", out)          # the git sections still print

    def test_no_gh_flag(self):
        code, out = self.run_sh("--no-gh")
        self.assertEqual(code, 3)
        self.assertIn("not read (--no-gh)", out)

    def test_missing_main_is_bad_input(self):
        self.r.git("update-ref", "-d", "refs/remotes/origin/main")
        self.assertEqual(self.run_sh()[0], 2)

    def test_unknown_option_and_other_remote(self):
        self.assertEqual(self.run_sh("--nope")[0], 2)
        self.assertEqual(self.run_sh("--remote", "elsewhere")[0], 2)

    def test_nothing_is_written(self):
        before = self.r.git("status", "--porcelain")
        self.run_sh()
        self.assertEqual(self.r.git("status", "--porcelain"), before)


if __name__ == "__main__":
    unittest.main()
