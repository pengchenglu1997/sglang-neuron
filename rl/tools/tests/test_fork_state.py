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
  case "$a" in number,headRefName,headRepositoryOwner) heads=1 ;; esac
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
        r.write("sglang_neuron/bound.py", "b\n")
        r.commit("engine only, rooted on upstream-main")
        r.git("update-ref", "refs/remotes/origin/feat/bound", r.git("rev-parse", "HEAD"))
        r.git("update-ref", "refs/remotes/origin/feat/inpr", r.git("rev-parse", "HEAD"))
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
        self.canned("up_heads.txt", "22 fix/clean\n25 feat/dirty\n31 feat/inpr\n")

    def tearDown(self):
        self.r.close()
        shutil.rmtree(self.gh, ignore_errors=True)

    def canned(self, name, text):
        with open(os.path.join(self.gh, name), "w", newline="\n") as fh:
            fh.write(text)

    def clean_heads(self):
        """only heads that are clean: the state is then complete and clean"""
        self.canned("up_heads.txt", "22 fix/clean\n31 feat/inpr\n")

    def run_sh(self, *args, fail=False):
        env = dict(os.environ, FAKE_GH_DIR=self.gh, PATH=self.gh + os.pathsep + os.environ["PATH"])
        if fail:
            env["FAKE_GH_FAIL"] = "1"
        if "--fetch-now" in args:
            args = [a for a in args if a != "--fetch-now"]
        else:
            args = ["--no-fetch"] + list(args)
        p = subprocess.run(["bash", SH] + list(args), cwd=self.r.dir, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        return p.returncode, p.stdout.decode("utf-8", "replace") + p.stderr.decode("utf-8", "replace")

    def section(self, out, title):
        parts = out.split("== ")
        return [p for p in parts if p.startswith(title)][0]

    def test_branches_merged_state_and_delete_hint(self):
        code, out = self.run_sh()
        self.assertEqual(code, 4, out)          # feat/dirty holds a TEAM path
        sec = self.section(out, "branches")
        done = [l for l in sec.split("\n") if l.startswith("feat/done")][0]
        self.assertIn("main=yes", done)
        self.assertIn("not the head of an open upstream pull request: delete", done)   # it holds a team path
        bound = [l for l in sec.split("\n") if l.startswith("feat/bound")][0]
        self.assertIn("upstream-bound", bound)
        self.assertIn("keep", bound)
        self.assertNotIn("delete", bound)
        inpr = [l for l in sec.split("\n") if l.startswith("feat/inpr")][0]
        self.assertIn("head of upstream #31: keep", inpr)
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

    def test_carried_subject_without_a_hash_is_found(self):
        r = self.r
        r.git("checkout", "-q", "integration")
        r.git("checkout", "-q", "-b", "side2", "refs/remotes/origin/upstream-main")
        r.write("sglang_neuron/carried2.py", "c\n")
        r.commit("carried work 2")
        r.git("checkout", "-q", "integration")
        r.git("merge", "-q", "--no-ff", "-m", "Merge upstream PR 17 (owner/repo#17), not run by us", "side2")
        r.git("update-ref", "refs/remotes/origin/integration", r.git("rev-parse", "HEAD"))
        sec = self.section(self.run_sh()[1], "carried")
        self.assertIn("Merge upstream PR 17", sec)
        self.assertIn("Merge upstream PR #20", sec)

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

    def test_exit_0_only_when_everything_is_read_and_clean(self):
        self.clean_heads()
        code, out = self.run_sh()
        self.assertEqual(code, 0, out)

    def test_dirty_head_has_its_own_exit_code(self):
        code, out = self.run_sh()
        self.assertEqual(code, 4, out)
        self.assertIn("feat/dirty (upstream #25): exit 1", out)

    def test_no_upstream_remote_ref_is_not_read_and_exit_3(self):
        self.clean_heads()
        self.r.git("update-ref", "-d", "refs/remotes/upstream/main")
        code, out = self.run_sh()
        self.assertEqual(code, 3, out)
        self.assertIn("not read", self.section(out, "mirror"))

    def test_missing_integration_or_upstream_main_is_exit_3(self):
        self.clean_heads()
        self.r.git("update-ref", "-d", "refs/remotes/origin/integration")
        code, out = self.run_sh()
        self.assertEqual(code, 3, out)
        self.assertIn("not read", self.section(out, "engine paths"))
        self.assertIn("not read", self.section(out, "carried"))

    def test_head_without_its_branch_is_exit_3_even_inside_the_loop(self):
        self.canned("up_heads.txt", "22 fix/clean\n40 fix/gone\n")
        code, out = self.run_sh()
        self.assertEqual(code, 3, out)
        self.assertIn("fix/gone (upstream #40): not read, no origin/fix/gone", out)

    def test_heads_not_read_gives_no_delete_hint(self):
        code, out = self.run_sh("--no-gh")
        done = [l for l in self.section(out, "branches").split("\n") if l.startswith("feat/done")][0]
        self.assertIn("not read", done)
        self.assertNotIn("delete after", done)

    def test_fetch_is_the_default_and_a_failed_fetch_is_bad_input(self):
        code, out = self.run_sh("--fetch-now")      # the throw-away repository has no remote named origin
        self.assertEqual(code, 2, out)
        self.assertIn("git fetch origin failed", out)

    def test_age_of_the_last_fetch_is_printed_and_stale_is_warned(self):
        code, out = self.run_sh()
        self.assertIn("age of the remote-tracking refs is unknown", self.section(out, "fetch"))
        fh = self.r.git("rev-parse", "--git-path", "FETCH_HEAD")
        fh = fh if os.path.isabs(fh) else os.path.join(self.r.dir, fh)
        with open(fh, "w") as f:
            f.write("")
        self.assertNotIn("stale", self.section(self.run_sh()[1], "fetch"))
        self.assertIn("minutes ago", self.section(self.run_sh()[1], "fetch"))
        old = 1_700_000_000
        os.utime(fh, (old, old))
        self.assertIn("may be stale", self.section(self.run_sh()[1], "fetch"))

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
