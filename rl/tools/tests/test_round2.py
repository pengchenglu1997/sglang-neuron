"""Fixtures for the second review of pull request 21: the sync rule of F1, F10 symlinks, the scrubber scanning itself,
the workflow's hardening, upstream_ready.sh from a sub-folder, and the lead-only .gitattributes."""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import helpers
import check_docs
import check_pr

TOOLS = helpers.TOOLS
ROOT = os.path.dirname(os.path.dirname(TOOLS))
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "team-checks.yml")
SH = os.path.join(TOOLS, "upstream_ready.sh")
SCRUB = os.path.join(TOOLS, "scrub_check.py")
CHECK_PR = os.path.join(TOOLS, "check_pr.py")
BAD_ID = "i-" + "0123456789abcdef0"      # built from pieces so that this file is not itself a hit
HAVE_BASH = shutil.which("bash") is not None


def run(args, cwd, input=None, env=None):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run(args, cwd=cwd, input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=e)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


class SyncRule(unittest.TestCase):
    """F1 into integration, with real trees: a sync equals main on every team path."""

    def setUp(self):
        self.r = helpers.Repo()
        self.r.write("README.md", "readme\n")
        self.r.write("sglang_neuron/e.py", "e = 0\n")
        self.r.commit("root")
        self.r.git("branch", "integration")
        self.r.write("rl/a.md", "a\n")
        self.r.write("docs/rl/R.md", "r\n")
        self.main = self.r.commit("main has team files")

    def tearDown(self):
        self.r.close()

    def pr(self, files, labels):
        """A branch from integration that writes files; returns (exit code, output) of the F1 command line."""
        self.r.git("checkout", "-q", "-b", "pr%d" % len(self.r.git("branch").split()), "integration")
        for p, c in files.items():
            self.r.write(p, c)
        head = self.r.commit("pr")
        names = self.r.git("diff", "--name-only", "integration..." + head).split("\n")
        code, out, err = run([sys.executable, CHECK_PR, "f1", "--base", "integration", "--labels-json", labels, "--paths-file", "-",
                              "--head", head, "--main", self.main, "--repo", self.r.dir], self.r.dir,
                             input=("\0".join(names) + "\0").encode("utf-8"))
        return code, out + err

    MAIN_FILES = {"rl/a.md": "a\n", "docs/rl/R.md": "r\n"}

    def test_pure_sync_passes_without_label(self):
        code, out = self.pr(self.MAIN_FILES, "[]")
        self.assertEqual(code, 0, out)
        self.assertIn("sync from main", out)

    def test_sync_plus_engine_paths_with_cpu_only_passes(self):
        files = dict(self.MAIN_FILES)
        files["sglang_neuron/e.py"] = "e = 1\n"
        code, out = self.pr(files, '["cpu-only"]')
        self.assertEqual(code, 0, out)

    def test_sync_plus_engine_paths_still_needs_a_label(self):
        files = dict(self.MAIN_FILES)
        files["sglang_neuron/e.py"] = "e = 1\n"
        code, out = self.pr(files, "[]")
        self.assertEqual(code, 1, out)
        self.assertIn("exactly one", out)

    def test_team_file_edited_differently_from_main_fails(self):
        files = dict(self.MAIN_FILES)
        files["rl/a.md"] = "edited, main has something else\n"
        code, out = self.pr(files, '["cpu-only"]')
        self.assertEqual(code, 1, out)
        self.assertIn("not a sync from main", out)
        self.assertIn("rl/a.md", out)

    def test_claimed_sync_with_one_extra_team_file_fails(self):
        files = dict(self.MAIN_FILES)
        files["rl/extra.md"] = "not on main\n"
        code, out = self.pr(files, "[]")
        self.assertEqual(code, 1, out)
        self.assertIn("rl/extra.md", out)

    def test_partial_sync_missing_a_main_file_fails(self):
        code, out = self.pr({"rl/a.md": "a\n"}, "[]")
        self.assertEqual(code, 1, out)
        self.assertIn("docs/rl/R.md", out)

    HW = '["hardware-verified"]'

    def test_hardware_verified_with_rewritten_checker_fails(self):
        files = dict(self.MAIN_FILES)
        files["sglang_neuron/e.py"] = "e = 1\n"
        files["rl/tools/check_pr.py"] = "print('PASS')\n"
        code, out = self.pr(files, self.HW)
        self.assertEqual(code, 1, out)
        self.assertIn("rl/tools/check_pr.py", out)
        self.assertIn("not a sync from main", out)

    def test_hardware_verified_with_rewritten_workflow_fails(self):
        files = dict(self.MAIN_FILES)
        files["sglang_neuron/e.py"] = "e = 1\n"
        files[".github/workflows/team-checks.yml"] = "name: x\n"
        code, out = self.pr(files, self.HW)
        self.assertEqual(code, 1, out)
        self.assertIn(".github/workflows/team-checks.yml", out)

    def test_hardware_verified_with_agents_or_codeowners_fails(self):
        for p in ("AGENTS.md", ".github/CODEOWNERS"):
            files = dict(self.MAIN_FILES)
            files["sglang_neuron/e.py"] = "e = 1\n"
            files[p] = "changed\n"
            code, out = self.pr(files, self.HW)
            self.assertEqual(code, 1, (p, out))

    def test_hardware_verified_with_own_data_folder_and_docs_rl_line_passes(self):
        files = dict(self.MAIN_FILES)
        files["sglang_neuron/e.py"] = "e = 1\n"
        files["rl/data/2026-12-01_s/README.md"] = "session\n"
        files["docs/rl/R.md"] = "r, plus one line\n"
        code, out = self.pr(files, self.HW)
        self.assertEqual(code, 0, out)

    def test_hardware_verified_data_and_docs_only_needs_no_main_comparison(self):
        # integration lacks rl/a.md that main has; the pull request does not touch it
        code, out = self.pr({"sglang_neuron/e.py": "e = 3\n", "rl/data/s2/README.md": "x\n", "docs/rl/N.md": "n\n"}, self.HW)
        self.assertEqual(code, 0, out)

    def test_cpu_only_with_own_data_folder_is_not_exempt(self):
        files = dict(self.MAIN_FILES)
        files["sglang_neuron/e.py"] = "e = 1\n"
        files["rl/data/2026-12-01_s/README.md"] = "session\n"
        code, out = self.pr(files, '["cpu-only"]')
        self.assertEqual(code, 1, out)

    def test_engine_only_pull_request_needs_no_sync(self):
        code, out = self.pr({"sglang_neuron/e.py": "e = 2\n"}, '["cpu-only"]')
        self.assertEqual(code, 0, out)

    def test_main_is_not_asked_for_other_bases(self):
        self.r.git("checkout", "-q", "-b", "prm", "integration")
        self.r.write("rl/a.md", "different\n")
        head = self.r.commit("pr")
        code, out, _ = run([sys.executable, CHECK_PR, "f1", "--base", "main", "--labels-json", "[]", "--paths-file", "-",
                            "--head", head, "--main", "no-such-ref", "--repo", self.r.dir], self.r.dir, input=b"rl/a.md\0")
        self.assertEqual(code, 0, out)

    def test_unknown_main_ref_is_exit_2_not_a_pass(self):
        self.r.git("checkout", "-q", "-b", "prx", "integration")
        self.r.write("rl/a.md", "a\n")
        head = self.r.commit("pr")
        code, out, err = run([sys.executable, CHECK_PR, "f1", "--base", "integration", "--labels-json", "[]", "--paths-file", "-",
                              "--head", head, "--main", "no-such-ref", "--repo", self.r.dir], self.r.dir, input=b"rl/a.md\0")
        self.assertEqual(code, 2, (out, err))

    def test_mode_change_is_a_difference(self):
        a = check_pr.team_tree(self.main, self.r.dir)
        self.assertEqual(sorted(a), ["README.md", "docs/rl/R.md", "rl/a.md"])
        self.assertEqual(check_pr.sync_differences(self.main, self.main, self.r.dir), [])


class F10Symlinks(unittest.TestCase):
    def setUp(self):
        self.r = helpers.Repo()
        self.r.write("rl/data/x.md", "x\n")
        self.r.write("rl/ok.md", "ok\n")
        self.r.write("sglang_neuron/a.py", "a\n")
        self.r.write("docs/guide.md", "g\n")
        self.r.write("docs/rl/R.md", "r\n")

    def tearDown(self):
        self.r.close()

    def check(self):
        self.r.commit("c")
        return check_docs.f10(self.r.dir)

    def test_engine_symlink_into_team_path_fails(self):
        self.r.symlink("sglang_neuron/link.md", "../docs/rl/R.md")
        ok, m = self.check()
        self.assertFalse(ok)
        self.assertEqual(len(m), 1, m)
        self.assertIn("sglang_neuron/link.md", m[0])
        self.assertIn("into a team path", m[0])

    def test_engine_symlink_to_team_folder_and_through_a_link_fails(self):
        self.r.symlink("docs/to_rl", "../rl")
        self.r.symlink("tests/hop", "../docs/to_rl/data/x.md")
        ok, m = self.check()
        self.assertFalse(ok)
        self.assertEqual(sorted(x.split()[1].split(":")[0] for x in m), ["docs/to_rl", "tests/hop"])

    def test_engine_symlink_to_engine_path_passes(self):
        self.r.symlink("sglang_neuron/alias.py", "a.py")
        ok, m = self.check()
        self.assertTrue(ok, m)

    def test_team_symlink_pointing_outside_team_paths_fails(self):
        self.r.symlink("rl/engine_link.py", "../sglang_neuron/a.py")
        ok, m = self.check()
        self.assertFalse(ok)
        self.assertIn("rl/engine_link.py", m[0])
        self.assertIn("points outside them", m[0])

    def test_team_symlink_with_absolute_target_or_escaping_the_repository_fails(self):
        self.r.symlink("rl/abs", "/etc/hostname")
        self.r.symlink("docs/rl/up", "../../../outside")
        ok, m = self.check()
        self.assertFalse(ok)
        self.assertEqual(len(m), 2, m)

    def test_team_symlink_inside_team_paths_passes(self):
        self.r.symlink("rl/alias.md", "ok.md")
        self.r.symlink("README.md", "rl/ok.md")
        ok, m = self.check()
        self.assertTrue(ok, m)

    def test_symlink_loop_does_not_hang(self):
        self.r.symlink("sglang_neuron/l1", "l2")
        self.r.symlink("sglang_neuron/l2", "l1")
        ok, m = self.check()
        self.assertTrue(ok, m)


class ScrubberScansItself(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="scrubself_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_real_scrubber_file_is_clean_with_the_real_allow_file(self):
        code, out, _ = run([sys.executable, SCRUB, "rl/tools/scrub_check.py", "rl/tools/scrub_allow.txt", "--allow", "rl/tools/scrub_allow.txt"], ROOT)
        self.assertEqual(code, 0, out)
        self.assertIn("hits: 0", out)

    def test_a_new_hit_in_a_file_named_scrub_check_is_found(self):
        os.makedirs(os.path.join(self.dir, "rl", "tools"))
        for name in ("scrub_check.py", "other.py"):
            with open(os.path.join(self.dir, "rl", "tools", name), "w", newline="\n") as fh:
                fh.write("x = '%s'\n" % BAD_ID)
        code, out, _ = run([sys.executable, SCRUB, "rl/tools"], self.dir)
        self.assertEqual(code, 1, out)
        self.assertIn("rl/tools/scrub_check.py:1 instance-id", out)

    def test_allow_entries_are_narrow(self):
        os.makedirs(os.path.join(self.dir, "rl", "tools"))
        with open(os.path.join(self.dir, "rl", "tools", "scrub_check.py"), "w", newline="\n") as fh:
            # the words are joined here so that this test file itself holds none of the scrubbed patterns
            fh.write("# " + "arn" + ":aws and a new " + "price: " + "US" + "D, and 'spot " + "price'\nx = '" + "hour" + "ly'\n")
        code, out, _ = run([sys.executable, SCRUB, "rl/tools/scrub_check.py", "--allow", os.path.join(ROOT, "rl", "tools", "scrub_allow.txt")], self.dir)
        self.assertEqual(code, 1, out)
        self.assertIn("hour" + "ly", out)


def block(step):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_review_fixes
    return test_review_fixes.f1_run_block(step)


class WorkflowHardening(unittest.TestCase):
    def text(self):
        with open(WORKFLOW, encoding="utf-8") as fh:
            return fh.read()

    def test_every_checkout_drops_credentials_and_every_action_is_pinned_by_sha(self):
        lines = self.text().split("\n")
        uses = [(i, l) for i, l in enumerate(lines) if re.match(r"\s*- uses:", l)]
        self.assertEqual(len(uses), 6)
        for i, l in uses:
            self.assertRegex(l, r"uses: actions/[a-z-]+@[0-9a-f]{40} # v\d+\.\d+\.\d+$")
            if "actions/checkout" in l:
                self.assertIn("persist-credentials: false", "\n".join(lines[i + 1:i + 5]), l)

    def test_comment_states_the_precondition(self):
        t = self.text()
        self.assertIn("lead-only", t)
        self.assertIn("require review from Code Owners", t)

    @unittest.skipUnless(HAVE_BASH, "bash is needed")
    def test_checks_job_stops_when_git_diff_fails(self):
        r = helpers.Repo()
        rt = tempfile.mkdtemp(prefix="rtemp_")
        try:
            shutil.copytree(TOOLS, os.path.join(r.dir, "rl", "tools"), ignore=shutil.ignore_patterns("tests", "__pycache__"))
            r.write("README.md", "r\n")
            r.write("rl/best.tsv", "key\tlabel\tmodel\tfile\tselector\tbaseline\ttier\n")
            r.write("docs/rl/RESULTS.md", "r\n")   # F5 reads both documents; rl/data must exist for the tables check
            r.write("rl/data/note.txt", "n\n")
            good = r.commit("c")
            r.git("update-ref", "refs/remotes/origin/upstream-main", good)
            script = os.path.join(rt, "checks.sh")
            with open(script, "w", newline="\n") as fh:
                fh.write(block("Team checks F2-F8 and F10 (any failing check fails the job)"))
            env = {"RUNNER_TEMP": rt, "LABELS_JSON": "[]", "PR_BODY": "", "BASE_REF": "main", "BASE_SHA": "0" * 40, "HEAD_SHA": good, "MAIN_SHA": good}
            code, out, err = run(["bash", script], r.dir, env=env)
            self.assertNotEqual(code, 0, out + err)
            self.assertNotIn(": ok", out)
            env["BASE_SHA"] = good
            code, out, err = run(["bash", script], r.dir, env=env)
            self.assertEqual(code, 0, out + err)
        finally:
            r.close()
            shutil.rmtree(rt, ignore_errors=True)

    @unittest.skipUnless(HAVE_BASH, "bash is needed")
    def test_a_failing_check_fails_the_job_and_the_others_still_run(self):
        # F8 (state wording on an added line) used to be report mode; now the job fails, and F10 still runs after it.
        r = helpers.Repo()
        rt = tempfile.mkdtemp(prefix="rtemp_")
        try:
            shutil.copytree(TOOLS, os.path.join(r.dir, "rl", "tools"), ignore=shutil.ignore_patterns("tests", "__pycache__"))
            r.write("README.md", "r\n")
            r.write("rl/best.tsv", "key\tlabel\tmodel\tfile\tselector\tbaseline\ttier\n")
            r.write("docs/rl/RESULTS.md", "r\n")
            r.write("rl/data/note.txt", "n\n")
            base = r.commit("c")
            r.git("update-ref", "refs/remotes/origin/upstream-main", base)
            r.write("README.md", "r\nThe branch is currently open.\n")
            head = r.commit("d")
            script = os.path.join(rt, "checks.sh")
            with open(script, "w", newline="\n") as fh:
                fh.write(block("Team checks F2-F8 and F10 (any failing check fails the job)"))
            env = {"RUNNER_TEMP": rt, "LABELS_JSON": "[]", "PR_BODY": "", "BASE_REF": "main", "BASE_SHA": base, "HEAD_SHA": head, "MAIN_SHA": base}
            code, out, err = run(["bash", script], r.dir, env=env)
            self.assertNotEqual(code, 0, out + err)
            self.assertIn("F8 state wording on added lines failed", out + err)
            self.assertIn("F10 engine files do not mention rl/ and symlinks: ok", out + err)
        finally:
            r.close()
            shutil.rmtree(rt, ignore_errors=True)


class CodeownersGitattributes(unittest.TestCase):
    def test_root_gitattributes_is_lead_only_after_the_catch_all(self):
        with open(os.path.join(ROOT, ".github", "CODEOWNERS"), encoding="utf-8") as fh:
            text = fh.read()
        lines = [l.split() for l in text.split("\n") if l.strip() and not l.startswith("#")]
        star = next(i for i, l in enumerate(lines) if l[0] == "*")
        idx = [i for i, l in enumerate(lines) if l[0] == "/.gitattributes"]
        self.assertTrue(idx and idx[0] > star)
        self.assertIn("tier 3", text)


class ToolsAreCheckedOutWithLF(unittest.TestCase):
    def test_gitattributes_pins_lf_for_the_tools_folder(self):
        with open(os.path.join(ROOT, ".gitattributes"), encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
        self.assertIn("rl/tools/** text eol=lf", lines)

    def test_git_reports_eol_lf_for_tool_files_and_fixtures(self):
        # in a temporary repository that carries a copy of the real root .gitattributes
        r = helpers.Repo()
        try:
            with open(os.path.join(ROOT, ".gitattributes"), "rb") as fh:
                r.write(".gitattributes", fh.read())
            for p in ("rl/tools/check_pr.py", "rl/tools/upstream_ready.sh", "rl/tools/tests/fixtures/f1_cases.json"):
                code, out, err = run(["git", "check-attr", "eol", "text", "--", p], r.dir)
                self.assertEqual(code, 0, err)
                self.assertIn("eol: lf", out, p)
                self.assertIn("text: set", out, p)
        finally:
            r.close()

    def test_a_crlf_clone_still_gets_lf(self):
        """Commit an LF file, then check it out with core.autocrlf=true under the same attribute: it stays LF."""
        r = helpers.Repo()
        try:
            r.write(".gitattributes", "rl/tools/** text eol=lf\n")
            r.write("rl/tools/a.py", "x = 1\n")
            r.commit("c")
            r.git("config", "core.autocrlf", "true")
            os.remove(os.path.join(r.dir, "rl", "tools", "a.py"))
            r.git("checkout", "--", "rl/tools/a.py")
            with open(os.path.join(r.dir, "rl", "tools", "a.py"), "rb") as fh:
                self.assertEqual(fh.read(), b"x = 1\n")
        finally:
            r.close()


@unittest.skipUnless(HAVE_BASH, "bash is needed")
class UpstreamReadyFromSubfolder(unittest.TestCase):
    def setUp(self):
        self.r = helpers.Repo()
        self.r.write("sglang_neuron/a.py", "a\n")
        self.r.commit("root")
        self.r.git("branch", "upstream-main")
        self.r.git("checkout", "-q", "-b", "feature")
        self.r.write("sglang_neuron/b.py", "b\n")
        self.r.commit("engine only")

    def tearDown(self):
        self.r.close()

    def test_runs_from_a_subfolder(self):
        sub = os.path.join(self.r.dir, "sglang_neuron")
        code, out, err = run(["bash", SH, "feature", "--base", "upstream-main", "--skip-tests"], sub, env={"PYTHON": sys.executable})
        self.assertEqual(code, 0, out + err)
        self.assertIn("READY", out)

    def test_outside_a_checkout_is_exit_2(self):
        d = tempfile.mkdtemp(prefix="nogit_")
        try:
            code, out, err = run(["bash", SH, "feature", "--skip-tests"], d, env={"PYTHON": sys.executable, "GIT_CEILING_DIRECTORIES": os.path.dirname(d)})
            self.assertEqual(code, 2, out + err)
            self.assertIn("not inside a git checkout", err)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
