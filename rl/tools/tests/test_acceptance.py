"""Fixtures for the fixes of the fork acceptance test: the checks job fails on every base (report mode only for F5
switches and F10 on integration), F3 scrubs engine files too, the F8 rule of rl/METHOD.md section 1, F6 'Served:',
common.show never reads a git failure as a missing file, upstream_ready.sh and a missing module, the selftest template,
the Claude Code settings and hook, the lm_head PATCH_MANIFEST.txt, and the text fixes."""
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import helpers
import check_data
import check_docs
import check_pr
import common
from test_review_fixes import HAVE_BASH, f1_run_block, run

TOOLS = helpers.TOOLS
ROOT = os.path.dirname(os.path.dirname(TOOLS))
SH = os.path.join(TOOLS, "upstream_ready.sh")
SCRUB = os.path.join(TOOLS, "scrub_check.py")
HOOK = os.path.join(ROOT, ".claude", "hooks", "ask_before_publish.sh")
BAD_ID = "i-" + "0abc123def4567890"      # built from pieces so that this file is not itself a hit
CHECKS_STEP = "Team checks F2-F8 and F10 (any failing check fails the job)"


def read(rel):
    with io.open(os.path.join(ROOT, *rel.split("/")), encoding="utf-8", newline="") as fh:
        return fh.read()


@unittest.skipUnless(HAVE_BASH and shutil.which("python"), "bash and a python on PATH are needed")
class ChecksJobOnEveryBase(unittest.TestCase):
    """The violating pull requests the grader built, replayed through the real workflow step."""

    def setUp(self):
        self.r = helpers.Repo()
        self.rt = tempfile.mkdtemp(prefix="rtemp_")
        shutil.copytree(TOOLS, os.path.join(self.r.dir, "rl", "tools"), ignore=shutil.ignore_patterns("tests", "__pycache__"))
        self.r.write("README.md", "r\n")
        self.r.write("rl/best.tsv", "key\tlabel\tmodel\tfile\tselector\tbaseline\ttier\n")
        self.r.write("docs/rl/RESULTS.md", "r\n")
        self.r.write("rl/data/note.txt", "n\n")
        self.r.write("sglang_neuron/e.py", "e = 0\n")
        self.base = self.r.commit("base")
        self.r.git("update-ref", "refs/remotes/origin/upstream-main", self.base)
        self.script = os.path.join(self.rt, "checks.sh")
        with open(self.script, "w", newline="\n") as fh:
            fh.write(f1_run_block(CHECKS_STEP))

    def tearDown(self):
        self.r.close()
        shutil.rmtree(self.rt, ignore_errors=True)

    def job(self, base_ref, labels="[]", body=""):
        head = self.r.commit("pr")
        env = {"RUNNER_TEMP": self.rt, "LABELS_JSON": labels, "PR_BODY": body, "BASE_REF": base_ref,
               "BASE_SHA": self.base, "HEAD_SHA": head, "MAIN_SHA": self.base}
        code, out, err = run(["bash", self.script], self.r.dir, env=env)
        return code, out + err

    def test_engine_file_with_an_instance_id_fails_f3_on_integration_and_on_main(self):
        self.r.write("sglang_neuron/e.py", "e = 0\nhost = '%s'\n" % BAD_ID)
        for base in ("integration", "main"):
            code, out = self.job(base, '["cpu-only"]')
            self.assertNotEqual(code, 0, base + out)
            self.assertIn("F3 scrub of every changed file failed", out)
            self.assertIn("sglang_neuron/e.py", out)

    def test_hardware_verified_pull_request_with_a_mismatched_manifest_fails_f2_on_integration(self):
        self.r.write("sglang_neuron/e.py", "e = 1\n")
        code, out = self.job("integration", '["hardware-verified"]', "Evidence: rl/data/2026-12-01_x/")
        self.assertNotEqual(code, 0, out)
        self.assertIn("F2 hardware evidence failed", out)

    def test_data_readme_with_an_instance_id_fails_on_integration(self):
        d = "rl/data/2026-12-01_t/"
        text = "Served: none\nhost %s\n" % BAD_ID
        self.r.write(d + "README.md", text)
        self.r.write(d + "SHA256SUMS", "%s  README.md\n" % helpers.sha256(text))
        code, out = self.job("integration", '["hardware-verified"]')
        self.assertNotEqual(code, 0, out)
        self.assertIn("F3 scrub of every changed file failed", out)

    def test_f10_is_report_mode_on_integration_and_fails_on_main(self):
        self.r.write("sglang_neuron/e.py", "e = 0  # see rl/README.md\n")
        code, out = self.job("integration", '["cpu-only"]')
        self.assertEqual(code, 0, out)
        self.assertIn("report mode for base integration only", out)
        code, out = self.job("main", '["hardware-verified"]')
        self.assertNotEqual(code, 0, out)
        self.assertIn("F10 engine files do not mention rl/ and symlinks failed", out)

    def test_f8_fails_on_integration(self):
        self.r.write("docs/rl/RESULTS.md", "r\nThe lm_head switch has no effect on Qwen3-8B.\n")
        code, out = self.job("integration", '["hardware-verified"]')
        self.assertNotEqual(code, 0, out)
        self.assertIn("F8 state wording on added lines failed", out)

    def test_workflow_names_only_f5_switches_and_f10_for_report_mode(self):
        t = read(".github/workflows/team-checks.yml")
        calls = re.findall(r'integration_report "([^"]+)"', t)
        self.assertEqual(calls, ["F5 switches", "F10 engine files do not mention rl/ and symlinks"])
        self.assertIn("changed_alive.z\" --allow", t)
        self.assertNotIn("team_files.z", t)
        self.assertNotIn('[ "$BASE_REF" != main ]', t)


class F8NullClaims(unittest.TestCase):
    def setUp(self):
        self.r = helpers.Repo()
        self.r.write("docs/rl/RESULTS.md", "r\n")
        self.base = self.r.commit("base")

    def tearDown(self):
        self.r.close()

    def f8(self, line):
        self.r.write("docs/rl/RESULTS.md", "r\n" + line + "\n")
        head = self.r.commit("pr")
        self.r.git("reset", "-q", "--hard", self.base)
        return check_docs.f8(self.base, head, self.r.dir)

    def test_each_phrase_needs_paired_n(self):
        for line in ("The lm_head switch has no effect on Qwen3-8B.", "No difference at concurrency 32.",
                     "The kernel shows no change in latency.", "There were no differences."):
            ok, msgs = self.f8(line)
            self.assertFalse(ok, line)
            self.assertIn("needs paired(n)", msgs[0])

    def test_paired_on_the_same_line_passes(self):
        for line in ("No difference at paired(7): median delta inside the A/A spread.",
                     "no change at `paired(n)` with n = 7 (paired(n))"):
            ok, msgs = self.f8(line)
            self.assertTrue(ok, (line, msgs))

    def test_quoted_phrase_and_the_screen_wording_pass(self):
        for line in ('Never write "no effect"; write "not distinguished at screen tier".',
                     "The tool prints `no change` when the file is equal.",
                     "> no difference (a quoted block)",
                     "Switch 1 and switch 0 are not distinguished at screen tier."):
            ok, msgs = self.f8(line)
            self.assertTrue(ok, (line, msgs))


class F6Served(unittest.TestCase):
    def setUp(self):
        self.r = helpers.Repo()
        self.r.write("rl/README.md", "x\n")
        self.base = self.r.commit("base")

    def tearDown(self):
        self.r.close()

    def folder(self, readme, manifest=False):
        d = "rl/data/2026-12-01_t/"
        files = {"README.md": readme, "a.csv": "1\n"}
        if manifest:
            files["p/PATCH_MANIFEST.txt"] = "%s  env_registry.py\n" % ("0" * 64)
        for k, v in files.items():
            self.r.write(d + k, v)
        self.r.write(d + "SHA256SUMS", "".join("%s  %s\n" % (helpers.sha256(v), k) for k, v in files.items()))
        return check_data.check(self.base, self.r.commit("pr"), self.r.dir)

    def test_served_commit_without_manifest_fails(self):
        ok, msgs = self.folder("# s\nServed: fork 9857f66\n")
        self.assertFalse(ok)
        self.assertTrue(any("served an engine patch" in m for m in msgs), msgs)

    def test_served_commit_with_manifest_passes(self):
        ok, msgs = self.folder("# s\n- Served: fork 9857f66, fork 897d927\n", manifest=True)
        self.assertTrue(ok, msgs)

    def test_served_none_passes_without_manifest(self):
        ok, msgs = self.folder("# s\nServed: none\n")
        self.assertTrue(ok, msgs)

    def test_no_served_line_fails(self):
        ok, msgs = self.folder("# s\nno line\n")
        self.assertFalse(ok)
        self.assertTrue(any("no 'Served:' line" in m for m in msgs), msgs)


class ShowNeverHidesAGitFailure(unittest.TestCase):
    def setUp(self):
        self.r = helpers.Repo()
        self.r.write("a/b.txt", "x\n")
        self.head = self.r.commit("c")

    def tearDown(self):
        self.r.close()

    def test_present_missing_and_unknown_rev(self):
        self.assertEqual(common.show(self.head, "a/b.txt", cwd=self.r.dir), b"x\n")
        self.assertIsNone(common.show(self.head, "a/none.txt", cwd=self.r.dir))
        self.assertIsNone(common.show("f" * 40, "a/b.txt", cwd=self.r.dir))
        self.assertTrue(common.exists_at(self.head, "a", cwd=self.r.dir))

    def test_a_failing_git_raises(self):
        # A folder that is not a git repository: git fails with another message, which must not read as "missing".
        d = tempfile.mkdtemp(prefix="nogit_")
        try:
            env_ceiling = os.environ.get("GIT_CEILING_DIRECTORIES")
            os.environ["GIT_CEILING_DIRECTORIES"] = os.path.dirname(d)
            try:
                with self.assertRaises(RuntimeError):
                    common.show("HEAD", "a/b.txt", cwd=d)
            finally:
                if env_ceiling is None:
                    os.environ.pop("GIT_CEILING_DIRECTORIES", None)
                else:
                    os.environ["GIT_CEILING_DIRECTORIES"] = env_ceiling
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_check_data_reports_a_git_failure_as_bad_input(self):
        d = tempfile.mkdtemp(prefix="nogit_")
        try:
            os.environ["GIT_CEILING_DIRECTORIES"] = os.path.dirname(d)
            self.assertEqual(check_data.main(["--base", "HEAD", "--head", "HEAD", "--repo", d]), 2)
        finally:
            os.environ.pop("GIT_CEILING_DIRECTORIES", None)
            shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(HAVE_BASH, "bash is needed")
class UpstreamReadyTests(unittest.TestCase):
    def setUp(self):
        self.r = helpers.Repo()
        self.r.write("sglang_neuron/a.py", "a\n")
        self.r.commit("root")
        self.r.git("branch", "upstream-main")
        self.r.git("checkout", "-q", "-b", "up/x", "upstream-main")
        self.r.write("sglang_neuron/a.py", "b\n")

    def tearDown(self):
        self.r.close()

    def run_sh(self):
        self.r.commit("c")
        p = subprocess.run(["bash", SH, "up/x", "--base", "upstream-main"], cwd=self.r.dir,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=dict(os.environ, PYTHON=sys.executable))
        return p.returncode, p.stdout.decode("utf-8", "replace") + p.stderr.decode("utf-8", "replace")

    def test_no_test_is_exit_3_and_team_paths_exit_1(self):
        code, out = self.run_sh()
        self.assertEqual(code, 3, out)
        self.r.write("rl/README.md", "t\n")
        code, out = self.run_sh()
        self.assertEqual(code, 1, out)

    def test_missing_module_is_cannot_run_exit_2(self):
        self.r.write("tests/x/selftest_needs.py", "import torch_not_installed_here_xyz\n")
        code, out = self.run_sh()
        self.assertEqual(code, 2, out)
        self.assertIn("cannot run tests: needs torch_not_installed_here_xyz", out)
        self.assertNotIn("NOT READY: tests/x/selftest_needs.py failed", out)

    def test_failing_test_is_exit_1_even_with_a_missing_module(self):
        self.r.write("tests/x/selftest_needs.py", "import torch_not_installed_here_xyz\n")
        self.r.write("tests/x/selftest_fails.py", "raise SystemExit(1)\n")
        code, out = self.run_sh()
        self.assertEqual(code, 1, out)
        self.assertIn("NOT READY: tests/x/selftest_fails.py failed", out)

    def test_the_template_copied_into_tests_makes_the_branch_ready(self):
        with open(os.path.join(TOOLS, "selftest_template.py"), "rb") as fh:
            self.r.write("tests/x/selftest_a.py", fh.read())
        self.r.write("sglang_neuron/env_registry.py", "X = 1\n")
        code, out = self.run_sh()
        self.assertEqual(code, 0, out)
        self.assertIn("READY: up/x", out)


class SelftestTemplate(unittest.TestCase):
    def test_runs_and_names_no_team_folder(self):
        text = read("rl/tools/selftest_template.py")
        self.assertIsNone(re.search(r"(?<![A-Za-z0-9_.])rl/", text))     # check F10 once it is copied under tests/
        p = subprocess.run([sys.executable, os.path.join(TOOLS, "selftest_template.py")], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)


class ClaudeSettingsAndHook(unittest.TestCase):
    def test_settings_hold_the_ask_rules_and_the_hook(self):
        s = json.loads(read(".claude/settings.json"))
        ask = s["permissions"]["ask"]
        for tool in ("Bash", "PowerShell"):
            for rule in ("git push *", "gh pr create *", "aws * run-instances *"):
                self.assertIn("%s(%s)" % (tool, rule), ask)
        self.assertNotIn("allow", s["permissions"])
        hook = s["hooks"]["PreToolUse"][0]
        self.assertEqual(hook["matcher"], "Bash|PowerShell")
        self.assertIn(".claude/hooks/ask_before_publish.sh", hook["hooks"][0]["command"])

    def test_claude_folder_is_a_team_path(self):
        self.assertTrue(common.is_team(".claude/settings.json"))
        self.assertIn("/.claude/ ", read(".github/CODEOWNERS"))

    def test_the_scrub_does_not_flag_the_settings_or_the_hook(self):
        p = subprocess.run([sys.executable, SCRUB, ".claude", "--allow", os.path.join(TOOLS, "scrub_allow.txt")],
                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(p.returncode, 0, p.stdout.decode() + p.stderr.decode())

    @unittest.skipUnless(HAVE_BASH, "bash is needed")
    def test_hook_asks_for_writes_upstream_pushes_and_launches_only(self):
        def decide(cmd):
            data = json.dumps({"tool_name": "Bash", "tool_input": {"command": cmd}}).encode()
            p = subprocess.run(["bash", HOOK], input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(p.returncode, 0, p.stderr)
            out = p.stdout.decode().strip()
            if not out:
                return None
            return json.loads(out)["hookSpecificOutput"]["permissionDecision"]
        for cmd in ("gh pr comment 25 -R sgl-project/sglang-neuron --body x",
                    "gh api repos/sgl-project/sglang-neuron/issues/25/comments -f body=x",
                    "gh -R sgl-project/sglang-neuron pr ready 25",
                    "git push mine feat/x", "git -C . push origin main", "cd a && git push",
                    "gh pr create --base main", "aws --profile p ec2 run-instances --count 1"):
            self.assertEqual(decide(cmd), "ask", cmd)
        for cmd in ("gh pr view 25 -R sgl-project/sglang-neuron", "gh api repos/sgl-project/sglang-neuron/pulls/25",
                    "git status", "git log --grep push", "aws ec2 describe-instances", "python rl/tools/best.py"):
            self.assertIsNone(decide(cmd), cmd)


class LmHeadPatchManifest(unittest.TestCase):
    D = "rl/data/2026-11-26_trn2/"

    def test_manifest_equals_the_as_run_record_line_for_line(self):
        man = [l for l in read(self.D + "lm_head/PATCH_MANIFEST.txt").split("\n") if l and not l.startswith("#")]
        rec = [l for l in read(self.D + "lm_head/patch_sha256.txt").split("\n") if l]
        self.assertEqual(len(rec), 7)
        self.assertEqual(man, rec)

    def test_sums_file_lists_it(self):
        with open(os.path.join(ROOT, *(self.D + "lm_head/PATCH_MANIFEST.txt").split("/")), "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        self.assertEqual(read(self.D + "SHA256SUMS.patch_manifest"), "%s  lm_head/PATCH_MANIFEST.txt\n" % digest)

    def test_frozen_readme_and_record_are_unchanged(self):
        legacy = read("rl/data/LEGACY.sha256")
        for rel in ("README.md", "lm_head/patch_sha256.txt"):
            with open(os.path.join(ROOT, *(self.D + rel).split("/")), "rb") as fh:
                self.assertIn("%s  %s" % (hashlib.sha256(fh.read()).hexdigest(), self.D + rel), legacy)

    def test_f2_accepts_the_files_of_the_commit_that_ran(self):
        try:
            ran = common.git(["rev-parse", "--verify", "9857f66^{commit}"], cwd=ROOT).strip()
        except RuntimeError:
            self.skipTest("commit 9857f66 is not in this clone")
        paths = ["sglang_neuron/env_registry.py", "sglang_neuron/sampling.py"]
        ok, msgs = check_pr.f2(["hardware-verified"], paths, "Evidence: rl/data/2026-11-26_trn2/lm_head/", ran, "HEAD", ROOT)
        self.assertTrue(ok, msgs)


class TextFixes(unittest.TestCase):
    def test_agents_section_0_and_the_rules_of_the_acceptance_test(self):
        t = read("AGENTS.md")
        self.assertLessEqual(len(t.splitlines()), 80)
        for needle in ("gh api user --jq .login", "you act as the lead", "core.longpaths true", "torch (CPU build)",
                       "rl/tools/selftest_template.py", "cannot run tests: needs <module>", "Served: <fork commit>",
                       "rule kept by the lead, not enforced by GitHub", "public repository citations", "`.claude/**`"):
            self.assertIn(needle, t)
        self.assertNotIn("A rejected push to `origin` (expected", t)

    def test_readme_branch_table_agrees_with_agents(self):
        t = read("README.md")
        self.assertNotIn("Start new work here.", t)
        self.assertIn("starts on `integration`", t)
        self.assertIn("`cpu-only` (did not run on a device) or `hardware-verified`", t)
        self.assertIn("(`q8_rel`)", t)
        self.assertIn("q8_rel\t", read("rl/best.tsv"))

    def test_s4_prep_pins_the_default_llmbench_url(self):
        t = read("rl/s4_prep.sh")
        self.assertIn("${LLMBENCH_URL:-https://github.com/HolboxAI/llmbench}", t)
        self.assertIn("LB_PIN=08f416347db19bb45ff2489b8a69eacef62b6875", t)

    def test_top_level_rl_scripts_are_lf_on_every_platform(self):
        lines = [l.strip() for l in read(".gitattributes").split("\n") if l.strip() and not l.startswith("#")]
        for rule in ("rl/*.py text eol=lf", "rl/*.sh text eol=lf", ".claude/** text eol=lf"):
            self.assertIn(rule, lines)

    def test_rl_readme_corrects_the_frozen_session_readme(self):
        t = read("rl/README.md")
        self.assertIn("at the top level of this repository", t)
        self.assertIn("top level of `rl/`", t)


if __name__ == "__main__":
    unittest.main()
