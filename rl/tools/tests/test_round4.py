"""Fixtures for round 4 of pull request 21: main named by a sha (a tag called origin/main changes nothing), the F1 checker
isolated from the other files of the base tools folder, all of rl/tools/ lead-only, LF for the workflow and CODEOWNERS."""
import json
import os
import re
import shutil
import tempfile
import unittest

import helpers
from test_review_fixes import HAVE_BASH, WORKFLOW, ROOT, TOOLS, f1_run_block, run

FETCH_STEP = "Fetch main for the sync rule"


def step_block(name):
    return f1_run_block(name)


@unittest.skipUnless(HAVE_BASH and shutil.which("python"), "bash and a python on PATH are needed")
class MainIsNamedByShaAndCheckerIsIsolated(unittest.TestCase):
    def setUp(self):
        self.up = helpers.Repo()                      # stands for the fork on GitHub
        self.rtemp = tempfile.mkdtemp(prefix="rtemp_")
        for n in ("check_pr.py", "common.py"):
            with open(os.path.join(TOOLS, n), "rb") as fh:
                self.up.write("rl/tools/" + n, fh.read())
        self.up.write("rl/a.md", "a\n")
        self.main_sha = self.up.commit("main")
        self.work = None

    def tearDown(self):
        self.up.close()
        if self.work:
            shutil.rmtree(self.work, ignore_errors=True)
        shutil.rmtree(self.rtemp, ignore_errors=True)

    def clone(self):
        self.work = tempfile.mkdtemp(prefix="work_")
        code, out, err = run(["git", "clone", "-q", self.up.dir, self.work], self.rtemp)
        self.assertEqual(code, 0, err)
        w = helpers.Repo.__new__(helpers.Repo)
        w.dir, w.links = self.work, {}
        w.git("config", "user.email", "t" + chr(64) + "example.invalid")
        w.git("config", "user.name", "t")
        w.git("config", "core.autocrlf", "false")
        w.git("config", "commit.gpgsign", "false")
        return w

    def script(self, name, block):
        path = os.path.join(self.rtemp, name)
        with open(path, "w", newline="\n") as fh:
            fh.write(block)
        return path

    def fetch_and_f1(self, w, base, head, labels, base_ref="integration"):
        env_file = os.path.join(self.rtemp, "github_env")
        open(env_file, "w").close()
        env = {"RUNNER_TEMP": self.rtemp, "GITHUB_ENV": env_file}
        code, out, err = run(["bash", self.script("fetch.sh", step_block(FETCH_STEP))], w.dir, env=env)
        self.assertEqual(code, 0, out + err)
        with open(env_file, encoding="utf-8") as fh:
            sha = dict(l.strip().split("=", 1) for l in fh if "=" in l)["MAIN_SHA"]
        self.assertEqual(sha, self.main_sha)
        env.update({"LABELS_JSON": json.dumps(labels), "BASE_REF": base_ref, "BASE_SHA": base, "HEAD_SHA": head, "MAIN_SHA": sha})
        return run(["bash", self.script("f1.sh", step_block("F1 label from base and paths"))], w.dir, env=env)

    def test_a_tag_named_origin_main_does_not_change_the_verdict(self):
        w = self.clone()
        w.git("checkout", "-q", "-b", "integ")
        integ = w.git("rev-parse", "HEAD")
        w.git("checkout", "-q", "-b", "pr")
        w.write("rl/a.md", "not what main has\n")
        head = w.commit("edit a team file differently from main")
        w.git("tag", "origin/main", head)             # a permissive tree: it equals the pull request head
        self.assertEqual(w.git("rev-parse", "origin/main"), head)    # the short name really is hijacked
        code, out, _ = self.fetch_and_f1(w, integ, head, [])
        self.assertEqual(code, 1, out)
        self.assertIn("not a sync from main", out)

    def test_a_tag_named_origin_main_does_not_hide_the_checker_of_main(self):
        # integration bootstrap: the checker comes from main; the tag points at a tree with a fake checker that prints PASS
        w = self.clone()
        w.git("checkout", "-q", "--orphan", "boot")
        w.git("rm", "-rqf", ".")
        w.write("README.md", "r\n")
        base = w.commit("integration bootstrap base")
        w.git("checkout", "-q", "-b", "fake")
        w.write("rl/tools/check_pr.py", "import sys\nprint('PASS')\nsys.exit(0)\n")
        w.write("rl/tools/common.py", "x = 1\n")
        fake = w.commit("fake tools")
        w.git("tag", "origin/main", fake)
        w.git("checkout", "-q", "-b", "pr", base)
        w.write("sglang_neuron/a.py", "engine change, no label\n")
        head = w.commit("pr")
        block = step_block("F1 label from base and paths").replace("749704d8d9c57b67bb61e5ed38c769b0815dcaac", base)
        env_file = os.path.join(self.rtemp, "github_env")
        open(env_file, "w").close()
        env = {"RUNNER_TEMP": self.rtemp, "GITHUB_ENV": env_file}
        code, out, err = run(["bash", self.script("fetch.sh", step_block(FETCH_STEP))], w.dir, env=env)
        self.assertEqual(code, 0, out + err)
        env.update({"LABELS_JSON": "[]", "BASE_REF": "integration", "BASE_SHA": base, "HEAD_SHA": head, "MAIN_SHA": self.main_sha})
        code, out, _ = run(["bash", self.script("f1b.sh", block)], w.dir, env=env)
        self.assertEqual(code, 1, out)
        self.assertIn("exactly one", out)             # the real checker of main judged it, not the tag's fake

    def test_workflow_never_names_main_by_a_short_ref(self):
        with open(WORKFLOW, encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        for l in lines:
            if "origin/main" in l:
                self.assertTrue(l.lstrip().startswith("#") or "refs/heads/main:refs/remotes/origin/main" in l
                                or "refs/remotes/origin/main^{commit}" in l, l)
        self.assertNotIn("MAIN_REF", "\n".join(lines))

    def base_with_shadows(self):
        for n in ("json.py", "re.py", "hashlib.py", "subprocess.py", "argparse.py", "os.py"):
            self.up.write("rl/tools/" + n, "raise RuntimeError('shadowed " + n + "')\n")
        return self.up.commit("base with files that shadow the standard library")

    def test_files_beside_the_checker_cannot_shadow_the_standard_library(self):
        base = self.base_with_shadows()
        w = self.clone()
        w.git("checkout", "-q", "-b", "pr")
        w.write("rl/x.md", "x\n")
        head = w.commit("pr")
        env_file = os.path.join(self.rtemp, "github_env")
        open(env_file, "w").close()
        env = {"RUNNER_TEMP": self.rtemp, "GITHUB_ENV": env_file}
        code, out, err = run(["bash", self.script("fetch.sh", step_block(FETCH_STEP))], w.dir, env=env)
        self.assertEqual(code, 0, out + err)
        env.update({"LABELS_JSON": "[]", "BASE_REF": "main", "BASE_SHA": base, "HEAD_SHA": head, "MAIN_SHA": base})
        code, out, err = run(["bash", self.script("f1.sh", step_block("F1 label from base and paths"))], w.dir, env=env)
        self.assertEqual(code, 0, out + err)
        self.assertIn("PASS", out)
        # control: run in place, the shadowing file is imported and the checker dies
        tools = os.path.join(self.rtemp, "f1-from-base", "rl", "tools")
        self.assertTrue(os.path.exists(os.path.join(tools, "json.py")))
        code, out, err = run(["python", os.path.join(tools, "check_pr.py"), "classify", "--kind", "team"], w.dir, input=b"")
        self.assertNotEqual(code, 0)
        self.assertIn("shadowed", err)


class CodeownersAndAttributes(unittest.TestCase):
    def test_all_of_rl_tools_is_lead_only_with_a_reason(self):
        with open(os.path.join(ROOT, ".github", "CODEOWNERS"), encoding="utf-8") as fh:
            text = fh.read()
        rows = [l.split() for l in text.split("\n") if l.strip() and not l.startswith("#")]
        star = next(i for i, l in enumerate(rows) if l[0] == "*")
        idx = [i for i, l in enumerate(rows) if l[0] == "/rl/tools/"]
        self.assertTrue(idx and idx[-1] > star)
        self.assertEqual(rows[idx[-1]][1], "@pengchenglu1997")
        self.assertIn("All of rl/tools/ is tier 3", text)

    def test_gitattributes_pins_lf_for_workflow_and_codeowners(self):
        with open(os.path.join(ROOT, ".gitattributes"), encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
        self.assertIn(".github/workflows/*.yml text eol=lf", lines)
        self.assertIn(".github/CODEOWNERS text eol=lf", lines)

    def test_a_crlf_clone_gets_lf_for_the_workflow_and_codeowners(self):
        r = helpers.Repo()
        try:
            with open(os.path.join(ROOT, ".gitattributes"), "rb") as fh:
                r.write(".gitattributes", fh.read())
            r.write(".github/workflows/team-checks.yml", "name: x\non: y\n")
            r.write(".github/CODEOWNERS", "* @a\n")
            r.commit("c")
            r.git("config", "core.autocrlf", "true")
            for p in (".github/workflows/team-checks.yml", ".github/CODEOWNERS"):
                os.remove(os.path.join(r.dir, *p.split("/")))
                r.git("checkout", "--", p)
                with open(os.path.join(r.dir, *p.split("/")), "rb") as fh:
                    self.assertNotIn(b"\r", fh.read(), p)
        finally:
            r.close()


if __name__ == "__main__":
    unittest.main()
