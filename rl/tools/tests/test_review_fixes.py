"""Fixtures for the review corrections of pull request 21: each one fails (or passes) for its stated reason."""
import json
import os
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

TOOLS = helpers.TOOLS
ROOT = os.path.dirname(os.path.dirname(TOOLS))
SH = os.path.join(TOOLS, "upstream_ready.sh")
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "team-checks.yml")
SCRUB = os.path.join(TOOLS, "scrub_check.py")
CHECK_PR = os.path.join(TOOLS, "check_pr.py")
BAD_ID = "i-" + "0123456789abcdef0"      # built from pieces so that this file is not itself a hit
BS = chr(92)
HAVE_BASH = shutil.which("bash") is not None


def run(args, cwd, input=None, env=None):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run(args, cwd=cwd, input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=e)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def on_posix_or_skip(test):
    """Names with a backslash, a trailing space or a newline cannot exist on Windows file systems."""
    return unittest.skipIf(os.name == "nt", "file names with these characters cannot be created on Windows")(test)


class GitSuppliedPaths(unittest.TestCase):
    """Item 5: a name that git reports is never stripped or rewritten."""

    def test_is_team_is_exact(self):
        for p in ("rl" + BS + "evil.py", "docs" + BS + "rl" + BS + "x.md", "README.md ", " README.md", "./README.md"):
            self.assertFalse(common.is_team(p), repr(p))
        self.assertTrue(common.is_team("rl/a b.md "))

    def test_split_paths_keeps_every_byte(self):
        self.assertEqual(common.split_paths(b"README.md \0 a.md\0"), ["README.md ", " a.md"])
        self.assertEqual(common.split_paths(b"a\nb \n"), ["a", "b "])
        self.assertEqual(common.split_paths(b"x\ny\0z\0"), ["x\ny", "z"])

    def test_f1_cli_unlabelled_odd_names_fail(self):
        for name in ("rl" + BS + "evil.py", "README.md ", "docs" + BS + "rl" + BS + "x.md"):
            code, out, _ = run([sys.executable, CHECK_PR, "f1", "--base", "main", "--labels-json", "[]", "--paths-file", "-"],
                               ROOT, input=(name + "\0").encode())
            self.assertEqual(code, 1, (name, out))
            self.assertIn("exactly one", out)

    def test_classify_z_keeps_the_exact_name(self):
        raw = ("README.md \0rl/x.md\0sglang_neuron/a b.py\0").encode()
        code, out, _ = run([sys.executable, CHECK_PR, "classify", "--kind", "engine", "-z"], ROOT, input=raw)
        self.assertEqual(out.split("\0")[:-1], ["README.md ", "sglang_neuron/a b.py"])

    @on_posix_or_skip
    def test_git_diff_names_with_backslash_and_space(self):
        r = helpers.Repo()
        try:
            r.write("a.txt", "a\n")
            base = r.commit("base")
            r.git("checkout", "-q", "-b", "pr")
            r.write("rl" + BS + "evil.py", "x\n")
            r.write("README.md ", "x\n")
            head = r.commit("pr")
            paths = [p for _, p in common.changed(base, head, r.dir)]
            self.assertEqual(sorted(paths), sorted(["rl" + BS + "evil.py", "README.md "]))
            self.assertFalse(any(common.is_team(p) for p in paths))
        finally:
            r.close()


class LabelsJson(unittest.TestCase):
    """Item 12: a label name that holds a comma stays one name."""

    def test_comma_label_is_not_cpu_only(self):
        self.assertEqual(check_pr.labels_arg(None, '["cpu-only,x"]'), ["cpu-only,x"])
        code, out, _ = run([sys.executable, CHECK_PR, "f1", "--base", "integration", "--labels-json", '["cpu-only,x"]',
                            "--paths-file", "-"], ROOT, input=b"sglang_neuron/a.py\0")
        self.assertEqual(code, 1, out)
        self.assertIn("found 0", out)

    def test_plain_json_label_passes(self):
        code, out, _ = run([sys.executable, CHECK_PR, "f1", "--base", "integration", "--labels-json", '["cpu-only"]',
                            "--paths-file", "-"], ROOT, input=b"sglang_neuron/a.py\0")
        self.assertEqual(code, 0, out)

    def test_bad_json_is_exit_2(self):
        code, _, err = run([sys.executable, CHECK_PR, "f1", "--base", "main", "--labels-json", "{", "--paths-file", "-"], ROOT, input=b"")
        self.assertEqual(code, 2, err)


class ScrubMissingAndStrip(unittest.TestCase):
    """Items 6 and 13."""

    def setUp(self):
        self.r = helpers.Repo()

    def tearDown(self):
        self.r.close()

    def scrub(self, *args):
        return run([sys.executable, SCRUB] + list(args), self.r.dir)

    def test_missing_listed_path_is_exit_2(self):
        self.r.write("list.txt", "gone.md\n")
        code, out, err = self.scrub("--files-from", "list.txt")
        self.assertEqual(code, 2, (out, err))
        self.assertIn("does not exist", err)

    def test_missing_positional_path_is_exit_2(self):
        code, out, err = self.scrub("nothing/here.md")
        self.assertEqual(code, 2, (out, err))

    def test_missing_path_in_nul_list_is_exit_2(self):
        self.r.write("list.z", b"a.md\0")
        code, _, err = self.scrub("--files-from-0", "list.z")
        self.assertEqual(code, 2, err)

    @on_posix_or_skip
    def test_name_with_trailing_space_is_scanned(self):
        self.r.write("rl/x.md ", "host %s\n" % BAD_ID)
        self.r.write("list.z", b"rl/x.md \0")
        code, out, _ = self.scrub("--files-from-0", "list.z")
        self.assertEqual(code, 1, out)
        self.assertIn("instance-id", out)

    @on_posix_or_skip
    def test_name_with_newline_is_scanned_through_nul_list(self):
        self.r.write("rl/a\nb.md", "host %s\n" % BAD_ID)
        self.r.write("list.z", b"rl/a\nb.md\0")
        code, out, _ = self.scrub("--files-from-0", "list.z")
        self.assertEqual(code, 1, out)

    @on_posix_or_skip
    def test_list_lines_are_not_stripped(self):
        self.r.write("rl/x.md", "ok\n")
        self.r.write("list.txt", "rl/x.md \n")        # a name with a trailing space that does not exist
        code, _, err = self.scrub("--files-from", "list.txt")
        self.assertEqual(code, 2, err)

    def test_allow_with_matched_text_is_narrow(self):
        self.r.write("rl/x.md", "uses: a/checkout@v4\nping root@box\n")
        self.r.write("allow.txt", "rl/x.md\tuser-at-host\treason\tcheckout@\n")
        code, out, _ = self.scrub("rl/x.md", "--allow", "allow.txt")
        self.assertEqual(code, 1, out)
        self.assertIn("'root@'", out)
        self.assertIn("accepted on purpose: 1", out)

    def test_second_match_on_a_line_is_not_hidden(self):
        self.r.write("rl/x.md", "uses: a/checkout@v4 and root@box\n")
        self.r.write("allow.txt", "rl/x.md\tuser-at-host\treason\tcheckout@\n")
        code, out, _ = self.scrub("rl/x.md", "--allow", "allow.txt")
        self.assertEqual(code, 1, out)
        self.assertIn("'root@'", out)

    def test_whole_file_entry_still_accepts_every_hit(self):
        self.r.write("rl/x.md", "a/checkout@v4 and root@box\n")
        self.r.write("allow.txt", "rl/x.md\tuser-at-host\treason\n")
        code, _, _ = self.scrub("rl/x.md", "--allow", "allow.txt")
        self.assertEqual(code, 0)

    def test_allow_file_reason_is_scrubbed(self):
        self.r.write("rl/tools/scrub_allow.txt", "rl/x.md\tinstance-id\tseen on %s\n" % BAD_ID)
        code, out, _ = self.scrub("rl/tools/scrub_allow.txt")
        self.assertEqual(code, 1, out)
        self.assertIn("instance-id", out)

    def test_allow_file_path_columns_are_not_scrubbed(self):
        self.r.write("rl/tools/scrub_allow.txt", "rl/x.md\tuser-at-host\tclean reason\troot@\n")
        code, out, _ = self.scrub("rl/tools/scrub_allow.txt")
        self.assertEqual(code, 0, out)

    def test_real_allow_file_and_workflow_are_clean(self):
        code, out, _ = run([sys.executable, SCRUB, "rl/tools/scrub_allow.txt", ".github/workflows/team-checks.yml",
                            "--allow", "rl/tools/scrub_allow.txt"], ROOT)
        self.assertEqual(code, 0, out)


class DocsFrozenAndCpu(unittest.TestCase):
    """Items 7, 8 and 9."""

    def setUp(self):
        self.r = helpers.Repo()

    def tearDown(self):
        self.r.close()

    def test_f4_new_data_readme_with_cpu_quantity_fails(self):
        self.r.write("README.md", "r\n")
        base = self.r.commit("base")
        self.r.write("rl/data/2026-12-02_trn2/README.md", "simulator: 24 of 24 pass\n")
        self.r.commit("pr")
        ok, m = check_docs.f4(self.r.dir, None, base)
        self.assertFalse(ok)
        self.assertIn("rl/data/2026-12-02_trn2/README.md:1", m[0])

    def test_f4_existing_frozen_file_is_exempt_with_base(self):
        self.r.write("rl/data/old/README.md", "simulator: 24 of 24 pass\n")
        base = self.r.commit("base")
        self.assertTrue(check_docs.f4(self.r.dir, None, base)[0])
        self.assertFalse(check_docs.f4(self.r.dir, None, None)[0])      # no base, no LEGACY list: nothing is exempt

    def test_f4_legacy_listed_file_is_exempt(self):
        self.r.write("rl/data/old/README.md", "simulator: 24 of 24 pass\n")
        self.r.write("rl/data/LEGACY.sha256", "%s  old/README.md\n" % helpers.sha256("x"))
        self.r.commit("c")
        self.assertTrue(check_docs.f4(self.r.dir)[0])

    def test_f7_new_data_readme_dead_link_fails(self):
        self.r.write("README.md", "r\n")
        base = self.r.commit("base")
        self.r.write("rl/data/new/README.md", "[x](nothing.md)\n")
        self.r.commit("pr")
        self.assertFalse(check_docs.f7(self.r.dir, base)[0])

    def test_f8_existing_frozen_file_exempt_and_legacy_exempt(self):
        self.r.write("rl/docs/REC.md", "old\n")
        base = self.r.commit("base")
        self.r.git("checkout", "-q", "-b", "pr")
        self.r.write("rl/docs/REC.md", "old\ncurrently\n")       # an edit of a frozen record: exempt from F8 (F6 owns rl/data)
        self.r.write("rl/data/imp/README.md", "currently\n")
        self.r.write("rl/data/LEGACY.sha256", "%s  imp/README.md\n" % helpers.sha256("currently\n"))
        head = self.r.commit("pr")
        ok, m = check_docs.f8(base, head, self.r.dir)
        self.assertTrue(ok, m)

    def test_f8_new_docs_record_is_linted(self):
        base = self.r.commit("base")
        self.r.git("checkout", "-q", "-b", "pr")
        self.r.write("rl/docs/NEW.md", "has not been run\n")
        head = self.r.commit("pr")
        self.assertFalse(check_docs.f8(base, head, self.r.dir)[0])

    def test_cpu_rx_matches_plurals(self):
        for line in ("It ran on 8 CPUs.", "3 simulators pass.", "we own 2 laptops", "CPU-only 3 tests"):
            self.assertTrue(check_docs.CPU_RX.search(line), line)
        self.assertFalse(check_docs.CPU_RX.search("a eucpu thing"))

    def test_f4_plural_lines_fail(self):
        self.r.write("README.md", "It ran on 8 CPUs.\n3 simulators pass.\n2 laptops\n")
        self.r.commit()
        ok, m = check_docs.f4(self.r.dir)
        self.assertFalse(ok)
        self.assertEqual(len([x for x in m if x.startswith("F4: README.md:")]), 3, m)

    def test_f4_allow_is_per_line_not_per_file(self):
        self.r.write("README.md", "cpu 24 cases\ncpu 25 cases\n")
        self.r.write("rl/tools/scrub_allow.txt", "README.md\tpublic-cpu-lint\twhy\tcpu 24 cases\n")
        self.r.commit()
        ok, m = check_docs.f4(self.r.dir, os.path.join(self.r.dir, "rl/tools/scrub_allow.txt"))
        self.assertFalse(ok)
        self.assertEqual(len([x for x in m if "holds a digit" in x]), 1, m)
        self.assertIn("README.md:2", m[0])
        self.assertIn("Allow entry text: cpu 25 cases", m[0])

    def test_f4_whole_file_entry_without_line_is_not_accepted(self):
        self.r.write("README.md", "cpu 24 cases\n")
        self.r.write("rl/tools/scrub_allow.txt", "README.md\tpublic-cpu-lint\twhy\n")
        self.r.commit()
        self.assertFalse(check_docs.f4(self.r.dir, os.path.join(self.r.dir, "rl/tools/scrub_allow.txt"))[0])


class DataExistingFolder(unittest.TestCase):
    """Item 10."""

    def setUp(self):
        self.r = helpers.Repo()
        self.r.write("rl/data/2026-11-26_trn2/README.md", "r\n")
        self.r.write("rl/data/2026-11-26_trn2/a.csv", "1\n")
        self.r.write("rl/data/2026-11-26_trn2/SHA256SUMS", "%s  a.csv\n" % helpers.sha256("1\n"))
        self.base = self.r.commit("base")
        self.r.git("checkout", "-q", "-b", "pr")

    def tearDown(self):
        self.r.close()

    def check(self):
        return check_data.check(self.base, self.r.commit("pr"), self.r.dir)

    def test_unlisted_file_added_to_existing_folder_fails(self):
        self.r.write("rl/data/2026-11-26_trn2/extra.csv", "z\n")
        ok, m = self.check()
        self.assertFalse(ok)
        self.assertIn("need a new sums file", " ".join(m))

    def test_file_listed_in_a_new_sums_file_passes(self):
        self.r.write("rl/data/2026-11-26_trn2/extra.csv", "z\n")
        self.r.write("rl/data/2026-11-26_trn2/SHA256SUMS.2026-12-02", "%s  extra.csv\n" % helpers.sha256("z\n"))
        ok, m = self.check()
        self.assertTrue(ok, m)

    def test_wrong_hash_in_the_new_sums_file_fails(self):
        self.r.write("rl/data/2026-11-26_trn2/extra.csv", "z\n")
        self.r.write("rl/data/2026-11-26_trn2/SHA256SUMS.2026-12-02", "%s  extra.csv\n" % helpers.sha256("other\n"))
        ok, m = self.check()
        self.assertFalse(ok)
        self.assertIn("does not match", " ".join(m))

    def test_second_unlisted_file_fails_beside_a_listed_one(self):
        self.r.write("rl/data/2026-11-26_trn2/extra.csv", "z\n")
        self.r.write("rl/data/2026-11-26_trn2/sneaky.csv", "s\n")
        self.r.write("rl/data/2026-11-26_trn2/SHA256SUMS.2026-12-02", "%s  extra.csv\n" % helpers.sha256("z\n"))
        ok, m = self.check()
        self.assertFalse(ok)
        self.assertIn("sneaky.csv is added to an existing folder and is not listed", " ".join(m))

    def test_changing_the_folders_own_sums_still_fails_as_modified(self):
        self.r.write("rl/data/2026-11-26_trn2/SHA256SUMS", "changed\n")
        ok, m = self.check()
        self.assertFalse(ok)
        self.assertIn("never modified or deleted", " ".join(m))

    def test_folder_cap_counts_the_total_at_the_head(self):
        old = b"o" * (2 * 1024 * 1024)
        # the existing folder is already large: five 2 MB files, then one more file of 1 KB pushes it over 10 MB
        for i in range(5):
            self.r.write("rl/data/big/f%d.bin" % i, old + bytes([i]))
        self.r.write("rl/data/big/README.md", "r\n")
        self.r.write("rl/data/big/SHA256SUMS", "x\n")
        self.base = self.r.commit("big folder")
        self.r.write("rl/data/big/extra.csv", "z\n")
        self.r.write("rl/data/big/SHA256SUMS.d", "%s  extra.csv\n" % helpers.sha256("z\n"))
        ok, m = check_data.check(self.base, self.r.commit("pr2"), self.r.dir)
        self.assertFalse(ok)
        self.assertIn("10 MB folder cap", " ".join(m))


@unittest.skipUnless(HAVE_BASH, "bash is needed")
class UpstreamReadyReview(unittest.TestCase):
    """Items 2 and 3."""

    def setUp(self):
        self.r = helpers.Repo()
        self.r.write("sglang_neuron/a.py", "a\n")
        self.r.commit("root")
        self.r.git("branch", "upstream-main")

    def tearDown(self):
        self.r.close()

    def sh(self, branch, *extra, **env):
        e = {"PYTHON": sys.executable}
        e.update(env)
        return run(["bash", SH, branch, "--base", "upstream-main"] + list(extra), self.r.dir, env=e)

    def branch_with(self, name, files):
        self.r.git("checkout", "-q", "-b", name, "upstream-main")
        for p, c in files.items():
            self.r.write(p, c)
        self.r.commit(name)

    def test_orphan_branch_is_exit_2_not_ready(self):
        self.r.git("checkout", "-q", "--orphan", "orph")
        self.r.git("rm", "-rf", "-q", ".")
        self.r.write("README.md", "x\n")
        self.r.commit("orphan")
        code, out, err = self.sh("orph", "--skip-tests")
        self.assertEqual(code, 2, (out, err))
        self.assertNotIn("READY", out)
        self.assertIn("failed", err)

    def test_missing_interpreter_is_exit_2_not_ready(self):
        self.branch_with("ok", {"sglang_neuron/a.py": "b\n"})
        code, out, err = self.sh("ok", "--skip-tests", PYTHON="/nonexistent/python")
        self.assertEqual(code, 2, (out, err))
        self.assertNotIn("READY", out)
        self.assertIn("could not classify", err)

    def test_team_branch_with_missing_interpreter_is_not_ready_either(self):
        self.branch_with("teamy", {"rl/README.md": "x\n"})
        code, out, _ = self.sh("teamy", "--skip-tests", PYTHON="/nonexistent/python")
        self.assertEqual(code, 2)
        self.assertNotIn("READY", out)

    def test_no_cpu_test_is_not_checked_and_not_ready(self):
        self.branch_with("notests", {"sglang_neuron/a.py": "b\n"})
        code, out, _ = self.sh("notests")
        self.assertEqual(code, 3, out)
        self.assertIn("NOT CHECKED", out)
        self.assertNotRegex(out, r"(?<!NOT )READY:")

    def test_selftest_script_runs_and_passes(self):
        self.branch_with("st", {"sglang_neuron/a.py": "b\n", "tests/sampler/selftest_x.py": "print('selftest ran')\n"})
        code, out, _ = self.sh("st")
        self.assertEqual(code, 0, out)
        self.assertIn("tests/sampler/selftest_x.py passes", out)
        self.assertIn("READY: st", out)

    def test_failing_selftest_is_not_ready(self):
        self.branch_with("stf", {"sglang_neuron/a.py": "b\n", "tests/sampler/selftest_x.py": "import sys\nsys.exit(3)\n"})
        code, out, _ = self.sh("stf")
        self.assertEqual(code, 1, out)
        self.assertIn("NOT READY: tests/sampler/selftest_x.py failed", out)
        self.assertNotRegex(out, r"(?<!NOT )READY:")

    def test_skip_tests_ready_line_says_what_was_checked(self):
        self.branch_with("sk", {"sglang_neuron/a.py": "b\n"})
        code, out, _ = self.sh("sk", "--skip-tests")
        self.assertEqual(code, 0, out)
        self.assertIn("READY (TEAM-path check only, CPU tests skipped): sk", out)


def real_history_has(rev):
    return subprocess.run(["git", "cat-file", "-e", rev + "^{commit}"], cwd=ROOT, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


# sha256 of the four sglang_neuron files that fork pull request 13 changed, as recorded by the 2026-11-26 lm_head run
# (patch_sha256.txt of that session), written as PATCH_MANIFEST.txt lines.
PR13_HEAD = "1cb5b13"
PR13_MANIFEST = (
    "9e591eb29aa471334ff5480b32aab3e690396255c24b2a9bc71d343d61b3f817  env_registry.py\n"
    "eea962d545bbed3387e68943b1272c77df940dcb2b94c4e0cd48e6399e777f85  models/transformer/config.py\n"
    "54e81240d7c62bdae1071a5fac8819bac4a4623e549e9da7ef17630c5ebbc51b  models/transformer/model.py\n"
    "ba55c5ef8d301a6c6a77bfae275071adcf127c31599e983d9026c4654c565283  models/transformer/weights.py\n"
)


@unittest.skipUnless(real_history_has(PR13_HEAD), "the real history up to the head of pull request 13 is not fetched (use fetch-depth 0)")
class ReplayF2Pr13(unittest.TestCase):
    """Item 11: F2 on the real head of fork pull request 13 with a manifest that holds the recorded hashes."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pr13_")
        cls.dir = os.path.join(cls.tmp, "clone")
        for args in (["clone", "-q", "--shared", "--no-checkout", ROOT, cls.dir],):
            subprocess.run(["git"] + args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cls.g("config", "core.autocrlf", "false")
        cls.g("checkout", "-q", PR13_HEAD)
        with open(os.path.join(ROOT, "rl", "tools", "tests", "fixtures", "replay_prs.json"), encoding="utf-8") as fh:
            cls.paths = json.load(fh)["13"]["paths"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def g(cls, *args):
        r = subprocess.run(["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", "-c", "commit.gpgsign=false"] + list(args),
                           cwd=cls.dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode("utf-8", "replace"))
        return r.stdout.decode().strip()

    def with_manifest(self, manifest, edit=None):
        self.g("checkout", "-q", "--detach", PR13_HEAD)
        folder = os.path.join(self.dir, "rl", "data", "2026-11-26_trn2", "lm_head")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "PATCH_MANIFEST.txt"), "w", newline="\n") as fh:
            fh.write(manifest)
        if edit:
            with open(os.path.join(self.dir, "sglang_neuron", edit), "ab") as fh:
                fh.write(b"#")
        self.g("add", "-A")
        self.g("commit", "-q", "-m", "manifest")
        return self.g("rev-parse", "HEAD")

    def f2(self, head):
        return check_pr.f2(["hardware-verified"], self.paths, "rl/data/2026-11-26_trn2/lm_head/x_p1.csv", head, "origin/main", self.dir)

    def test_recorded_hashes_match_the_real_head(self):
        ok, m = self.f2(self.with_manifest(PR13_MANIFEST))
        self.assertTrue(ok, m)
        self.assertIn("4 changed sglang_neuron file(s) match 1 manifest(s)", " ".join(m))

    def test_one_byte_changed_fails(self):
        ok, m = self.f2(self.with_manifest(PR13_MANIFEST, edit="models/transformer/model.py"))
        self.assertFalse(ok)
        self.assertIn("the merged text was not run", " ".join(m))
        self.assertIn("models/transformer/model.py", " ".join(m))

    def test_real_body_without_a_cited_folder_fails(self):
        head = self.with_manifest(PR13_MANIFEST)
        ok, m = check_pr.f2(["hardware-verified"], self.paths, "Pre-transposed LM head; no data folder cited.", head, "origin/main", self.dir)
        self.assertFalse(ok)

    def test_upstream_ready_on_the_real_head_is_not_ready(self):
        """upstream_ready.sh on the head of pull request 13 (README.md and docs/rl on it): NOT READY."""
        if not HAVE_BASH:
            self.skipTest("bash is needed")
        base = None
        for cand in ("origin/upstream-main", "upstream-main", "fork/upstream-main"):
            if real_history_has(cand):
                base = cand
                break
        if base is None:
            self.skipTest("no upstream-main ref in this checkout")
        code, out, err = run(["bash", SH, PR13_HEAD, "--base", base, "--skip-tests"], ROOT, env={"PYTHON": sys.executable})
        self.assertEqual(code, 1, (out, err))
        self.assertIn("README.md", out)
        self.assertNotIn("READY", out.replace("NOT READY", ""))


WORKFLOW_STEP = "F1 label from base and paths"


def f1_run_block():
    with open(WORKFLOW, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    i = next(k for k, l in enumerate(lines) if ("name: " + WORKFLOW_STEP) in l)
    j = next(k for k in range(i, len(lines)) if lines[k].strip() == "run: |")
    body = []
    for l in lines[j + 1:]:
        if l.strip() and not l.startswith(" " * 10):
            break
        body.append(l[10:])
    return "\n".join(body) + "\n"


@unittest.skipUnless(HAVE_BASH and shutil.which("python"), "bash and a python on PATH are needed")
class WorkflowF1RunsTheBaseCopy(unittest.TestCase):
    """Item 4: the required F1 step runs check_pr.py and common.py from the base commit."""

    def setUp(self):
        self.r = helpers.Repo()
        self.tools = {}
        for n in ("check_pr.py", "common.py"):
            with open(os.path.join(TOOLS, n), "rb") as fh:
                self.tools[n] = fh.read()
        self.rtemp = tempfile.mkdtemp(prefix="rtemp_")

    def tearDown(self):
        self.r.close()
        shutil.rmtree(self.rtemp, ignore_errors=True)

    def base_with_tools(self):
        self.r.write("README.md", "r\n")
        for n, data in self.tools.items():
            self.r.write("rl/tools/" + n, data)
        return self.r.commit("base")

    def step(self, base, head, labels, base_ref="main"):
        script = os.path.join(self.rtemp, "step.sh")
        with open(script, "w", newline="\n") as fh:
            fh.write(f1_run_block())
        env = {"RUNNER_TEMP": self.rtemp, "LABELS_JSON": json.dumps(labels), "BASE_REF": base_ref,
               "BASE_SHA": base, "HEAD_SHA": head}
        return run(["bash", script], self.r.dir, env=env)

    def tamper(self):
        text = self.tools["check_pr.py"].decode("utf-8")
        marker = "    ev = [l for l in labels"
        self.assertIn(marker, text)
        return text.replace(marker, "    return True, ['F1: tampered']\n" + marker, 1).encode("utf-8")

    def test_pull_request_cannot_loosen_f1_in_the_same_pull_request(self):
        base = self.base_with_tools()
        self.r.git("checkout", "-q", "-b", "pr")
        self.r.write("rl/tools/check_pr.py", self.tamper())
        self.r.write("sglang_neuron/a.py", "engine change, no label\n")
        head = self.r.commit("pr")
        code, out, _ = self.step(base, head, [])
        self.assertNotEqual(code, 0, out)
        self.assertIn("copy of check_pr.py and common.py at the base commit", out)
        self.assertIn("exactly one", out)

    def test_control_the_pull_requests_own_copy_would_have_passed(self):
        base = self.base_with_tools()
        self.r.git("checkout", "-q", "-b", "pr")
        self.r.write("rl/tools/check_pr.py", self.tamper())
        self.r.write("sglang_neuron/a.py", "engine change, no label\n")
        self.r.commit("pr")
        code, out, _ = run([sys.executable, os.path.join(self.r.dir, "rl", "tools", "check_pr.py"), "f1", "--base", "main",
                            "--labels-json", "[]", "--paths-file", "-"], self.r.dir, input=b"sglang_neuron/a.py\0")
        self.assertEqual(code, 0, out)

    def test_labelled_engine_pull_request_passes(self):
        base = self.base_with_tools()
        self.r.git("checkout", "-q", "-b", "pr")
        self.r.write("sglang_neuron/a.py", "x\n")
        head = self.r.commit("pr")
        code, out, _ = self.step(base, head, ["hardware-verified"])
        self.assertEqual(code, 0, out)

    def test_team_only_pull_request_passes_without_label(self):
        base = self.base_with_tools()
        self.r.git("checkout", "-q", "-b", "pr")
        self.r.write("rl/x.md", "x\n")
        head = self.r.commit("pr")
        code, out, _ = self.step(base, head, [])
        self.assertEqual(code, 0, out)

    def test_comma_label_does_not_count(self):
        base = self.base_with_tools()
        self.r.git("checkout", "-q", "-b", "pr")
        self.r.write("sglang_neuron/a.py", "x\n")
        head = self.r.commit("pr")
        code, out, _ = self.step(base, head, ["cpu-only,x"], base_ref="integration")
        self.assertNotEqual(code, 0, out)

    def test_base_without_tools_uses_the_pull_requests_copy(self):
        self.r.write("README.md", "r\n")
        base = self.r.commit("base")
        self.r.git("checkout", "-q", "-b", "pr")
        for n, data in self.tools.items():
            self.r.write("rl/tools/" + n, data)
        head = self.r.commit("pr")
        code, out, _ = self.step(base, head, [])
        self.assertEqual(code, 0, out)
        self.assertIn("the base has no rl/tools/check_pr.py yet", out)

    def test_codeowners_lists_the_f1_files_for_the_lead(self):
        with open(os.path.join(ROOT, ".github", "CODEOWNERS"), encoding="utf-8") as fh:
            lines = [l.split() for l in fh if l.strip() and not l.startswith("#")]
        star = next(i for i, l in enumerate(lines) if l[0] == "*")
        for path in ("/rl/tools/check_pr.py", "/rl/tools/common.py", "/rl/tools/tests/"):
            idx = [i for i, l in enumerate(lines) if l[0] == path]
            self.assertTrue(idx and idx[0] > star, path)       # the last matching line wins, so it must come after "*"


if __name__ == "__main__":
    unittest.main()
