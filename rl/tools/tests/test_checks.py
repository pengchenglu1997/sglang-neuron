"""F2, F3 (scrubber), F4, F6, F7, F8, F10 and upstream_ready.sh, each on a throw-away repository."""
import os
import subprocess
import sys
import unittest

import helpers
import check_data
import check_docs
import check_pr

TOOLS = helpers.TOOLS
SH = os.path.join(TOOLS, "upstream_ready.sh")
BAD_ID = "i-" + "0123456789abcdef0"      # built from pieces so that this file is not itself a hit


class F2(unittest.TestCase):
    def setUp(self):
        self.r = helpers.Repo()
        self.src = "x = 1\n"
        self.r.write("sglang_neuron/a.py", "x = 0\n")
        self.r.write("README.md", "r\n")
        self.base = self.r.commit("base")

    def tearDown(self):
        self.r.close()

    def head(self, manifest_sha, source=None):
        self.r.git("checkout", "-q", "-b", "pr")
        self.r.write("sglang_neuron/a.py", source or self.src)
        self.r.write("rl/data/2026-12-01_t/PATCH_MANIFEST.txt", "# commit abc\n%s  a.py\n" % manifest_sha)
        return self.r.commit("pr")

    def run_f2(self, labels, body, head):
        return check_pr.f2(labels, ["sglang_neuron/a.py"], body, head, self.base, self.r.dir)

    def test_match_passes(self):
        h = self.head(helpers.sha256(self.src))
        ok, m = self.run_f2(["hardware-verified"], "evidence rl/data/2026-12-01_t/serving/x.csv", h)
        self.assertTrue(ok, m)

    def test_mismatch_fails(self):
        h = self.head(helpers.sha256("other\n"))
        ok, m = self.run_f2(["hardware-verified"], "see rl/data/2026-12-01_t/x.csv", h)
        self.assertFalse(ok)
        self.assertIn("merged text was not run", " ".join(m))

    def test_no_citation_fails(self):
        h = self.head(helpers.sha256(self.src))
        ok, m = self.run_f2(["hardware-verified"], "no evidence", h)
        self.assertFalse(ok)
        self.assertIn("cites no rl/data", " ".join(m))

    def test_missing_folder_fails(self):
        h = self.head(helpers.sha256(self.src))
        ok, m = self.run_f2(["hardware-verified"], "see rl/data/2099-01-01_none/x", h)
        self.assertFalse(ok)
        self.assertIn("exists neither", " ".join(m))

    def test_file_not_in_manifest_fails(self):
        self.r.git("checkout", "-q", "-b", "pr")
        self.r.write("sglang_neuron/a.py", self.src)
        self.r.write("rl/data/2026-12-01_t/PATCH_MANIFEST.txt", "%s  b.py\n" % helpers.sha256(self.src))
        h = self.r.commit("pr")
        ok, m = self.run_f2(["hardware-verified"], "rl/data/2026-12-01_t/", h)
        self.assertFalse(ok)

    def test_folder_on_main_counts(self):
        self.r.write("rl/data/2026-12-01_t/PATCH_MANIFEST.txt", "%s  a.py\n" % helpers.sha256(self.src))
        base = self.r.commit("data")
        self.r.git("checkout", "-q", "-b", "pr")
        self.r.write("sglang_neuron/a.py", self.src)
        h = self.r.commit("pr")
        ok, m = check_pr.f2(["hardware-verified"], ["sglang_neuron/a.py"], "rl/data/2026-12-01_t/x", h, base, self.r.dir)
        self.assertTrue(ok, m)

    def test_cpu_only_not_checked(self):
        ok, m = check_pr.f2(["cpu-only"], ["sglang_neuron/a.py"], "", "HEAD", "main", self.r.dir)
        self.assertTrue(ok)


class Scrub(unittest.TestCase):
    def run_scrub(self, text, name="rl/x.md", allow=None):
        r = helpers.Repo()
        try:
            r.write(name, text)
            args = [sys.executable, os.path.join(TOOLS, "scrub_check.py"), name]
            if allow:
                r.write("allow.txt", allow)
                args += ["--allow", "allow.txt"]
            p = subprocess.run(args, cwd=r.dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return p.returncode, p.stdout.decode()
        finally:
            r.close()

    def test_instance_id_fails(self):
        code, out = self.run_scrub("host %s up\n" % BAD_ID)
        self.assertEqual(code, 1)
        self.assertIn("instance-id", out)

    def test_clean_passes(self):
        self.assertEqual(self.run_scrub("plain text\n")[0], 0)

    def test_allow_entry_with_reason(self):
        code, out = self.run_scrub("host %s\n" % BAD_ID, allow="rl/x.md\tinstance-id\ta documented example\n")
        self.assertEqual(code, 0)
        self.assertIn("a documented example", out)

    def test_carriage_return_fails(self):
        self.assertEqual(self.run_scrub("a\r\nb\r\n")[0], 1)


class Data(unittest.TestCase):
    def setUp(self):
        self.r = helpers.Repo()
        self.r.write("rl/README.md", "x\n")
        self.r.write("rl/data/old/README.md", "old\n")
        self.r.write("rl/data/old/a.csv", "1\n")
        self.base = self.r.commit("base")
        self.r.git("checkout", "-q", "-b", "pr")

    def tearDown(self):
        self.r.close()

    def run_check(self):
        h = self.r.commit("pr")
        return check_data.check(self.base, h, self.r.dir)

    def good_folder(self, extra=None):
        d = "rl/data/2026-12-01_t/"
        self.r.write(d + "README.md", "r\nServed: none\n")   # F6: a new folder says what it served
        self.r.write(d + "a.csv", "1,2\n")
        self.r.write(d + "SHA256SUMS", "%s  README.md\n%s  a.csv\n" % (helpers.sha256("r\nServed: none\n"), helpers.sha256("1,2\n")))

    def test_good_folder(self):
        self.good_folder()
        ok, m = self.run_check()
        self.assertTrue(ok, m)

    def test_missing_sums_and_readme(self):
        self.r.write("rl/data/2026-12-01_t/a.csv", "1\n")
        ok, m = self.run_check()
        self.assertFalse(ok)
        t = " ".join(m)
        self.assertIn("no README.md", t)
        self.assertIn("no SHA256SUMS", t)

    def test_wrong_hash(self):
        self.good_folder()
        self.r.write("rl/data/2026-12-01_t/a.csv", "9,9\n")
        ok, m = self.run_check()
        self.assertFalse(ok)
        self.assertIn("does not match", " ".join(m))

    def test_unlisted_file(self):
        self.good_folder()
        self.r.write("rl/data/2026-12-01_t/extra.csv", "z\n")
        ok, m = self.run_check()
        self.assertFalse(ok)
        self.assertIn("not listed", " ".join(m))

    def test_modifying_existing_data_fails(self):
        self.r.write("rl/data/old/a.csv", "2\n")
        ok, m = self.run_check()
        self.assertFalse(ok)
        self.assertIn("never modified or deleted", " ".join(m))

    def test_deleting_existing_data_fails(self):
        self.r.remove("rl/data/old/a.csv")
        ok, m = self.run_check()
        self.assertFalse(ok)

    def test_file_cap(self):
        self.good_folder()
        big = b"x" * (2 * 1024 * 1024 + 1)
        self.r.write("rl/data/2026-12-01_t/big.bin", big)
        self.r.write("rl/data/2026-12-01_t/SHA256SUMS", "%s  README.md\n%s  a.csv\n%s  big.bin\n"
                     % (helpers.sha256("r\n"), helpers.sha256("1,2\n"), helpers.sha256(big)))
        ok, m = self.run_check()
        self.assertFalse(ok)
        self.assertIn("2 MB file cap", " ".join(m))

    def test_legacy_sums(self):
        self.r.write("rl/data/LEGACY.sha256", "%s  rl/data/old/a.csv\n" % helpers.sha256("WRONG"))
        ok, m = self.run_check()
        self.assertFalse(ok)
        self.assertIn("differs from LEGACY.sha256", " ".join(m))

    def test_legacy_sums_relative_to_data(self):
        self.r.write("rl/data/LEGACY.sha256", "%s  old/a.csv\n" % helpers.sha256("1\n"))
        ok, m = self.run_check()
        self.assertTrue(ok, m)


class Docs(unittest.TestCase):
    def setUp(self):
        self.r = helpers.Repo()

    def tearDown(self):
        self.r.close()

    def test_f4_cpu_digit(self):
        self.r.write("docs/rl/RESULTS.md", "The simulator run passed 24 of 24 cases.\n")
        self.r.commit()
        ok, m = check_docs.f4(self.r.dir)
        self.assertFalse(ok)
        self.assertIn("F4: docs/rl/RESULTS.md:1", m[0])

    def test_f4_code_span_and_fence_are_exempt(self):
        self.r.write("README.md", "CPU test: `bash tests/rlkit/run_all.sh 24`.\n\n```\ncpu 24\n```\nThe laptop is not named here.\n")
        self.r.commit()
        ok, m = check_docs.f4(self.r.dir)
        self.assertTrue(ok, m)

    def test_f4_allow_entry(self):
        self.r.write("rl/README.md", "cpu 24 cases\n")
        self.r.write("rl/tools/scrub_allow.txt", "rl/README.md\tpublic-cpu-lint\twhy\tcpu 24 cases\n")
        self.r.commit()
        ok, m = check_docs.f4(self.r.dir, os.path.join(self.r.dir, "rl/tools/scrub_allow.txt"))
        self.assertTrue(ok, m)

    def test_f7_dead_link_and_path(self):
        self.r.write("docs/rl/A.md", "[x](B.md) [y](../../README.md) `rl/nothing.py` rl/data/<s>/ [z](https://example.invalid/a)\n")
        self.r.write("README.md", "r\n")
        self.r.commit()
        ok, m = check_docs.f7(self.r.dir)
        t = " ".join(m)
        self.assertFalse(ok)
        self.assertIn("link B.md", t)
        self.assertIn("path rl/nothing.py", t)
        self.assertNotIn("README.md does not", t)
        self.assertEqual(len(m), 2, m)

    def test_f7_agents_line_cap(self):
        self.r.write("AGENTS.md", "x\n" * 81)
        self.r.commit()
        ok, m = check_docs.f7(self.r.dir)
        self.assertFalse(ok)
        self.assertIn("81 lines", " ".join(m))

    def test_f7_pinned_link(self):
        self.r.write("rl/a.py", "x\n")
        sha = self.r.commit()
        self.r.write("README.md", "[a](https://github.com/o/sglang-neuron/tree/%s/rl/a.py) [b](https://github.com/o/sglang-neuron/tree/%s/rl/zz.py)\n" % (sha, sha))
        self.r.commit()
        ok, m = check_docs.f7(self.r.dir)
        self.assertFalse(ok)
        self.assertEqual(len(m), 1, m)
        self.assertIn("rl/zz.py", m[0])

    def test_f8_added_lines_only(self):
        self.r.write("docs/rl/R.md", "old line currently open\n")
        base = self.r.commit()
        self.r.git("checkout", "-q", "-b", "pr")
        self.r.write("docs/rl/R.md", "old line currently open\n> quoted: currently open\nPR 5 is a draft for now.\nclean\n")
        head = self.r.commit()
        ok, m = check_docs.f8(base, head, self.r.dir)
        self.assertFalse(ok)
        self.assertEqual(len(m), 1, m)
        self.assertIn("R.md:3", m[0])

    def test_f8_new_data_readme_is_linted(self):
        base = self.r.commit()
        self.r.git("checkout", "-q", "-b", "pr")
        self.r.write("rl/data/s/README.md", "currently\n")
        head = self.r.commit()
        ok, m = check_docs.f8(base, head, self.r.dir)
        self.assertFalse(ok)
        self.assertIn("rl/data/s/README.md:1", m[0])

    def test_f10(self):
        self.r.write("sglang_neuron/a.py", "import os\nfrom rl.x import y\n")
        self.r.write("tests/README.md", "see docs/rl/RESULTS.md\n")
        self.r.write("docs/guides/g.md", "the url/ and world/ folders\n")
        self.r.write("rl/ok.py", "from rl import z\n")
        self.r.commit()
        ok, m = check_docs.f10(self.r.dir)
        self.assertFalse(ok)
        self.assertEqual(sorted(x.split()[1] for x in m), ["sglang_neuron/a.py:2", "tests/README.md:1"])


class UpstreamReady(unittest.TestCase):
    def setUp(self):
        self.r = helpers.Repo()
        self.r.write("sglang_neuron/a.py", "a\n")
        self.r.write("docs/guide.md", "g\n")
        self.r.commit("root")
        self.r.git("branch", "upstream-main")
        self.r.git("checkout", "-q", "-b", "main-like")
        self.r.write("rl/README.md", "team\n")
        self.r.write("README.md", "team\n")
        self.r.commit("team files on a main-like branch")

    def tearDown(self):
        self.r.close()

    def run_sh(self, branch):
        p = subprocess.run(["bash", SH, branch, "--base", "upstream-main", "--skip-tests"], cwd=self.r.dir,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.returncode, p.stdout.decode()

    def test_clean_engine_branch_is_ready(self):
        self.r.git("checkout", "-q", "-b", "up/ok", "upstream-main")
        self.r.write("sglang_neuron/a.py", "b\n")
        self.r.commit("engine")
        code, out = self.run_sh("up/ok")
        self.assertEqual(code, 0, out)
        self.assertIn("READY", out)

    def test_branch_cut_from_team_main_is_not_ready(self):
        code, out = self.run_sh("main-like")
        self.assertEqual(code, 1, out)
        self.assertIn("rl/README.md", out)
        self.assertIn("README.md", out)

    def test_one_team_file_is_not_ready(self):
        self.r.git("checkout", "-q", "-b", "up/bad", "upstream-main")
        self.r.write("sglang_neuron/a.py", "c\n")
        self.r.write("rl/README.md", "touched\n")
        self.r.commit("engine and team")
        code, out = self.run_sh("up/bad")
        self.assertEqual(code, 1, out)
        self.assertIn("rl/README.md", out)
        self.assertNotIn("sglang_neuron/a.py", out)

    def test_unknown_branch(self):
        self.assertEqual(self.run_sh("nope")[0], 2)


if __name__ == "__main__":
    unittest.main()
