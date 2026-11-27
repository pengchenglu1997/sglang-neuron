"""rewrite_links.py and check_tables.py on throw-away repositories."""
import io
import os
import unittest
from contextlib import redirect_stdout

import helpers
import check_tables
import rewrite_links

OLD = "https://github.com/pengchenglu1997/trainium-sglang-rl"


class Rewrite(unittest.TestCase):
    def setUp(self):
        r = self.r = helpers.Repo()
        r.write("rl/data/s/README.md", "x\n")
        r.write("rl/data/LEGACY.sha256", "0" * 64 + "  rl/data/s/frozen.md\n")
        r.write("rl/REPRODUCE.md", "x\n")
        r.write("README.md", "[data](%s/tree/main/data/s) and the repo %s, cite %s/blob/main/REPRODUCE.md.\n" % (OLD, OLD, OLD))
        r.write("docs/rl/RESULTS.md", "see [folder](%s/tree/main/data/s/) and (%s/tree/abc1234/data/s).\n" % (OLD, OLD))
        r.write("docs/rl/GONE.md", "[x](%s/tree/main/nothing/here) clone %s.git\n" % (OLD, OLD))
        r.write("tests/rlkit_sampler/README.md", "kit: [c](%s)\n" % OLD)
        r.write("rl/data/s/frozen.md", "link %s\n" % OLD)
        r.write("rl/docs/NOTE.md", "link %s\n" % OLD)
        r.write("crlf.md", "a %s/tree/main/data/s\r\nb\r\n" % OLD)
        r.commit("c")

    def tearDown(self):
        self.r.close()

    def run_main(self, *args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = rewrite_links.main(["--repo", self.r.dir] + list(args))
        return code, buf.getvalue()

    def read(self, rel):
        with open(os.path.join(self.r.dir, *rel.split("/")), "rb") as fh:
            return fh.read().decode("utf-8")

    def test_dry_run_prints_old_and_new_and_writes_nothing(self):
        before = self.r.git("status", "--porcelain")
        code, out = self.run_main()
        self.assertEqual(self.r.git("status", "--porcelain"), before)
        self.assertIn("README.md:1: %s/tree/main/data/s -> rl/data/s" % OLD, out)
        self.assertIn("dry run", out)

    def test_link_target_is_relative_to_its_file_and_prose_from_the_root(self):
        code, out = self.run_main("--exclude", "docs/rl/GONE.md", "--apply")
        self.assertEqual(code, 1, out)               # the bare address and the pinned link are left for a person
        self.assertIn("[data](rl/data/s) and the repo %s, cite rl/REPRODUCE.md." % OLD, self.read("README.md"))
        self.assertIn("[folder](../../rl/data/s)", self.read("docs/rl/RESULTS.md"))

    def test_bare_address_and_pinned_links_are_manual_and_untouched(self):
        code, out = self.run_main("--exclude", "docs/rl/GONE.md", "--apply")
        self.assertEqual(code, 1, out)
        self.assertIn("README.md:1: %s -> MANUAL" % OLD, out)
        self.assertIn("a link pinned to abc1234", out)
        self.assertIn("and (%s/tree/abc1234/data/s)." % OLD, self.read("docs/rl/RESULTS.md"))
        self.assertNotIn("rl/data/s).", self.read("docs/rl/RESULTS.md").split("and (")[1])
        self.assertNotIn("the repo rl/", self.read("README.md"))

    def test_protected_files_are_left_alone(self):
        self.run_main("--exclude", "docs/rl/GONE.md", "--apply")
        for rel in ("tests/rlkit_sampler/README.md", "rl/data/s/frozen.md", "rl/docs/NOTE.md", "docs/rl/GONE.md"):
            self.assertIn(OLD, self.read(rel), rel)

    def test_unmapped_targets_are_reported_and_fail(self):
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("rl/nothing/here does not exist", out)
        self.assertIn("a clone address", out)
        self.assertIn(OLD, self.read("docs/rl/GONE.md"))

    def test_exclude_prefix_skips_files(self):
        code, out = self.run_main("--exclude", "docs/", "--exclude", "README.md")
        self.assertNotIn("docs/rl", out)
        self.assertNotIn("README.md:", out)

    def test_line_endings_are_kept(self):
        self.run_main("--exclude", "docs/rl/GONE.md", "--apply")
        self.assertEqual(self.read("crlf.md"), "a rl/data/s\r\nb\r\n")

    def test_second_run_finds_nothing_more_to_map(self):
        self.run_main("--exclude", "docs/rl/GONE.md", "--apply")
        code, out = self.run_main("--exclude", "docs/rl/GONE.md")
        self.assertEqual(code, 1, out)               # the two MANUAL hits remain for a person
        self.assertEqual(out.count("-> rl/"), 0)
        self.assertEqual(out.count("-> MANUAL"), 2)

    def test_bad_repo(self):
        self.assertEqual(rewrite_links.main(["--repo", os.path.join(self.r.dir, "nope")]), 2)


MAKE = '''import os, sys
d = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(d, "tables.txt"), "w", newline="\\n").write("rows %s\\n" % os.path.basename(d.rstrip("/")))
'''


class Tables(unittest.TestCase):
    def setUp(self):
        self.r = helpers.Repo()

    def tearDown(self):
        self.r.close()

    def folder(self, name, script=MAKE, tables="rows d\n"):
        self.r.write("rl/data/%s/make_tables.py" % name, script)
        if tables is not None:
            self.r.write("rl/data/%s/tables.txt" % name, tables)

    def run_main(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = check_tables.main(["--root", self.r.dir])
        return code, buf.getvalue()

    def test_equal_output_passes_and_nothing_changes(self):
        self.folder("s")
        code, out = self.run_main()
        self.assertEqual(code, 0, out)
        with open(os.path.join(self.r.dir, "rl", "data", "s", "tables.txt")) as fh:
            self.assertEqual(fh.read(), "rows d\n")
        self.assertFalse(os.path.exists(os.path.join(self.r.dir, "rl", "data", "s", "__pycache__")))

    def test_difference_names_the_line(self):
        self.folder("s", tables="rows d\nextra\n")
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("differs from the script's output at line 2", out)

    def test_failing_script_fails(self):
        self.folder("s", script="raise SystemExit(3)\n")
        self.assertEqual(self.run_main()[0], 1)

    def test_folder_without_tables_is_not_checked(self):
        self.folder("s", tables=None)
        code, out = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("0 folder(s)", out)

    def test_no_data_folder_is_bad_input(self):
        self.assertEqual(self.run_main()[0], 2)

    def test_real_repository(self):
        root = os.path.dirname(os.path.dirname(helpers.TOOLS))
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = check_tables.main(["--root", root])
        self.assertEqual(code, 0, buf.getvalue())


if __name__ == "__main__":
    unittest.main()
