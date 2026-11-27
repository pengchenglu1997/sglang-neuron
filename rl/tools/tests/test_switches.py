"""switches.py on a throw-away repository: what counts as a switch the fork adds, defaults, purposes, F5 check."""
import io
import os
import unittest
from contextlib import redirect_stdout, redirect_stderr

import helpers
import switches

REGISTRY = '''import os


def zero_or_one(name, environ=os.environ):
    raw = environ.get(name, "0")
    if raw not in ("0", "1"):
        raise ValueError(name)
    return raw == "1"
'''
BASE_A = 'import os\nOLD = os.environ.get("OLD_UPSTREAM_VAR", "0")\n'
NEW_A = '''import os
from sglang_neuron import env_registry

OLD = os.environ.get("OLD_UPSTREAM_VAR", "0")
REUSED = os.environ.get("OLD_UPSTREAM_VAR", "1")
SAMPLER_ENV = "NEW_SAMPLER_FLAG"


def sampler_on(environ=os.environ):
    """NEW_SAMPLER_FLAG: 1 runs the sampler as one kernel; any other value is refused."""
    return env_registry.zero_or_one(SAMPLER_ENV, environ)


# NEW_LADDER_FLAG=0 compiles one context instead of three; a short server has no use for the rest.
ladder = __import__("os").environ.get("NEW_LADDER_FLAG", "1") != "0"
mode = os.getenv("NEW_MODE_FLAG", "auto")  # which implementation runs
bare = os.environ.get("NEW_BARE_FLAG")
indexed = os.environ["NEW_INDEXED_FLAG"]
'''


class SwitchesTest(unittest.TestCase):
    def setUp(self):
        self.r = helpers.Repo()
        self.r.write("sglang_neuron/env_registry.py", REGISTRY)
        self.r.write("sglang_neuron/a.py", BASE_A)
        self.r.write("README.md", "x\n")
        self.r.commit("upstream")
        self.r.git("branch", "upstream-main")
        self.r.git("checkout", "-q", "-b", "work")
        self.r.write("sglang_neuron/a.py", NEW_A)
        self.r.write("sglang_neuron/notes.txt", 'environ.get("NEW_IN_TEXT_FILE")\n')
        self.r.write("rl/tools/switches_exceptions.txt", "# none\n")
        self.r.commit("fork change")

    def tearDown(self):
        self.r.close()

    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = switches.main(["--root", self.r.dir, "--base", "upstream-main"] + list(args))
        return code, out.getvalue() + err.getvalue()

    def derived(self):
        return {s["name"]: s for s in switches.collect(self.r.dir, "upstream-main", [("work", "main")])}

    def test_names_found_and_excluded(self):
        got = self.derived()
        self.assertEqual(sorted(got), ["NEW_BARE_FLAG", "NEW_INDEXED_FLAG", "NEW_LADDER_FLAG", "NEW_MODE_FLAG",
                                       "NEW_SAMPLER_FLAG"])
        self.assertNotIn("OLD_UPSTREAM_VAR", got)          # upstream already has it, a second read changes nothing
        self.assertNotIn("NEW_IN_TEXT_FILE", got)          # not a .py file

    def test_defaults_and_on_state(self):
        got = self.derived()
        self.assertEqual((got["NEW_SAMPLER_FLAG"]["default"], got["NEW_SAMPLER_FLAG"]["on"]), ("0", "no"))
        self.assertEqual((got["NEW_LADDER_FLAG"]["default"], got["NEW_LADDER_FLAG"]["on"]), ("1", "yes"))
        self.assertEqual((got["NEW_MODE_FLAG"]["default"], got["NEW_MODE_FLAG"]["on"]), ("auto", "mode"))
        self.assertEqual((got["NEW_BARE_FLAG"]["default"], got["NEW_BARE_FLAG"]["on"]), ("unset", "no"))

    def test_purpose_from_docstring_comment_and_trailing_comment(self):
        got = self.derived()
        self.assertEqual(got["NEW_SAMPLER_FLAG"]["purpose"], "1 runs the sampler as one kernel")
        self.assertTrue(got["NEW_LADDER_FLAG"]["purpose"].startswith("NEW_LADDER_FLAG=0 compiles one context"))
        self.assertNotIn("a short server", got["NEW_LADDER_FLAG"]["purpose"])
        self.assertEqual(got["NEW_MODE_FLAG"]["purpose"], "which implementation runs")
        self.assertEqual(got["NEW_BARE_FLAG"]["purpose"], "no description in code")

    def test_table_has_branch_column(self):
        code, out = self.run_main("--branch", "work=main")
        self.assertEqual(code, 0, out)
        self.assertIn("| `NEW_SAMPLER_FLAG` | `0` | no |", out)
        self.assertIn("| `main` |", out)

    def test_branch_label_defaults_to_the_checked_out_name(self):
        code, out = self.run_main()
        self.assertEqual(code, 0, out)
        self.assertIn("`work`", out)

    def test_two_branches_list_both(self):
        self.r.git("checkout", "-q", "-b", "other", "upstream-main")
        self.r.write("sglang_neuron/b.py", 'import os\nX = os.environ.get("NEW_SAMPLER_FLAG", "0")\n')
        self.r.commit("other")
        code, out = self.run_main("--branch", "work=main", "--branch", "other")
        self.assertEqual(code, 0, out)
        line = [l for l in out.split("\n") if "NEW_SAMPLER_FLAG" in l][0]
        self.assertIn("`main`, `other`", line)

    def test_check_fails_for_a_default_on_switch_without_a_reason(self):
        code, out = self.run_main("--check", "--branch", "work")
        self.assertEqual(code, 1)
        self.assertIn("NEW_LADDER_FLAG", out)
        self.assertIn("NEW_MODE_FLAG", out)
        self.assertNotIn("NEW_SAMPLER_FLAG:", out)

    def test_check_passes_with_listed_exceptions(self):
        self.r.write("rl/tools/switches_exceptions.txt", "NEW_LADDER_FLAG\tcompiles fewer graphs\nNEW_MODE_FLAG\tauto picks\n")
        self.r.commit("exceptions")
        self.assertEqual(self.run_main("--check", "--branch", "work")[0], 0)

    def test_exception_without_reason_is_bad_input(self):
        self.r.write("rl/tools/switches_exceptions.txt", "NEW_LADDER_FLAG\n")
        self.assertEqual(self.run_main("--check", "--branch", "work")[0], 2)

    def test_block_stale_write_fresh(self):
        self.r.write("rl/tools/switches_exceptions.txt", "NEW_LADDER_FLAG\tr\nNEW_MODE_FLAG\tr\n")
        self.r.write("README.md", "a\n" + switches.BEGIN + "\nold\n" + switches.END + "\nz\n")
        self.r.commit("readme")
        code, out = self.run_main("--check", "README.md", "--branch", "work=main")
        self.assertEqual(code, 1)
        self.assertIn("stale", out)
        self.assertEqual(self.run_main("--write", "README.md", "--branch", "work=main")[0], 0)
        self.assertEqual(self.run_main("--check", "README.md", "--branch", "work=main")[0], 0)
        self.assertEqual(self.run_main("--check", "README.md", "--branch", "work=other")[0], 1)

    def test_no_block_passes_unless_required(self):
        self.r.write("rl/tools/switches_exceptions.txt", "NEW_LADDER_FLAG\tr\nNEW_MODE_FLAG\tr\n")
        code, out = self.run_main("--check", "README.md", "--branch", "work")
        self.assertEqual(code, 0, out)
        self.assertIn("not checked", out)
        self.assertEqual(self.run_main("--check", "README.md", "--branch", "work", "--require-block")[0], 1)

    def test_mixed_line_endings_are_refused_and_a_crlf_block_is_found(self):
        self.r.write("rl/tools/switches_exceptions.txt", "NEW_LADDER_FLAG\tr\nNEW_MODE_FLAG\tr\n")
        self.r.write("README.md", "a\r\n" + switches.BEGIN + "\r\nold\r\n" + switches.END + "\r\nz\r\n")
        self.r.commit("readme")
        self.assertEqual(self.run_main("--write", "README.md", "--branch", "work=main")[0], 0)
        self.assertEqual(self.run_main("--check", "README.md", "--branch", "work=main", "--require-block")[0], 0)
        path = os.path.join(self.r.dir, "README.md")
        with open(path, "rb") as fh:
            raw = fh.read()
        with open(path, "wb") as fh:
            fh.write(raw.replace(b"\r\n", b"\n", 2))
        code, out = self.run_main("--check", "README.md", "--branch", "work=main")
        self.assertEqual(code, 1, out)
        self.assertIn("mixed line endings", out)

    def test_unknown_base_and_branch_are_bad_input(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(switches.main(["--root", self.r.dir, "--base", "nope"]), 2)
            self.assertEqual(switches.main(["--root", self.r.dir, "--base", "upstream-main", "--branch", "nope"]), 2)

    def test_branch_without_changes_has_no_switch(self):
        code, out = self.run_main("--branch", "upstream-main=up")
        self.assertEqual(code, 0)
        self.assertIn("(none)", out)


if __name__ == "__main__":
    unittest.main()
