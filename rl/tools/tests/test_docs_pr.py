"""Fixtures of the documentation step: numbers_moved.py, integration_lines.py, and the generated blocks of the real
README.md and docs/rl/RESULTS.md (their markers must exist, or F5 would pass without checking anything)."""
import io
import os
import subprocess
import sys
import unittest

import helpers

MIG = os.path.join(helpers.TOOLS, "migration")
NUMBERS = os.path.join(MIG, "numbers_moved.py")
LINES = os.path.join(MIG, "integration_lines.py")
ROOT = os.path.dirname(os.path.dirname(helpers.TOOLS))
BEST = os.path.join(helpers.TOOLS, "best.py")
SWITCHES = os.path.join(helpers.TOOLS, "switches.py")


def run(args, cwd):
    r = subprocess.run([sys.executable] + args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")


BASE_DOC = """# Results
Sampler kernel: 3,337.3 and 3,326.7 tok/s at concurrency 32 (screen).
Pass 2 ran 1,234.5 tok/s on 20 of 20 prompts, run on 2026-11-24, see section 8.4.
Only here: 4,321.9 tok/s on a path that has no data file, and 20 of 20 again.
Kernel tests: 47 pass in the simulator, 9,999,111 entries.
Rounded: 3,337 tok/s.
"""
HEAD_DOC = """# Results
The headline lives in the table below.
Rounded: still the same 3,337 tok/s.
"""
DATA_README = "Every number of the session, read from the files:\n3337.3 and 3326.7 are the two passes.\n"
CSV = "concurrency,output_token_throughput_tps\n32,1234.4999\n"


class NumbersMoved(unittest.TestCase):
    def setUp(self):
        self.r = helpers.Repo()
        self.r.write("docs/rl/RESULTS.md", BASE_DOC)
        self.r.write("rl/data/2026-11-26_x/README.md", DATA_README)
        self.r.write("rl/data/2026-11-26_x/run.csv", CSV)
        self.base = self.r.commit("base")
        self.r.write("docs/rl/RESULTS.md", HEAD_DOC)
        self.head = self.r.commit("head")

    def tearDown(self):
        self.r.close()

    def numbers(self, *extra):
        code, out, err = run([NUMBERS, "--base", self.base, "--head", self.head, "--scope", "docs/rl"] + list(extra), self.r.dir)
        self.assertEqual(code, 0, out + err)
        return out

    def test_a_moved_number_names_its_home_and_an_unplaced_one_is_listed_with_its_sentence(self):
        out = self.numbers("--list-moved")
        self.assertIn("3326.7 -> rl/data/2026-11-26_x/README.md:2 (session README)", out)
        self.assertIn("1234.5 -> rl/data/2026-11-26_x/run.csv:2 (csv", out)   # the csv holds 1234.4999: it rounds to the written figure
        tail = out.split("## removed, no home")[1]
        self.assertIn("4321.9", tail)
        self.assertIn("no data file", tail)
        self.assertNotIn("3326.7", tail)
        self.assertNotIn("1234.5", tail)

    def test_a_cpu_or_simulator_sentence_is_tagged(self):
        tail = self.numbers().split("## removed, no home")[1]
        self.assertIn("9999111", tail)
        self.assertIn("cut by the public-text rule", tail)

    def test_dates_section_numbers_and_identifiers_are_not_numerals(self):
        out = self.numbers("--list-moved")
        for token in ("2026", "8.4", "-11-24"):
            self.assertNotIn("\n" + token + " ", out)
        self.assertNotIn("8.4 ", out.split("## removed, no home")[1])

    def test_a_number_that_still_occurs_is_not_removed(self):
        out = self.numbers("--list-moved")
        self.assertNotIn("\n3337 -> ", out)   # 3,337 stays in the head text, so it is not itemised

    def test_a_small_number_needs_a_neighbour_of_its_sentence_in_the_data_line(self):
        tail = self.numbers("--list-moved").split("## removed, no home")[1]
        self.assertNotIn("[20]", tail)   # "20 of 20" is still in no sentence of the head, but 20 alone proves nothing
        self.r.write("rl/data/2026-11-26_x/note.md", "20 prompts and 1234.5 together\n")
        self.r.commit("more data")
        out = self.numbers("--list-moved")
        self.assertIn("20 -> rl/data/2026-11-26_x/note.md:1", out)

    def test_audit_lists_a_head_number_with_no_data_file(self):
        self.r.write("docs/rl/RESULTS.md", HEAD_DOC + "A typo: 3,331.7 tok/s.\n")
        self.head = self.r.commit("typo")
        out = self.numbers("--audit")
        self.assertIn("## audit", out)
        self.assertIn("3331.7", out.split("## audit")[1])

    def test_elsewhere_file_is_searched_last(self):
        sc = os.path.join(self.r.dir, "elsewhere.txt")
        with open(sc, "w", newline="\n") as fh:
            fh.write("a copy: 4,321.9 tok/s\n")
        out = self.numbers("--list-moved", "--elsewhere", sc)
        self.assertIn("4321.9 -> ", out)
        self.assertIn("file named by --elsewhere", out)


class IntegrationLines(unittest.TestCase):
    def test_kept_reworded_and_dropped_lines(self):
        r = helpers.Repo()
        try:
            r.write("docs/rl/K.md", "one\n")
            main = r.commit("main")
            r.git("branch", "integration")
            r.git("checkout", "-q", "integration")
            r.write("docs/rl/K.md", "one\nThe top-k kernel ran on a device once at Qwen3-0.6B\nand in two passes.\n"
                    "Changed 2026-11-26: the kernel is new.\nTen simulator tests, 129 rows, all pass.\nQuite another text entirely here.\n"
                    "The kernel ran on a device once at Qwen3-0.6B in two runs.\n")
            integration = r.commit("integration")
            r.git("checkout", "-q", "main")
            r.write("docs/rl/K.md", "one\nThe top-k kernel ran on a device once at Qwen3-0.6B and in two passes.\n"
                    "The kernel ran on a device once at Qwen3-0.6B (tier: once).\n")
            head = r.commit("head")
            code, out, err = run([LINES, "--main", main, "--integration", integration, "--head", head, "--repo", r.dir], r.dir)
            self.assertEqual(code, 0, out + err)
            self.assertIn("lines only on integration: 6; on main now: 2; dropped: 4", out)   # two kept lines: wrapped differently
            self.assertIn("[state] Changed 2026-11-26", out)
            self.assertIn("[cpu or simulator quantity] Ten simulator tests", out)
            self.assertIn("[cut] Quite another text", out)
            self.assertIn("reworded in place", out)
        finally:
            r.close()


class GeneratedBlocksOfTheRealDocs(unittest.TestCase):
    def read(self, rel):
        with io.open(os.path.join(ROOT, *rel.split("/")), encoding="utf-8", newline="") as fh:
            return fh.read()

    def test_readme_and_results_hold_the_best_block_and_readme_the_switches_block(self):
        for rel in ("README.md", "docs/rl/RESULTS.md"):
            self.assertRegex(self.read(rel), r"(?m)^<!-- best:begin [^>]*-->\s*$", rel)
            self.assertRegex(self.read(rel), r"(?m)^<!-- best:end -->\s*$", rel)
        self.assertRegex(self.read("README.md"), r"(?m)^<!-- switches:begin [^>]*-->\s*$")
        self.assertRegex(self.read("README.md"), r"(?m)^<!-- switches:end -->\s*$")

    def test_the_best_blocks_equal_what_best_py_renders(self):
        code, out, err = run([BEST, "--check", "README.md", "docs/rl/RESULTS.md", "--require-block", "--root", ROOT], ROOT)
        self.assertEqual(code, 0, out + err)

    def test_agents_md_is_short_and_states_the_label_rule_of_f1(self):
        text = self.read("AGENTS.md")
        self.assertLessEqual(len(text.splitlines()), 80)
        for needle in ("exactly one label", "hardware-verified", "cpu-only", "sync from `main`", "check F1"):
            self.assertIn(needle, text)

    def test_agents_commands_are_runnable_as_written(self):
        text = self.read("AGENTS.md")
        self.assertIn("git fetch --multiple origin upstream", text)  # "git fetch origin upstream" reads upstream as a ref
        self.assertNotIn("check_data.py rl/data", text)              # check_data.py takes --base and --head only
        self.assertIn("check_data.py --base origin/main --head HEAD", text)
        self.assertNotIn("--base main --head", text)                 # a cpu-only change targets integration
        code, out, err = run([os.path.join(helpers.TOOLS, "check_data.py"), "--help"], ROOT)
        self.assertIn("--base", out + err)

    def test_one_tier_word_and_no_cpu_quantity_in_kernels(self):
        for rel in ("docs/rl/RESULTS.md", "docs/rl/KERNELS.md"):
            self.assertNotIn("single run", self.read(rel), rel)
        self.assertIn("**`once`**", self.read("rl/METHOD.md"))
        self.assertNotIn("1e-5", self.read("docs/rl/KERNELS.md"))

    def test_team_checks_job_fails_on_every_base(self):
        # The job used to be report-only for base integration; now only F5 switches and F10 are, and only there
        # (test_acceptance.ChecksJobOnEveryBase replays the violating pull requests).
        text = self.read(".github/workflows/team-checks.yml")
        self.assertNotIn('[ "$BASE_REF" != main ]', text)
        self.assertIn('if [ "$BASE_REF" != integration ]; then step "$@"; return; fi', text)
        self.assertNotIn("exit 0\n", text.split("Team checks F2-F8 and F10", 1)[1])
        self.assertNotIn("test_docs_blocks", text)

    def test_no_hand_written_state_table_is_left_in_the_readme(self):
        text = self.read("README.md")
        for needle in ("Open work branches", "Sent upstream", "What `main` adds to upstream"):
            self.assertNotIn(needle, text)
        self.assertIn("rl/tools/fork_state.sh", text)


if __name__ == "__main__":
    unittest.main()
