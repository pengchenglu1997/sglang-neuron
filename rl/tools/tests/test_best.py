"""best.py: rl/best.tsv rows against the result files, the generated block, and the real repository."""
import io
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr

import helpers
import best

HEADER = "key\tlabel\tmodel\tfile\tselector\tbaseline\ttier\n"
SEL = "output_token_throughput_tps:concurrency=32"


def csv_text(c32, n_error=0):
    return ("concurrency,n_error,output_token_throughput_tps\n1,0,100.5\n8,0,500.25\n32,%s,%s\n" % (n_error, c32))


class BestTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="best_")
        self.data("s/a_p1.csv", csv_text(3337.261))
        self.data("s/a_p2.csv", csv_text(3326.7))
        self.data("s/b_p1.csv", csv_text(2378.2))
        self.data("s/b_p2.csv", csv_text(2376.04))
        self.data("s/c_p1.csv", csv_text(3300.0))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def data(self, rel, text):
        p = os.path.join(self.root, "rl", "data", *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", newline="") as fh:
            fh.write(text)

    def tsv(self, *rows):
        os.makedirs(os.path.join(self.root, "rl"), exist_ok=True)
        with open(os.path.join(self.root, "rl", "best.tsv"), "w", newline="") as fh:
            fh.write(HEADER + "".join("\t".join(r) + "\n" for r in rows))

    def doc(self, rel, text):
        p = os.path.join(self.root, *rel.split("/"))
        with open(p, "wb") as fh:
            fh.write(text.encode("utf-8"))
        return rel

    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = best.main(["--root", self.root] + list(args))
        return code, out.getvalue() + err.getvalue()

    def good(self):
        self.tsv(("k_a", "kernel on", "M", "s/a_p1.csv;s/a_p2.csv", SEL, "k_b", "screen"),
                 ("k_b", "other engine", "M", "s/b_p1.csv;s/b_p2.csv", SEL, "-", "screen"))

    def test_render_copies_the_file_values(self):
        self.good()
        code, out = self.run_main()
        self.assertEqual(code, 0, out)
        self.assertIn("3,337.3 and 3,326.7 tok/s at concurrency 32", out)
        self.assertIn("k_b: 2,378.2 and 2,376.0 tok/s", out)
        self.assertIn("| screen |", out)
        self.assertNotIn("%", out)

    def test_json_rows(self):
        self.good()
        code, out = self.run_main("--json")
        self.assertEqual(code, 0)
        self.assertIn('"tier": "screen"', out)
        self.assertIn('"files": ["rl/data/s/a_p1.csv"', out)

    def test_check_passes(self):
        self.good()
        self.assertEqual(self.run_main("--check")[0], 0)

    def test_missing_file_fails(self):
        self.tsv(("k_a", "x", "M", "s/none.csv", SEL, "-", "screen"))
        code, out = self.run_main("--check")
        self.assertEqual(code, 1)
        self.assertIn("does not exist", out)

    def test_selector_without_a_row_fails(self):
        self.tsv(("k_a", "x", "M", "s/a_p1.csv", "output_token_throughput_tps:concurrency=64", "-", "screen"))
        code, out = self.run_main("--check")
        self.assertEqual(code, 1)
        self.assertIn("0 rows", out)

    def test_cell_with_a_failed_request_fails(self):
        self.data("s/e.csv", csv_text(3000.0, n_error=2))
        self.tsv(("k_a", "x", "M", "s/e.csv", SEL, "-", "screen"))
        code, out = self.run_main("--check")
        self.assertEqual(code, 1)
        self.assertIn("failed request", out)

    def test_bad_tier_unknown_baseline_duplicate_and_hub_id(self):
        self.tsv(("k_a", "x", "M", "s/a_p1.csv", SEL, "nope", "fast"),
                 ("k_a", "see L-I2-59", "M", "s/a_p1.csv", SEL, "-", "paired(0)"))
        code, out = self.run_main("--check")
        self.assertEqual(code, 1)
        for want in ("tier 'fast'", "baseline 'nope'", "duplicate key", "hub row id", "paired(0)"):
            self.assertIn(want, out)

    def test_file_outside_data_fails(self):
        self.tsv(("k_a", "x", "M", "../best.tsv", SEL, "-", "screen"))
        code, out = self.run_main("--check")
        self.assertEqual(code, 1)
        self.assertIn("path under rl/data", out)

    def test_bad_header_is_bad_input(self):
        os.makedirs(os.path.join(self.root, "rl"), exist_ok=True)
        with open(os.path.join(self.root, "rl", "best.tsv"), "w") as fh:
            fh.write("key\tlabel\n")
        self.assertEqual(self.run_main("--check")[0], 2)

    def test_not_distinguished_within_ten_percent(self):
        self.tsv(("k_a", "x", "M", "s/a_p1.csv;s/a_p2.csv", SEL, "k_c", "screen"),
                 ("k_c", "y", "M", "s/c_p1.csv", SEL, "-", "screen"))
        self.assertIn("not distinguished at screen tier", self.run_main()[1])
        self.good()
        self.assertNotIn("not distinguished", self.run_main()[1])

    def test_paired_tier_is_not_marked(self):
        self.tsv(("k_a", "x", "M", "s/a_p1.csv;s/a_p2.csv", SEL, "k_c", "paired(7)"),
                 ("k_c", "y", "M", "s/c_p1.csv", SEL, "-", "paired(7)"))
        out = self.run_main()[1]
        self.assertNotIn("not distinguished", out)
        self.assertIn("paired(7)", out)

    def test_block_check_write_and_stale(self):
        self.good()
        text = "intro\n" + best.BEGIN + "\nold\n" + best.END + "\noutro\n"
        self.doc("README.md", text)
        code, out = self.run_main("--check", "README.md")
        self.assertEqual(code, 1)
        self.assertIn("stale", out)
        self.assertEqual(self.run_main("--write", "README.md")[0], 0)
        self.assertEqual(self.run_main("--check", "README.md")[0], 0)
        with open(os.path.join(self.root, "README.md"), "rb") as fh:
            got = fh.read().decode()
        self.assertTrue(got.startswith("intro\n") and got.endswith("\noutro\n"))
        self.assertIn("3,337.3", got)

    def test_write_keeps_crlf(self):
        self.good()
        self.doc("README.md", "a\r\n" + best.BEGIN + "\r\nold\r\n" + best.END + "\r\nz\r\n")
        self.assertEqual(self.run_main("--write", "README.md")[0], 0)
        with open(os.path.join(self.root, "README.md"), "rb") as fh:
            raw = fh.read()
        self.assertNotIn(b"\n", raw.replace(b"\r\n", b""))
        self.assertEqual(self.run_main("--check", "README.md")[0], 0)

    def test_no_block_passes_unless_required(self):
        self.good()
        self.doc("README.md", "no block\n")
        code, out = self.run_main("--check", "README.md")
        self.assertEqual(code, 0)
        self.assertIn("not checked", out)
        self.assertEqual(self.run_main("--check", "README.md", "--require-block")[0], 1)

    def test_broken_marker_pair_fails(self):
        self.good()
        self.doc("README.md", best.BEGIN + "\ntext\n")
        self.assertEqual(self.run_main("--check", "README.md")[0], 1)

    def test_lf_block_in_a_crlf_file_is_found_and_checked(self):
        self.good()
        self.doc("README.md", "a\r\n" + best.BEGIN + "\r\nold\r\n" + best.END + "\r\nz\r\n")
        self.assertEqual(self.run_main("--write", "README.md")[0], 0)
        code, out = self.run_main("--check", "README.md", "--require-block")
        self.assertEqual(code, 0, out)

    def test_mixed_line_endings_are_refused(self):
        self.good()
        wrong = best.BEGIN + "\n| k_a | M | x | 9,999.9 tok/s | - | screen | f |\n" + best.END
        self.doc("README.md", "a\r\n" + wrong + "\r\nz\r\n")
        code, out = self.run_main("--check", "README.md")
        self.assertEqual(code, 1, out)
        self.assertIn("mixed line endings", out)
        self.assertEqual(self.run_main("--write", "README.md")[0], 2)

    def test_selector_metric_must_be_a_known_one(self):
        self.data("s/e_p1.csv", csv_text(100.0, n_error=5))
        self.tsv(("k_a", "x", "M", "s/e_p1.csv", "n_error:concurrency=32", "-", "screen"))
        code, out = self.run_main("--check")
        self.assertEqual(code, 1, out)
        self.assertIn("metric", out)

    def test_real_repository_rows_are_sound(self):
        root = os.path.dirname(os.path.dirname(helpers.TOOLS))
        code, out = self.run_real(root)
        self.assertEqual(code, 0, out)
        self.assertIn("3,337.3 and 3,326.7", self.render_real(root))

    def run_real(self, root):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = best.main(["--root", root, "--check"])
        return code, buf.getvalue()

    def render_real(self, root):
        buf = io.StringIO()
        with redirect_stdout(buf):
            best.main(["--root", root])
        return buf.getvalue()


if __name__ == "__main__":
    unittest.main()
