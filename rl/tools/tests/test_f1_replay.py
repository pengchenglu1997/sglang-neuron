"""F1 fixtures (each fails or passes for its stated reason), the CLI, and the replay of fork pull requests 3, 13 and 15."""
import io
import json
import os
import subprocess
import sys
import unittest

import helpers  # noqa: F401  (puts rl/tools on sys.path)
import check_pr

CHECK_PR = os.path.join(helpers.TOOLS, "check_pr.py")


def load(name):
    with open(os.path.join(helpers.FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


class F1Fixtures(unittest.TestCase):
    def test_cases(self):
        cases = load("f1_cases.json")
        self.assertGreater(len(cases), 15)
        for c in cases:
            ok, msgs = check_pr.f1(c["base"], c["labels"], c["paths"])
            text = " ".join(msgs)
            self.assertEqual(ok, c["expect"] == "pass", "%s: %s" % (c["name"], text))
            self.assertIn(c["reason"], text, c["name"])

    def test_cli_exit_codes_and_nul_input(self):
        paths = b"sglang_neuron/a.py\0rl/x.md\0"
        r = subprocess.run([sys.executable, CHECK_PR, "f1", "--base", "main", "--labels", "", "--paths-file", "-"],
                           input=paths, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn(b"exactly one", r.stdout)
        r = subprocess.run([sys.executable, CHECK_PR, "f1", "--base", "main", "--labels", "hardware-verified", "--paths-file", "-"],
                           input=b"sglang_neuron/a.py\nrl/x.md\n", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_classify(self):
        r = subprocess.run([sys.executable, CHECK_PR, "classify", "--kind", "engine"],
                           input=b"rl/a\0docs/rl/b.md\0docs/x.md\0README.md\0tests/README.md\0.github/CODEOWNERS\0",
                           stdout=subprocess.PIPE)
        self.assertEqual(r.stdout.decode().split(), ["docs/x.md", "tests/README.md"])


class Replay(unittest.TestCase):
    def test_recorded_pull_requests(self):
        for num, pr in sorted(load("replay_prs.json").items()):
            if num.startswith("_"):
                continue
            ok, msgs = check_pr.f1(pr["base"], pr["labels"], pr["paths"])
            self.assertEqual(ok, pr["expect"] == "pass", "PR %s: %s" % (num, msgs))

    def test_never_looser_than_the_old_rule_for_engine_changes(self):
        """The old rule: exactly one evidence label. F1 never passes an engine pull request the old rule rejected."""
        import itertools
        label_sets = [[], ["cpu-only"], ["hardware-verified"], ["cpu-only", "hardware-verified"], ["bug"]]
        for base, labels, paths in itertools.product(["main", "integration", "upstream-main", "feat/x"], label_sets,
                                                     [["sglang_neuron/a.py"], ["sglang_neuron/a.py", "rl/b.md"], ["tests/t.py"]]):
            old_ok = len([l for l in labels if l in ("cpu-only", "hardware-verified")]) == 1
            new_ok, _ = check_pr.f1(base, labels, paths)
            self.assertFalse(new_ok and not old_ok, (base, labels, paths))

    def test_only_documented_relaxation(self):
        """The one relaxation: a team-only pull request into main without a label."""
        import itertools
        label_sets = [[], ["cpu-only"], ["hardware-verified"], ["cpu-only", "hardware-verified"]]
        for base, labels, paths in itertools.product(["main", "integration", "upstream-main"], label_sets,
                                                     [["rl/a.py"], ["README.md", "docs/rl/x.md"], ["sglang_neuron/a.py"], []]):
            old_ok = len(labels) == 1
            new_ok, _ = check_pr.f1(base, labels, paths)
            if new_ok and not old_ok:
                self.assertEqual(base, "main", (base, labels, paths))
                self.assertTrue(paths and all(check_pr.common.is_team(p) for p in paths), (base, labels, paths))
                self.assertLess(len(labels), 2)


if __name__ == "__main__":
    unittest.main()
