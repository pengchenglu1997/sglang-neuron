#!/usr/bin/env python3
# --- template header: delete it in the copy, from this line to the line "end of template header" ---
# Template of the CPU selftest that a change for upstream brings with it (AGENTS.md section 4). Copy it to
# tests/<area>/selftest_<topic>.py on the branch cut from origin/upstream-main, delete this header, set MODULE, write the
# docstring, and replace the placeholder check with one that fails on the code before the change and passes after it.
# upstream_ready.sh runs every tests/**/selftest_*.py as "python <script>" from the top of the branch (exit 0 is a pass)
# and calls a branch NOT READY while a selftest still holds the marker line below. Unchanged, the placeholder fails.
# The copy travels upstream: no path, name or folder of the team tooling in it (check F10 fails such an engine file).
# SELFTEST_TEMPLATE_UNCHANGED
# --- end of template header ---
"""CPU selftest of <the change>: <the behaviour it checks>.

Standard library only (if the module under test needs more, say so here). It loads the changed file by its path, so
importing the package and its device dependencies is not needed.
"""
import importlib.util
import os
import sys

TOP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODULE = os.path.join(TOP, "sglang_neuron", "env_registry.py")   # replace: the file the change touches


def load(path, name="module_under_test"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    failures = []

    def check(cond, what):
        print(("ok   " if cond else "FAIL ") + what)
        if not cond:
            failures.append(what)

    # replace: one check per behaviour the change adds or fixes, e.g. check(load(MODULE).f(2) == 4, "f doubles")
    check(False, "placeholder: replace this check with one that tests the change")
    print("%d check(s) failed" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
