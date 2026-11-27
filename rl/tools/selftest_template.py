#!/usr/bin/env python3
"""Template of a CPU selftest that a change for upstream brings with it (AGENTS.md section 4).

Copy it to tests/<area>/selftest_<topic>.py on the branch cut from origin/upstream-main, then replace the check below
with one that fails on the code before the change and passes after it. upstream_ready.sh runs every
tests/**/selftest_*.py as "python <script>" from the top of the branch: exit status 0 is a pass, anything else a failure.

Rules for the copy (it travels upstream):
  - the standard library only, unless the module under test itself needs more (then say so in the docstring);
  - load the changed file by its path, so that importing the package (and its device dependencies) is not needed;
  - no path, name or folder of the team tooling of this fork: the copy is an engine file, and check F10 fails an engine
    file that mentions that folder.
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

    # replace: one check per behaviour the change adds or fixes
    check(os.path.isfile(MODULE), "the file under test exists: %s" % os.path.relpath(MODULE, TOP))
    print("%d check(s) failed" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
