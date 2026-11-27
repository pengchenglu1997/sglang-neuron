"""Test helpers: temporary git repositories and the tools folder on sys.path."""
import os
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)


class Repo(object):
    """A throw-away repository with LF files and a fixed identity."""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="teamchecks_")
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "t" + chr(64) + "example.invalid")
        self.git("config", "user.name", "t")
        self.git("config", "core.autocrlf", "false")
        self.git("config", "commit.gpgsign", "false")

    def git(self, *args):
        r = subprocess.run(["git"] + list(args), cwd=self.dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            raise RuntimeError("git %s: %s" % (" ".join(args), r.stderr.decode("utf-8", "replace")))
        return r.stdout.decode("utf-8", "replace").strip()

    def write(self, path, content):
        full = os.path.join(self.dir, *path.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(content if isinstance(content, bytes) else content.encode("utf-8"))

    def remove(self, path):
        os.remove(os.path.join(self.dir, *path.split("/")))

    def commit(self, msg="c"):
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", msg)
        return self.git("rev-parse", "HEAD")

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def sha256(data):
    import hashlib
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()
