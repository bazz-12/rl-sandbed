"""
go2_provenance.py - [P3.4] Where a run, an export and a card come from: git state, code fingerprint, file hashes.
Standard library only (no Genesis, no torch), so every script can use it: go2_train.py writes the run's manifest
with it, go2_export.py hashes the exported files, go2_card.py checks that the code has not changed since training.

    python go2_provenance.py                 # this folder: git commit, dirty flag, package versions, code fingerprint
    python go2_provenance.py logs/go2-v1.0   # compare the current code with the code that trained the run
"""

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256_file(path, chunk=1 << 20):
    """sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def code_fingerprint(folder=HERE):
    """{file: sha256[:16]} of every .py file of the code folder (not recursive).
    The code a run used, even without git."""
    return {name: sha256_file(os.path.join(folder, name))[:16]
            for name in sorted(os.listdir(folder)) if name.endswith(".py")}


def git_info(folder=HERE):
    """Git state of the repository holding `folder`: {"commit", "dirty", "untracked_py", "diff"}.
    All values are None without git (not installed, or not a repository).
    dirty = tracked files differ from the commit; untracked_py = .py files git does not track (code a clone would miss);
    diff = `git diff HEAD` when dirty (what a checkout of the commit lacks)."""
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=folder, stderr=subprocess.DEVNULL).decode()

    try:
        commit = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
        untracked = [line[3:] for line in git("status", "--porcelain").splitlines() if line.startswith("??")]
        return {"commit": commit, "dirty": dirty, "untracked_py": [f for f in untracked if f.endswith(".py")],
                "diff": git("diff", "HEAD") if dirty else ""}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "untracked_py": None, "diff": None}


def environment():
    """Interpreter, host and package versions (the packages that are importable)."""
    out = {"python": sys.version.split()[0], "host": platform.node(), "platform": platform.platform()}
    from importlib import metadata

    for dist in ("torch", "genesis-world", "rsl-rl-lib", "tensordict", "numpy", "onnx", "onnxruntime"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    return out


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def compare_code(manifest, folder=HERE):
    """Files whose content differs between the run's manifest fingerprint and `folder` now: (changed, added, removed).
    None when the manifest has no fingerprint (runs trained before P3.4)."""
    then = (manifest or {}).get("code_sha256")
    if not then:
        return None
    current = code_fingerprint(folder)
    changed = sorted(f for f in then if f in current and current[f] != then[f])
    return changed, sorted(set(current) - set(then)), sorted(set(then) - set(current))


def main():
    if len(sys.argv) > 1:
        path = os.path.join(sys.argv[1], "manifest.json")
        with open(path) as f:
            manifest = json.load(f)
        diff = compare_code(manifest)
        print(f"{path}: commit {manifest.get('git_hash')} dirty {manifest.get('git_dirty')}")
        if diff is None:
            print("no code fingerprint in this manifest (trained before P3.4)")
        else:
            changed, added, removed = diff
            print("code unchanged since training" if not (changed or removed) else
                  f"changed since training: {changed}; removed: {removed}")
            if added:
                print(f"new files since training (not used by it): {added}")
        return
    info = git_info()
    print(f"commit {info['commit']}  dirty {info['dirty']}  untracked .py {info['untracked_py']}")
    print("  ".join(f"{k} {v}" for k, v in environment().items()))
    for name, digest in code_fingerprint().items():
        print(f"  {digest}  {name}")


if __name__ == "__main__":
    main()
