"""
bilbyflow-init — copy the runnable scripts into your working directory.

    bilbyflow-init                        # copies to ./
    bilbyflow-init /path/to/my/run        # copies there
    bilbyflow-init --list                 # just show what would be copied

The scripts are copies, not symlinks — edit them freely. They import from
the installed package, so `pip install -e .` keeps them current with your
code changes; only the orchestration (which banks to build, what to plot,
CLI flags) lives in the copy.
"""

import argparse
import os
import shutil
import importlib.resources


def _scripts_dir():
    """Installed location of bilbyflow/scripts/."""
    ref = importlib.resources.files("bilbyflow.scripts")
    # files() returns a Traversable; for an editable install it's already
    # a Path, for a wheel it's a zipfile entry — as_posix works for both
    return str(ref)


# only copy actual runnable scripts, not __init__.py or utils
_SKIP = {"__init__.py", "utils.py", "__pycache__"}


def main():
    p = argparse.ArgumentParser(
        description="Copy BilbyFlow scripts into your working directory")
    p.add_argument("dest", nargs="?", default=".",
                   help="target directory (default: current directory)")
    p.add_argument("--list", action="store_true",
                   help="show what would be copied and exit")
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite existing files")
    args = p.parse_args()

    src = _scripts_dir()
    scripts = sorted(f for f in os.listdir(src)
                     if f.endswith(".py") and f not in _SKIP)

    if args.list:
        print(f"{len(scripts)} scripts in {src}:")
        for s in scripts:
            print(f"  {s}")
        return

    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)

    copied, skipped = 0, 0
    for s in scripts:
        dst = os.path.join(dest, s)
        if os.path.exists(dst) and not args.overwrite:
            print(f"  skip {s} (exists, use --overwrite)")
            skipped += 1
            continue
        shutil.copy2(os.path.join(src, s), dst)
        print(f"  {s} -> {dest}/")
        copied += 1

    print(f"\n{copied} copied, {skipped} skipped -> {dest}")
    if copied:
        print("these are copies — edit freely, they import from the "
              "installed package")


if __name__ == "__main__":
    main()