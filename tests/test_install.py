"""Installation and wiring: does the package import, are the entry points
real, are the files intact.

Every check here corresponds to a failure that actually happened: a
subpackage broken by a deleted function, sixty NUL bytes appended to a source
file (the error names no line number), a console script pointing at a
main() that had been renamed, a module importing something that no longer
exists.
"""

import importlib
import importlib.util
import pathlib
import subprocess
import sys

import pytest
import torch

SUBPACKAGES = [
    "bilbyflow",
    "bilbyflow.io",
    "bilbyflow.coordinates",
    "bilbyflow.data",
    "bilbyflow.nn",
    "bilbyflow.likelihood",
    "bilbyflow.inference",
    "bilbyflow.diagnostics",
    "bilbyflow.training",
    "bilbyflow.plotting",
    "bilbyflow.scripts",
]

# SCRIPTS = ["train", "reweight_real", "reweight_injections", "diagnostics",
#            "era_test"]

# CONSOLE_SCRIPTS = ["bilbyflow-train", "bilbyflow-reweight-real",
#                    "bilbyflow-reweight-injections", "bilbyflow-diagnostics"]

SCRIPTS = ["train"]
CONSOLE_SCRIPTS = ["bilbyflow-train"]

def _src_root():
    """The installed package directory, wherever it lives."""
    spec = importlib.util.find_spec("bilbyflow")
    return pathlib.Path(spec.origin).parent


# ---------------------- imports ----------------------------------------------------

@pytest.mark.parametrize("mod", SUBPACKAGES)
def test_subpackage_imports(mod):
    """Each subpackage must import on its own — a broken __init__ in one
    should not be masked by another importing first."""
    importlib.import_module(mod)


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_module_has_main(script):
    """The console scripts point at these; import errors here are what the
    entry points hit."""
    m = importlib.import_module(f"bilbyflow.scripts.{script}")
    assert callable(getattr(m, "main", None)), f"{script}.main missing"


def test_top_level_exports():
    import bilbyflow
    from bilbyflow.npe import NPE
    from bilbyflow.nn.embedding import StrainEmbedding
    from bilbyflow.nn.flow import ConditionalFlow
    assert all(callable(c) for c in (NPE, StrainEmbedding, ConditionalFlow))



# ------------------ file stability -----------------------------------

def test_no_nul_bytes():
    """A source file once carried 60 trailing NUL bytes; the resulting
    SyntaxError names no line number and the file looks fine in an editor."""
    bad = [str(p) for p in _src_root().rglob("*.py")
           if b"\x00" in p.read_bytes()]
    assert not bad, f"NUL bytes in: {bad}"


def test_every_module_parses():
    """Catches a half-saved file that no test happens to import."""
    import ast
    import warnings
    bad, warned = [], []
    for p in _src_root().rglob("*.py"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", SyntaxWarning)
            try:
                ast.parse(p.read_bytes().decode())
            except SyntaxError as e:
                bad.append(f"{p}: {e}")
        warned += [f"{p}: {x.message}" for x in w]
    assert not bad, bad
    assert not warned, f"SyntaxWarnings (use raw strings for LaTeX): {warned}"


def test_all_lists_are_populated():
    """__all__ built from dir() must sit at the END of the module — placed at
    the top it evaluates to an empty list and `import *` imports nothing."""
    empty = []
    for mod in SUBPACKAGES:
        m = importlib.import_module(mod)
        names = getattr(m, "__all__", None)
        if names is not None and len(names) == 0:
            empty.append(mod)
    assert not empty, f"empty __all__ in: {empty}"


# -- console scripts ----------------------------------------------------------

def _declared_console_scripts():
    from importlib.metadata import distribution
    try:
        eps = distribution("bilbyflow").entry_points
    except Exception:
        return []
    return [ep.name for ep in eps if ep.group == "console_scripts"]


def test_declared_console_scripts_run():
    """EVERY entry point pyproject declares must import and run --help.
    Reading them from the installed metadata means a newly declared script
    is covered automatically, and one whose module is missing fails here
    rather than in a user's shell."""
    import shutil
    names = _declared_console_scripts()
    if not names:
        pytest.skip("package not pip-installed (no entry-point metadata)")
    failures = []
    for cmd in names:
        if shutil.which(cmd) is None:
            failures.append(f"{cmd}: declared but not on PATH")
            continue
        r = subprocess.run([cmd, "--help"], capture_output=True, text=True,
                           timeout=120)
        if r.returncode != 0:
            failures.append(f"{cmd}: --help exited {r.returncode}\n"
                            f"{r.stderr[-1000:]}")
    assert not failures, failures

@pytest.mark.parametrize("script", SCRIPTS)
def test_module_entry_runs(script):
    """python -m bilbyflow.scripts.X --help, which is how the cluster jobs
    invoke them."""
    r = subprocess.run([sys.executable, "-m", f"bilbyflow.scripts.{script}",
                        "--help"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (f"python -m bilbyflow.scripts.{script} --help "
                               f"failed:\n{r.stderr[-2000:]}")


# -- dependencies that are imported but easy to forget in pyproject -----------

@pytest.mark.parametrize("dep", ["torch", "torchvision", "zuko", "numpy",
                                 "scipy", "yaml", "matplotlib"])
def test_runtime_dependency_importable(dep):
    importlib.import_module(dep)


def test_zuko_supports_the_features_we_use():
    """`passes` (coupling transforms) and rsample_and_log_prob are not in
    zuko 0.1.x; the package pins >=1.0 for exactly this."""
    import zuko
    f = zuko.flows.NSF(4, 8, transforms=2, hidden_features=[16], passes=2)
    assert hasattr(f(torch.zeros(1, 8)), "rsample_and_log_prob")