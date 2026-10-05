"""Keep the Apple Silicon manifest and its Python constraints consistent."""

from pathlib import Path
import re

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_macos_constraints_are_unique_exact_pins():
    seen = set()
    for line in (ROOT / "constraints_macos.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        req = Requirement(line)
        name = canonicalize_name(req.name)
        assert name not in seen
        seen.add(name)
        assert not req.extras and req.marker is None and req.url is None
        specs = list(req.specifier)
        assert len(specs) == 1 and specs[0].operator == "=="
        assert "*" not in specs[0].version
    assert {"torch", "numpy", "jax", "jaxlib", "numpyro", "flax", "optax"} <= seen


def test_macos_manifest_uses_adjacent_constraints_and_matching_pip_pins():
    manifest = yaml.safe_load((ROOT / "environment_macos.yml").read_text())
    pip = next(item["pip"] for item in manifest["dependencies"] if isinstance(item, dict))
    assert pip[0] == "-c constraints_macos.txt"
    pins = {}
    for line in (ROOT / "constraints_macos.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            req = Requirement(line)
            pins[canonicalize_name(req.name)] = str(req.specifier)
    for line in pip[1:]:
        if line.startswith("git+"):
            assert re.fullmatch(r"git\+https://github\.com/kaifuchenlab/MEBOCOST\.git@[0-9a-f]{40}", line)
        else:
            req = Requirement(line)
            assert str(req.specifier) == pins[canonicalize_name(req.name)]
    for line in manifest["dependencies"]:
        if not isinstance(line, str) or "=" not in line:
            continue
        name, version, *_ = line.split("::")[-1].split("=")
        name = {"pytorch": "torch"}.get(name, name)
        # Conda's igraph is the C library, not the Python distribution.
        if name == "igraph":
            continue
        if canonicalize_name(name) in pins:
            assert pins[canonicalize_name(name)] == f"=={version}"


def test_installation_guides_apply_macos_constraints_to_project_install():
    for name in ("README.md", "docs/installation.md"):
        assert "pip install -c constraints_macos.txt ." in (ROOT / name).read_text()
