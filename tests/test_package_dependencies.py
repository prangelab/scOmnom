"""Keep Python graph dependency metadata distinct from Conda package names."""

from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import yaml


ROOT = Path(__file__).resolve().parents[1]


def project_requirements():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return {canonicalize_name(req.name): req
            for req in map(Requirement, project["project"]["dependencies"])}


def test_python_requires_igraph_without_legacy_metapackage():
    requirements = project_requirements()
    assert "igraph" in requirements
    assert "python-igraph" not in requirements
    assert requirements["igraph"].url is None
    assert requirements["igraph"].marker is None


def test_tested_igraph_version_satisfies_python_requirement():
    assert project_requirements()["igraph"].specifier.contains("1.0.0")


def test_conda_keeps_python_binding_package_name():
    for name in ("environment.yml", "environment_linux.yml", "environment_macos.yml"):
        manifest = yaml.safe_load((ROOT / name).read_text())
        names = {line.split("::")[-1].split("=")[0]
                 for line in manifest["dependencies"] if isinstance(line, str)}
        assert "python-igraph" in names
        assert "igraph" in names
