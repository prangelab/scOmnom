"""Keep Linux scientific pins and the CUDA build selection consistent."""

from pathlib import Path
import re

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import yaml


ROOT = Path(__file__).resolve().parents[1]


def read_pins():
    pins = {}
    for line in (ROOT / "constraints_linux.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        req = Requirement(line)
        name = canonicalize_name(req.name)
        assert name not in pins
        assert not req.extras and req.marker is None and req.url is None
        specs = list(req.specifier)
        assert len(specs) == 1 and specs[0].operator == "=="
        assert "*" not in specs[0].version
        pins[name] = specs[0].version
    return pins


def test_linux_scientific_constraints_are_exact():
    pins = read_pins()
    assert {"torch", "numpy", "scipy", "numba", "llvmlite", "jax", "jaxlib",
            "numpyro", "flax", "optax", "lightning", "torchmetrics"} <= pins.keys()
    assert pins["decoupler"] == "2.1.3"
    assert pins["scib-metrics"] == "0.5.7"


def test_linux_manifest_matches_python_pins():
    manifest = yaml.safe_load((ROOT / "environment_linux.yml").read_text())
    pins = read_pins()
    pip = next(item["pip"] for item in manifest["dependencies"] if isinstance(item, dict))
    assert pip[0] == "-c constraints_linux.txt"
    for line in pip[1:]:
        if line.startswith("git+"):
            assert re.fullmatch(r"git\+https://github\.com/kaifuchenlab/MEBOCOST\.git@[0-9a-f]{40}", line)
        else:
            req = Requirement(line)
            assert str(req.specifier) == "==" + pins[canonicalize_name(req.name)]
    for line in manifest["dependencies"]:
        if not isinstance(line, str) or "=" not in line:
            continue
        name, version, *_ = line.split("::")[-1].split("=")
        name = canonicalize_name({"pytorch": "torch"}.get(name, name))
        if name in pins and name != "igraph":
            assert pins[name] == version


def test_linux_preserves_gpu_builds_and_channel_order():
    manifest = yaml.safe_load((ROOT / "environment_linux.yml").read_text())
    assert manifest["channels"] == ["pytorch", "nvidia", "conda-forge", "bioconda", "defaults"]
    for spec in (
        "pytorch::pytorch=2.5.1=py3.11_cuda12.4_cudnn9.1.0_0",
        "pytorch::torchvision=0.20.1=py311_cu124",
        "pytorch::torchaudio=2.5.1=py311_cu124",
        "pytorch::pytorch-cuda=12.4=hc786d27_7",
        "nvidia::cuda-version=13.1=hd92462c_3",
        "nvidia::cuda-opencl=13.1.115=h4f1e1d6_0",
    ):
        assert spec in manifest["dependencies"]


def test_linux_conda_pip_overlap_uses_same_versions():
    manifest = yaml.safe_load((ROOT / "environment_linux.yml").read_text())
    pins = read_pins()
    for name in ("requests", "umap-learn", "annotated-types", "imageio",
                 "lazy-loader", "tifffile", "typing-inspection"):
        assert f"{name}={pins[name]}" in manifest["dependencies"]


def test_linux_pins_match_accepted_transitive_versions():
    pins = read_pins()
    assert "python-igraph" not in pins
    for name, version in {"igraph": "1.0.0", "annotated-types": "0.8.0",
                          "imageio": "2.38.0", "lazy-loader": "0.6",
                          "tifffile": "2026.3.3", "typing-inspection": "0.4.4"}.items():
        assert pins[name] == version


def test_linux_guides_keep_constraints_for_project_install():
    for name in ("README.md", "docs/installation.md"):
        doc = (ROOT / name).read_text()
        assert "pip install -c constraints_linux.txt ." in doc
        assert "CONDA_CHANNEL_PRIORITY=flexible conda env create -f environment_linux.yml" in doc
