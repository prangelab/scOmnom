"""Validate the Linux artifact lock and its separate pip installation stage."""

import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import yaml


ROOT = Path(__file__).resolve().parents[1]


def read_lock():
    return json.loads((ROOT / "locks/linux-64-conda.artifacts.json").read_text())


def test_explicit_lock_matches_artifact_manifest():
    lock = read_lock()
    assert lock["schema_version"] == 1
    assert lock["platform"] == "linux-64"
    packages = lock["packages"]
    assert lock["artifact_count"] == len(packages) == 688
    assert len({p["name"] for p in packages}) == len(packages)
    assert len({p["url"] for p in packages}) == len(packages)
    lines = (ROOT / "locks/linux-64-conda.explicit.txt").read_text().splitlines()
    entries = [line for line in lines if line and not line.startswith("#")]
    assert entries == ["@EXPLICIT", *(p["url"] + "#" + p["md5"] for p in packages)]


def test_lock_contains_only_reviewed_public_channels_and_valid_hashes():
    for item in read_lock()["packages"]:
        assert set(item) == {"name", "version", "build", "url", "sha256", "md5"}
        url = urlsplit(item["url"])
        assert url.scheme == "https" and url.port is None
        assert not (url.username or url.password or url.query or url.fragment)
        parts = url.path.strip("/").split("/")
        if url.hostname == "conda.anaconda.org":
            assert len(parts) == 3
            assert parts[0] in {"pytorch", "nvidia", "conda-forge", "bioconda"}
        else:
            assert url.hostname == "repo.anaconda.com"
            assert len(parts) == 4 and parts[:2] in (["pkgs", "main"], ["pkgs", "r"])
        assert parts[-2] in {"linux-64", "noarch"}
        basename = unquote(parts[-1])
        expected = f'{item["name"]}-{item["version"]}-{item["build"]}'
        assert basename in {expected + ".conda", expected + ".tar.bz2"}
        assert re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        assert re.fullmatch(r"[0-9a-f]{32}", item["md5"])


def test_lock_matches_declared_conda_versions_builds_and_channels():
    packages = {p["name"]: p for p in read_lock()["packages"]}
    manifest = yaml.safe_load((ROOT / "environment_linux.yml").read_text())
    for spec in manifest["dependencies"]:
        if not isinstance(spec, str):
            continue
        channel, _, spec = spec.rpartition("::")
        name, *pins = spec.split("=")
        item = packages[name]
        if pins:
            assert item["version"] == pins[0], name
        if len(pins) > 1:
            assert item["build"] == pins[1], name
        if channel:
            assert urlsplit(item["url"]).path.split("/")[1] == channel
    assert packages["r-base"]["version"] == "4.5.3"


def test_separate_pip_stage_matches_yaml_exactly():
    manifest = yaml.safe_load((ROOT / "environment_linux.yml").read_text())
    expected = next(item["pip"] for item in manifest["dependencies"] if isinstance(item, dict))
    lines = (ROOT / "requirements_linux.txt").read_text().splitlines()
    assert [line for line in lines if line and not line.startswith("#")] == expected
    assert expected[0] == "-c constraints_linux.txt"


def test_guides_include_both_installation_stages():
    for name in ("README.md", "docs/installation.md"):
        doc = (ROOT / name).read_text()
        assert "micromamba create -n scOmnom_env --file locks/linux-64-conda.explicit.txt" in doc
        assert "python -m pip install -r requirements_linux.txt" in doc
        assert "pip install -c constraints_linux.txt ." in doc
