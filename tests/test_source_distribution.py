"""Verify that source releases retain their platform installation inputs."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
INSTALLATION_FILES = (
    "environment.yml", "environment_linux.yml", "environment_macos.yml",
    "constraints_linux.txt", "constraints_macos.txt", "docs/installation.md",
    "changelog.md",
)


def test_sdist_contains_exact_installation_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", "MANIFEST.in", *INSTALLATION_FILES):
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    shutil.copytree(ROOT / "src" / "scomnom", source / "src" / "scomnom",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    subprocess.run(
        [sys.executable, "-c", "from setuptools.build_meta import build_sdist; build_sdist('dist')"],
        cwd=source, env={**os.environ, "PYTHONPATH": ""},
        check=True, capture_output=True, text=True,
    )
    archives = list((source / "dist").glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0]) as archive:
        prefix = archive.getnames()[0].split("/")[0]
        for name in INSTALLATION_FILES:
            member = archive.extractfile(f"{prefix}/{name}")
            assert member is not None, name
            assert member.read() == (ROOT / name).read_bytes(), name
