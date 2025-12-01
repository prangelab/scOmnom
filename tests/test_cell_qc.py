# tests/test_cell_qc.py

import numpy as np
import pytest
import scanpy as sc
import pandas as pd
from pathlib import Path

from scomnom.cell_qc import _compute_cell_counts, run_cell_qc
from scomnom.config import CellQCConfig


# -----------------------------------------------------------------------------
# Synthetic AnnData
# -----------------------------------------------------------------------------
def synthetic_adata(n=50, g=10):
    X = np.random.poisson(1.0, (n, g))
    adata = sc.AnnData(X)
    adata.obs_names = [f"cell{i}" for i in range(n)]
    adata.var_names = [f"gene{i}" for i in range(g)]
    return adata


# -----------------------------------------------------------------------------
# Fixtures: mock all IO + plotting + reads
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_plots(monkeypatch):
    import scomnom.plot_utils as pu

    monkeypatch.setattr(pu, "setup_scanpy_figs", lambda *a, **k: None)
    monkeypatch.setattr(pu, "plot_read_comparison", lambda *a, **k: None)
    monkeypatch.setattr(pu, "save_multi", lambda *a, **k: None)

    return True


@pytest.fixture
def mock_io(monkeypatch):
    import scomnom.io_utils as io

    # RAW loader → returns raw_map + filtered + unfiltered read dicts
    monkeypatch.setattr(
        io,
        "load_raw_data",
        lambda cfg, record_pre_filter_counts=False, plot_dir=None: (
            {"S1": synthetic_adata(30), "S2": synthetic_adata(50)},
            {"S1": 3000, "S2": 5000},   # raw_filtered_reads
            {"S1": 3500, "S2": 5500},   # raw_unfiltered_reads
        ),
    )

    # Filtered loader
    monkeypatch.setattr(
        io,
        "load_filtered_data",
        lambda cfg: (
            {"S1": synthetic_adata(40), "S2": synthetic_adata(60)},
            {"S1": 4000, "S2": 6000},   # reads
        ),
    )

    # CellBender loader
    monkeypatch.setattr(
        io,
        "load_cellbender_data",
        lambda cfg: (
            {"S1": synthetic_adata(20), "S2": synthetic_adata(80)},
            {"S1": 2000, "S2": 8000},   # reads
        ),
    )

    return True


# -----------------------------------------------------------------------------
# Minimal config factory
# -----------------------------------------------------------------------------
def make_cfg(tmp_path, **kwargs):
    return CellQCConfig(
        output_dir=tmp_path,
        figdir_name="figs",
        figure_formats=["png"],
        raw_sample_dir=None,
        filtered_sample_dir=None,
        cellbender_dir=None,
        **kwargs,
    )


# -----------------------------------------------------------------------------
# Basic internal test: _compute_cell_counts
# -----------------------------------------------------------------------------
def test_compute_cell_counts():
    smap = {
        "A": synthetic_adata(10),
        "B": synthetic_adata(35),
    }
    out = _compute_cell_counts(smap)
    assert out == {"A": 10, "B": 35}


# -----------------------------------------------------------------------------
# Case 1: RAW present → all comparisons
# -----------------------------------------------------------------------------
def test_run_cell_qc_raw(tmp_path, mock_plots, mock_io):
    cfg = make_cfg(
        tmp_path,
        raw_sample_dir=tmp_path / "raw",  # triggers load_raw_data mock
    )
    (tmp_path / "raw").mkdir(exist_ok=True)

    # Should run without error
    run_cell_qc(cfg)


# -----------------------------------------------------------------------------
# Case 2: No RAW, but CR + CB present
# -----------------------------------------------------------------------------
def test_run_cell_qc_cr_and_cb(tmp_path, mock_plots, mock_io):
    cfg = make_cfg(
        tmp_path,
        raw_sample_dir=None,
        filtered_sample_dir=tmp_path / "cr",
        cellbender_dir=tmp_path / "cb",
    )
    (tmp_path / "cr").mkdir(exist_ok=True)
    (tmp_path / "cb").mkdir(exist_ok=True)

    run_cell_qc(cfg)


# -----------------------------------------------------------------------------
# Case 3: Only one source → fallback cell-count barplot
# -----------------------------------------------------------------------------
def test_run_cell_qc_single_source(tmp_path, mock_plots, mock_io):
    cfg = make_cfg(
        tmp_path,
        raw_sample_dir=None,
        filtered_sample_dir=tmp_path / "cr",
        cellbender_dir=None,
    )
    (tmp_path / "cr").mkdir(exist_ok=True)

    run_cell_qc(cfg)
