from __future__ import annotations
import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import anndata as ad
import scanpy as sc

from .config import LoadAndQCConfig

LOGGER = logging.getLogger(__name__)


def find_raw_dirs(sample_dir: Path, pattern: str) -> List[Path]:
    return [Path(p) for p in glob.glob(str(sample_dir / pattern))]


def find_cellbender_dirs(cb_dir: Path, pattern: str) -> List[Path]:
    return [Path(p) for p in glob.glob(str(cb_dir / pattern))]


def read_raw_10x(raw_dir: Path) -> ad.AnnData:
    adata = sc.read_10x_mtx(str(raw_dir), var_names="gene_symbols", cache=True)
    adata.var_names_make_unique()
    return adata


def read_cellbender_h5(cb_folder: Path, sample: str, h5_suffix: str) -> Optional[ad.AnnData]:
    h5_path = cb_folder / f"{sample}{h5_suffix}"
    if not h5_path.exists():
        LOGGER.warning("CellBender output file missing for %s", sample)
        return None
    adata = sc.read_10x_h5(str(h5_path), gex_only=True)
    adata.var_names_make_unique()
    return adata


def save_adata(adata: ad.AnnData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(str(out_path), compression="gzip")
    LOGGER.info("Wrote %s", out_path)