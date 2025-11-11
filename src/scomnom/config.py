from __future__ import annotations
from pydantic import BaseModel, Field, validator
from pathlib import Path
from typing import Optional, Dict, List

class LoadAndQCConfig(BaseModel):
    # I/O
    sample_dir: Path = Field(..., description="Directory with 10x raw folders: <sample>.raw_feature_bc_matrix")
    cellbender_dir: Optional[Path] = Field(None, description="Directory with CellBender outputs: <sample>.cellbender_filtered.output/")
    metadata_tsv: Optional[Path] = Field(None, description="TSV with per-sample metadata indexed by sample ID")
    output_dir: Path = Field(..., description="Directory for outputs (h5ad + figures)")
    output_name: str = Field("adata.preprocessed.h5ad", description="Output h5ad filename")

    # Parallelism
    n_jobs: int = Field(4, ge=1)

    # QC thresholds
    min_cells: int = Field(3, ge=1, description="Minimum cells per gene")
    min_genes: int = Field(200, ge=1, description="Minimum genes per cell")
    min_cells_per_sample: int = Field(20, ge=0)
    max_pct_mt: float = Field(5.0, ge=0.0, description="Mitochondrial % threshold after CB + filters")

    # Light prefilter before QC
    min_genes_prefilter: int = Field(50, ge=0, description="Remove barcodes with fewer detected genes before QC")
    min_umis_prefilter: int = Field(20, ge=0, description="Remove barcodes with fewer UMIs before QC")

    # Gene flags
    mt_prefix: str = Field("MT-", description="Prefix for mitochondrial genes")
    ribo_prefixes: List[str] = Field(default_factory=lambda: ["RPS", "RPL"])
    hb_regex: str = Field("^HB[^(P)]", description="Regex for hemoglobin genes")

    # HVG / PCA / neighbors
    n_top_genes: int = 2000
    max_pcs_plot: int = 50
    # Batch key
    batch_key: str = "sample_id"

    # Plots
    make_figures: bool = True
    figdir_name: str = "figures"

    # File patterns
    raw_pattern: str = "*.raw_feature_bc_matrix"
    cellbender_pattern: str = "*.cellbender_filtered.output"
    cellbender_h5_suffix: str = ".cellbender_out.h5"

    @property
    def figdir(self) -> Path:
        return self.output_dir / self.figdir_name

    @validator("output_name")
    def ensure_h5ad(cls, v: str) -> str:
        return v if v.endswith(".h5ad") else f"{v}.h5ad"
