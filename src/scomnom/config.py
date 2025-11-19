from __future__ import annotations
from pydantic import BaseModel, Field, validator
from pathlib import Path
from typing import Optional, Dict, List
from matplotlib.figure import Figure

class LoadAndQCConfig(BaseModel):
    # I/O
    raw_sample_dir: Optional[Path] = Field(None, description="Directory with 10x raw_feature_bc_matrix folders")
    filtered_sample_dir: Optional[Path] = Field(None,description="Directory with 10x filtered_feature_bc_matrix folders")
    cellbender_dir: Optional[Path] = Field(None, description="Directory with CellBender outputs")
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
    batch_key: Optional[str] = None

    # Plots
    make_figures: bool = True
    figdir_name: str = "figures"
    figure_formats: List[str] = Field(
        default_factory=lambda: ["png", "pdf"],
        description="Figure formats to save (e.g. ['png','pdf'])."
    )

    # File patterns
    raw_pattern: str = "*.raw_feature_bc_matrix"
    filtered_pattern: str = "*.filtered_feature_bc_matrix"
    cellbender_pattern: str = "*.cellbender_filtered.output"
    cellbender_h5_suffix: str = ".cellbender_out.h5"

    @property
    def figdir(self) -> Path:
        return self.output_dir / self.figdir_name

    @validator("output_name")
    def ensure_h5ad(cls, v: str) -> str:
        return v if v.endswith(".h5ad") else f"{v}.h5ad"

    @validator("figure_formats", each_item=True)
    def validate_formats(cls, fmt: str):
        supported = Figure().canvas.get_supported_filetypes()
        if fmt.lower() not in supported:
            raise ValueError(
                f"Unsupported figure format '{fmt}'. "
                f"Supported formats include: {', '.join(sorted(supported))}"
            )
        return fmt.lower()

    from pydantic import model_validator
    @model_validator(mode="after")
    def check_exactly_one_source(self):
        sources = [
            self.raw_sample_dir,
            self.filtered_sample_dir,
            self.cellbender_dir,
        ]
        if sum(x is not None for x in sources) != 1:
            raise ValueError(
                "Exactly one of raw_sample_dir, filtered_sample_dir, cellbender_dir must be set."
            )
        return self


class IntegrationConfig(BaseModel):
    # I/O
    input_path: Path = Field(..., description="Preprocessed h5ad from load_and_filter")
    output_path: Optional[Path] = Field(
        None,
        description="Output integrated h5ad. Defaults to <input_stem>.integrated.h5ad",
    )

    # Integration method control
    methods: Optional[List[str]] = Field(
        None,
        description=(
            "Integration methods to run. "
            "Supported: Scanorama, Harmony, scVI, scANVI. "
            "Default: all except scANVI."
        ),
    )

    # Keys
    batch_key: Optional[str] = Field(
        None, description="Batch/sample key in adata.obs (default: auto-detect)"
    )
    label_key: str = Field(
        "leiden",
        description="Label/cluster key in adata.obs used for scib-metrics & scANVI",
    )

    # Compute
    benchmark_n_jobs: int = Field(1, ge=1, description="Parallel workers for scib-metrics")

    # Logging
    logfile: Optional[Path] = Field(
        None,
        description="Optional log file path; defaults to <input_dir>/integration.log",
    )

    @validator("methods")
    def normalize_methods(cls, v):
        if v is None:
            return None
        return [m.strip() for m in v]


class CellQCConfig(BaseModel):
    # Paths
    output_dir: Path

    # Figure handling
    figdir_name: str = "figures"
    figure_formats: List[str] = ["png", "pdf"]

    # 10x directory patterns (same defaults as LoadAndQCConfig for compatibility)
    raw_pattern: str = "*.raw_feature_bc_matrix"
    filtered_pattern: str = "*.filtered_feature_bc_matrix"
    cellbender_pattern: str = "*.cellbender_filtered.output"
    cellbender_h5_suffix: str = ".cellbender_out.h5"

    # These are assigned dynamically by cell-qc
    raw_sample_dir: Optional[Path] = None
    filtered_sample_dir: Optional[Path] = None
    cellbender_dir: Optional[Path] = None

    # Metadata not used in cell-qc, but included for compatibility with io_utils
    metadata_tsv: Optional[Path] = None
    batch_key: Optional[str] = None

    make_figures: bool = True

    @validator("figure_formats", each_item=True)
    def validate_formats(cls, fmt):
        supported = Figure().canvas.get_supported_filetypes()
        fmt = fmt.lower()
        if fmt not in supported:
            raise ValueError(f"Unsupported figure format '{fmt}'. Supported: {', '.join(sorted(supported))}")
        return fmt

