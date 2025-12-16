from __future__ import annotations

from numba.core.types import Boolean
from pydantic import BaseModel, Field, validator, model_validator
from pathlib import Path
from typing import Optional, Dict, List, Literal
from matplotlib.figure import Figure
import multiprocessing


class LoadAndFilterConfig(BaseModel):

    # ---- Input ----
    raw_sample_dir: Optional[Path] = None
    filtered_sample_dir: Optional[Path] = None
    cellbender_dir: Optional[Path] = None

    metadata_tsv: Optional[Path] = None
    batch_key: Optional[str] = None

    # ---- Output ----
    output_dir: Path
    output_name: str = "adata.filtered"
    save_h5ad: bool = False

    # ---- Compute ----
    n_jobs: int = 4

    # ---- QC ----
    min_cells: int = 3
    min_genes: int = 500
    min_cells_per_sample: int = 20
    max_pct_mt: float = 5.0
    n_top_genes: int = 2000

    # ---- Doublets (SOLO) ----
    doublet_mode: Literal["fixed", "rate"] = "rate"
    doublet_score_threshold: float = 0.25
    expected_doublet_rate: float = 0.1
    apply_doublet_score: Optional[bool] = None
    apply_doublet_score_path: Optional[Path] = "results/adata.merged.zarr"

    # ---- Patterns ----
    raw_pattern: str = "*.raw_feature_bc_matrix"
    filtered_pattern: str = "*.filtered_feature_bc_matrix"
    cellbender_pattern: str = "*.cellbender_filtered.output"
    cellbender_h5_suffix: str = ".cellbender_out_filtered.h5"
    cellbender_barcode_suffix: str = Field(
        ".cellbender_out_cell_barcodes.csv",
        description="Suffix for CellBender barcode file "
                    "(e.g. '_cellbender_out_cell_barcodes.csv')."
    )

    # ---- Figures ----
    make_figures: bool = True
    figdir_name: str = "figures"
    figure_formats: List[str] = Field(default_factory=lambda: ["png", "pdf"])

    # ---- Logging ----
    logfile: Optional[Path] = None

    @property
    def figdir(self) -> Path:
        return self.output_dir / self.figdir_name

    # ---- Validators ----
    @model_validator(mode="after")
    def check_inputs(self):
        # --apply-doublet-score mode
        if self.apply_doublet_score is not None:
            # metadata not required
            return self

        # Normal modes require metadata
        if self.metadata_tsv is None:
            raise ValueError(
                "metadata_tsv is required unless --apply-doublet-score is used"
            )

        if self.filtered_sample_dir is not None:
            if self.raw_sample_dir or self.cellbender_dir:
                raise ValueError("filtered cannot be combined with raw or cellbender")

        if self.raw_sample_dir is None and self.filtered_sample_dir is None:
            raise ValueError("Must provide raw_sample_dir or filtered_sample_dir")

        return self

    @model_validator(mode="after")
    def check_doublet_config(self):
        if self.doublet_mode == "fixed":
            if not (0 < self.doublet_score_threshold < 1):
                raise ValueError("doublet_score_threshold must be in (0, 1)")
        elif self.doublet_mode == "rate":
            if not (0 < self.expected_doublet_rate < 0.5):
                raise ValueError("expected_doublet_rate must be in (0, 0.5)")
        else:
            raise ValueError("doublet_mode must be 'fixed' or 'rate'")
        return self


class ProcessAndIntegrateConfig(BaseModel):

    input_path: Path
    output_dir: Path
    output_name: str = "adata.integrated"
    save_h5ad: bool = False

    batch_key: Optional[str] = None
    label_key: str = "leiden"

    methods: Optional[List[str]] = None
    benchmark_n_jobs: int = 16

    figdir_name: str = "figures"
    figure_formats: List[str] = Field(default_factory=lambda: ["png", "pdf"])

    logfile: Optional[Path] = None

    @validator("methods")
    def normalize_methods(cls, v):
        if v is None:
            return None
        return [m.lower() for m in v]


# ---------------------------------------------------------------------
# CLUSTER AND ANNOTATE CONFIG
# ---------------------------------------------------------------------
class ClusterAnnotateConfig(BaseModel):
    # I/O
    input_path: Path = Field(..., description="Integrated h5ad from the integration step")
    output_path: Optional[Path] = Field(
        None,
        description="Output clustered/annotated h5ad. Defaults to <input>.clustered.annotated.h5ad",
    )

    # Embedding / batch / labels
    embedding_key: str = Field(
        "X_integrated",
        description="Key in .obsm to use for neighbors / silhouette (default: X_integrated from integration)",
    )
    batch_key: Optional[str] = Field(
        None,
        description="Batch/sample key in adata.obs (default: auto-detect)",
    )
    label_key: str = Field(
        "leiden",
        description="Final cluster label key in adata.obs",
    )
    final_auto_idents_key: str = "final_auto_idents"

    # bioARI
    bio_guided_clustering: bool = True
    w_hom: float = 0.15
    w_frag: float = 0.10
    w_bioari: float = 0.15

    # -------------------------------
    # ssGSEA defaults
    # -------------------------------
    run_ssgsea: bool = True

    # Default gene sets: Hallmark + Reactome (MSigDB)
    # User may override with CLI like:
    #   --ssgsea-gene-sets /path/to/hallmark.gmt,/path/to/reactome.gmt
    ssgsea_gene_sets: List[str] = Field(
        default_factory=lambda: ["HALLMARK", "REACTOME"],
        description="Gene set collections to use for ssGSEA."
    )

    # Expression source: raw counts if present
    ssgsea_use_raw: bool = True

    # Gene set size filters
    ssgsea_min_size: int = 10
    ssgsea_max_size: int = 500

    # Rank-based normalization is standard for ssGSEA
    ssgsea_sample_norm_method: str = "rank"

    # Parallel workers — auto-detect CPU cores − 1 (but minimum = 1)
    ssgsea_nproc: int = Field(
        default_factory=lambda: max(1, multiprocessing.cpu_count() - 1),
        description="Number of parallel worker processes for ssGSEA."
    )

    # Figure handling
    figdir_name: str = "figures"
    make_figures: bool = True
    figure_formats: List[str] = Field(
        default_factory=lambda: ["png", "pdf"],
        description="Figure formats to save",
    )

    # Resolution sweep
    res_min: float = Field(0.1, ge=0.0)
    res_max: float = Field(2.5, ge=0.0)
    n_resolutions: int = Field(25, ge=2)
    penalty_alpha: float = Field(
        0.02,
        ge=0.0,
        description="Penalty term for number of clusters in penalized silhouette",
    )

    # Stability analysis
    stability_repeats: int = Field(5, ge=1)
    subsample_frac: float = Field(
        0.8,
        gt=0.0,
        le=1.0,
        description="Fraction of cells to use per subsample for stability ARI",
    )
    random_state: int = 42

    tiny_cluster_size: int = 20
    min_cluster_size: int = 20
    min_plateau_len: int = 3
    max_cluster_jump_frac: float = 0.4
    stability_threshold: float = 0.85

    w_stab: float = 0.50
    w_sil: float = 0.35
    w_tiny: float = 0.15

    # Celltypist annotation
    celltypist_model: Optional[str] = Field(
        "Immune_All_Low.pkl",
        description="Path or name of CellTypist model (.pkl file). If None, skip annotation.",
    )
    celltypist_majority_voting: bool = True
    celltypist_label_key: str = "celltypist_label"
    final_label_key: str = Field(
        "leiden",
        description="Final annotation label. "
                    "If CellTypist is run, this will be overwritten to celltypist_cluster_label."
    )

    # cluster-collapsed CellTypist label
    celltypist_cluster_label_key: str = "celltypist_cluster_label"

    annotation_csv: Optional[Path] = None
    available_models: Optional[List[Dict[str, str]]] = None

    # Logging
    logfile: Optional[Path] = None

    @property
    def figdir(self) -> Path:
        base = self.output_path.parent if self.output_path is not None else self.input_path.parent
        return base / self.figdir_name

    @model_validator(mode="after")
    def check_resolution_range(self):
        if self.res_min >= self.res_max:
            raise ValueError("res_min must be < res_max")
        return self

    @validator("figure_formats", each_item=True)
    def validate_formats(cls, fmt: str):
        supported = Figure().canvas.get_supported_filetypes()
        fmt = fmt.lower()
        if fmt not in supported:
            raise ValueError(
                f"Unsupported figure format '{fmt}'. "
                f"Supported formats include: {', '.join(sorted(supported))}"
            )
        return fmt
