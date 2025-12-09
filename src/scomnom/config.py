from __future__ import annotations
from pydantic import BaseModel, Field, validator, model_validator
from pathlib import Path
from typing import Optional, Dict, List
from matplotlib.figure import Figure
import multiprocessing


# -------------------------------
# LoadDataConfig
# -------------------------------
class LoadDataConfig(BaseModel):
    """
    Configuration for the *load-only* module:
    - Load RAW OR filtered OR CellBender matrices
    - Attach metadata
    - Write per-sample Zarr stores
    - Merge sequentially into final Zarr
    """

    # ---------------------------------------------------------
    # I/O paths
    # ---------------------------------------------------------
    raw_sample_dir: Optional[Path] = Field(
        None,
        description="Directory containing <sample>.raw_feature_bc_matrix folders."
    )
    filtered_sample_dir: Optional[Path] = Field(
        None,
        description="Directory containing <sample>.filtered_feature_bc_matrix folders."
    )
    cellbender_dir: Optional[Path] = Field(
        None,
        description="Directory containing <sample>.cellbender_filtered.output folders."
    )

    metadata_tsv: Path = Field(
        ...,
        description="TSV file with per-sample metadata indexed by sample ID."
    )

    output_dir: Path = Field(
        ..., description="Directory where merged Zarr will be written."
    )
    output_name: str = Field(
        "adata.loaded",
        description="Base name for merged output ('.zarr' will be appended)."
    )

    batch_key: Optional[str] = Field(
        None,
        description="Column name in metadata_tsv to use as batch/sample ID. "
                    "If None, it is inferred automatically from metadata header."
    )

    # ---------------------------------------------------------
    # Compute settings
    # ---------------------------------------------------------
    n_jobs: int = Field(
        4,
        ge=1,
        description="Parallel workers for reading individual samples & writing per-sample Zarrs."
    )

    # ---------------------------------------------------------
    # File pattern overrides (rarely needed)
    # ---------------------------------------------------------
    raw_pattern: str = "*.raw_feature_bc_matrix"
    filtered_pattern: str = "*.filtered_feature_bc_matrix"
    cellbender_pattern: str = "*.cellbender_filtered.output"
    cellbender_h5_suffix: str = ".cellbender_out.h5"

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    @validator("output_name")
    def ensure_suffix(cls, v: str):
        return v if v.endswith(".zarr") else f"{v}.zarr"

    @model_validator(mode="after")
    def check_exactly_one_source(self):
        """
        For load_data: exactly ONE of raw / filtered / cellbender must be provided.
        """
        sources = [
            self.raw_sample_dir,
            self.filtered_sample_dir,
            self.cellbender_dir,
        ]
        if sum(x is not None for x in sources) != 1:
            raise ValueError(
                "Exactly one of raw_sample_dir, filtered_sample_dir, or cellbender_dir must be set."
            )
        return self


class LoadAndQCConfig(BaseModel):
    # I/O
    raw_sample_dir: Optional[Path] = Field(None, description="Directory with 10x raw_feature_bc_matrix folders")
    filtered_sample_dir: Optional[Path] = Field(None,description="Directory with 10x filtered_feature_bc_matrix folders")
    cellbender_dir: Optional[Path] = Field(None, description="Directory with CellBender outputs")
    metadata_tsv: Optional[Path] = Field(None, description="TSV with per-sample metadata indexed by sample ID")
    output_dir: Path = Field(..., description="Directory for outputs (anndata + figures)")
    output_name: str = Field("adata.preprocessed", description="Output filename")
    save_h5ad: bool = False
    n_jobs: int = Field(4, ge=1)

    # QC thresholds
    min_cells: int = Field(3, ge=1, description="Minimum cells per gene")
    min_genes: int = Field(500, ge=1, description="Minimum genes per cell")
    min_cells_per_sample: int = Field(20, ge=0)
    max_pct_mt: float = Field(5.0, ge=0.0, description="Mitochondrial % threshold after CB + filters")
    doublet_score_threshold: float = 0.25

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
