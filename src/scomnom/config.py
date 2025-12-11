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
        "adata.merged",
        description="Base name for merged output ('.zarr' will be appended)."
    )

    batch_key: Optional[str] = Field(
        None,
        description="Column name in metadata_tsv to use as batch/sample ID. "
                    "If None, it is inferred automatically from metadata header."
    )
    save_h5ad: Optional[bool] = Field(
        False,
        description="If set will save a h5ad file to <output_dir>.h5ad."
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


# -------------------------------
# QCFilterConfig
# -------------------------------
class QCFilterConfig(BaseModel):
    """
    Configuration for the qc-and-filter stage.

    - Loads a merged dataset from .zarr or .h5ad
    - Applies QC filters (min_genes, min_cells, max_pct_mt, min_cells_per_sample)
    - Writes filtered dataset as adata.filtered.zarr (+ optional .h5ad)
    """

    # Input: merged data from load-data
    input_path: Path = Field(
        ...,
        description="Path to merged dataset from load-data (.zarr directory or .h5ad file).",
    )

    # Output directory (directory, not file; must not end with .zarr)
    output_dir: Optional[Path] = Field(
        None,
        description="Directory for filtered output. Defaults to parent of input_path.",
    )

    # Batch / sample key (optional override)
    batch_key: Optional[str] = Field(
        None,
        description="Column in adata.obs to treat as batch/sample ID. "
                    "If None, io_utils.infer_batch_key() will choose.",
    )

    # QC thresholds
    min_cells: int = Field(
        3,
        description="[QC] Minimum cells per gene.",
    )
    min_genes: int = Field(
        500,
        description="[QC] Minimum genes per cell. Lower to ~200 for snRNA-seq.",
    )
    min_cells_per_sample: int = Field(
        20,
        description="[QC] Minimum cells per sample.",
    )
    max_pct_mt: float = Field(
        5.0,
        description="[QC] Max mitochondrial percentage. Increase to ~30–50% for snRNA-seq.",
    )

    # Plotting
    make_figures: bool = Field(
        True,
        description="Whether to generate QC plots.",
    )
    figdir_name: str = Field(
        "figures",
        description="Subdirectory under output_dir for figures.",
    )
    figure_formats: List[str] = Field(
        default_factory=lambda: ["png", "pdf"],
        description="Figure formats for plots.",
    )

    # Optional H5AD
    save_h5ad: bool = Field(
        False,
        description="Also write an .h5ad copy of the filtered dataset.",
    )

    # We keep the name fixed in practice, but expose as a constant-like field
    output_name: str = Field(
        "adata.filtered",
        description="Base name for filtered dataset ('.zarr' appended automatically).",
    )

    @property
    def figdir(self) -> Path:
        return self.output_dir / self.figdir_name

    @model_validator(mode="after")
    def _validate_paths(self):
        # ---- input_path: must exist and be .zarr dir OR .h5ad file ----
        if not self.input_path.exists():
            raise ValueError(f"input_path does not exist: {self.input_path}")

        if self.input_path.is_dir():
            # must be a .zarr directory
            if self.input_path.suffix != ".zarr":
                raise ValueError(
                    f"input_path directory must end with '.zarr', got: {self.input_path}"
                )
        else:
            # must be a file with .h5ad suffix
            if self.input_path.suffix.lower() != ".h5ad":
                raise ValueError(
                    f"input_path file must be a '.h5ad', got: {self.input_path}"
                )

        # ---- output_dir: default and checks ----
        if self.output_dir is None:
            # Parent of the input (directory or file)
            self.output_dir = self.input_path.parent

        # Must not look like a .zarr directory name
        if self.output_dir.suffix == ".zarr":
            raise ValueError(
                f"output_dir must be a plain directory, not a '.zarr' path: {self.output_dir}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        return self


# -------------------------------
# LoadAndFilterConfig
# -------------------------------
class LoadAndFilterConfig(BaseModel):
    """
    Unified configuration for the combined load-and-filter module.

    Performs:
      - Load raw / filtered / CellBender matrices
      - Per-sample QC and filtering (memory-safe)
      - Merging filtered samples
      - Metadata attachment
    """

    # ---------------------------------------------------------
    # Input sources (exactly one required)
    # ---------------------------------------------------------
    raw_sample_dir: Optional[Path] = None
    filtered_sample_dir: Optional[Path] = None
    cellbender_dir: Optional[Path] = None

    metadata_tsv: Path = Field(
        ...,
        description="TSV with per-sample metadata indexed by sample_id."
    )

    # The batch/sample key used throughout the pipeline
    batch_key: Optional[str] = Field(
        None,
        description="Column in metadata_tsv defining sample/batch. "
                    "If None, inferred from metadata header."
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    output_dir: Path = Field(
        ...,
        description="Directory for merged AnnData and figures."
    )

    output_name: str = Field(
        "adata.merged",
        description="Base name for merged dataset ('.zarr' added automatically)."
    )

    save_h5ad: bool = False

    # ---------------------------------------------------------
    # Compute
    # ---------------------------------------------------------
    n_jobs: int = Field(4, ge=1)

    # ---------------------------------------------------------
    # QC thresholds
    # ---------------------------------------------------------
    min_cells: int = 3              # gene filter
    min_genes: int = 500            # cell filter
    min_cells_per_sample: int = 20  # sample filter
    max_pct_mt: float = 5.0

    # ---------------------------------------------------------
    # File patterns
    # ---------------------------------------------------------
    raw_pattern: str = "*.raw_feature_bc_matrix"
    filtered_pattern: str = "*.filtered_feature_bc_matrix"
    cellbender_pattern: str = "*.cellbender_filtered.output"
    cellbender_h5_suffix: str = ".cellbender_out.h5"

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    make_figures: bool = True
    figdir_name: str = "figures"

    figure_formats: List[str] = Field(
        default_factory=lambda: ["png", "pdf"],
        description="Figure formats to save."
    )

    @validator("figure_formats", each_item=True)
    def validate_formats(cls, fmt):
        supported = Figure().canvas.get_supported_filetypes()
        fmt = fmt.lower()
        if fmt not in supported:
            raise ValueError(
                f"Unsupported figure format '{fmt}'. Supported: {sorted(supported)}"
            )
        return fmt

    @validator("output_name")
    def strip_zarr_suffix(cls, v):
        return v.replace(".zarr", "")

    @property
    def figdir(self) -> Path:
        return self.output_dir / self.figdir_name

    # ---------------------------------------------------------
    # Validators
    # ---------------------------------------------------------
    @model_validator(mode="after")
    def check_input_sources(self):
        sources = [
            self.raw_sample_dir,
            self.filtered_sample_dir,
            self.cellbender_dir,
        ]
        if sum(x is not None for x in sources) != 1:
            raise ValueError(
                "Exactly one of raw_sample_dir, filtered_sample_dir, or "
                "cellbender_dir must be provided."
            )
        return self

    @model_validator(mode="after")
    def prepare_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self


# ---------------------------------------------------------------------
# PROCESS + INTEGRATE CONFIG
# ---------------------------------------------------------------------
class ProcessAndIntegrateConfig(BaseModel):

    # ---- Input + output ----
    input_path: Path = Field(
        ...,
        description="Path to input dataset (Zarr or H5AD) produced by load-and-filter."
    )
    output_dir: Optional[Path] = Field(
        "results",
        description="Directory to store integration output + figures.",
    )
    output_name: Optional[str] = Field(
        "adata.integrated",
        description="Stem for integrated output (e.g. adata.integrated.zarr)."
    )
    save_h5ad: Optional[bool] = Field(
        False, description="Optional additional H5AD output."
    )

    # ---- Metadata keys ----
    batch_key: str = Field(
        "sample_id",
        description="Batch/sample key used for integration."
    )
    label_key: str = Field(
        "leiden",
        description="Cell-type/cluster labels used for scANVI + scIB metrics."
    )

    # ---- Integration method selection ----
    methods: Optional[List[str]] = Field(
        ["BBKNN", "scVI", "scANVI"],
        description="Integration methods to run and benchmark."
    )

    # ---- Benchmarking ----
    benchmark_n_jobs: int = Field(
        16,
        description="Parallel jobs for scIB benchmarking."
    )

    # ---- Doublet reuse options ----
    reuse_scvi_model: bool = Field(
        True,
        description="Reuse SCVI model trained for SOLO doublet detection."
    )
    scvi_refine_after_solo: bool = Field(
        default=True,
        description="After SOLO cleanup, perform a short SCVI fine-tuning pass on the filtered dataset."
    )
    scvi_refine_epochs: int = Field(
        default=15,
        description="Number of epochs for SCVI fine-tuning after SOLO cleanup."
    )

    # ---- QC / cleanup thresholds ----
    doublet_score_threshold: float = Field(
        0.25,
        description="Threshold for SOLO doublet_score (> threshold = doublet).",
    )

    max_pct_mt: float = Field(
        5.0,
        description="Maximum pct_counts_mt allowed after SOLO cleanup.",
    )

    min_cells_per_sample: int = Field(
        20,
        description="Minimum cells per sample required after SOLO cleanup.",
    )

    # ---- HVG / PCA ----
    n_top_genes: int = Field(
        3000,
        description="Number of highly variable genes for SC integration.",
    )

    # ---- Figures ----
    figdir_name: str = Field(
        "figures",
        description="Name of figure output directory inside output_dir."
    )

    figure_formats: List[str] = Field(
        default_factory=lambda: ["png", "pdf"],
        description="Figure formats to save."
    )
    max_pcs_plot: int = 50

    # ---- Logging ----
    logfile: Optional[Path] = Field(
        None,
        description="Write log to file instead of stdout only."
    )

    @validator("methods")
    def normalize_methods(cls, v):
        if v is None:
            return None
        return [m.strip() for m in v]


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
